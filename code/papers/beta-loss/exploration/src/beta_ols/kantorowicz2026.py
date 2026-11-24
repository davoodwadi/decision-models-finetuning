"""Kantorowicz-Reznichenko et al. (2026) contractual-rights conjoint (`computer_choice`, `laundry_choice`).

Paper estimand: Figure 3 Panels A and B (`dataverse_files/script_final.R`, `amces_computers.csv`,
`amces_laundry_machines.csv`), two separate respondent-clustered OLS AMCE regressions (`cregg::amce`) on stacked
profile rows (each displayed product is a row, y = 1 if chosen): 9 attributes / 19 non-reference AMCEs for laptop
computers (13,421 tasks = 26,842 profile rows across 2,238 respondents) and 9 attributes / 19 non-reference AMCEs for
laundry machines (13,402 tasks = 26,804 profile rows across 2,239 respondents). Both profiles' attributes are
independently randomized, so own-attribute rows are unbiased.
Our regression: identical. Two regressions ("kantorowicz2026/computers", "kantorowicz2026/laundry"), 38 AMCEs total.
`split_by_task` splits the module into two pure-product pseudo-datasets (`kantorowicz2026/computer_choice`,
`kantorowicz2026/laundry_choice`) for training, following the `berke2023` pattern.
Coding decisions: data/kantorowicz2026/DECISIONS.md.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.kantorowicz2026 import KANTOROWICZ2026_PERSONA_COLS, load_kantorowicz2026_df
from data.kantorowicz2026.stimulus import (
    COMPUTER_ATTRIBUTE_SPECS,
    COMPUTER_CHOICE_OPTIONS,
    LAUNDRY_ATTRIBUTE_SPECS,
    LAUNDRY_CHOICE_OPTIONS,
)

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    regression_rows_from_parts,
)

NAME = "kantorowicz2026"
PERSONA_COLS = KANTOROWICZ2026_PERSONA_COLS

# 1. Levels: two independent 9-attribute product conjoints; each attribute's first level in `stimulus.py` is the
#    published `cregg::amce` reference level. Short attribute / level names keep beta tables readable and align the
#    four shared legal-right attributes across both products in `wide_ols_table`.
TASKS = {
    "computer_choice": f"{NAME}/computers",
    "laundry_choice": f"{NAME}/laundry",
}
TASK_LABELS = {
    f"{NAME}/computers": "Computers",
    f"{NAME}/laundry": "Laundry",
}
EXPECTED_OPTIONS_BY_TASK = {
    "computer_choice": [text for _, text in COMPUTER_CHOICE_OPTIONS],
    "laundry_choice": [text for _, text in LAUNDRY_CHOICE_OPTIONS],
}
SPECS_BY_TASK = {
    "computer_choice": COMPUTER_ATTRIBUTE_SPECS,
    "laundry_choice": LAUNDRY_ATTRIBUTE_SPECS,
}
CODE_PREFIX_BY_TASK = {
    "computer_choice": "c",
    "laundry_choice": "l",
}

SHORT_ATTR_NAMES = {
    "Price": "Price",
    "Brand": "Brand",
    "RAM": "RAM",
    "Storage.Capacity": "Storage",
    "Processor": "Processor",
    "Depth": "Depth",
    "Noise": "Noise",
    "Load.Capacity": "Load capacity",
    "Warranty": "Warranty",
    "Privacy": "Privacy",
    "Liability": "Liability",
    "Conflict.Resolution": "Conflict resolution",
}

SHORT_LEVEL_NAMES = {
    "Collected, processed and stored": "No privacy",
    "Collected and processed for marketing purposes": "No privacy (marketing)",
    "Collected and processed for the development of more sustainable products": "No privacy (sustainability)",
    "Not collected nor processed": "Full privacy",
}


def _short_level(display_val: str) -> str:
    return SHORT_LEVEL_NAMES[display_val] if display_val in SHORT_LEVEL_NAMES else display_val


# 2. Design: [Intercept, one dummy per non-reference level] for each product regression (19 targets each, 38 total)
TARGET_COLUMNS_BY_GROUP: dict[str, list[str]] = {}
DESIGN_COLUMNS_BY_GROUP: dict[str, list[str]] = {}
for _task_v, _group in TASKS.items():
    _targets: list[str] = []
    for _rds_col, _prompt_label, _level_map in SPECS_BY_TASK[_task_v]:
        _short_attr = SHORT_ATTR_NAMES[_rds_col]
        _display_levels = list(_level_map.values())
        _ref = _short_level(_display_levels[0])
        for _display_val in _display_levels[1:]:
            _targets.append(f"{_short_attr}: {_short_level(_display_val)} vs {_ref}")
    TARGET_COLUMNS_BY_GROUP[_group] = _targets
    DESIGN_COLUMNS_BY_GROUP[_group] = ["Intercept", *_targets]


def _parse_qid_codes(qid: str, task_v: str) -> tuple[str, str]:
    # `kantorowicz2026:computer_choice:c012012010_c201210321` -> ("c012012010", "c201210321")
    parts = qid.split(":")
    if len(parts) != 3 or parts[0] != NAME or parts[1] != task_v:
        raise ValueError(f"Unexpected kantorowicz2026 qid {qid!r} for task {task_v!r}")
    codes = parts[2].split("_")
    if len(codes) != 2:
        raise ValueError(f"Expected two profile codes in qid {qid!r}, got {codes}")
    return codes[0], codes[1]


def design_row(task_v: str, description: str, expected_code: str, context: str) -> list[float]:
    # One displayed product profile description -> design row, cross-checked against its `qid` compact code
    specs = SPECS_BY_TASK[task_v]
    prefix = CODE_PREFIX_BY_TASK[task_v]
    if not expected_code.startswith(prefix) or len(expected_code) != 1 + len(specs):
        raise ValueError(f"{context}: invalid profile code {expected_code!r} for {task_v!r}")

    clauses = description.split("; ")
    if len(clauses) != len(specs):
        raise ValueError(f"{context}: expected {len(specs)} attributes in description, got {len(clauses)}")
    parsed: dict[str, str] = {}
    for clause in clauses:
        if ": " not in clause:
            raise ValueError(f"{context}: malformed attribute clause {clause!r}")
        label, value = clause.split(": ", 1)
        if label in parsed:
            raise ValueError(f"{context}: duplicate attribute label {label!r}")
        parsed[label] = value

    row = [1.0]
    for attr_idx, (rds_col, prompt_label, level_map) in enumerate(specs):
        if prompt_label not in parsed:
            raise KeyError(f"{context}: missing attribute {prompt_label!r} ({rds_col}) in option description")
        display_val = parsed[prompt_label]
        display_levels = list(level_map.values())
        if display_val not in display_levels:
            raise ValueError(f"{context}: invalid {rds_col} level {display_val!r} (expected one of {display_levels})")
        level_idx = display_levels.index(display_val)
        code_digit = int(expected_code[1 + attr_idx])
        if level_idx != code_digit:
            raise ValueError(
                f"{context}: description {rds_col}={display_val!r} (idx {level_idx}) != qid code digit {code_digit}"
            )
        for non_ref_idx in range(1, len(display_levels)):
            row.append(float(level_idx == non_ref_idx))
    return row


# 3. Choice rows: one row per answered paired choice task (computers first, then laundry machines)
def load_choice_df() -> pd.DataFrame:
    kantorowicz_df = load_kantorowicz2026_df()
    choice_df = kantorowicz_df[kantorowicz_df["v"].isin(TASKS)].reset_index(drop=True)
    if len(choice_df) == 0:
        raise ValueError(f"No {sorted(TASKS)} rows found in kantorowicz2026_df")
    return choice_df


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    each task gives 2 profile rows; each row's outcome is that profile's own choice probability
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    num_options = []
    y_values = []
    outcome_rows = []
    for t_idx, row in enumerate(r_df.itertuples(index=False)):
        task_v = str(row.v)
        if task_v not in TASKS:
            raise ValueError(f"{row.CASEID} task {t_idx}: unexpected task variable {task_v!r}")
        context = f"{row.CASEID} {task_v} task {t_idx}"
        expected_opts = EXPECTED_OPTIONS_BY_TASK[task_v]
        if list(row.option_texts) != expected_opts:
            raise ValueError(f"{context}: expected {expected_opts}, got {row.option_texts}")
        one_hot = check_one_hot([float(v) for v in row.one_hot], 2, context)
        if len(row.option_descriptions) != 2:
            raise ValueError(f"{context}: expected 2 product profiles, got {len(row.option_descriptions)}")
        code_a, code_b = _parse_qid_codes(str(row.qid), task_v)
        group = TASKS[task_v]
        first_option = len(y_values)
        for opt_idx, (description, code) in enumerate(zip(row.option_descriptions, (code_a, code_b), strict=True)):
            outcome_rows.append(
                OutcomeRow(
                    group=group,
                    x=design_row(task_v, description, code, f"{context} opt {opt_idx}"),
                    option_idx=[first_option + opt_idx],
                    weights=[1.0],
                )
            )
        num_options.append(2)
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (sigma from the full human data; 0.5 by design of 2-option rows)
def regression_rows(choice_df: pd.DataFrame) -> pd.DataFrame:
    return regression_rows_from_parts(choice_df, respondent_beta_part, DESIGN_COLUMNS_BY_GROUP)


def regression_specs(rows: pd.DataFrame) -> dict[str, RegressionSpec]:
    # Specs for the product groups present in `rows` (both on the full data, one on a single-product pseudo dataset)
    present_groups = set(rows["group"])
    specs = {}
    for group in TASKS.values():
        if group not in present_groups:
            continue
        target_cols = TARGET_COLUMNS_BY_GROUP[group]
        specs[group] = RegressionSpec(
            name=group,
            design_columns=DESIGN_COLUMNS_BY_GROUP[group],
            target_columns=target_cols,
            target_labels=[f"{TASK_LABELS[group]} | {col}" for col in target_cols],
            sigma=outcome_sigma(rows.loc[rows["group"] == group, "y"].to_numpy()),
            cov_type="cluster",
        )
    return specs

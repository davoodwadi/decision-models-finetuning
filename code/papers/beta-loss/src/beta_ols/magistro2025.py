"""Magistro et al. (2025) generative AI vs. offshoring conjoint (`conjoint_ai`, `conjoint_offshoring`).

Paper estimand: Online Supporting Information Figures A1 and B1 (`dataverse_files/main.R`, `appendix.R`, `cregg::cj`
AMCEs), respondent-clustered OLS regressions on the 5-point company-decision support score (`qtable_1..4`, 1 = Strongly
oppose to 5 = Strongly support) split by country (US vs. Canada) and economic shock arm (AI vs. Offshoring).
Our regression: 4 separate respondent-clustered OLS regressions (`magistro2025/us_ai`, `magistro2025/us_offshoring`,
`magistro2025/ca_ai`, `magistro2025/ca_offshoring`), 4 attributes / 8 non-reference AMCEs per regression (32 AMCEs
total). Each task contributes one outcome row equal to the expected 1-5 support score (`sum_k k * p_k`; exact integer
1-5 for humans).
Coding decisions: data/magistro2025/DECISIONS.md.
"""

from __future__ import annotations

import re

import pandas as pd
import torch

from data.magistro2025 import MAGISTRO2025_PERSONA_COLS, load_magistro2025_df
from data.magistro2025.stimulus import SUPPORT_OPTIONS

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    regression_rows_from_parts,
)

NAME = "magistro2025"
PERSONA_COLS = MAGISTRO2025_PERSONA_COLS

# 1. Levels: 2 countries x 2 between-subject conjoint arms = 4 regressions; each attribute's reference is "Unchanged".
CONJOINT_TASKS = ("conjoint_ai", "conjoint_offshoring")
GROUP_BY_COUNTRY_TASK: dict[tuple[str, str], str] = {
    ("US", "conjoint_ai"): f"{NAME}/us_ai",
    ("US", "conjoint_offshoring"): f"{NAME}/us_offshoring",
    ("CA", "conjoint_ai"): f"{NAME}/ca_ai",
    ("CA", "conjoint_offshoring"): f"{NAME}/ca_offshoring",
}
GROUPS = tuple(GROUP_BY_COUNTRY_TASK.values())
GROUP_LABELS = {
    f"{NAME}/us_ai": "US AI",
    f"{NAME}/us_offshoring": "US Offshoring",
    f"{NAME}/ca_ai": "Canada AI",
    f"{NAME}/ca_offshoring": "Canada Offshoring",
}
EXPECTED_OPTION_CODES = [code for code, _ in SUPPORT_OPTIONS]
EXPECTED_OPTION_TEXTS = [text for _, text in SUPPORT_OPTIONS]
SCORE_WEIGHTS = [float(code) for code, _ in SUPPORT_OPTIONS]

# Attribute coding matching `stimulus.make_profile_code` (`p{0|1|2}c{0|1|2}f{0|1|2}d{0|1|2}`):
# 0 = Unchanged (reference), 1 = Decrease, 2 = Increase (or deeper price decrease $300).
ATTR_SPECS: list[tuple[str, str, str, list[tuple[int, str, str]]]] = [
    ("p", "Price", "$600", [(1, "$450", "$600 -> $450"), (2, "$300", "$600 -> $300")]),
    ("c", "Customer service jobs", "100", [(1, "75", "100 -> 75"), (2, "125", "100 -> 125")]),
    ("f", "Factory jobs", "400", [(1, "300", "400 -> 300"), (2, "500", "400 -> 500")]),
    ("d", "Data science jobs", "200", [(1, "150", "200 -> 150"), (2, "250", "200 -> 250")]),
]

# 2. Design: [Intercept, 8 non-reference dummies]; shared by both arms
TARGET_COLUMNS = [
    f"{attr_name}: {lvl_label} vs {ref_label}"
    for _, attr_name, ref_label, non_refs in ATTR_SPECS
    for _, lvl_label, _ in non_refs
]
DESIGN_COLUMNS = ["Intercept", *TARGET_COLUMNS]

PROFILE_CODE_RE = re.compile(r"^(US|CA)_p([012])c([012])f([012])d([012])$")


def _parse_profile_codes(qid: str, task_v: str, question: str, context: str) -> tuple[str, dict[str, int]]:
    # `magistro2025:conjoint_ai:US_p2c0f2d1` -> ("US", {"p": 2, "c": 0, "f": 2, "d": 1}), cross-checked against text
    parts = qid.split(":")
    if len(parts) != 3 or parts[0] != NAME or parts[1] != task_v:
        raise ValueError(f"{context}: unexpected magistro2025 qid {qid!r} for task {task_v!r}")
    match = PROFILE_CODE_RE.match(parts[2])
    if match is None:
        raise ValueError(f"{context}: invalid profile code {parts[2]!r} in qid {qid!r}")
    country_code = match.group(1)
    codes = {
        "p": int(match.group(2)),
        "c": int(match.group(3)),
        "f": int(match.group(4)),
        "d": int(match.group(5)),
    }
    for key, attr_name, _, non_refs in ATTR_SPECS:
        lvl = codes[key]
        if lvl != 0:
            expected_snip = dict((code, snip) for code, _, snip in non_refs)[lvl]
            if expected_snip not in question:
                raise ValueError(
                    f"{context}: qid code {key}={lvl} expects {expected_snip!r} ({attr_name}) in question text"
                )
    return country_code, codes


def design_row(codes: dict[str, int]) -> list[float]:
    row = [1.0]
    for key, _, _, non_refs in ATTR_SPECS:
        lvl = codes[key]
        for code, _, _ in non_refs:
            row.append(float(lvl == code))
    return row


# 3. Choice rows: one row per answered conjoint task (AI and Offshoring arms; excludes pre-treatment policy items)
def load_choice_df() -> pd.DataFrame:
    magistro_df = load_magistro2025_df()
    choice_df = magistro_df[magistro_df["v"].isin(CONJOINT_TASKS)].reset_index(drop=True)
    if len(choice_df) == 0:
        raise ValueError(f"No {list(CONJOINT_TASKS)} rows found in magistro2025_df")
    return choice_df


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    each task has 5 ordered options (1..5) and gives 1 outcome row equal to the expected support score
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    num_options = []
    y_values = []
    outcome_rows = []
    for t_idx, row in enumerate(r_df.itertuples(index=False)):
        task_v = str(row.v)
        if task_v not in CONJOINT_TASKS:
            raise ValueError(f"{row.CASEID} task {t_idx}: unexpected task variable {task_v!r}")
        context = f"{row.CASEID} {task_v} task {t_idx}"
        if list(row.option_codes) != EXPECTED_OPTION_CODES:
            raise ValueError(f"{context}: expected option_codes {EXPECTED_OPTION_CODES}, got {row.option_codes}")
        if list(row.option_texts) != EXPECTED_OPTION_TEXTS:
            raise ValueError(f"{context}: expected option_texts {EXPECTED_OPTION_TEXTS}, got {row.option_texts}")
        one_hot = check_one_hot([float(v) for v in row.one_hot], len(EXPECTED_OPTION_CODES), context)
        country_code, codes = _parse_profile_codes(str(row.qid), task_v, str(row.question), context)
        first_option = len(y_values)
        outcome_rows.append(
            OutcomeRow(
                group=GROUP_BY_COUNTRY_TASK[(country_code, task_v)],
                x=design_row(codes),
                option_idx=[first_option + k for k in range(len(SCORE_WEIGHTS))],
                weights=list(SCORE_WEIGHTS),
            )
        )
        num_options.append(len(EXPECTED_OPTION_CODES))
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (respondent-clustered SEs; sigma from the full human data per country x arm)
def regression_rows(choice_df: pd.DataFrame) -> pd.DataFrame:
    return regression_rows_from_parts(choice_df, respondent_beta_part, {g: DESIGN_COLUMNS for g in GROUPS})


def regression_specs(rows: pd.DataFrame) -> dict[str, RegressionSpec]:
    present_groups = set(rows["group"])
    specs = {}
    for group in GROUPS:
        if group not in present_groups:
            continue
        specs[group] = RegressionSpec(
            name=group,
            design_columns=DESIGN_COLUMNS,
            target_columns=TARGET_COLUMNS,
            target_labels=[f"{GROUP_LABELS[group]} | {col}" for col in TARGET_COLUMNS],
            sigma=outcome_sigma(rows.loc[rows["group"] == group, "y"].to_numpy()),
            cov_type="cluster",
        )
    return specs

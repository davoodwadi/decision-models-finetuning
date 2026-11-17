"""Grummon et al. (2023) front-of-package label vending-machine RCT (`bev_choice`, `snack_choice`).

Paper estimand: two separate primary outcomes, calories selected from beverages and from snacks, each regressed on
trial arm vs the Calorie-label control (beverages: two-part model; snacks: OLS; robust SEs). With only arm dummies
the two-part model's average differential effects equal differences in arm means, i.e. the OLS betas.
Our regression: 2 HC1 OLS models ("grummon2023/bev", "grummon2023/snack") sharing one design row per respondent
(intercept + 4 arm dummies). Outcome = calories of the chosen product (model: expected calories over 16 products).
Respondents must have both tasks (7,943 of 7,945; 2 have no beverage choice).
Coding decisions: data/nutrition/grummon2023/PLAN.md.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.grummon2023 import GRUMMON2023_PERSONA_COLS, load_grummon2023_df
from data.nutrition.grummon2023.stimulus import ARM_NAMES, build_product_catalog, load_analytic_cohort

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    regression_rows_from_parts,
)

NAME = "grummon2023"
PERSONA_COLS = GRUMMON2023_PERSONA_COLS

# 1. Levels: 5 randomized label arms; "Calorie" (control) is the reference
CONTROL_ARM = "Calorie"
TREAT_ARMS = ["Green", "TrafficLight", "PhysicalActivity", "Warning"]
if list(ARM_NAMES.values()) != [CONTROL_ARM, *TREAT_ARMS]:
    raise AssertionError(f"Unexpected Grummon 2023 arm order: {list(ARM_NAMES.values())}")
TASKS = {"bev_choice": f"{NAME}/bev", "snack_choice": f"{NAME}/snack"}   # task -> regression, in prompt order

# 2. Design: [Intercept, one dummy per treatment arm]; shared by both regressions
TARGET_COLUMNS = [f"{arm} vs {CONTROL_ARM}" for arm in TREAT_ARMS]
DESIGN_COLUMNS = ["Intercept", *TARGET_COLUMNS]


def design_row(arm: str) -> list[float]:
    if arm not in [CONTROL_ARM, *TREAT_ARMS]:
        raise ValueError(f"Unexpected Grummon 2023 arm {arm!r}")
    row = [1.0]
    for treat_arm in TREAT_ARMS:
        row.append(float(arm == treat_arm))
    return row


# 3. Choice rows: both tasks of every respondent who answered both, ordered by respondent then (bev, snack).
#    Each row carries `option_kcal`: the true calories of its 16 products in option order (authors' `item_cal_*`).
def load_choice_df() -> pd.DataFrame:
    grummon2023_df = load_grummon2023_df()
    _, bev_chars, snack_chars = load_analytic_cohort()
    catalog = build_product_catalog(bev_chars, snack_chars)
    kcal_by_task = {
        "bev_choice": [float(item["true_cal"]) for item in catalog["bev"]],
        "snack_choice": [float(item["true_cal"]) for item in catalog["snack"]],
    }
    item_numbers = {
        "bev_choice": [int(item["item_num"]) for item in catalog["bev"]],
        "snack_choice": [int(item["item_num"]) for item in catalog["snack"]],
    }

    task_rows = grummon2023_df[grummon2023_df["v"].isin(list(TASKS))].copy()
    tasks_per_case = task_rows.groupby("CASEID")["v"].nunique()
    complete_caseids = tasks_per_case[tasks_per_case == len(TASKS)].index
    task_rows = task_rows[task_rows["CASEID"].isin(complete_caseids)]

    # Option codes are the authors' item numbers 1..16, the same order as the catalog
    for task, numbers in item_numbers.items():
        for codes in task_rows.loc[task_rows["v"] == task, "option_codes"]:
            if list(codes) != numbers:
                raise ValueError(f"{task}: option codes {list(codes)} do not match catalog item numbers {numbers}")
    task_rows["option_kcal"] = [kcal_by_task[task] for task in task_rows["v"]]

    # Order: respondents by numeric id (`grummon2023_<id>`, so arms are interleaved), tasks as in TASKS
    task_rows["case_number"] = [int(str(c).rsplit("_", 1)[-1]) for c in task_rows["CASEID"]]
    task_rows["task_order"] = [list(TASKS).index(task) for task in task_rows["v"]]
    task_rows = task_rows.sort_values(["case_number", "task_order"]).drop(columns=["case_number", "task_order"])
    return task_rows.reset_index(drop=True)


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    2 tasks of 16 options; each task gives one outcome row = calories of the chosen product (sum of p_k * kcal_k)
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    if list(r_df["v"]) != list(TASKS):
        raise ValueError(f"{r_df['CASEID'].iloc[0]}: expected tasks {list(TASKS)}, got {list(r_df['v'])}")
    if r_df["arm"].nunique() != 1:
        raise ValueError(f"{r_df['CASEID'].iloc[0]}: arm differs between tasks")
    x_row = design_row(str(r_df["arm"].iloc[0]))
    num_options = []
    y_values = []
    outcome_rows = []
    for row in r_df.itertuples(index=False):
        one_hot = check_one_hot([float(v) for v in row.one_hot], 16, f"{row.CASEID} {row.v}")
        first_option = len(y_values)
        outcome_rows.append(
            OutcomeRow(
                group=TASKS[row.v],
                x=x_row,
                option_idx=list(range(first_option, first_option + 16)),
                weights=list(row.option_kcal),
            )
        )
        num_options.append(16)
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (sigma = SD of chosen calories per task, from the full human data)
def regression_rows(choice_df: pd.DataFrame) -> pd.DataFrame:
    design_columns_by_group = {group: DESIGN_COLUMNS for group in TASKS.values()}
    return regression_rows_from_parts(choice_df, respondent_beta_part, design_columns_by_group)


def regression_specs(rows: pd.DataFrame) -> dict[str, RegressionSpec]:
    task_labels = {f"{NAME}/bev": "Beverage", f"{NAME}/snack": "Snack"}
    specs = {}
    for group in TASKS.values():
        specs[group] = RegressionSpec(
            name=group,
            design_columns=DESIGN_COLUMNS,
            target_columns=TARGET_COLUMNS,
            target_labels=[f"{task_labels[group]} kcal | {col}" for col in TARGET_COLUMNS],
            sigma=outcome_sigma(rows.loc[rows["group"] == group, "y"].to_numpy()),
            cov_type="HC1",
        )
    return specs

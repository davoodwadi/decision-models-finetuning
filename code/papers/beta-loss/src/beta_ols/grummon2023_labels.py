"""Grummon et al. (2023) vending-machine RCT, label-choice coding (a second coding of the same data as `grummon2023_calories`).

Not a paper estimand (the paper regresses calories; that coding is `grummon2023_calories`). This coding regresses the
choice itself on the label printed on each product, so the betas say how a label changes a product's choice share.
Our regression: 2 respondent-clustered OLS models ("grummon2023_labels/bev", "grummon2023_labels/snack") on product
rows (each of the 16 displayed products is a row, y = 1 if chosen), with product fixed effects, arm dummies (nuisance:
they absorb the share unlabelled products lose as a group) and 5 label targets per task. The arm's poster is the same
for every product, so its effect is not separable: each label beta is "this label, after that arm's poster".
Same task rows, respondents and prompts as `grummon2023_calories` (its `load_choice_df`).
Coding decisions: data/nutrition/grummon2023/DECISIONS.md.
"""

from __future__ import annotations

import re

import pandas as pd
import torch

from beta_ols import grummon2023_calories
from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    regression_rows_from_parts,
)

NAME = "grummon2023_labels"
PERSONA_COLS = grummon2023_calories.PERSONA_COLS
NUM_PRODUCTS = 16

# 1. Levels: the 5 arms ("Calorie" control is the reference) and the labels each treatment arm prints on products
CONTROL_ARM = grummon2023_calories.CONTROL_ARM
TREAT_ARMS = grummon2023_calories.TREAT_ARMS
TASKS = {"bev_choice": f"{NAME}/bev", "snack_choice": f"{NAME}/snack"}   # task -> regression, in prompt order
CALORIE_LABEL_PATTERN = re.compile(r"^\d+ CALORIES PER SERVING \(white calorie label\)$")
GREEN_OFTEN = "CHOOSE OFTEN (green label, smiling face)"
TRAFFIC_LIGHT_LABELS = [   # reference level (RARELY) first
    "CHOOSE RARELY (red label, frowning face)",
    "CHOOSE SOMETIMES (yellow label, neutral face)",
    "CHOOSE OFTEN (green label, smiling face)",
]
MINUTES_LABEL_PATTERN = re.compile(r"^(\d+) MINUTES \(orange label, walking person icon\)$")
OCTAGON_LABEL_PATTERN = re.compile(r"^HIGH IN [A-Z ]+ \(black octagon\)$")

# 2. Design: [Intercept, products 2..16 (product 1 = reference), arm dummies (nuisance), label targets]; both tasks
#    use the same columns ("product k" = the k-th displayed product of that task)
PRODUCT_COLUMNS = [f"product {k}" for k in range(2, NUM_PRODUCTS + 1)]
ARM_COLUMNS = [f"arm: {arm}" for arm in TREAT_ARMS]
TARGET_COLUMNS = [
    "Green: CHOOSE OFTEN",
    "TrafficLight: CHOOSE OFTEN",
    "TrafficLight: CHOOSE SOMETIMES",
    "PhysicalActivity: minutes / 100",
    "Warning: octagons",
]
DESIGN_COLUMNS = ["Intercept", *PRODUCT_COLUMNS, *ARM_COLUMNS, *TARGET_COLUMNS]


def printed_labels(description: str, context: str) -> list[str]:
    # The labels printed on one product besides its calorie label; the calorie label must be there exactly once
    labels = re.findall(r"\[([^\]]+)\]", description)
    calorie_labels = [label for label in labels if CALORIE_LABEL_PATTERN.match(label)]
    if len(calorie_labels) != 1:
        raise ValueError(f"{context}: expected one calorie label, found {len(calorie_labels)} in {description!r}")
    return [label for label in labels if not CALORIE_LABEL_PATTERN.match(label)]


def label_values(arm: str, description: str, context: str) -> list[float]:
    # Target column values of one product in one arm, in TARGET_COLUMNS order; raises on any label the arm should not show
    labels = printed_labels(description, context)
    green_often = 0.0
    traffic_often = 0.0
    traffic_sometimes = 0.0
    minutes_per_100 = 0.0
    octagons = 0.0
    if arm == CONTROL_ARM:
        if labels:
            raise ValueError(f"{context}: the control arm shows only the calorie label, got {labels}")
    elif arm == "Green":
        if labels not in ([], [GREEN_OFTEN]):
            raise ValueError(f"{context}: Green arm label must be none or {GREEN_OFTEN!r}, got {labels}")
        green_often = float(labels == [GREEN_OFTEN])
    elif arm == "TrafficLight":
        if len(labels) != 1 or labels[0] not in TRAFFIC_LIGHT_LABELS:
            raise ValueError(f"{context}: TrafficLight arm needs one of {TRAFFIC_LIGHT_LABELS}, got {labels}")
        traffic_sometimes = float(labels[0] == TRAFFIC_LIGHT_LABELS[1])
        traffic_often = float(labels[0] == TRAFFIC_LIGHT_LABELS[2])
    elif arm == "PhysicalActivity":
        if len(labels) != 1 or not MINUTES_LABEL_PATTERN.match(labels[0]):
            raise ValueError(f"{context}: PhysicalActivity arm needs one walking-minutes label, got {labels}")
        minutes_per_100 = float(MINUTES_LABEL_PATTERN.match(labels[0]).group(1)) / 100.0
    elif arm == "Warning":
        for label in labels:
            if not OCTAGON_LABEL_PATTERN.match(label):
                raise ValueError(f"{context}: Warning arm shows only octagons, got {label!r}")
        octagons = float(len(labels))
    else:
        raise ValueError(f"{context}: unexpected Grummon 2023 arm {arm!r}")
    return [green_often, traffic_often, traffic_sometimes, minutes_per_100, octagons]


def design_row(product_index: int, arm: str, description: str, context: str) -> list[float]:
    # One displayed product (0-based position in its task) -> design row
    row = [1.0]
    for k in range(2, NUM_PRODUCTS + 1):
        row.append(float(product_index + 1 == k))
    for treat_arm in TREAT_ARMS:
        row.append(float(arm == treat_arm))
    row.extend(label_values(arm, description, context))
    return row


# 3. Choice rows: the same rows as `grummon2023_calories` (both tasks of every respondent who answered both)
def load_choice_df() -> pd.DataFrame:
    return grummon2023_calories.load_choice_df()


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    each task gives 16 product rows; each row's outcome is that product's own choice probability
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    # Tasks in TASKS order, each at most once (both on the full data; one on a single-task pseudo dataset)
    task_order = [task for task in TASKS if task in set(r_df["v"])]
    if list(r_df["v"]) != task_order or len(task_order) == 0:
        raise ValueError(f"{r_df['CASEID'].iloc[0]}: expected tasks in order {list(TASKS)}, got {list(r_df['v'])}")
    if r_df["arm"].nunique() != 1:
        raise ValueError(f"{r_df['CASEID'].iloc[0]}: arm differs between tasks")
    arm = str(r_df["arm"].iloc[0])
    num_options = []
    y_values = []
    outcome_rows = []
    for row in r_df.itertuples(index=False):
        context = f"{row.CASEID} {row.v}"
        one_hot = check_one_hot([float(v) for v in row.one_hot], NUM_PRODUCTS, context)
        if len(row.option_descriptions) != NUM_PRODUCTS:
            raise ValueError(f"{context}: expected {NUM_PRODUCTS} products, got {len(row.option_descriptions)}")
        first_option = len(y_values)
        for product_index, description in enumerate(row.option_descriptions):
            outcome_rows.append(
                OutcomeRow(
                    group=TASKS[row.v],
                    x=design_row(product_index, arm, description, f"{context} product {product_index + 1}"),
                    option_idx=[first_option + product_index],
                    weights=[1.0],
                )
            )
        num_options.append(NUM_PRODUCTS)
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (sigma from the full human data; sqrt(1/16 x 15/16) by design of 16-option rows)
def regression_rows(choice_df: pd.DataFrame) -> pd.DataFrame:
    design_columns_by_group = {group: DESIGN_COLUMNS for group in TASKS.values()}
    return regression_rows_from_parts(choice_df, respondent_beta_part, design_columns_by_group)


def regression_specs(rows: pd.DataFrame) -> dict[str, RegressionSpec]:
    # Specs for the task groups present in `rows` (both on the full data, one on a single-task pseudo dataset)
    task_labels = {f"{NAME}/bev": "Beverage", f"{NAME}/snack": "Snack"}
    present_groups = set(rows["group"])
    specs = {}
    for group in TASKS.values():
        if group not in present_groups:
            continue
        specs[group] = RegressionSpec(
            name=group,
            design_columns=DESIGN_COLUMNS,
            target_columns=TARGET_COLUMNS,
            target_labels=[f"{task_labels[group]} choice | {col}" for col in TARGET_COLUMNS],
            sigma=outcome_sigma(rows.loc[rows["group"] == group, "y"].to_numpy()),
            cov_type="cluster",
        )
    return specs

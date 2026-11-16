"""Grummon et al. (2025) front-of-package label RCT: healthier-product identification (`idbev`..`idmeal`).

Paper estimand (eTable 9, "Correctly identified healthier product"): average differential effects vs the Positive
labels from a mixed-effects logistic model over the 6 repeated binary items (a secondary outcome; the primary
outcome, purchase healthfulness, is not in this data). Option 1 is always the healthier product.
Our regression: one OLS ("grummon2025") at the paper's unit of analysis, the item: 6 rows per respondent,
y = 1 if the healthier product was picked, on arm dummies (Positive = reference), SEs clustered by respondent.
Every respondent has all 6 items and one arm, so the betas equal the regression of each respondent's share correct.
Fitting single answers makes sigma the SD of a single 0/1 answer, like every other choice dataset here.
OLS is linear in the model's probabilities (required by the loss); it gives the difference in mean proportions,
not the mixed-logit effects (3 of 5 published effects are about 0.01 higher).
Three respondents who never answered the items are not in the parquet (the authors coded them as incorrect).
Role in this project: test.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.grummon2025 import GRUMMON2025_PERSONA_COLS, PRIMARY_CHOICE_STEMS, TREAT_ARMS, load_grummon2025_df

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    regression_rows_from_parts,
)

NAME = "grummon2025"
PERSONA_COLS = GRUMMON2025_PERSONA_COLS
ITEMS_PER_RESPONDENT = len(PRIMARY_CHOICE_STEMS)   # 6 identification items

# 1. Levels: 6 randomized label arms; "Positive" is the reference
CONTROL_ARM = "Positive"

# 2. Design: [Intercept, one dummy per treatment arm]
TARGET_COLUMNS = [f"{arm} vs {CONTROL_ARM}" for arm in TREAT_ARMS]
DESIGN_COLUMNS = ["Intercept", *TARGET_COLUMNS]


def design_row(arm: str) -> list[float]:
    if arm not in [CONTROL_ARM, *TREAT_ARMS]:
        raise ValueError(f"Unexpected Grummon 2025 arm {arm!r}")
    row = [1.0]
    for treat_arm in TREAT_ARMS:
        row.append(float(arm == treat_arm))
    return row


# 3. Choice rows: the 6 identification items of every respondent
def load_choice_df() -> pd.DataFrame:
    grummon2025_df = load_grummon2025_df()
    choice_df = grummon2025_df[grummon2025_df["v"].isin(PRIMARY_CHOICE_STEMS)].reset_index(drop=True)
    items_per_case = choice_df.groupby("CASEID").size()
    if int(items_per_case.min()) != ITEMS_PER_RESPONDENT or int(items_per_case.max()) != ITEMS_PER_RESPONDENT:
        raise ValueError(f"Expected every Grummon 2025 respondent to have all {ITEMS_PER_RESPONDENT} items")
    return choice_df


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    6 tasks of 3 options (healthier, less healthy, "They're the same"); each task gives one outcome row =
#    p(option 1), the probability of picking the healthier product
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    if len(r_df) != ITEMS_PER_RESPONDENT:
        raise ValueError(f"{r_df['CASEID'].iloc[0]}: expected {ITEMS_PER_RESPONDENT} items, got {len(r_df)}")
    if r_df["arm"].nunique() != 1:
        raise ValueError(f"{r_df['CASEID'].iloc[0]}: arm differs between items")
    x_row = design_row(str(r_df["arm"].iloc[0]))
    num_options = []
    y_values = []
    outcome_rows = []
    for row in r_df.itertuples(index=False):
        one_hot = check_one_hot([float(v) for v in row.one_hot], 3, f"{row.CASEID} {row.v}")
        healthier_option = len(y_values)
        outcome_rows.append(OutcomeRow(group=NAME, x=x_row, option_idx=[healthier_option], weights=[1.0]))
        num_options.append(3)
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (sigma = SD of the single 0/1 answer, from the full human data)
def regression_rows(choice_df: pd.DataFrame) -> pd.DataFrame:
    return regression_rows_from_parts(choice_df, respondent_beta_part, {NAME: DESIGN_COLUMNS})


def regression_specs(rows: pd.DataFrame) -> dict[str, RegressionSpec]:
    return {
        NAME: RegressionSpec(
            name=NAME,
            design_columns=DESIGN_COLUMNS,
            target_columns=TARGET_COLUMNS,
            target_labels=TARGET_COLUMNS,
            sigma=outcome_sigma(rows.loc[rows["group"] == NAME, "y"].to_numpy()),
            cov_type="cluster",
        )
    }

"""Kreps et al. (2020) COVID-19 vaccine conjoint (`vax_choice`).

Paper estimand: Table 3 Model 1, respondent-clustered OLS AMCEs of `vaxbin` (1 if that profile was chosen) on the two
displayed profiles per task; replication file `kreps_etal_vax_acceptance_replication.do` line 62. Tasks have a third
option ("I would choose not to get either vaccine"); it has no row, and choosing it gives y = 0 on both profile rows.
Our regression: identical. One regression ("covid"). Role in this project: locked test.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.covid import AMCE_TERMS, COVID_PERSONA_COLS, load_covid_df
from data.covid import _profile_attr_dict as profile_attr_dict

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    parse_treatment_term,
    regression_rows_from_parts,
)

NAME = "covid"
PERSONA_COLS = COVID_PERSONA_COLS

# 1. Levels: the paper's Table 3 terms (attribute code column, non-reference level)
TERM_DUMMIES = [parse_treatment_term(term) for term, _ in AMCE_TERMS]

# 2. Design: [Intercept, one dummy per Table 3 term]
TARGET_COLUMNS = [term for term, _ in AMCE_TERMS]
TARGET_LABELS = [label for _, label in AMCE_TERMS]
DESIGN_COLUMNS = ["Intercept", *TARGET_COLUMNS]


def design_row(profile_id: str) -> list[float]:
    # One displayed vaccine profile (e.g. "p123") -> design row
    codes = profile_attr_dict(profile_id)
    row = [1.0]
    for column, level in TERM_DUMMIES:
        row.append(float(codes[column] == level))
    return row


# 3. Choice rows: one row per answered vaccine choice
def load_choice_df() -> pd.DataFrame:
    covid_df = load_covid_df()
    choice_df = covid_df[covid_df["v"] == "vax_choice"].reset_index(drop=True)
    if len(choice_df) == 0:
        raise ValueError("No `vax_choice` rows found in covid_df")
    return choice_df


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    each task has 3 options (A, B, neither) and gives 2 profile rows; each row's outcome is that profile's probability
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    num_options = []
    y_values = []
    outcome_rows = []
    for t_idx, row in enumerate(r_df.itertuples(index=False)):
        one_hot = check_one_hot([float(v) for v in row.one_hot], 3, f"{row.CASEID} task {t_idx}")
        profile_a, profile_b = str(row.qid).split(":")[-1].split("_")
        first_option = len(y_values)
        outcome_rows.append(OutcomeRow(group=NAME, x=design_row(profile_a), option_idx=[first_option], weights=[1.0]))
        outcome_rows.append(OutcomeRow(group=NAME, x=design_row(profile_b), option_idx=[first_option + 1], weights=[1.0]))
        num_options.append(3)
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (sigma from the full human data; below 0.5 because "neither" leaves both rows at 0)
def regression_rows(choice_df: pd.DataFrame) -> pd.DataFrame:
    return regression_rows_from_parts(choice_df, respondent_beta_part, {NAME: DESIGN_COLUMNS})


def regression_specs(rows: pd.DataFrame) -> dict[str, RegressionSpec]:
    return {
        NAME: RegressionSpec(
            name=NAME,
            design_columns=DESIGN_COLUMNS,
            target_columns=TARGET_COLUMNS,
            target_labels=TARGET_LABELS,
            sigma=outcome_sigma(rows.loc[rows["group"] == NAME, "y"].to_numpy()),
            cov_type="cluster",
        )
    }

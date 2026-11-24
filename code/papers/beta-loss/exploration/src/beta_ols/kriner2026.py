"""Goldfarb, Kriner et al. (2026) airline sustainable-aviation-fuel conjoint (`flight_choice`).

Paper estimand: Table 2, respondent-clustered OLS AMCEs on flight rows (each displayed flight is a row, y = 1 if
chosen), replication file `Goldfarb_etal_SAF_JCP_replication.do` line 5. The two flights' attributes are independently
randomized, so own-attribute rows are unbiased.
Our regression: identical. One regression ("kriner2026"). Role in this project: validation.
Coding decisions: data/Kriner2026/DECISIONS.md.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.Kriner2026 import AMCE_TERMS, KRINER2026_PERSONA_COLS, load_kriner2026_df, parse_flight_description
from data.Kriner2026.stimulus import CHOICE_OPTIONS

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    parse_treatment_term,
    regression_rows_from_parts,
)

NAME = "kriner2026"
PERSONA_COLS = KRINER2026_PERSONA_COLS
EXPECTED_OPTIONS = [text for _, text in CHOICE_OPTIONS]

# 1. Levels: the paper's Table 2 terms (attribute code column, non-reference level); references are the omitted levels
TERM_DUMMIES = [parse_treatment_term(term) for term, _ in AMCE_TERMS]

# 2. Design: [Intercept, one dummy per Table 2 term]
TARGET_COLUMNS = [term for term, _ in AMCE_TERMS]
TARGET_LABELS = [label for _, label in AMCE_TERMS]
DESIGN_COLUMNS = ["Intercept", *TARGET_COLUMNS]


def design_row(description: str) -> list[float]:
    # One displayed flight -> design row (attribute codes from the description)
    codes = parse_flight_description(description)
    row = [1.0]
    for column, level in TERM_DUMMIES:
        row.append(float(codes[column] == level))
    return row


# 3. Choice rows: one row per answered flight choice
def load_choice_df() -> pd.DataFrame:
    kriner2026_df = load_kriner2026_df()
    choice_df = kriner2026_df[kriner2026_df["v"] == "flight_choice"].reset_index(drop=True)
    if len(choice_df) == 0:
        raise ValueError("No `flight_choice` rows found in kriner2026_df")
    return choice_df


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    each task gives 2 flight rows; each row's outcome is that flight's own probability
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    num_options = []
    y_values = []
    outcome_rows = []
    for t_idx, row in enumerate(r_df.itertuples(index=False)):
        if list(row.option_texts) != EXPECTED_OPTIONS:
            raise ValueError(f"{row.CASEID} task {t_idx}: expected {EXPECTED_OPTIONS}, got {row.option_texts}")
        one_hot = check_one_hot([float(v) for v in row.one_hot], 2, f"{row.CASEID} task {t_idx}")
        if len(row.option_descriptions) != 2:
            raise ValueError(f"{row.CASEID} task {t_idx}: expected 2 flights, got {len(row.option_descriptions)}")
        first_option = len(y_values)
        for opt_idx, description in enumerate(row.option_descriptions):
            outcome_rows.append(
                OutcomeRow(group=NAME, x=design_row(description), option_idx=[first_option + opt_idx], weights=[1.0])
            )
        num_options.append(2)
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (sigma from the full human data; 0.5 by design of 2-option rows)
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

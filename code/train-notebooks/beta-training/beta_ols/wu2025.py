"""Wu et al. (2025) electricity-plan conjoint (`plan_choice`).

Paper estimand: respondent-clustered OLS AMCEs on option rows (each displayed plan is a row, y = 1 if chosen).
Our regression: identical. One regression ("wu2025").
Coding decisions: data/wu2025/DECISIONS.md.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.wu2025 import WU2025_PERSONA_COLS, load_wu2025_df
from data.wu2025.stimulus import ATTR_LOCAL, ATTR_MONEY, ATTR_PERCENT, CONJOINT_OPTIONS

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    regression_rows_from_parts,
)

NAME = "wu2025"
PERSONA_COLS = WU2025_PERSONA_COLS

# 1. Levels: the three randomized plan attributes; the first level of each is the OLS reference
LEVELS = {
    ATTR_MONEY: ["$105", "$110", "$115", "$120", "$125"],
    ATTR_PERCENT: ["10%", "30%", "50%", "100%"],
    ATTR_LOCAL: ["No", "Yes"],
}

# 2. Design: [Intercept, one dummy per non-reference level]; the dummy column names are also the target labels
TARGET_COLUMNS = []
for _attr, _levels in LEVELS.items():
    for _level in _levels[1:]:
        TARGET_COLUMNS.append(f"{_attr}: {_level} vs {_levels[0]}")
DESIGN_COLUMNS = ["Intercept", *TARGET_COLUMNS]


def design_row(description: str) -> list[float]:
    # One displayed plan ("Monthly bill: $110; Percentage of renewables: 30%; ...") -> design row
    levels = {}
    for clause in description.split("; "):
        attr, value = clause.split(": ", 1)
        levels[attr] = value
    row = [1.0]
    for attr, attr_levels in LEVELS.items():
        if levels[attr] not in attr_levels:
            raise ValueError(f"Invalid Wu 2025 level {attr}={levels[attr]!r}")
        for level in attr_levels[1:]:
            row.append(float(levels[attr] == level))
    return row


# 3. Choice rows: one row per answered conjoint task
def load_choice_df() -> pd.DataFrame:
    wu2025_df = load_wu2025_df()
    choice_df = wu2025_df[wu2025_df["v"] == "plan_choice"].reset_index(drop=True)
    if len(choice_df) == 0:
        raise ValueError("No `plan_choice` rows found in wu2025_df")
    return choice_df


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    each task gives 2 option rows; each row's outcome is that option's own probability
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    num_options = []
    y_values = []
    outcome_rows = []
    for t_idx, row in enumerate(r_df.itertuples(index=False)):
        if list(row.option_texts) != CONJOINT_OPTIONS:
            raise ValueError(f"{row.CASEID} task {t_idx}: expected {CONJOINT_OPTIONS}, got {row.option_texts}")
        one_hot = check_one_hot([float(v) for v in row.one_hot], 2, f"{row.CASEID} task {t_idx}")
        if len(row.option_descriptions) != 2:
            raise ValueError(f"{row.CASEID} task {t_idx}: expected 2 plans, got {len(row.option_descriptions)}")
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


# 5. Regression table and specs (sigma from the full human data)
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

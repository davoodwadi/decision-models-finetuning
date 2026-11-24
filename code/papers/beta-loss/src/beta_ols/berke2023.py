"""Berke et al. (2023) drone-delivery conjoint (`delivery_choice_v0..v3`).

Paper estimand: 4 separate vendor-specific choice models (Section 3.4, Table 4). Dominated pairs were excluded when
building choice sets (p. 7), so ground and drone attributes are correlated and own-attribute rows would be biased.
Our regression: 4 respondent-clustered OLS models ("berke2023/v0".."v3"), each on stacked option rows (Ground, Drone)
whose attributes are coded THIS option minus the OTHER option, plus a drone constant. The slopes equal the
pair-difference regression (`y_drone` on `X_drone - X_ground`) exactly. Intercept and drone constant are not targets.
Coding decisions: data/berke2023/DECISIONS.md.
"""

from __future__ import annotations

import pandas as pd
import torch

from data.berke2023 import BERKE2023_PERSONA_COLS, load_berke2023_df
from data.berke2023.stimulus import CHOICE_OPTIONS, VENDOR_TYPES

from beta_ols.core import (
    OutcomeRow,
    RegressionSpec,
    RespondentBetaPart,
    check_one_hot,
    outcome_sigma,
    regression_rows_from_parts,
)

NAME = "berke2023"
PERSONA_COLS = BERKE2023_PERSONA_COLS
EXPECTED_OPTIONS = [text for _, text in CHOICE_OPTIONS]   # ["Ground vehicle", "Drone"], the prompt order

# 1. Levels: per vendor, fee and wait have 4 levels coded 0..3 (level 0 = $0 fee, shortest wait = reference);
#    the actual values differ by vendor (data/berke2023/stimulus.py COST_LEVELS / TIME_LEVELS)
ATTRIBUTES = ["privacy", "fee_L1", "fee_L2", "fee_L3", "wait_L1", "wait_L2", "wait_L3"]
TARGET_LABELS = {
    "privacy_diff": "Drone privacy: YES vs NO",
    "fee_L1_diff": "Fee L1 vs $0 ($1)",
    "fee_L2_diff": "Fee L2 vs $0 ($3 fast / $5 meds)",
    "fee_L3_diff": "Fee L3 vs $0 ($5 fast / $10 meds)",
    "wait_L1_diff": "Wait L1 vs L0 (20m fast / 2h meds)",
    "wait_L2_diff": "Wait L2 vs L0 (30m fast / 6h meds)",
    "wait_L3_diff": "Wait L3 vs L0 (45m fast / 1d meds)",
}

# 2. Design: [Intercept, is_drone, privacy_diff, fee_L1..L3_diff, wait_L1..L3_diff]
TARGET_COLUMNS = [f"{attr}_diff" for attr in ATTRIBUTES]
DESIGN_COLUMNS = ["Intercept", "is_drone", *TARGET_COLUMNS]
GROUPS = [f"{NAME}/v{v_idx}" for v_idx in range(len(VENDOR_TYPES))]


def parse_design(qid: str) -> dict[str, int]:
    # `qid` ends in `gc<k>_gt<k>_dc<k>_dt<k>_dp<k>`: ground cost, ground time, drone cost, drone time, drone privacy
    design = {}
    for token in qid.split(":")[-1].split("_"):
        design[token[:2]] = int(token[2:])
    return design


def attribute_codes(cost_idx: int, time_idx: int, privacy: int) -> dict[str, float]:
    # One option's level dummies vs level 0, plus its privacy flag (ground has no privacy attribute: 0)
    return {
        "privacy": float(privacy),
        "fee_L1": float(cost_idx == 1),
        "fee_L2": float(cost_idx == 2),
        "fee_L3": float(cost_idx == 3),
        "wait_L1": float(time_idx == 1),
        "wait_L2": float(time_idx == 2),
        "wait_L3": float(time_idx == 3),
    }


def design_row(own_codes: dict[str, float], other_codes: dict[str, float], is_drone: bool) -> list[float]:
    # One option row: own attributes minus the other option's, with the drone constant
    row = [1.0, float(is_drone)]
    for attr in ATTRIBUTES:
        row.append(own_codes[attr] - other_codes[attr])
    return row


# 3. Choice rows: one row per answered delivery choice
def load_choice_df() -> pd.DataFrame:
    berke2023_df = load_berke2023_df()
    choice_df = berke2023_df[berke2023_df["v"].str.startswith("delivery_choice_v")].reset_index(drop=True)
    if len(choice_df) == 0:
        raise ValueError("No `delivery_choice_v*` rows found in berke2023_df")
    return choice_df


# 4. Respondent beta part (the ONE coding used by the human OLS and the training loss):
#    each task gives 2 option rows in prompt order (Ground, Drone); each row's outcome is that option's probability
def respondent_beta_part(r_df: pd.DataFrame) -> RespondentBetaPart:
    num_options = []
    y_values = []
    outcome_rows = []
    for t_idx, row in enumerate(r_df.itertuples(index=False)):
        if list(row.option_texts) != EXPECTED_OPTIONS:
            raise ValueError(f"{row.CASEID} task {t_idx}: expected {EXPECTED_OPTIONS}, got {row.option_texts}")
        one_hot = check_one_hot([float(v) for v in row.one_hot], 2, f"{row.CASEID} task {t_idx}")
        v_idx = int(str(row.v).rsplit("v", 1)[-1])
        design = parse_design(str(row.qid))
        ground_codes = attribute_codes(design["gc"], design["gt"], privacy=0)
        drone_codes = attribute_codes(design["dc"], design["dt"], privacy=design["dp"])
        first_option = len(y_values)
        ground_row = OutcomeRow(
            group=GROUPS[v_idx],
            x=design_row(ground_codes, drone_codes, is_drone=False),
            option_idx=[first_option],
            weights=[1.0],
        )
        drone_row = OutcomeRow(
            group=GROUPS[v_idx],
            x=design_row(drone_codes, ground_codes, is_drone=True),
            option_idx=[first_option + 1],
            weights=[1.0],
        )
        outcome_rows.append(ground_row)
        outcome_rows.append(drone_row)
        num_options.append(2)
        y_values.extend(one_hot)
    return RespondentBetaPart(
        num_options=num_options, y_flat=torch.tensor(y_values, dtype=torch.float64), outcome_rows=outcome_rows
    )


# 5. Regression table and specs (sigma from the full human data of each vendor; 0.5 by design of stacked rows)
def regression_rows(choice_df: pd.DataFrame) -> pd.DataFrame:
    design_columns_by_group = {group: DESIGN_COLUMNS for group in GROUPS}
    return regression_rows_from_parts(choice_df, respondent_beta_part, design_columns_by_group)


def regression_specs(rows: pd.DataFrame) -> dict[str, RegressionSpec]:
    # Specs for the vendor groups present in `rows` (all four on the full data, one on a single-vendor pseudo dataset)
    present_groups = set(rows["group"])
    specs = {}
    for v_idx, group in enumerate(GROUPS):
        if group not in present_groups:
            continue
        vendor_label = VENDOR_TYPES[v_idx][0]
        specs[group] = RegressionSpec(
            name=group,
            design_columns=DESIGN_COLUMNS,
            target_columns=TARGET_COLUMNS,
            target_labels=[f"v{v_idx}: {vendor_label} | {TARGET_LABELS[col]}" for col in TARGET_COLUMNS],
            sigma=outcome_sigma(rows.loc[rows["group"] == group, "y"].to_numpy()),
            cov_type="cluster",
        )
    return specs

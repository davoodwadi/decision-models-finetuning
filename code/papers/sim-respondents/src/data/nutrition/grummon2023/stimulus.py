"""Survey catalog, arm-specific visual stimulus builders, and Audit UI indexers for Grummon et al. (2023)."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRUMMON2023_DIR = Path(__file__).resolve().parent
RB_DATA_DIR = GRUMMON2023_DIR / "ResearchBox 1895" / "Data"
RAW_CSV = RB_DATA_DIR / "VLO-Survey-2022-3+FINAL_December+5,+2022_09.36-NoIDs.csv"
BEV_DTA = RB_DATA_DIR / "BeverageCharacteristics.dta"
SNACK_DTA = RB_DATA_DIR / "SnackCharacteristics.dta"

TOOFAST_SEC = 191.2

HEADER_FIXES = {
    "snack _healthyY_Cal": "snack_healthyY_Cal",
    "snack _healthyG_Cal": "snack_healthyG_Cal",
    "stigma_personal_CAL": "stigma_personal_Cal",
    "stigma_obesity_CAL": "stigma_obesity_Cal",
    "stigma_blame_CAL": "stigma_blame_Cal",
    "disgust_CAL": "disgust_Cal",
}

ARM_ORDER = [
    (1, "Cal", "Calorie"),
    (2, "G", "Green"),
    (3, "TLL", "TrafficLight"),
    (4, "PA", "PhysicalActivity"),
    (5, "W", "Warning"),
]
ARM_SUFFIX = {code: sfx for code, sfx, _ in ARM_ORDER}
ARM_NAMES = {code: name for code, _, name in ARM_ORDER}
PLURAL_LABEL_ARMS = {"TrafficLight", "Warning"}

# Normalized product display names (fixing double space in #3 beverage and restoring package oz on #12 and #13 snacks)
PRODUCT_NAME_OVERRIDES = {
    ("bev", 3): "Mountain Dew - 20oz",
    ("snack", 12): "Oreo Cream Mini Cookie - 3oz",
    ("snack", 13): "Nature Valley Oats and Honey Granola Bar - 1.49oz",
}

# Per eTable 1 Footnote a: Diet Mountain Dew - 20oz (beverage #6, 10 kcal in nutrition table) mistakenly
# displayed a "0 CALORIES PER SERVING" front-of-package label on its product image in all 5 trial arms.
DISPLAYED_CALORIE_OVERRIDES = {
    ("bev", 6): 0,
}

# Pre-task bonus instruction shown to all participants ([020]-[021] in VLOSurveyItems.docx)
VENDING_BONUS_INTRO = (
    "In this survey, you will see beverages and snacks you might see in a vending machine. "
    "We will ask you to select the beverage and the snack you would most like to purchase. "
    'We will randomly select 100 respondents to receive a "bonus" for completing this part of the survey. '
    "If you are selected for the bonus, you will have one of your product selections delivered to you.\n"
    "You should choose the beverage and snack you most want to purchase because your choices could be delivered to you."
)

BEV_SECTION_INTRO = (
    "The next section is about beverage choices. You will see beverages you might see in a vending machine. "
    "We will ask you to select the beverage you would most like to purchase.\n"
    "If you see a beverage brand you want but don't see the flavor you want, please still select that beverage "
    "regardless of the flavor available."
)

SNACK_SECTION_INTRO = (
    "The next section is about snack choices. You will see snacks you might see in a vending machine. "
    "We will ask you to select the snack you would most like to purchase.\n"
    "If you see a snack brand you want but don't see the flavor you want, please still select that snack "
    "regardless of the flavor available."
)

# Arm-specific digital posters shown immediately before the beverage / snack vending tasks (eFigure, Supplement 2 p. 4)
BEV_POSTERS = {
    "Calorie": "",
    "Green": (
        "Poster shown — Healthy Beverage Guide: Look for beverages with green labels for a healthier choice.\n"
        "• [CHOOSE OFTEN (green label, smiling face)]: Water; Unsweetened seltzer water."
    ),
    "TrafficLight": (
        "Poster shown — Healthy Beverage Guide: Look for beverages with green labels for a healthier choice.\n"
        "• [CHOOSE OFTEN (green label, smiling face)]: Water; Unsweetened seltzer water.\n"
        "• [CHOOSE SOMETIMES (yellow label, neutral face)]: 100% juice; Diet drinks.\n"
        "• [CHOOSE RARELY (red label, frowning face)]: Regular soda; Sweetened teas; Lemonade and sports drinks."
    ),
    "PhysicalActivity": (
        "Poster shown — DID YOU KNOW IT TAKES 115 MINS OF BRISK WALKING TO WORK OFF A BOTTLE OF SODA? "
        "Look for labels with 0 minutes for healthier drink choices [orange label with walking person icon]."
    ),
    "Warning": (
        "Poster shown — Healthy Beverage Guide: Look for beverages with no octagon labels for a healthier choice.\n"
        "• [HIGH IN SUGAR (black octagon)]: Label shows when products have a lot of sugar."
    ),
}

SNACK_POSTERS = {
    "Calorie": "",
    "Green": (
        "Poster shown — Healthy Snack Guide: Look for snacks with green labels for a healthier choice.\n"
        "• [CHOOSE OFTEN (green label, smiling face)]: Baked chips; Pretzels; Select cookies."
    ),
    "TrafficLight": (
        "Poster shown — Healthy Snack Guide: Look for snacks with green labels for a healthier choice.\n"
        "• [CHOOSE OFTEN (green label, smiling face)]: Baked chips; Pretzels; Select cookies.\n"
        "• [CHOOSE SOMETIMES (yellow label, neutral face)]: Trail mix; Select granola bars; Select chips and popcorn.\n"
        "• [CHOOSE RARELY (red label, frowning face)]: Select cookies; Candy and chocolate bars; Select chips and crackers."
    ),
    "PhysicalActivity": (
        "Poster shown — DID YOU KNOW IT TAKES 110 MINS OF BRISK WALKING TO WORK OFF A CHOCOLATE BAR? "
        "Look for labels with fewer than 45 minutes for healthier snack choices [orange label with walking person icon]."
    ),
    "Warning": (
        "Poster shown — Healthy Snack Guide: Look for snacks with fewer or no octagon labels for a healthier choice.\n"
        "• [HIGH IN CALORIES / HIGH IN SUGAR / HIGH IN SODIUM / HIGH IN SATURATED FAT (black octagons)]: "
        "Labels show when products have a lot of calories, sugar, sodium, or saturated fat."
    ),
}

# Standalone label artwork shown on label reaction & perception screens (Figure 1A & VLOSurveyItems.docx [139]-[300])
STANDALONE_LABEL_STIMULUS = {
    "Calorie": "Label shown: [140 CALORIES PER SERVING (white calorie label)].",
    "Green": "Label shown: [CHOOSE OFTEN (green label, smiling face)].",
    "TrafficLight": (
        "Labels shown: [CHOOSE OFTEN (green label, smiling face)], "
        "[CHOOSE SOMETIMES (yellow label, neutral face)], and "
        "[CHOOSE RARELY (red label, frowning face)]."
    ),
    "PhysicalActivity": "Label shown: [63 MINUTES (orange label, walking person icon)].",
    "Warning": (
        "Labels shown: [HIGH IN CALORIES (black octagon)], [HIGH IN SATURATED FAT (black octagon)], "
        "[HIGH IN SUGAR (black octagon)], and [HIGH IN SODIUM (black octagon)]."
    ),
}

# Summary of labels present in the preceding vending task for retrospective noticing items ([129], [134])
RETROSPECTIVE_TASK_CONTEXT = {
    "Calorie": (
        "In the preceding beverage and snack vending machine tasks, all products displayed a white "
        "[CALORIES PER SERVING] label on the front of the package (and no other front-of-package nutrition labels)."
    ),
    "Green": (
        "In the preceding beverage and snack vending machine tasks, all products displayed a white "
        "[CALORIES PER SERVING] label, and healthier products also displayed a green [CHOOSE OFTEN] label with a smiling face."
    ),
    "TrafficLight": (
        "In the preceding beverage and snack vending machine tasks, all products displayed a white "
        "[CALORIES PER SERVING] label alongside a traffic light label: green [CHOOSE OFTEN], yellow [CHOOSE SOMETIMES], "
        "or red [CHOOSE RARELY]."
    ),
    "PhysicalActivity": (
        "In the preceding beverage and snack vending machine tasks, all products displayed a white "
        "[CALORIES PER SERVING] label alongside an orange [MINUTES] physical activity label with a walking person icon."
    ),
    "Warning": (
        "In the preceding beverage and snack vending machine tasks, all products displayed a white "
        "[CALORIES PER SERVING] label, and products exceeding nutrient thresholds also displayed black octagon "
        "[HIGH IN CALORIES / SATURATED FAT / SUGAR / SODIUM] warning labels."
    ),
}

EXTENT_5 = [
    (1, "Not at all"),
    (2, "Very little"),
    (3, "Somewhat"),
    (4, "Quite a bit"),
    (5, "A great deal"),
]
HEALTHY_5 = [
    (1, "Very unhealthy"),
    (2, "Somewhat unhealthy"),
    (3, "Neither unhealthy nor healthy"),
    (4, "Somewhat healthy"),
    (5, "Very healthy"),
]
TRUST_5 = [
    (1, "Completely distrust"),
    (2, "Somewhat distrust"),
    (3, "Neither trust nor distrust"),
    (4, "Somewhat trust"),
    (5, "Completely trust"),
]
CONTROL_3 = [
    (1, "Less in control of making healthy eating decisions"),
    (2, "Neither less nor more in control of making healthy eating decisions"),
    (3, "More in control of making healthy eating decisions"),
]
NOTICED_3 = [
    (1, "Yes"),
    (0, "No"),
    (2, "Not sure"),
]
INFL_3 = [
    (1, "Yes"),
    (0, "No"),
    (2, "Did not see the label"),
]

# Single-product healthfulness target mapping (verified against CODEBOOK.csv, VLOSurveyItems.docx, and Table 2)
HEALTHY_ITEM_TARGETS = {
    "bev_healthyR": ("bev", 7, "How unhealthy or healthy do you think this beverage is?"),
    "bev_healthyY": ("bev", 8, "How unhealthy or healthy do you think this beverage is?"),
    "bev_healthyG": ("bev", 15, "How unhealthy or healthy do you think this beverage is?"),
    "snack_healthyR": ("snack", 4, "How unhealthy or healthy do you think this product is?"),
    "snack_healthyY": ("snack", 5, "How unhealthy or healthy do you think this product is?"),
    "snack_healthyG": ("snack", 11, "How unhealthy or healthy do you think this product is?"),
}

# Singular vs. plural stems for all 22 label-specific items ([128] of VLOSurveyItems.docx)
LABEL_REACTION_SPECS = OrderedDict(
    [
        (
            "infl_bev",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "choice",
                "include": False,
                "singular": "Did the label influence which beverage you selected?",
                "plural": "Did the labels influence which beverage you selected?",
                "options": INFL_3,
                "tags": ["prior-task-dep", "grummon2025-similar"],
                "note": "Retrospective self-report about prior beverage selection; similar to Grummon 2025 secondary item rused_trial (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "help_bev",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": False,
                "singular": "How much did you use this label to help you decide which beverage to select?",
                "plural": "How much did you use these labels to help you decide which beverage to select?",
                "options": EXTENT_5,
                "tags": ["prior-task-dep", "grummon2025-similar"],
                "note": "Retrospective self-report about prior beverage selection; similar to Grummon 2025 secondary items rused_trial / rhelp (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "infl_snack",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "choice",
                "include": False,
                "singular": "Did the label influence which snack you selected?",
                "plural": "Did the labels influence which snack you selected?",
                "options": INFL_3,
                "tags": ["prior-task-dep", "grummon2025-similar"],
                "note": "Retrospective self-report about prior snack selection; similar to Grummon 2025 secondary item rused_trial (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "help_snack",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": False,
                "singular": "How much did you use this label to help you decide which snack to select?",
                "plural": "How much did you use these labels to help you decide which snack to select?",
                "options": EXTENT_5,
                "tags": ["prior-task-dep", "grummon2025-similar"],
                "note": "Retrospective self-report about prior snack selection; similar to Grummon 2025 secondary items rused_trial / rhelp (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "attn",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label grab your attention?",
                "plural": "How much do these labels grab your attention?",
                "options": EXTENT_5,
                "tags": [],
                "note": "Standalone label evaluation (Table 2).",
            },
        ),
        (
            "elab",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label make you think about the health effects of foods and beverages?",
                "plural": "How much do these labels make you think about the health effects of foods and beverages?",
                "options": EXTENT_5,
                "tags": ["grummon2025-similar"],
                "note": "Standalone label evaluation (Table 2); construct parallel to Grummon 2025 secondary item elaborationhealth (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "pme_disc",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label discourage you from wanting to buy unhealthy foods and beverages?",
                "plural": "How much do these labels discourage you from wanting to buy unhealthy foods and beverages?",
                "options": EXTENT_5,
                "tags": [],
                "note": "Perceived message effectiveness — discouragement (Table 2).",
            },
        ),
        (
            "pme_enc",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label encourage you to want to buy healthy foods and beverages?",
                "plural": "How much do these labels encourage you to want to buy healthy foods and beverages?",
                "options": EXTENT_5,
                "tags": [],
                "note": "Perceived message effectiveness — encouragement (Table 2).",
            },
        ),
        (
            "affect_worried",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label make you feel worried?",
                "plural": "How much do these labels make you feel worried?",
                "options": EXTENT_5,
                "tags": ["in-grummon2025"],
                "note": "Same wording/scale as Grummon 2025 secondary item raffect_worried (also in the Grummon 2025 test set grummon2025_US_long.parquet, evaluated on different FOPL designs).",
            },
        ),
        (
            "affect_scared",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label make you feel scared?",
                "plural": "How much do these labels make you feel scared?",
                "options": EXTENT_5,
                "tags": ["in-grummon2025"],
                "note": "Same wording/scale as Grummon 2025 secondary item raffect_scared (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "affect_guilty",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label make you feel guilty?",
                "plural": "How much do these labels make you feel guilty?",
                "options": EXTENT_5,
                "tags": ["in-grummon2025"],
                "note": "Same wording/scale as Grummon 2025 secondary item raffect_guilty (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "affect_ashamed",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label make you feel ashamed?",
                "plural": "How much do these labels make you feel ashamed?",
                "options": EXTENT_5,
                "tags": ["in-grummon2025"],
                "note": "Same wording/scale as Grummon 2025 secondary item raffect_ashamed (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "affect_sad",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much does this label make you feel sad?",
                "plural": "How much do these labels make you feel sad?",
                "options": EXTENT_5,
                "tags": ["in-grummon2025"],
                "note": "Same wording/scale as Grummon 2025 secondary item raffect_sad (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "learn",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "To what extent did you learn something new from this label?",
                "plural": "To what extent did you learn something new from these labels?",
                "options": EXTENT_5,
                "tags": [],
                "note": "Positive label perception — learned something new (Table 2).",
            },
        ),
        (
            "trust",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How much do you trust or distrust the information in this label?",
                "plural": "How much do you trust or distrust the information in these labels?",
                "options": TRUST_5,
                "tags": ["in-grummon2025"],
                "note": "Same wording/scale as Grummon 2025 secondary item rtrust (also in grummon2025_US_long.parquet).",
            },
        ),
        (
            "control",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "Overall, does this label make you feel…",
                "plural": "Overall, do these labels make you feel…",
                "options": CONTROL_3,
                "tags": ["kev-type-check"],
                "note": "3-point ordered bipolar scale (1=Less in control, 2=Neither, 3=More in control); verify score vs choice.",
            },
        ),
        (
            "react_manip",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "This label is trying to manipulate me.",
                "plural": "These labels are trying to manipulate me.",
                "options": EXTENT_5,
                "tags": [],
                "note": "Message reactance — manipulation (Table 2).",
            },
        ),
        (
            "react_annoy",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "This label annoys me.",
                "plural": "These labels annoy me.",
                "options": EXTENT_5,
                "tags": [],
                "note": "Message reactance — annoyance (Table 2).",
            },
        ),
        (
            "stigma_personal",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How personally stigmatized does this label make you feel?",
                "plural": "How personally stigmatized do these labels make you feel?",
                "options": EXTENT_5,
                "tags": [],
                "note": "Perceived personal stigma (Table 2).",
            },
        ),
        (
            "stigma_obesity",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "This label stigmatizes people with obesity.",
                "plural": "These labels stigmatize people with obesity.",
                "options": EXTENT_5,
                "tags": [],
                "note": "Perceived obesity stigma item 1 (Table 2).",
            },
        ),
        (
            "stigma_blame",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "This label increases blame toward people for being overweight.",
                "plural": "These labels increase blame toward people for being overweight.",
                "options": EXTENT_5,
                "tags": [],
                "note": "Perceived obesity stigma item 2 (Table 2).",
            },
        ),
        (
            "disgust",
            {
                "block": "Label Reactions and Perceptions",
                "kev_type": "score",
                "include": True,
                "singular": "How disgusted do people with obesity make you feel?",
                "plural": "How disgusted do people with obesity make you feel?",
                "options": EXTENT_5,
                "tags": [],
                "note": "Disgust toward people with obesity, displayed alongside the participant's assigned label(s) (Table 2).",
            },
        ),
    ]
)

OUTCOME_STEMS = [
    "bev_choice",
    "snack_choice",
    "bev_healthyR",
    "bev_healthyY",
    "bev_healthyG",
    "snack_healthyR",
    "snack_healthyY",
    "snack_healthyG",
    "noticed_cal",
    "noticed_trial",
    *LABEL_REACTION_SPECS.keys(),
]


def load_analytic_cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean the N=7,945 analytic cohort and 16-item beverage/snack lookup tables."""
    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    if len(raw_df) != 10_055:
        raise ValueError(f"Expected 10,055 raw Qualtrics rows, got {len(raw_df)}")

    missing_headers = [c for c in HEADER_FIXES if c not in raw_df.columns]
    if missing_headers:
        raise KeyError(f"Missing expected raw headers: {missing_headers}")

    df_all = raw_df.rename(columns=HEADER_FIXES).replace([-99, "-99", -99.0], np.nan)
    for col in ["Q_TotalDuration", "confirmkeepdata", "consent", "age", "foplcond"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="raise")

    clean_df = df_all[
        (df_all["svy_status"] == "complete")
        & (df_all["Q_TotalDuration"] >= TOOFAST_SEC)
        & (df_all["confirmkeepdata"] != 0)
        & (df_all["consent"] != 0)
        & (df_all["age"] >= 18)
    ].copy().reset_index(drop=True)

    if len(clean_df) != 7_945:
        raise ValueError(f"Expected N=7,945 analytic respondents, got {len(clean_df)}")

    clean_df["foplcond"] = clean_df["foplcond"].astype(int)
    clean_df["arm_suffix"] = clean_df["foplcond"].map(ARM_SUFFIX)
    clean_df["arm"] = clean_df["foplcond"].map(ARM_NAMES)
    clean_df["CASEID"] = [f"grummon2023_{i:04d}" for i in range(1, len(clean_df) + 1)]

    clean_df["snack_healthyY_W"] = pd.to_numeric(clean_df["snack_healthyY_W"], errors="raise")
    if int((clean_df["snack_healthyY_W"] == 10).sum()) != 55:
        raise ValueError("Expected exactly 55 rows with snack_healthyY_W == 10")
    clean_df.loc[clean_df["snack_healthyY_W"] == 10, "snack_healthyY_W"] = 5

    for stem in OUTCOME_STEMS:
        merged = pd.Series(np.nan, index=clean_df.index, dtype="float64")
        for cond_code, sfx, _ in ARM_ORDER:
            arm_col = f"{stem}_{sfx}"
            if arm_col not in clean_df.columns:
                raise KeyError(f"Missing expected arm column {arm_col!r}")
            in_arm = clean_df["foplcond"] == cond_code
            if len(clean_df.loc[~in_arm, arm_col].dropna()) != 0:
                raise ValueError(f"Cross-arm leakage detected in {arm_col}")
            merged.loc[in_arm] = pd.to_numeric(clean_df.loc[in_arm, arm_col], errors="raise")
        clean_df[stem] = merged

    bev_chars = pd.read_stata(BEV_DTA, convert_categoricals=False).sort_values("itemnumber_bev").reset_index(drop=True)
    snack_chars = pd.read_stata(SNACK_DTA, convert_categoricals=False).sort_values("itemnumber_snack").reset_index(drop=True)
    if len(bev_chars) != 16 or len(snack_chars) != 16:
        raise ValueError("Expected 16 beverages and 16 snacks")

    return clean_df, bev_chars, snack_chars


def build_product_catalog(bev_chars: pd.DataFrame, snack_chars: pd.DataFrame) -> dict[str, list[dict]]:
    """Standardize the 16 beverage and 16 snack items with exact nutrition and warning flags."""
    catalog: dict[str, list[dict]] = {"bev": [], "snack": []}
    specs = [
        ("bev", bev_chars, "bev"),
        ("snack", snack_chars, "snack"),
    ]
    for kind, table, sfx in specs:
        for _, row in table.iterrows():
            item_num = int(row[f"itemnumber_{sfx}"])
            raw_label = " ".join(str(row[f"label_{sfx}"]).split())
            display_name = PRODUCT_NAME_OVERRIDES[(kind, item_num)] if (kind, item_num) in PRODUCT_NAME_OVERRIDES else raw_label
            true_cal = int(round(float(row[f"item_cal_{sfx}"])))
            shown_cal = (
                DISPLAYED_CALORIE_OVERRIDES[(kind, item_num)]
                if (kind, item_num) in DISPLAYED_CALORIE_OVERRIDES
                else true_cal
            )
            col_code = str(row["item_col_str_jo"]).strip()
            if col_code not in {"G", "Y", "R"}:
                raise ValueError(f"Unexpected traffic-light color {col_code!r} on {kind} #{item_num}")
            catalog[kind].append(
                {
                    "kind": kind,
                    "item_num": item_num,
                    "raw_label": str(row[f"label_{sfx}"]),
                    "name": display_name,
                    "true_cal": true_cal,
                    "shown_cal": shown_cal,
                    "svgs": float(row[f"item_svgs_{sfx}"]),
                    "color": col_code,
                    "phys_min": int(round(float(row["item_phys_jo"]))),
                    "warn_calories": int(row[f"warn_calories_{sfx}"]) == 1,
                    "warn_satfat": int(row[f"warn_satfat_{sfx}"]) == 1,
                    "warn_sugar": int(row[f"warn_sugar_{sfx}"]) == 1,
                    "warn_sodium": int(row[f"warn_sodium_{sfx}"]) == 1,
                }
            )
    return catalog


def format_product_fop_label(prod: dict, arm_name: str) -> str:
    """Render the exact front-of-package badge(s) shown next to `prod` in `arm_name` (Figure 1A & 1B)."""
    cal_badge = f"[{prod['shown_cal']} CALORIES PER SERVING (white calorie label)]"
    badges = [cal_badge]

    if arm_name == "Calorie":
        pass
    elif arm_name == "Green":
        if prod["color"] == "G":
            badges.append("[CHOOSE OFTEN (green label, smiling face)]")
    elif arm_name == "TrafficLight":
        tll_map = {
            "G": "[CHOOSE OFTEN (green label, smiling face)]",
            "Y": "[CHOOSE SOMETIMES (yellow label, neutral face)]",
            "R": "[CHOOSE RARELY (red label, frowning face)]",
        }
        badges.append(tll_map[prod["color"]])
    elif arm_name == "PhysicalActivity":
        badges.append(f"[{prod['phys_min']} MINUTES (orange label, walking person icon)]")
    elif arm_name == "Warning":
        # Ordered left-to-right matching Figure 1A: Calories, Saturated Fat, Sugar, Sodium
        if prod["warn_calories"]:
            badges.append("[HIGH IN CALORIES (black octagon)]")
        if prod["warn_satfat"]:
            badges.append("[HIGH IN SATURATED FAT (black octagon)]")
        if prod["warn_sugar"]:
            badges.append("[HIGH IN SUGAR (black octagon)]")
        if prod["warn_sodium"]:
            badges.append("[HIGH IN SODIUM (black octagon)]")
    else:
        raise ValueError(f"Unknown arm_name: {arm_name!r}")

    return "Front-of-package label: " + "; ".join(badges)


def build_vending_context(kind: str, arm_name: str) -> str:
    """Build the pre-choice instruction + arm digital poster shown before `bev_choice` or `snack_choice`."""
    if kind == "bev":
        parts = [VENDING_BONUS_INTRO, BEV_SECTION_INTRO]
        poster = BEV_POSTERS[arm_name]
    elif kind == "snack":
        parts = [VENDING_BONUS_INTRO, SNACK_SECTION_INTRO]
        poster = SNACK_POSTERS[arm_name]
    else:
        raise ValueError(f"Unexpected vending kind: {kind!r}")
    if poster:
        parts.append(poster)
    return "\n\n".join(parts)


def get_arm_variant_spec(
    stem: str,
    arm_name: str,
    product_catalog: dict[str, list[dict]],
) -> dict:
    """Return `{context, question_stem, option_codes, option_texts, option_descriptions}` for `(stem, arm_name)`."""
    if stem == "bev_choice":
        prods = product_catalog["bev"]
        return {
            "context": build_vending_context("bev", arm_name),
            "question_stem": (
                "Imagine you are looking at a vending machine because you want to purchase a beverage. "
                "Take a few minutes to look at the beverages below. Make sure to scroll to see all of the options. "
                "Then, click on the item you would most like to purchase."
            ),
            "option_codes": [p["item_num"] for p in prods],
            "option_texts": [p["name"] for p in prods],
            "option_descriptions": [format_product_fop_label(p, arm_name) for p in prods],
        }

    if stem == "snack_choice":
        prods = product_catalog["snack"]
        return {
            "context": build_vending_context("snack", arm_name),
            "question_stem": (
                "Imagine you are looking at a vending machine because you want to purchase a snack. "
                "Take a few minutes to look at the products below. Make sure to scroll to see all of the options. "
                "Then, click on the item you would most like to purchase."
            ),
            "option_codes": [p["item_num"] for p in prods],
            "option_texts": [p["name"] for p in prods],
            "option_descriptions": [format_product_fop_label(p, arm_name) for p in prods],
        }

    if stem in HEALTHY_ITEM_TARGETS:
        kind, item_num, q_stem = HEALTHY_ITEM_TARGETS[stem]
        prod = product_catalog[kind][item_num - 1]
        if prod["item_num"] != item_num:
            raise ValueError(f"Catalog index mismatch for {stem}: expected #{item_num}, got #{prod['item_num']}")
        fop = format_product_fop_label(prod, arm_name)
        return {
            "context": f"Product shown: {prod['name']}\n{fop}",
            "question_stem": q_stem,
            "option_codes": [c for c, _ in HEALTHY_5],
            "option_texts": [t for _, t in HEALTHY_5],
            "option_descriptions": None,
        }

    if stem == "noticed_cal":
        return {
            "context": RETROSPECTIVE_TASK_CONTEXT[arm_name],
            "question_stem": (
                "When you were selecting a beverage and a snack, did you notice any calorie labels "
                "on or next to the products?"
            ),
            "option_codes": [c for c, _ in NOTICED_3],
            "option_texts": [t for _, t in NOTICED_3],
            "option_descriptions": None,
        }

    if stem == "noticed_trial":
        return {
            "context": RETROSPECTIVE_TASK_CONTEXT[arm_name],
            "question_stem": (
                "When you were selecting a beverage and a snack, did you notice any other nutrition labels "
                "on or next to the products?"
            ),
            "option_codes": [c for c, _ in NOTICED_3],
            "option_texts": [t for _, t in NOTICED_3],
            "option_descriptions": None,
        }

    if stem in LABEL_REACTION_SPECS:
        spec = LABEL_REACTION_SPECS[stem]
        q_stem = spec["plural"] if arm_name in PLURAL_LABEL_ARMS else spec["singular"]
        return {
            "context": STANDALONE_LABEL_STIMULUS[arm_name],
            "question_stem": q_stem,
            "option_codes": [c for c, _ in spec["options"]],
            "option_texts": [t for _, t in spec["options"]],
            "option_descriptions": None,
        }

    raise KeyError(f"Unknown outcome stem: {stem!r}")


def _count_codes(series: pd.Series, codes: list[int]) -> list[int]:
    vc = series.dropna().astype(int).value_counts()
    return [int(vc[c]) if c in vc.index else 0 for c in codes]


def build_survey_audit_artifacts(
    clean_df: pd.DataFrame | None = None,
    bev_chars: pd.DataFrame | None = None,
    snack_chars: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_dict, details_payload)` covering the entire VLOSurveyItems.docx flow."""
    if clean_df is None or bev_chars is None or snack_chars is None:
        clean_df, bev_chars, snack_chars = load_analytic_cohort()
    product_catalog = build_product_catalog(bev_chars, snack_chars)

    rows: list[dict[str, str]] = []
    tags_by_key: dict[str, list[str]] = {}
    items_detail: dict[str, dict] = {}
    study_contexts: list[dict] = []

    def add_entry(
        *,
        qid: str,
        csv_cols: list[str],
        family: str,
        block: str,
        n_answered: int,
        include: bool,
        kev_type: str,
        context_source: str,
        review_note: str,
        question_text: str,
        context_preview: str,
        options: list[dict],
        paired_variants: list[dict],
        tags: list[str],
        q_type_meta: str = "MC",
        selector_meta: str = "SAVR",
        sample_texts: list[str] | None = None,
    ) -> None:
        idx = len(rows)
        row_key = f"{idx}:{qid}"
        opts_preview = (
            f"[{len(options)}] " + " | ".join(o["option"] for o in options)
            if options
            else ""
        )
        rows.append(
            {
                "catalog_idx": str(idx),
                "QuestionID": qid,
                "csv_cols": ";".join(csv_cols),
                "n_cols": str(len(csv_cols)),
                "family": family,
                "BlockName": block,
                "n_subitems": "1",
                "n_answered": str(n_answered),
                "include": "True" if include else "False",
                "kev_type": kev_type,
                "context_source": context_source,
                "context_sep": "",
                "drop_options": "",
                "review_note": review_note,
                "QuestionText": question_text,
                "Context_preview": context_preview,
                "Rows_preview": "",
                "Options_preview": opts_preview,
            }
        )
        if tags:
            tags_by_key[row_key] = list(tags)

        qtext_variants = []
        db_variants = []
        if paired_variants:
            seen_stems: OrderedDict[str, tuple[str, int]] = OrderedDict()
            seen_dbs: OrderedDict[str, tuple[str, int]] = OrderedDict()
            for pv in paired_variants:
                st = pv["stem"]
                if st not in seen_stems:
                    seen_stems[st] = (pv["in_block_db_qid"], pv["n"])
                else:
                    lbl, prev_n = seen_stems[st]
                    seen_stems[st] = (lbl, prev_n + pv["n"])
                db_t = pv["in_block_db_text"]
                if db_t:
                    if db_t not in seen_dbs:
                        seen_dbs[db_t] = (pv["in_block_db_qid"], pv["n"])
                    else:
                        lbl, prev_n = seen_dbs[db_t]
                        seen_dbs[db_t] = (lbl, prev_n + pv["n"])
            qtext_variants = [
                {"label": lbl, "text": txt, "n": n}
                for txt, (lbl, n) in seen_stems.items()
            ]
            db_variants = [
                {"label": lbl, "text": txt, "n": n}
                for txt, (lbl, n) in seen_dbs.items()
            ]

        items_detail[row_key] = {
            "QuestionType": q_type_meta,
            "Selector": selector_meta,
            "rows": [],
            "options": options,
            "row_breakdowns": [],
            "qtext_variants": qtext_variants,
            "db_variants": db_variants,
            "opt_variant_sets": [],
            "paired_variants": paired_variants,
            "paired_variants_total": len(paired_variants),
            "sample_texts": sample_texts if sample_texts is not None else [],
            "in_block_db_text": context_preview,
        }

        if family == "0_DB":
            study_contexts.append(
                {
                    "qid": qid,
                    "block": block,
                    "is_question": False,
                    "n_variants": max(1, len(paired_variants)),
                    "preview": question_text[:140],
                    "full_text": question_text,
                }
            )

    def add_outcome(stem: str, block: str, kev_type: str, include: bool, tags: list[str], note: str) -> None:
        arm_cols = [f"{stem}_{sfx}" for _, sfx, _ in ARM_ORDER]
        paired_variants: list[dict] = []
        ref_spec = get_arm_variant_spec(stem, "Calorie", product_catalog)
        pooled_counts = [0] * len(ref_spec["option_texts"])

        for cond_code, _, arm_name in ARM_ORDER:
            spec = get_arm_variant_spec(stem, arm_name, product_catalog)
            sub_s = clean_df.loc[clean_df["foplcond"] == cond_code, stem]
            counts = _count_codes(sub_s, spec["option_codes"])
            for idx_c, c_val in enumerate(counts):
                pooled_counts[idx_c] += c_val
            opt_descs = spec["option_descriptions"]
            pv_opts = []
            for idx_o, (opt_txt, cnt) in enumerate(zip(spec["option_texts"], counts, strict=True)):
                entry: dict[str, object] = {"option": opt_txt, "count": cnt}
                if opt_descs is not None:
                    entry["description"] = opt_descs[idx_o]
                pv_opts.append(entry)
            paired_variants.append(
                {
                    "n": int(sub_s.notna().sum()),
                    "stem": spec["question_stem"],
                    "in_block_db_qid": arm_name,
                    "in_block_db_text": spec["context"],
                    "qid_texts": {},
                    "options": pv_opts,
                    "row_counts": [],
                }
            )

        pooled_opts: list[dict] = []
        ref_descs = ref_spec["option_descriptions"]
        for idx_o, (opt_txt, cnt) in enumerate(zip(ref_spec["option_texts"], pooled_counts, strict=True)):
            entry = {"option": opt_txt, "count": cnt}
            if ref_descs is not None:
                entry["description"] = ref_descs[idx_o]
            pooled_opts.append(entry)

        family = "1_Choice_Multi" if kev_type == "choice" else "2_Ordinal_Scale"
        add_entry(
            qid=stem,
            csv_cols=arm_cols,
            family=family,
            block=block,
            n_answered=int(clean_df[stem].notna().sum()),
            include=include,
            kev_type=kev_type,
            context_source="in_block_db",
            review_note=note,
            question_text=ref_spec["question_stem"],
            context_preview=ref_spec["context"],
            options=pooled_opts,
            paired_variants=paired_variants,
            tags=tags,
        )

    # --- 1. Introduction: Consent & Captcha ([005]-[017]) ---
    consent_counts = _count_codes(clean_df["consent"], [1, 0])
    add_entry(
        qid="consent",
        csv_cols=["consent"],
        family="0_Consent_Check",
        block="Introduction",
        n_answered=int(clean_df["consent"].notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note="Informed consent screen ([006]-[014]).",
        question_text="Do you consent to be part of the research study?",
        context_preview="",
        options=[{"option": "Yes", "count": consent_counts[0]}, {"option": "No", "count": consent_counts[1]}],
        paired_variants=[],
        tags=[],
    )
    add_entry(
        qid="captcha",
        csv_cols=["Q_RecaptchaScore"],
        family="0_Consent_Check",
        block="Introduction",
        n_answered=int(clean_df["Q_RecaptchaScore"].notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note="Qualtrics reCAPTCHA check ([015]-[017]).",
        question_text="Please complete the captcha.",
        context_preview="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="Captcha",
        selector_meta="",
    )

    # --- 2. Vending Machine Selection Tasks ([018]-[064]) ---
    add_entry(
        qid="vending_intro",
        csv_cols=[],
        family="0_DB",
        block="Initial Instructions",
        n_answered=len(clean_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Incentive-compatibility bonus instructions shown to all participants ([018]-[021]).",
        question_text=VENDING_BONUS_INTRO,
        context_preview="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )
    add_outcome(
        stem="bev_choice",
        block="Beverage Selection Task",
        kev_type="choice",
        include=True,
        tags=[],
        note="Primary outcome 1: 16-option beverage vending machine selection with arm-specific poster and per-product FOP badges (order of bev vs snack randomized via exptorder).",
    )
    add_outcome(
        stem="snack_choice",
        block="Snack Selection Task",
        kev_type="choice",
        include=True,
        tags=[],
        note="Primary outcome 2: 16-option snack vending machine selection with arm-specific poster and per-product FOP badges.",
    )

    # --- 3. Single-Product Healthfulness Perceptions ([065]-[126]) ---
    for stem, prod_desc in [
        ("bev_healthyR", "Beverage healthfulness (Red / unhealthy): Gatorade Fruit Punch - 20oz (Table 2)."),
        ("bev_healthyY", "Beverage healthfulness (Yellow / moderately healthy): G2 Grape - 20oz (Table 2)."),
        ("bev_healthyG", "Beverage healthfulness (Green / healthy): Schweppes Black Cherry Seltzer - 20oz (Table 2)."),
    ]:
        add_outcome(
            stem=stem,
            block="Healthfulness Perceptions – Beverages",
            kev_type="score",
            include=True,
            tags=[],
            note=prod_desc,
        )

    for stem, prod_desc in [
        ("snack_healthyR", "Snack healthfulness (Red / unhealthy): Lay's Classic Potato Chips - 1oz (Table 2)."),
        ("snack_healthyY", "Snack healthfulness (Yellow / moderately healthy): Ritz Bits Cheese Cracker Sandwiches - 1oz (Table 2; 55 raw code-10 typos repaired to 5 per 01_DataPrep.do)."),
        ("snack_healthyG", "Snack healthfulness (Green / healthy): Snyder's Old Tyme Pretzels - 1.58oz (Table 2)."),
    ]:
        add_outcome(
            stem=stem,
            block="Healthfulness Perceptions – Snacks",
            kev_type="score",
            include=True,
            tags=[],
            note=prod_desc,
        )

    # --- 4. Label Reactions and Perceptions ([127]-[302]) ---
    add_outcome(
        stem="noticed_cal",
        block="Label Reactions and Perceptions",
        kev_type="choice",
        include=False,
        tags=["prior-task-dep"],
        note="Retrospective recall of calorie labels from preceding vending selection task ([129]-[133]).",
    )
    add_outcome(
        stem="noticed_trial",
        block="Label Reactions and Perceptions",
        kev_type="choice",
        include=False,
        tags=["prior-task-dep", "in-grummon2025"],
        note="Retrospective recall of non-calorie labels ([134]-[138]); direct counterpart to Grummon 2025 item noticed_trial (same Yes / No / Not sure options; also in grummon2025_US_long.parquet).",
    )

    for stem, spec in LABEL_REACTION_SPECS.items():
        add_outcome(
            stem=stem,
            block=spec["block"],
            kev_type=spec["kev_type"],
            include=spec["include"],
            tags=spec["tags"],
            note=spec["note"],
        )

    # --- 5. Demographics & Covariates ([303]-[592]) ---
    demo_specs = [
        (
            "age",
            ["age"],
            "0_Demographics",
            "How old are you? [free numeric response, 18–115]",
            [],
            ["in-grummon2025-persona"],
            "Respondent age in years; also a persona attribute in Grummon 2025 (AGE).",
            "TE",
        ),
        (
            "gender",
            ["gender", "gender_4_TEXT"],
            "0_Demographics",
            "How do you identify?",
            [(1, "Woman"), (2, "Man"), (3, "Non-binary"), (4, "Prefer to self-describe")],
            ["in-grummon2025-persona"],
            "Gender identity; same coding as Grummon 2025 gender (1=Woman -> Female, 2=Man -> Male).",
            "MC",
        ),
        (
            "latino",
            ["latino"],
            "0_Demographics",
            "Are you of Hispanic, Latino, or Spanish origin?",
            [(1, "Yes"), (0, "No")],
            ["in-grummon2025-persona"],
            "Hispanic/Latino ethnicity; also a persona attribute in Grummon 2025 (HISPANIC).",
            "MC",
        ),
        (
            "Race",
            ["Race_1", "Race_2", "Race_3", "Race_4", "Race_5", "Race_6", "Race_6_TEXT"],
            "0_Demographics",
            "What is your race? (Check all that apply).",
            [
                (1, "American Indian or Alaska Native"),
                (2, "Asian"),
                (3, "Black or African American"),
                (4, "Native Hawaiian or Other Pacific Islander"),
                (5, "White"),
                (6, "Another race"),
            ],
            ["in-grummon2025-persona"],
            "Multi-select race checkboxes; also a persona attribute in Grummon 2025 (RACE).",
            "MC_Multi",
        ),
        (
            "educ",
            ["educ"],
            "0_Demographics",
            "What is the highest degree or level of school you have completed?",
            [
                (1, "Less than high school"),
                (2, "High school graduate (or GED)"),
                (3, "Some college or technical school"),
                (4, "Associate's degree"),
                (5, "Bachelor's degree"),
                (6, "Graduate or professional degree"),
            ],
            ["in-grummon2025-persona"],
            "6-level education collected right after consent; identical coding to Grummon 2025 educ (EDULEVEL).",
            "MC",
        ),
        (
            "dietqual",
            ["dietqual"],
            "0_Demographics",
            "In general, how healthy is your overall diet? Would you say…",
            [(1, "Poor"), (2, "Fair"), (3, "Good"), (4, "Very good"), (5, "Excellent")],
            [],
            "Self-rated overall diet quality covariate (Table 1).",
            "MC",
        ),
        (
            "numeracy",
            ["numeracy"],
            "0_Demographics",
            "Out of 1,000 people in a small town 500 are members of a choir. Out of these 500 members in the choir 100 are men. Out of the 500 inhabitants that are not in the choir 300 are men. What is the probability that a randomly drawn man is a member of the choir? (0–100%)",
            [],
            [],
            "Mental math numeracy item (correct answer = 25%; Table 1).",
            "TE",
        ),
        (
            "healthlit",
            ["healthlit"],
            "0_Demographics",
            "How often do you need to have someone help you when you read instructions, pamphlets, or other written materials from your doctor or pharmacy?",
            [(1, "Never"), (2, "Rarely"), (3, "Sometimes"), (4, "Most of the time"), (5, "Always")],
            [],
            "Single-item Morris health literacy screener (note: Grummon 2025 HEALTHLIT instead uses a 4-item Newest Vital Sign nutrition facts quiz, 0–4 correct).",
            "MC",
        ),
        (
            "NFPuse",
            ["NFPuse"],
            "0_Demographics",
            "How often do you use the Nutrition Facts label when deciding to buy a food product?",
            [(1, "Never"), (2, "Rarely"), (3, "Sometimes"), (4, "Most of the time"), (5, "Always")],
            ["in-grummon2025-persona"],
            "Nutrition Facts label usage frequency; identical item and 1–5 scale to Grummon 2025 nfpuse (NFPUSE).",
            "MC",
        ),
        (
            "vendingfreq_bev",
            ["vendingfreq_bev"],
            "0_Demographics",
            "In a typical week, how often do you buy beverages from vending machines?",
            [
                (1, "Never or less than 1 time per week"),
                (2, "1 to 2 times per week"),
                (3, "3 to 4 times per week"),
                (4, "5 or more times per week"),
            ],
            [],
            "Baseline beverage vending machine purchase frequency (Table 1).",
            "MC",
        ),
        (
            "vendingfreq_snack",
            ["vendingfreq_snack"],
            "0_Demographics",
            "In a typical week, how often do you buy snacks from vending machines?",
            [
                (1, "Never or less than 1 time per week"),
                (2, "1 to 2 times per week"),
                (3, "3 to 4 times per week"),
                (4, "5 or more times per week"),
            ],
            [],
            "Baseline snack vending machine purchase frequency (Table 1).",
            "MC",
        ),
        (
            "partyid",
            ["partyid", "partyid_8_TEXT"],
            "0_Demographics",
            "Do you consider yourself a Democrat, a Republican, an Independent, or something else?",
            [
                (1, "Strong Democrat"),
                (2, "Moderate Democrat"),
                (3, "Lean Democrat"),
                (4, "Independent"),
                (5, "Lean Republican"),
                (6, "Moderate Republican"),
                (7, "Strong Republican"),
                (8, "Something else"),
            ],
            [],
            "7-point partisan identification + Something else (Table 1).",
            "MC",
        ),
        (
            "hhsize_num",
            ["hhsize_num"],
            "0_Demographics",
            "How many people are in your household, including you? [1–20]",
            [],
            ["in-grummon2025-persona"],
            "Household size (also collected in Grummon 2025 hhsize_num).",
            "TE",
        ),
        (
            "children_num",
            ["children_num"],
            "0_Demographics",
            "How many children (ages 0–18) currently live in your household? [0–15]",
            [],
            ["in-grummon2025-persona"],
            "Number of children in household (also collected in Grummon 2025 children_num).",
            "TE",
        ),
        (
            "height_weight",
            ["height_feet", "height_inches", "weight_lbs"],
            "0_Demographics",
            "How tall are you (feet and inches)? / How much do you weigh (pounds)?",
            [],
            [],
            "Self-reported height and weight used to compute BMI category in Table 1.",
            "TE",
        ),
        (
            "income_10cat",
            ["income_10cat"],
            "0_Demographics",
            "Which of the following categories best describes your total household income before taxes in the last 12 months?",
            [
                (1, "Less than $10,000"),
                (2, "$10,000 to $14,999"),
                (3, "$15,000 to $24,999"),
                (4, "$25,000 to $34,999"),
                (5, "$35,000 to $49,999"),
                (6, "$50,000 to $74,999"),
                (7, "$75,000 to $99,999"),
                (8, "$100,000 to $149,999"),
                (9, "$150,000 to $199,999"),
                (10, "$200,000 or more"),
            ],
            ["in-grummon2025-persona"],
            "10-category household income; identical brackets to Grummon 2025 income (US_INC).",
            "MC",
        ),
        (
            "fpl_income",
            [f"fpl_{k}" for k in range(1, 21)],
            "0_Demographics",
            "Which of the following categories best describes your total household income before taxes in the last 12 months? [household-size-specific Federal Poverty Level brackets]",
            [
                (1, "<150% FPL threshold for household size"),
                (2, "150%–199% FPL threshold"),
                (3, "200%–249% FPL threshold"),
                (4, "250%–299% FPL threshold"),
                (5, ">=300% FPL threshold"),
            ],
            [],
            "Household-size-conditioned FPL bracket follow-up ([433]-[592]; Table 1).",
            "MC",
        ),
    ]

    for qid, cols, fam, q_txt, opt_pairs, d_tags, note, q_meta in demo_specs:
        if qid == "Race":
            opt_list = []
            for code_idx, label in opt_pairs:
                col_s = pd.to_numeric(clean_df[f"Race_{code_idx}"], errors="raise")
                opt_list.append({"option": label, "count": int((col_s == 1).sum())})
            n_ans = int(clean_df[[f"Race_{k}" for k in range(1, 7)]].notna().any(axis=1).sum())
            samples: list[str] = []
        elif qid == "fpl_income":
            fpl_merged = clean_df[cols].apply(pd.to_numeric, errors="raise").bfill(axis=1).iloc[:, 0]
            cnts = _count_codes(fpl_merged, [c for c, _ in opt_pairs])
            opt_list = [{"option": lbl, "count": cnt} for (_, lbl), cnt in zip(opt_pairs, cnts, strict=True)]
            n_ans = int(fpl_merged.notna().sum())
            samples = []
        elif opt_pairs:
            s_num = pd.to_numeric(clean_df[cols[0]], errors="raise")
            cnts = _count_codes(s_num, [c for c, _ in opt_pairs])
            opt_list = [{"option": lbl, "count": cnt} for (_, lbl), cnt in zip(opt_pairs, cnts, strict=True)]
            n_ans = int(s_num.notna().sum())
            samples = []
        else:
            opt_list = []
            n_ans = int(clean_df[cols].notna().any(axis=1).sum())
            s_clean = clean_df[cols[0]].dropna().astype(str)
            samples = s_clean.head(8).tolist()

        add_entry(
            qid=qid,
            csv_cols=cols,
            family=fam,
            block="Demographics",
            n_answered=n_ans,
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=q_txt,
            context_preview="",
            options=opt_list,
            paired_variants=[],
            tags=d_tags,
            q_type_meta=q_meta,
            selector_meta="SAVR" if opt_pairs else "",
            sample_texts=samples,
        )

    # --- 6. Closure & Debrief ([593]-[609]) ---
    any_s = clean_df["anythingelse"].dropna().astype(str)
    any_s = any_s[any_s.str.strip() != ""]
    add_entry(
        qid="anythingelse",
        csv_cols=["anythingelse"],
        family="0_TE",
        block="Closure",
        n_answered=int(len(any_s)),
        include=False,
        kev_type="",
        context_source="",
        review_note="Open-ended final comments box ([594]-[596]).",
        question_text="Is there anything else you want to tell us?",
        context_preview="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="TE",
        selector_meta="ML",
        sample_texts=any_s.head(8).tolist(),
    )

    keep_s = pd.to_numeric(clean_df["keepdata"], errors="raise")
    keep_cnts = _count_codes(keep_s, [1, 0])
    add_entry(
        qid="keepdata",
        csv_cols=["keepdata", "confirmkeepdata"],
        family="0_Consent_Check",
        block="Debrief",
        n_answered=int(keep_s.notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note="Post-debrief data-inclusion consent check ([597]-[607]; respondents confirming removal were excluded per CONSORT).",
        question_text="Please choose one of the options below.",
        context_preview="Debriefing form explaining study purpose and $2.50 gift card bonus.",
        options=[
            {"option": "I agree to have my data included in this study.", "count": keep_cnts[0]},
            {"option": "I no longer wish to participate and/or I would like my data removed from the study.", "count": keep_cnts[1]},
        ],
        paired_variants=[],
        tags=[],
    )

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 2,
        "schema_kind": "question_audit",
        "study_contexts": {"__all__": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload


def write_grummon2023_audit_files(out_dir: Path = GRUMMON2023_DIR) -> tuple[Path, Path, Path]:
    """Generate `question_audit.csv`, `question_tags.json`, and `.audit_details.json` in `out_dir`."""
    audit_df, tags_by_key, details_payload = build_survey_audit_artifacts()
    csv_path = out_dir / "question_audit.csv"
    tags_path = out_dir / "question_tags.json"
    details_path = out_dir / ".audit_details.json"

    audit_df.to_csv(csv_path, index=False)
    tags_path.write_text(json.dumps(tags_by_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    details_path.write_text(json.dumps(details_payload, ensure_ascii=False), encoding="utf-8")
    return csv_path, tags_path, details_path

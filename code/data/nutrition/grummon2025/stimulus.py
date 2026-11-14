"""Survey catalog, arm-specific visual stimulus builders, and Audit UI indexers for Grummon et al. (2025)."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import pandas as pd

GRUMMON2025_DIR = Path(__file__).resolve().parent
RAW_CSV = GRUMMON2025_DIR / "ParentFile_wide.csv"

# Canonical 6 trial arms (Positive is the control/reference arm in OLS and eTable 9)
ARM_NAMES = [
    "Positive",
    "NutritionInfo",
    "HighIn",
    "Positive+NutritionInfo",
    "Positive+HighIn",
    "Spectrum",
]

SHOPPING_INTRO_TEXT = (
    "You will now view an online grocery store and complete a shopping task. Then, you will answer survey questions.\n"
    "The online grocery store looks like a typical store, but only offers the following types of foods: "
    "Beverages, Breads, Cereals, Soups, Boxed and frozen meals, and Snacks.\n"
    "Enter the online store and select the items you would like to purchase among the types of foods offered in this store. "
    "You should shop as you normally would for these types of foods. For example, if you usually buy bread but not soup, "
    "you should select your preferred bread from the store, but you don't need to select a soup.\n"
    "Please spend about $35 or less. You will not be asked to spend your own money. However, you should make your selections "
    "carefully because we will choose 1 in 50 participants to receive the groceries they selected, plus any money left over. "
    "In total, about 100 participants will receive the groceries they selected, plus any money left over in their budget. "
    "This means you should select products you actually want."
)

CONSUMER_UNDERSTANDING_INTRO = (
    "The next questions are about different products. "
    "You may or may not have seen these products during the shopping task."
)

DEBRIEF_TEXT = (
    "Thank you for taking part in this research study. In the survey we indicated that 1 in 50 participants would be "
    "randomly selected to have the products they chose in the shopping task delivered to them. In actuality, all "
    "participants selected for this bonus will receive a $35 gift card, equal to your budget for the shopping task. "
    "This study was conducted by researchers at Stanford University and Harvard Pilgrim Health Care Institute to "
    "determine how effective different front-of-package labels are at changing which products people choose."
)

FOPL_STIMULUS = {
    "Positive": (
        "Blue star badge showing '[* GOOD]' (1 star), '[** BETTER]' (2 stars), or '[*** BEST]' (3 stars) "
        "on qualifying products; no front-of-package label on other products."
    ),
    "NutritionInfo": (
        "White box with magnifying glass icon on all products reading 'Nutrition Info | Per serving | "
        "Saturated Fat: Low (green) / Med (yellow) / High (red) | "
        "Sodium: Low (green) / Med (yellow) / High (red) | "
        "Added Sugars: Low (green) / Med (yellow) / High (red) | FDA.gov'."
    ),
    "HighIn": (
        "White box with magnifying glass icon reading 'High In | [Saturated Fat / Sodium / Added Sugars] | FDA.gov' "
        "when a product is high in those nutrients; no front-of-package label on other products."
    ),
    "Positive+NutritionInfo": (
        "Blue star badge ('[* GOOD]', '[** BETTER]', or '[*** BEST]' when earned) alongside white box on all products "
        "reading 'Nutrition Info | Per serving | Saturated Fat: Low (green) / Med (yellow) / High (red) | "
        "Sodium: Low (green) / Med (yellow) / High (red) | Added Sugars: Low (green) / Med (yellow) / High (red) | FDA.gov'."
    ),
    "Positive+HighIn": (
        "Blue star badge ('[* GOOD]', '[** BETTER]', or '[*** BEST]' when earned) alongside white box "
        "reading 'High In | [Saturated Fat / Sodium / Added Sugars] | FDA.gov' (when high in those nutrients); "
        "no front-of-package label on other products."
    ),
    "Spectrum": (
        "Color-coded 1-to-5 star badge on all products reading '[* POOR]' (1 star, red), '[** FAIR]' (2 stars, orange), "
        "'[*** GOOD]' (3 stars, yellow), '[**** BETTER]' (4 stars, light green), or '[***** BEST]' (5 stars, dark green)."
    ),
}

HEALTHIER_PROMPT = (
    "Click on the product you think is healthier, or click 'They're the same' "
    "if you think the two products are equally healthy."
)

# Each entry maps product_name -> (Positive, NutritionInfo, HighIn, Spectrum) base badge strings
PRODUCT_BASE_LABELS: OrderedDict[str, OrderedDict[str, tuple[str | None, str, str | None, str]]] = OrderedDict(
    [
        (
            "idbev",
            OrderedDict(
                [
                    (
                        "Ocean Spray 100% Cranberry juice",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: Low (green), Sodium: Low (green), Added Sugars: Low (green) | FDA.gov",
                            None,
                            "[** FAIR (2 stars, orange)]",
                        ),
                    ),
                    (
                        "Ocean Spray Cranberry Juice Cocktail",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: Low (green), Sodium: Low (green), Added Sugars: High (red) | FDA.gov",
                            "High In: Added Sugars | FDA.gov",
                            "[* POOR (1 star, red)]",
                        ),
                    ),
                ]
            ),
        ),
        (
            "idbread",
            OrderedDict(
                [
                    (
                        "Mission Extra Thin Corn Tortillas",
                        (
                            "[* GOOD (1 star, blue)]",
                            "Nutrition Info Per serving - Saturated Fat: Low (green), Sodium: Low (green), Added Sugars: Low (green) | FDA.gov",
                            None,
                            "[*** GOOD (3 stars, yellow)]",
                        ),
                    ),
                    (
                        "Mission Flour Burrito Tortillas",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: Med (yellow), Sodium: High (red), Added Sugars: Low (green) | FDA.gov",
                            "High In: Sodium | FDA.gov",
                            "[* POOR (1 star, red)]",
                        ),
                    ),
                ]
            ),
        ),
        (
            "idcereal",
            OrderedDict(
                [
                    (
                        "Post Selects Great Grains Crunchy Pecans",
                        (
                            "[* GOOD (1 star, blue)]",
                            "Nutrition Info Per serving - Saturated Fat: Low (green), Sodium: Med (yellow), Added Sugars: Med (yellow) | FDA.gov",
                            None,
                            "[*** GOOD (3 stars, yellow)]",
                        ),
                    ),
                    (
                        "General Mills Chocolate Chex Gluten Free Cereal",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: Low (green), Sodium: Med (yellow), Added Sugars: High (red) | FDA.gov",
                            "High In: Added Sugars | FDA.gov",
                            "[** FAIR (2 stars, orange)]",
                        ),
                    ),
                ]
            ),
        ),
        (
            "idsoup",
            OrderedDict(
                [
                    (
                        "Campbell's Healthy Request Chunk Sirloin Burger Soup",
                        (
                            "[* GOOD (1 star, blue)]",
                            "Nutrition Info Per serving - Saturated Fat: Med (yellow), Sodium: Med (yellow), Added Sugars: Low (green) | FDA.gov",
                            None,
                            "[*** GOOD (3 stars, yellow)]",
                        ),
                    ),
                    (
                        "Dinty Moore Beef Stew",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: Med (yellow), Sodium: High (red), Added Sugars: Low (green) | FDA.gov",
                            "High In: Sodium | FDA.gov",
                            "[* POOR (1 star, red)]",
                        ),
                    ),
                ]
            ),
        ),
        (
            "idsnack",
            OrderedDict(
                [
                    (
                        "Sun Chips Original multigrain snacks",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: Low (green), Sodium: Med (yellow), Added Sugars: Low (green) | FDA.gov",
                            None,
                            "[** FAIR (2 stars, orange)]",
                        ),
                    ),
                    (
                        "Nutella & Go! Cup",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: High (red), Sodium: Med (yellow), Added Sugars: High (red) | FDA.gov",
                            "High In: Saturated Fat, Added Sugars | FDA.gov",
                            "[* POOR (1 star, red)]",
                        ),
                    ),
                ]
            ),
        ),
        (
            "idmeal",
            OrderedDict(
                [
                    (
                        "Healthy Choice Cafe Steamers Chicken Marinara",
                        (
                            "[* GOOD (1 star, blue)]",
                            "Nutrition Info Per serving - Saturated Fat: Med (yellow), Sodium: Med (yellow), Added Sugars: Low (green) | FDA.gov",
                            None,
                            "[*** GOOD (3 stars, yellow)]",
                        ),
                    ),
                    (
                        "Lean Cuisine Turkey with Dressing",
                        (
                            None,
                            "Nutrition Info Per serving - Saturated Fat: Med (yellow), Sodium: High (red), Added Sugars: Med (yellow) | FDA.gov",
                            "High In: Sodium | FDA.gov",
                            "[** FAIR (2 stars, orange)]",
                        ),
                    ),
                ]
            ),
        ),
    ]
)

EXTENT_5 = [
    (1, "Not at all"),
    (2, "Very little"),
    (3, "Somewhat"),
    (4, "Quite a bit"),
    (5, "A great deal"),
]
# Note: eTable 3 & DataDictionary.docx use "Neither disagree nor agree" on rhelp/runderstand
# and "Neither agree nor disagree" on feltreal/picksimilar/similarusual.
DISAGREE_AGREE_5 = [
    (1, "Strongly disagree"),
    (2, "Somewhat disagree"),
    (3, "Neither disagree nor agree"),
    (4, "Somewhat agree"),
    (5, "Strongly agree"),
]
AGREE_DISAGREE_5 = [
    (1, "Strongly disagree"),
    (2, "Somewhat disagree"),
    (3, "Neither agree nor disagree"),
    (4, "Somewhat agree"),
    (5, "Strongly agree"),
]
TRUST_5 = [
    (1, "Completely distrust"),
    (2, "Somewhat distrust"),
    (3, "Neither trust nor distrust"),
    (4, "Somewhat trust"),
    (5, "Completely trust"),
]
SUPPORT_5 = [
    (1, "Strongly oppose"),
    (2, "Somewhat oppose"),
    (3, "Neither oppose nor support"),
    (4, "Somewhat support"),
    (5, "Strongly support"),
]
NOTICED_3 = [
    (1, "Yes"),
    (0, "No"),
    (2, "Not sure"),
]
YES_NO_2 = [
    (0, "No"),
    (1, "Yes"),
]

SECONDARY_OUTCOME_SPECS: OrderedDict[str, dict] = OrderedDict(
    [
        (
            "elaborationhealth",
            {
                "block": "Thinking About Healthfulness",
                "kev_type": "score",
                "include": False,
                "context_kind": "shopping_retrospective",
                "question_stem": (
                    "When you were selecting foods and beverages in the shopping task, "
                    "how much did you think about the overall healthfulness of the products?"
                ),
                "options": EXTENT_5,
                "tags": ["prior-task-dep", "grummon2023-similar"],
                "note": (
                    "Retrospective shopping-task cognitive elaboration (eTable 3, p. 7; eTable 9); "
                    "construct parallel to Grummon 2023 standalone item `elab`."
                ),
            },
        ),
        (
            "noticed_trial",
            {
                "block": "Label Noticing and Use",
                "kev_type": "choice",
                "include": False,
                "context_kind": "shopping_retrospective",
                "question_stem": (
                    "When you were selecting foods and beverages in the shopping task, did you notice "
                    "any other nutrition labels next to the products (other than the Nutrition Facts Panel)?"
                ),
                "options": NOTICED_3,
                "tags": ["prior-task-dep", "has-dont-know", "in-grummon2023"],
                "note": (
                    "Raw 3-option retrospective label-noticing item (`1=Yes`, `0=No`, `2=Not sure`; eTable 3, p. 8); "
                    "direct counterpart to `noticed_trial` in Grummon 2023."
                ),
            },
        ),
        (
            "rnoticed_trial",
            {
                "block": "Label Noticing and Use",
                "kev_type": "choice",
                "include": False,
                "context_kind": "shopping_retrospective",
                "question_stem": (
                    "When you were selecting foods and beverages in the shopping task, did you notice "
                    "any other nutrition labels next to the products (other than the Nutrition Facts Panel)?"
                ),
                "options": YES_NO_2,
                "tags": ["prior-task-dep", "in-grummon2023"],
                "note": (
                    "Derived binary recode of `noticed_trial` (`1=Yes` [2,281], `0=No or Not sure` [3,351]) "
                    "used in eTable 9 and `Grummon2025.ipynb` `QUESTIONS`."
                ),
            },
        ),
        (
            "rused_trial",
            {
                "block": "Label Noticing and Use",
                "kev_type": "choice",
                "include": False,
                "context_kind": "shopping_retrospective",
                "question_stem": (
                    "While you were shopping, did you use these other nutrition labels to decide "
                    "which foods and beverages to select?"
                ),
                "options": YES_NO_2,
                "tags": ["prior-task-dep", "low-n", "grummon2023-similar"],
                "note": (
                    "Displayed only to the n=2,281 respondents who answered `1=Yes` on `noticed_trial` "
                    "(eTable 3, p. 8); parallel to `infl_bev`/`infl_snack` in Grummon 2023."
                ),
            },
        ),
        (
            "raffect_worried",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": "How much do these labels make you feel worried?",
                "options": EXTENT_5,
                "tags": ["in-grummon2023"],
                "note": (
                    "Negative emotional reaction matrix item 1/5 (eTable 3, p. 8); same stem/scale as "
                    "`affect_worried` in Grummon 2023 (evaluated on 2025 FOPL designs)."
                ),
            },
        ),
        (
            "raffect_scared",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": "How much do these labels make you feel scared?",
                "options": EXTENT_5,
                "tags": ["in-grummon2023"],
                "note": (
                    "Negative emotional reaction matrix item 2/5 (eTable 3, p. 8); same stem/scale as "
                    "`affect_scared` in Grummon 2023."
                ),
            },
        ),
        (
            "raffect_guilty",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": "How much do these labels make you feel guilty?",
                "options": EXTENT_5,
                "tags": ["in-grummon2023"],
                "note": (
                    "Negative emotional reaction matrix item 3/5 (eTable 3, p. 8); same stem/scale as "
                    "`affect_guilty` in Grummon 2023."
                ),
            },
        ),
        (
            "raffect_ashamed",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": "How much do these labels make you feel ashamed?",
                "options": EXTENT_5,
                "tags": ["in-grummon2023"],
                "note": (
                    "Negative emotional reaction matrix item 4/5 (eTable 3, p. 8); same stem/scale as "
                    "`affect_ashamed` in Grummon 2023."
                ),
            },
        ),
        (
            "raffect_sad",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": "How much do these labels make you feel sad?",
                "options": EXTENT_5,
                "tags": ["in-grummon2023"],
                "note": (
                    "Negative emotional reaction matrix item 5/5 (eTable 3, p. 8); same stem/scale as "
                    "`affect_sad` in Grummon 2023."
                ),
            },
        ),
        (
            "rtrust",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": "How much do you trust or distrust the information in these labels?",
                "options": TRUST_5,
                "tags": ["in-grummon2023"],
                "note": (
                    "Standalone label trust item (eTable 3, p. 8; eTable 9); identical wording and 5-point scale "
                    "to `trust` in Grummon 2023."
                ),
            },
        ),
        (
            "rhelp",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": (
                    "How much do you agree or disagree with the following statement: "
                    "These labels would help me choose healthier foods and beverages."
                ),
                "options": DISAGREE_AGREE_5,
                "tags": ["grummon2023-similar"],
                "note": (
                    "Standalone perceived helpfulness item (eTable 3, p. 8; eTable 9; note verbatim option 3 is "
                    "'Neither disagree nor agree', whereas `Grummon2025.ipynb` Cell 3 used 'Neither agree nor disagree')."
                ),
            },
        ),
        (
            "runderstand",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": (
                    "How much do you agree or disagree with the following statement: "
                    "It is easy to understand the information in these labels."
                ),
                "options": DISAGREE_AGREE_5,
                "tags": [],
                "note": (
                    "Standalone perceived understandability item (eTable 3, p. 9; eTable 9; unique to Grummon 2025, "
                    "not in Grummon 2023)."
                ),
            },
        ),
        (
            "rsupport",
            {
                "block": "Label Reactions, Perceptions, and Policy Support",
                "kev_type": "score",
                "include": False,
                "context_kind": "standalone_label",
                "question_stem": "Would you oppose or support a policy requiring these labels on foods and beverages?",
                "options": SUPPORT_5,
                "tags": [],
                "note": (
                    "Standalone FOP labeling policy support item (eTable 3, p. 9; eTable 9; unique to Grummon 2025, "
                    "not in Grummon 2023)."
                ),
            },
        ),
        (
            "feltreal",
            {
                "block": "Online Store Realism",
                "kev_type": "score",
                "include": False,
                "context_kind": "shopping_retrospective",
                "question_stem": "This online grocery store felt like a real online grocery store.",
                "options": AGREE_DISAGREE_5,
                "tags": ["prior-task-dep"],
                "note": "Retrospective evaluation of the simulated online grocery store interface (DataDictionary.docx, p. 2).",
            },
        ),
        (
            "picksimilar",
            {
                "block": "Online Store Realism",
                "kev_type": "score",
                "include": False,
                "context_kind": "shopping_retrospective",
                "question_stem": (
                    "If I completed this shopping task in an actual store, "
                    "I would pick similar products as I chose in this study."
                ),
                "options": AGREE_DISAGREE_5,
                "tags": ["prior-task-dep"],
                "note": "Retrospective realism of simulated shopping selections (DataDictionary.docx, p. 2).",
            },
        ),
        (
            "similarusual",
            {
                "block": "Online Store Realism",
                "kev_type": "score",
                "include": False,
                "context_kind": "shopping_retrospective",
                "question_stem": "The foods I selected in this online grocery store were similar to my usual grocery purchases.",
                "options": AGREE_DISAGREE_5,
                "tags": ["prior-task-dep"],
                "note": "Retrospective similarity of simulated selections to usual purchases (DataDictionary.docx, p. 2).",
            },
        ),
    ]
)

PRODUCT_PAIR_NOTES = {
    "idbev": (
        "Consumer understanding 1/6 — beverages: Ocean Spray 100% Cranberry juice (healthier) vs. "
        "Ocean Spray Cranberry Juice Cocktail (eTable 3, p. 6; eTable 9). Note: neither beverage earns a Guiding Star "
        "(`pos_b=None`), so with `context_source=''` `Positive` shows `None`/`None`, `HighIn` matches `Positive+HighIn`, "
        "and `NutritionInfo` matches `Positive+NutritionInfo`; selecting `in_block_db` adds the arm's store label rule."
    ),
    "idbread": (
        "Consumer understanding 2/6 — breads: Mission Extra Thin Corn Tortillas (healthier, 1 Guiding Star) vs. "
        "Mission Flour Burrito Tortillas (High In Sodium; eTable 3, p. 7; eTable 9)."
    ),
    "idcereal": (
        "Consumer understanding 3/6 — cereals: Post Selects Great Grains Crunchy Pecans (healthier, 1 Guiding Star) vs. "
        "General Mills Chocolate Chex Gluten Free Cereal (High In Added Sugars; eTable 3, p. 7; eTable 9)."
    ),
    "idsoup": (
        "Consumer understanding 4/6 — soups: Campbell's Healthy Request Chunk Sirloin Burger Soup (healthier, 1 Guiding Star) vs. "
        "Dinty Moore Beef Stew (High In Sodium; eTable 3, p. 7; eTable 9)."
    ),
    "idsnack": (
        "Consumer understanding 5/6 — snacks: Sun Chips Original multigrain snacks (healthier) vs. "
        "Nutella & Go! Cup (High In Saturated Fat, Added Sugars; eTable 3, p. 7; eTable 9). Note: neither snack earns a "
        "Guiding Star (`pos_b=None`), so with `context_source=''` `Positive` shows `None`/`None`, `HighIn` matches "
        "`Positive+HighIn`, and `NutritionInfo` matches `Positive+NutritionInfo`; selecting `in_block_db` disambiguates all 6 arms."
    ),
    "idmeal": (
        "Consumer understanding 6/6 — boxed & frozen meals: Healthy Choice Cafe Steamers Chicken Marinara "
        "(healthier, 1 Guiding Star) vs. Lean Cuisine Turkey with Dressing (High In Sodium; eTable 3, p. 7; eTable 9)."
    ),
}


def format_product_label(stem: str, product_name: str, arm_name: str) -> str:
    """Render the front-of-package label string shown on `product_name` in `arm_name`."""
    pos_b, ninfo_b, highin_b, spec_b = PRODUCT_BASE_LABELS[stem][product_name]
    if arm_name == "Positive":
        badge = pos_b
    elif arm_name == "NutritionInfo":
        badge = ninfo_b
    elif arm_name == "HighIn":
        badge = highin_b
    elif arm_name == "Positive+NutritionInfo":
        badge = f"{pos_b}; {ninfo_b}" if pos_b is not None else ninfo_b
    elif arm_name == "Positive+HighIn":
        parts = [b for b in (pos_b, highin_b) if b is not None]
        badge = "; ".join(parts) if parts else None
    elif arm_name == "Spectrum":
        badge = spec_b
    else:
        raise ValueError(f"Unknown Grummon 2025 arm_name: {arm_name!r}")

    if badge is None:
        return "Front-of-package label: None"
    return f"Front-of-package label: {badge}"


def load_analytic_cohort(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    """Load `ParentFile_wide.csv` (`N = 5,636`) and validate trial arms and participant IDs."""
    if not raw_csv.exists():
        raise FileNotFoundError(f"Grummon 2025 raw CSV not found: {raw_csv}")
    df = pd.read_csv(raw_csv)
    if len(df) != 5_636:
        raise ValueError(f"Expected N=5,636 rows in {raw_csv.name}, got {len(df)}")
    observed_arms = set(df["foplcond"].dropna().unique())
    if observed_arms != set(ARM_NAMES) or df["foplcond"].isna().any():
        raise ValueError(f"Unexpected foplcond values: {sorted(observed_arms)} vs {ARM_NAMES}")
    if df["participantid"].isna().any() or df["participantid"].nunique() != len(df):
        raise ValueError("Expected 5,636 unique non-null participantid values")
    df = df.copy()
    df["participantid"] = df["participantid"].astype(str)
    df["CASEID"] = "grummon2025_" + df["participantid"]
    df["arm"] = df["foplcond"].astype(str)
    return df


def get_arm_variant_spec(stem: str, arm_name: str) -> dict:
    """Return `{context, question_stem, option_codes, option_texts, option_descriptions}` for `(stem, arm_name)`."""
    if arm_name not in FOPL_STIMULUS:
        raise ValueError(f"Unknown Grummon 2025 arm_name: {arm_name!r}")

    if stem in PRODUCT_BASE_LABELS:
        p1, p2 = list(PRODUCT_BASE_LABELS[stem].keys())
        return {
            "context": f"{CONSUMER_UNDERSTANDING_INTRO}\nLabels shown on products: {FOPL_STIMULUS[arm_name]}",
            "question_stem": HEALTHIER_PROMPT,
            "option_codes": [1, 2, 3],
            "option_texts": [p1, p2, "They're the same"],
            "option_descriptions": [
                format_product_label(stem, p1, arm_name),
                format_product_label(stem, p2, arm_name),
                None,
            ],
        }

    if stem in SECONDARY_OUTCOME_SPECS:
        spec = SECONDARY_OUTCOME_SPECS[stem]
        if spec["context_kind"] == "standalone_label":
            ctx = f"Labels shown on products: {FOPL_STIMULUS[arm_name]}"
        elif spec["context_kind"] == "shopping_retrospective":
            ctx = (
                "In the preceding simulated online grocery shopping task ($35 budget across beverages, breads, "
                f"cereals, soups, boxed/frozen meals, and snacks), products displayed the following front-of-package "
                f"labeling system: {FOPL_STIMULUS[arm_name]}"
            )
        else:
            raise ValueError(f"{stem}: unexpected context_kind {spec['context_kind']!r}")
        return {
            "context": ctx,
            "question_stem": spec["question_stem"],
            "option_codes": [c for c, _ in spec["options"]],
            "option_texts": [t for _, t in spec["options"]],
            "option_descriptions": None,
        }

    raise KeyError(f"Unknown Grummon 2025 outcome stem: {stem!r}")


def _count_codes(series: pd.Series, codes: list[int]) -> list[int]:
    vc = series.dropna().astype(int).value_counts()
    return [int(vc[c]) if c in vc.index else 0 for c in codes]


def build_survey_audit_artifacts(
    clean_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_dict, details_payload)` covering the complete Grummon et al. (2025) survey flow."""
    if clean_df is None:
        clean_df = load_analytic_cohort()

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
        in_block_db_text: str,
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
                "kev_type": kev_type if include else "",
                "context_source": context_source,
                "context_sep": "",
                "drop_options": "",
                "review_note": review_note,
                "QuestionText": question_text,
                "Context_preview": context_preview if context_source == "in_block_db" else "",
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
            "in_block_db_text": in_block_db_text,
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

    def add_outcome(
        *,
        stem: str,
        block: str,
        kev_type: str,
        include: bool,
        context_source: str,
        tags: list[str],
        note: str,
    ) -> None:
        paired_variants: list[dict] = []
        ref_spec = get_arm_variant_spec(stem, "Positive")
        pooled_counts = [0] * len(ref_spec["option_texts"])

        col_num = pd.to_numeric(clean_df[stem], errors="raise")
        for arm_name in ARM_NAMES:
            spec = get_arm_variant_spec(stem, arm_name)
            sub_s = col_num.loc[clean_df["arm"] == arm_name]
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
        csv_cols = [stem, f"scr_{stem}"] if stem in PRODUCT_BASE_LABELS else [stem]
        add_entry(
            qid=stem,
            csv_cols=csv_cols,
            family=family,
            block=block,
            n_answered=int(col_num.notna().sum()),
            include=include,
            kev_type=kev_type,
            context_source=context_source,
            review_note=note,
            question_text=ref_spec["question_stem"],
            context_preview=ref_spec["context"],
            in_block_db_text=ref_spec["context"],
            options=pooled_opts,
            paired_variants=paired_variants,
            tags=tags,
        )

    # --- 1. Eligibility & Pre-Shopping Task Instructions (eTable 3, pp. 5-6) ---
    add_entry(
        qid="age",
        csv_cols=["age"],
        family="0_Demographics",
        block="Eligibility",
        n_answered=int(clean_df["age"].notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note="Eligibility age screener (>=18 required; mapped to persona AGE).",
        question_text="How old are you? [free numeric response, 18-115]",
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=["in-grummon2023-persona"],
        q_type_meta="TE",
        selector_meta="",
        sample_texts=clean_df["age"].dropna().astype(int).astype(str).head(8).tolist(),
    )

    add_entry(
        qid="shopping_intro",
        csv_cols=[],
        family="0_DB",
        block="Introduction to Shopping Task",
        n_answered=len(clean_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Online grocery store and $35 incentive-compatibility instructions (eTable 3, pp. 5-6).",
        question_text=SHOPPING_INTRO_TEXT,
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )

    instruct_cnts = _count_codes(pd.to_numeric(clean_df["shoppingtaskinstruct"], errors="raise"), [1, 2, 3])
    add_entry(
        qid="shoppingtaskinstruct",
        csv_cols=["shoppingtaskinstruct"],
        family="0_Consent_Check",
        block="Introduction to Shopping Task",
        n_answered=int(clean_df["shoppingtaskinstruct"].notna().sum()),
        include=False,
        kev_type="",
        context_source="qid:shopping_intro",
        review_note="Comprehension check with custom validation forcing response 2 before entering store (eTable 3, p. 6).",
        question_text=(
            "We care about the quality of our data. For us to get the most accurate information, "
            "it is important that you understand these instructions. Please select the answer option "
            "that is correct based on the information you have read above:"
        ),
        context_preview="",
        in_block_db_text=SHOPPING_INTRO_TEXT,
        options=[
            {"option": "I have to pay for the items I select with my own money", "count": instruct_cnts[0]},
            {"option": "There is a chance I will be chosen to receive the groceries I selected", "count": instruct_cnts[1]},
            {"option": "No participants in this study will receive the groceries they selected", "count": instruct_cnts[2]},
        ],
        paired_variants=[],
        tags=["attn-check"],
    )

    basket_cols = [
        "mean_npm_100", "mean_us_points", "mean_energykcal_100g", "mean_sugar_100g",
        "mean_sodium_100g", "mean_sat_fat_100g", "mean_fib_100g", "mean_protein_100g",
        "count_highin", "tt_energykcal", "tt_sugar", "tt_sodium", "tt_sat_fat",
        "tt_fib", "tt_protein", "mean_dietary_fiber_g", "mean_protein_g",
        "mean_calories", "mean_sugars_g", "mean_saturated_fat_g", "mean_sodium_mg",
        "tt_qty", "tt_spent",
    ]
    basket_samples = [
        f"NPM={r.mean_npm_100:.1f}, items={int(r.tt_qty)}, spent=${r.tt_spent:.2f}, high_in={int(r.count_highin)}"
        for _, r in clean_df.head(6).iterrows()
    ]
    add_entry(
        qid="shopping_task_basket",
        csv_cols=basket_cols,
        family="0_TE",
        block="Shopping Task (Online Store)",
        n_answered=int(clean_df["mean_npm_100"].notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note=(
            "Continuous basket-level outcomes from the multi-item naturalistic online grocery store "
            "(4,000+ products, $35 budget; primary trial outcome `mean_npm_100` in Figure 2 & eTable 5; "
            "individual item selections are not discrete survey choices in ParentFile_wide.csv)."
        ),
        question_text="[Naturalistic online grocery store shopping basket summary: Ofcom NPM score, Guiding Stars, nutrient densities, item count, and total spend]",
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="TE",
        selector_meta="",
        sample_texts=basket_samples,
    )

    # --- 2. Consumer Understanding: 6 Product-Pair Comparisons (eTable 3, pp. 6-7) ---
    add_entry(
        qid="consumer_understanding_intro",
        csv_cols=[],
        family="0_DB",
        block="Consumer Understanding",
        n_answered=len(clean_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Section introduction before the 6 randomized product-pair comparisons (eTable 3, p. 6).",
        question_text=CONSUMER_UNDERSTANDING_INTRO,
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )

    for stem in PRODUCT_BASE_LABELS:
        item_tags = ["arm-prompt-collision"] if stem in {"idbev", "idsnack"} else []
        add_outcome(
            stem=stem,
            block="Consumer Understanding",
            kev_type="choice",
            include=True,
            context_source="",
            tags=item_tags,
            note=PRODUCT_PAIR_NOTES[stem],
        )

    # --- 3. Secondary Post-Shopping Perceptions, Emotions, Policy Support & Store Realism ---
    for stem, spec in SECONDARY_OUTCOME_SPECS.items():
        add_outcome(
            stem=stem,
            block=spec["block"],
            kev_type=spec["kev_type"],
            include=spec["include"],
            context_source="in_block_db",
            tags=spec["tags"],
            note=spec["note"],
        )

    # --- 4. Demographics & Nutrition Literacy Covariates (eTable 3, pp. 9-11) ---
    demo_specs = [
        (
            "gender",
            ["gender", "gender_4_text"],
            "How do you identify?",
            [(1, "Woman"), (2, "Man"), (3, "Non-binary"), (4, "Prefer to self-describe")],
            ["in-grummon2023-persona"],
            "Gender identity; mapped to persona SEX (`1->Female`, `2->Male`, `3->Non-binary`, `4->None`).",
            "MC",
        ),
        (
            "latino",
            ["latino"],
            "Are you of Hispanic, Latino, or Spanish origin?",
            [(1, "Yes"), (0, "No")],
            ["in-grummon2023-persona"],
            "Hispanic/Latino ethnicity; mapped to persona HISPANIC (`No` / `Yes`).",
            "MC",
        ),
        (
            "race",
            ["race_1", "race_2", "race_3", "race_4", "race_5", "race_6", "race_6_text", "racecat"],
            "What is your race? (Check all that apply).",
            [
                (1, "American Indian or Alaska Native"),
                (2, "Asian"),
                (3, "Black or African American"),
                (4, "Native Hawaiian or Other Pacific Islander"),
                (5, "White"),
                (6, "Another race"),
            ],
            ["in-grummon2023-persona"],
            "Multi-select race checkboxes + derived `racecat`; mapped to persona RACE.",
            "MC_Multi",
        ),
        (
            "educ",
            ["educ"],
            "What is the highest degree or level of school you have completed?",
            [
                (1, "Less than high school"),
                (2, "High school graduate (or GED)"),
                (3, "Some college or technical school"),
                (4, "Associate's degree"),
                (5, "Bachelor's degree"),
                (6, "Graduate or professional degree"),
            ],
            ["in-grummon2023-persona"],
            "6-level educational attainment; mapped to persona EDULEVEL.",
            "MC",
        ),
        (
            "healthlit_nvs",
            [
                "healthlittotalcals", "healthlitcarbs", "healthlitsatfat", "healthlitdvcalorie",
                "scr_lit1", "scr_lit2", "scr_lit3", "scr_lit4", "scr_litall",
            ],
            (
                "Newest Vital Sign (NVS) ice-cream Nutrition Facts quiz (4 numeric free-response math items "
                "summed into `scr_litall` = 0..4 correct; eTable 3, pp. 9-10)."
            ),
            [(i, f"{i} out of 4 correct") for i in range(5)],
            [],
            "Derived 0..4 nutrition literacy score (`scr_litall`); mapped to persona HEALTHLIT (distinct from Grummon 2023's single-item `MEDHELP`).",
            "MC",
        ),
        (
            "nfpuse",
            ["nfpuse"],
            "How often do you use the Nutrition Facts label when deciding to buy packaged foods?",
            [(1, "Never"), (2, "Rarely"), (3, "Sometimes"), (4, "Most of the time"), (5, "Always")],
            ["in-grummon2023-persona"],
            "Nutrition Facts label usage frequency; mapped to persona NFPUSE.",
            "MC",
        ),
        (
            "onlineamt",
            ["onlineamt"],
            "In the past year, about what percentage of your groceries did you buy online?",
            [(1, "Less than 25%"), (2, "25-49%"), (3, "50-74%"), (4, "75% or more")],
            [],
            "Online grocery shopping share; mapped to persona ONLINEAMT (unique to Grummon 2025).",
            "MC",
        ),
        (
            "hhsize_num",
            ["hhsize_num"],
            "How many people are in your household, including you? [1-20]",
            [],
            ["in-grummon2023-persona"],
            "Household size; mapped to global persona HHSIZE (`1`, `2`, `3`, `4`, `More than 4`).",
            "TE",
        ),
        (
            "children_num",
            ["children_num"],
            "How many children (ages 0-18) currently live in your household? [0-15]",
            [],
            ["in-grummon2023-persona"],
            "Number of children in household; mapped to persona CHILDREN (`0`, `1`, `2`, `3 or more`).",
            "TE",
        ),
        (
            "income",
            ["income"],
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
            ["in-grummon2023-persona"],
            "10-category household income; mapped to global persona US_INC.",
            "MC",
        ),
        (
            "wic",
            ["wic"],
            "Do you or anyone else in your household currently receive WIC benefits?",
            [(0, "No"), (1, "Yes"), (2, "Don't know")],
            [],
            "Household WIC receipt; mapped to persona WIC (`No`, `Yes`, `Don't know`; unique to Grummon 2025).",
            "MC",
        ),
        (
            "snap",
            ["snap"],
            (
                "Do you or anyone in your household currently get SNAP (Supplemental Nutrition Assistance Program) "
                "or Food Stamps?"
            ),
            [(0, "No"), (1, "Yes"), (2, "Don't know")],
            [],
            "Household SNAP / Food Stamps receipt; mapped to persona SNAP (`No`, `Yes`, `Don't know`; unique to Grummon 2025).",
            "MC",
        ),
    ]

    for qid, cols, q_txt, opt_pairs, d_tags, note, q_meta in demo_specs:
        if qid == "race":
            opt_list = []
            for code_idx, label in opt_pairs:
                col_s = pd.to_numeric(clean_df[f"race_{code_idx}"], errors="raise")
                opt_list.append({"option": label, "count": int((col_s == 1).sum())})
            n_ans = int(clean_df[[f"race_{k}" for k in range(1, 7)]].notna().any(axis=1).sum())
            samples: list[str] = []
        elif qid == "healthlit_nvs":
            s_num = pd.to_numeric(clean_df["scr_litall"], errors="raise")
            cnts = _count_codes(s_num, [c for c, _ in opt_pairs])
            opt_list = [{"option": lbl, "count": cnt} for (_, lbl), cnt in zip(opt_pairs, cnts, strict=True)]
            n_ans = int(s_num.notna().sum())
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
            samples = clean_df[cols[0]].dropna().astype(int).astype(str).head(8).tolist()

        add_entry(
            qid=qid,
            csv_cols=cols,
            family="0_Demographics",
            block="Demographics",
            n_answered=n_ans,
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=q_txt,
            context_preview="",
            in_block_db_text="",
            options=opt_list,
            paired_variants=[],
            tags=d_tags,
            q_type_meta=q_meta,
            selector_meta="SAVR" if opt_pairs else "",
            sample_texts=samples,
        )

    # --- 5. Debrief & Closure (eTable 3, p. 11) ---
    add_entry(
        qid="debrief",
        csv_cols=[],
        family="0_DB",
        block="Debrief",
        n_answered=len(clean_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Post-survey debriefing screen explaining the $35 gift card bonus (eTable 3, p. 11).",
        question_text=DEBRIEF_TEXT,
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 2,
        "schema_kind": "question_audit",
        "study_contexts": {"__all__": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload


def write_grummon2025_audit_files(out_dir: Path = GRUMMON2025_DIR) -> tuple[Path, Path, Path]:
    """Generate `question_audit.csv`, `question_tags.json`, and `.audit_details.json` in `out_dir`."""
    audit_df, tags_by_key, details_payload = build_survey_audit_artifacts()
    csv_path = out_dir / "question_audit.csv"
    tags_path = out_dir / "question_tags.json"
    details_path = out_dir / ".audit_details.json"

    audit_df.to_csv(csv_path, index=False)
    tags_path.write_text(json.dumps(tags_by_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    details_path.write_text(json.dumps(details_payload, ensure_ascii=False), encoding="utf-8")
    return csv_path, tags_path, details_path

"""Survey catalog, conjoint vaccine profile stimulus builders, and Audit UI indexers for Kreps et al. (2020)."""

from __future__ import annotations

from collections import OrderedDict
import itertools
import json
from pathlib import Path

import pandas as pd

COVID_DIR = Path(__file__).resolve().parent
RAW_DTA = COVID_DIR / "dataverse_files" / "Kreps_etal_vax_replication_data.dta"

MAX_PAIRED_CHOICE_VARIANTS = 200

CONSENT_TEXT = (
    "We are asking you to participate in a research study to better understand public opinion on political affairs. "
    "We will describe this study to you and answer any of your questions. This study is being led by Sarah Kreps and "
    "Doug Kriner, Department of Government at Cornell University.\n"
    "What the study is about: The purpose of this research is to better understand public opinion on major questions "
    "in contemporary society and politics.\n"
    "What we will ask you to do: We will ask you to read and reply to a few brief questions. After these questions, "
    "you will be asked a series of background questions. These will include questions about your age, gender, race, "
    "income, and education. At no point will you be asked any question which could personally identify you. Thus, we "
    "assure you that your responses will be anonymous and confidential. We estimate that the survey will take you "
    "about 10 minutes to complete."
)

INTRO_CONJOINT_TEXT = (
    "As you may know, scientists around the world are working to develop a vaccine for Covid-19. "
    "You will read about a series of hypothetical vaccines and answer a few questions about them.\n"
    "Respondents taking the survey on mobile devices: please turn your phone on its side so that you can easily see "
    "the comparisons between 2 hypothetical vaccines."
)

CONJOINT_PAIR_INTRO = (
    "As you may know, scientists around the world are working to develop a vaccine for Covid-19. "
    "Please consider the hypothetical vaccines described below:"
)

CONJOINT_SINGLE_INTRO = (
    "As you may know, scientists around the world are working to develop a vaccine for Covid-19. "
    "Please consider the hypothetical vaccine described below:"
)

EMPLOYMENT_INTRO_TEXT = "Next, we'd like to ask you a few questions about the impact of Covid-19."

DEMOGRAPHICS_INTRO_TEXT = "Finally, we would like to ask you a few questions about yourself:"

END_OF_SURVEY_TEXT = (
    "Thank you for completing this survey!\n"
    "We welcome your feedback on our survey. Please let us know below whether you encountered an issue while taking "
    "the survey. You can also let us know if we can improve our survey in any way."
)

CHOICE_STEM = (
    "If you had to choose, would you choose to get Vaccine A or Vaccine B, "
    "or would you choose not to be vaccinated?"
)

CHOICE_OPTIONS: list[tuple[int, str]] = [
    (1, "Vaccine A"),
    (2, "Vaccine B"),
    (3, "I would choose not to get either vaccine"),
]

LIKERT_STEM = "How likely or unlikely would you be to get the vaccine described above?"

# Verbatim descending display order in Qualtrics (`Kreps et al JNO 2020 survey.pdf`, pp. 3-11),
# paired with Stata `vaxord` numeric codes (7 = Extremely likely ... 1 = Extremely unlikely).
LIKERT_OPTIONS_DESC: list[tuple[int, str]] = [
    (7, "Extremely likely"),
    (6, "Moderately likely"),
    (5, "Slightly likely"),
    (4, "Neither likely nor unlikely"),
    (3, "Slightly unlikely"),
    (2, "Moderately unlikely"),
    (1, "Extremely unlikely"),
]

# Verbatim Table 1 attribute labels and levels from Kreps et al. (2020, JAMA Network Open, p. 3)
# and Qualtrics survey (`Kreps et al JNO 2020 survey.pdf`, pp. 3-11).
EFFICACY_LEVELS: OrderedDict[int, str] = OrderedDict([(1, "50%"), (2, "70%"), (3, "90%")])
DURATION_LEVELS: OrderedDict[int, str] = OrderedDict([(1, "1 year"), (2, "5 years")])
MAJORSIDE_LEVELS: OrderedDict[int, str] = OrderedDict([(1, "1 in 10,000"), (2, "1 in 1,000,000")])
MINORSIDE_LEVELS: OrderedDict[int, str] = OrderedDict([(1, "1 in 10"), (2, "1 in 30")])
FDA_RAW_TO_DISPLAY: dict[str, str] = {
    "The vaccine has been approved and licensed by the US Food and Drug Administration.": (
        "The vaccine has been approved and licensed by the US Food and Drug Administration."
    ),
    "The vaccine has received an emergency use authorization from the US Food and Drug Administration.  This allows the expedited use of promising drugs that the FDA has found it reasonable to believe may be effective in combatting the virus": (
        "The vaccine has received an emergency use authorization from the US Food and Drug Administration. "
        "This allows the expedited use of promising drugs that the FDA has found it reasonable to believe may be "
        "effective in combatting the virus."
    ),
}
FDA_LEVELS: OrderedDict[int, str] = OrderedDict(
    [
        (1, "The vaccine has been approved and licensed by the US Food and Drug Administration."),
        (
            2,
            "The vaccine has received an emergency use authorization from the US Food and Drug Administration. "
            "This allows the expedited use of promising drugs that the FDA has found it reasonable to believe may be "
            "effective in combatting the virus.",
        ),
    ]
)
ORIGIN_LEVELS: OrderedDict[int, str] = OrderedDict(
    [(1, "United States"), (2, "United Kingdom"), (3, "China")]
)
ENDORSED_LEVELS: OrderedDict[int, str] = OrderedDict(
    [
        (1, "President Donald Trump"),
        (2, "Vice President Joe Biden"),
        (3, "US Centers for Disease Control and Prevention"),
        (4, "World Health Organization"),
    ]
)

ATTRIBUTE_SPECS: list[tuple[str, str, OrderedDict[int, str]]] = [
    ("efficacy_n", "Efficacy (protection against severe symptoms)", EFFICACY_LEVELS),
    ("duration_n", "Protection duration", DURATION_LEVELS),
    ("majorside_n", "Risk of severe side effects (hospitalization or death)", MAJORSIDE_LEVELS),
    ("minorside_n", "Risk of mild side effects (flu-like symptoms)", MINORSIDE_LEVELS),
    ("fda_n", "Government authorization", FDA_LEVELS),
    ("origin_n", "Vaccine origin", ORIGIN_LEVELS),
    ("endorsed_n", "Recommended by", ENDORSED_LEVELS),
]

ATTR_CODE_COLS = [col for col, _, _ in ATTRIBUTE_SPECS]
CONJOINT_CSV_COLS = [
    "choice_set",
    "efficacy",
    "duration",
    "majorside",
    "minorside",
    "fda",
    "origin",
    "endorsed",
    "efficacy1",
    "efficacy2",
    "efficacy3",
    "duration1",
    "duration2",
    "majorside1",
    "majorside2",
    "minorside1",
    "minorside2",
    "fda1",
    "fda2",
    "origin1",
    "origin2",
    "origin3",
    "endorsed1",
    "endorsed2",
    "endorsed3",
    "endorsed4",
    *ATTR_CODE_COLS,
]


def _build_profile_catalog() -> tuple[dict[tuple[int, ...], str], dict[str, tuple[int, ...]]]:
    code_ranges = [list(levels.keys()) for _, _, levels in ATTRIBUTE_SPECS]
    tuple_to_pid: dict[tuple[int, ...], str] = {}
    pid_to_tuple: dict[str, tuple[int, ...]] = {}
    for idx, combo in enumerate(itertools.product(*code_ranges), start=1):
        pid = f"p{idx:03d}"
        tuple_to_pid[combo] = pid
        pid_to_tuple[pid] = combo
    if len(tuple_to_pid) != 576:
        raise ValueError(f"Expected 576 full-factorial profiles, got {len(tuple_to_pid)}")
    return tuple_to_pid, pid_to_tuple


TUPLE_TO_PROFILE_ID, PROFILE_ID_TO_TUPLE = _build_profile_catalog()


def format_profile_description(profile_id: str) -> str:
    """Format a single vaccine profile inline description for Kev `choice` `option_descriptions`."""
    codes = PROFILE_ID_TO_TUPLE[profile_id]
    parts = [
        f"{label}: {levels[code]}"
        for (_, label, levels), code in zip(ATTRIBUTE_SPECS, codes, strict=True)
    ]
    return "; ".join(parts)


def format_profile_block_context(profile_id: str) -> str:
    """Format the stimulus block shown above a single-profile `vax_likert` evaluation."""
    codes = PROFILE_ID_TO_TUPLE[profile_id]
    bullets = [
        f"- {label}: {levels[code]}"
        for (_, label, levels), code in zip(ATTRIBUTE_SPECS, codes, strict=True)
    ]
    return CONJOINT_SINGLE_INTRO + "\n" + "\n".join(bullets)


# Post-conjoint survey items in exact Qualtrics flow (`Kreps et al JNO 2020 survey.pdf`, pp. 11-15)
POST_CONJOINT_SPECS: OrderedDict[str, dict] = OrderedDict(
    [
        (
            "work_status",
            {
                "csv_cols": ["work_status", "work_impacted"],
                "block": "Employment status",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "in_block_db",
                "in_block_db_text": EMPLOYMENT_INTRO_TEXT,
                "question_stem": "Have you experienced any recent changes in your employment status?",
                "options": [
                    (1, "No change"),
                    (2, "Reduced hours"),
                    (3, "Getting laid off"),
                    (4, "Prefer not to say"),
                ],
                "tags": ["has-dont-know"],
                "note": (
                    "Employment impact item (survey p. 11). Qualtrics option 5 (`Not applicable`) was recoded to `NaN` "
                    "(n=384) in `Kreps_etal_vax_replication_data.dta`; code 4 (`Prefer not to say`, n=54) remains."
                ),
            },
        ),
        (
            "work_home",
            {
                "csv_cols": ["work_home", "work_home_ord"],
                "block": "Employment status",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "in_block_db",
                "in_block_db_text": EMPLOYMENT_INTRO_TEXT,
                "question_stem": "Are you able to work from home?",
                "options": [
                    (1, "Yes"),
                    (2, "No"),
                    (3, "To some extent"),
                    (4, "Prefer not to say"),
                ],
                "tags": ["has-dont-know"],
                "note": (
                    "Remote-work ability item (survey p. 12). Qualtrics option 5 (`Not applicable`) was recoded to `NaN` "
                    "(n=477) in `Kreps_etal_vax_replication_data.dta`; code 4 (`Prefer not to say`, n=30) remains."
                ),
            },
        ),
        (
            "personal_contact1",
            {
                "csv_cols": ["personal_contact1", "knowcovid"],
                "block": "Covid Questions",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": "Do you personally know someone who has been officially tested positive for Covid-19?",
                "options": [(1, "Yes"), (2, "No")],
                "tags": [],
                "note": "Personal acquaintance with a confirmed COVID-19 case (survey p. 12; Table 3 covariate `knowcovid`).",
            },
        ),
        (
            "personal_contact2",
            {
                "csv_cols": ["personal_contact2", "knowcovidsevere"],
                "block": "Covid Questions",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Do you personally know someone who has been hospitalized or died as a result of having COVID-19?"
                ),
                "options": [(1, "Yes"), (2, "No")],
                "tags": [],
                "note": "Personal acquaintance with COVID-19 hospitalization or death (survey p. 12).",
            },
        ),
        (
            "covid_future",
            {
                "csv_cols": ["covid_future", "worst_to_come"],
                "block": "Covid Questions",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Which comes closer to your view about where the U.S. stands in the coronavirus outbreak: "
                    "the worst is behind us, or the worst is yet to come?"
                ),
                "options": [
                    (1, "The worst is behind us"),
                    (2, "The worst is yet to come"),
                    (3, "Unsure"),
                ],
                "tags": ["has-dont-know"],
                "note": "US pandemic trajectory outlook fielded July 9, 2020 (survey p. 12; Table 3 covariate `worst_to_come`).",
            },
        ),
        (
            "covid_fed_approve",
            {
                "csv_cols": ["covid_fed_approve"],
                "block": "Covid Questions",
                "family": "2_Ordinal_Scale",
                "kev_type": "score",
                "reversed": True,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Do you approve or disapprove of the way the federal government is handling the response "
                    "to the coronavirus?"
                ),
                "options": [
                    (1, "Strongly approve"),
                    (2, "Approve"),
                    (3, "Disapprove"),
                    (4, "Strongly disapprove"),
                ],
                "tags": ["unclean-text"],
                "note": (
                    "Excluded: Kreps_etal_vax_replication_data.dta recoded `covid_fed_approve` (`5 - x`) in place "
                    "without updating the Qualtrics Stata value-label table (`1=Strongly approve .. 4=Strongly disapprove`), "
                    "leaving numeric values and Stata labels inverted with no raw pre-cleaning export in Dataverse."
                ),
            },
        ),
        (
            "covid_china",
            {
                "csv_cols": ["covid_china", "china_hid"],
                "block": "Covid Questions",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Do you believe China hid the coronavirus from the rest of the world after its outbreak "
                    "in Wuhan, China?"
                ),
                "options": [(1, "Yes"), (2, "No"), (3, "Unsure")],
                "tags": ["has-dont-know"],
                "note": "Belief about Wuhan outbreak disclosure (survey p. 13).",
            },
        ),
        (
            "flu_vaccine",
            {
                "csv_cols": ["flu_vaccine"],
                "block": "Pharma/vaccine items",
                "family": "2_Ordinal_Scale",
                "kev_type": "score",
                "reversed": False,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": "How often in the past have you gotten a flu vaccination?",
                "options": [
                    (1, "Never"),
                    (2, "Once or twice"),
                    (3, "Most years"),
                    (4, "Every year"),
                ],
                "tags": [],
                "note": "4-point ascending past flu vaccination frequency (survey p. 13; Table 3 covariate `flu_vaccine`).",
            },
        ),
        (
            "vaccine_safety",
            {
                "csv_cols": ["vaccine_safety", "vax_safety"],
                "block": "Pharma/vaccine items",
                "family": "2_Ordinal_Scale",
                "kev_type": "score",
                "reversed": True,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": "In general, how safe do you think vaccines are?",
                "options": [
                    (1, "Extremely safe"),
                    (2, "Very safe"),
                    (3, "Somewhat safe"),
                    (4, "Not at all safe"),
                ],
                "tags": ["has-dont-know"],
                "note": (
                    "4-point descending general vaccine safety perception (survey p. 13). Qualtrics option 5 "
                    "(`Don't know`) was recoded to `NaN` (n=112) in `Kreps_etal_vax_replication_data.dta`."
                ),
            },
        ),
        (
            "mandatory_vax",
            {
                "csv_cols": ["mandatory_vax", "mandatory_vax_bin"],
                "block": "Pharma/vaccine items",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Thinking about childhood diseases, such as measles, mumps, rubella, and polio: should all "
                    "children be required to be vaccinated, or should parents be able to decide NOT to vaccinate "
                    "their children?"
                ),
                "options": [
                    (1, "Require all be vaccinated"),
                    (2, "Parents should decide"),
                    (3, "Don't know"),
                ],
                "tags": ["has-dont-know"],
                "note": "Childhood vaccination mandate preference (survey pp. 13-14).",
            },
        ),
        (
            "insurance",
            {
                "csv_cols": ["insurance", "no_insurance"],
                "block": "Pharma/vaccine items",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": "Do you currently have health insurance?",
                "options": [
                    (1, "Yes"),
                    (2, "No"),
                    (3, "Prefer not to say"),
                ],
                "tags": ["has-dont-know"],
                "note": "Current health insurance coverage (survey p. 14; Table 3 covariate `no_insurance`).",
            },
        ),
        (
            "pharma",
            {
                "csv_cols": ["pharma", "pharma_favor"],
                "block": "Pharma/vaccine items",
                "family": "2_Ordinal_Scale",
                "kev_type": "score",
                "reversed": True,
                "include": False,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Is your view of the pharmaceutical industry very positive, somewhat positive, neutral, "
                    "somewhat negative, or very negative?"
                ),
                "options": [
                    (1, "Very positive"),
                    (2, "Somewhat positive"),
                    (3, "Neutral"),
                    (4, "Somewhat negative"),
                    (5, "Very negative"),
                ],
                "tags": [],
                "note": "5-point descending pharmaceutical industry favorability (survey p. 14; Table 3 covariate `pharma_favor`).",
            },
        ),
        (
            "pres_approval",
            {
                "csv_cols": ["pres_approval"],
                "block": "Demographics",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": False,
                "context_source": "in_block_db",
                "in_block_db_text": DEMOGRAPHICS_INTRO_TEXT,
                "question_stem": (
                    "Do you approve or disapprove of the way Donald Trump is handling his job as president?"
                ),
                "options": [
                    (1, "Approve"),
                    (0, "Disapprove or No opinion [collapsed in .dta]"),
                ],
                "tags": ["unclean-text", "has-dont-know"],
                "note": (
                    "Asked inside the Demographics block (survey p. 15) with 3 Qualtrics options (`Approve`, "
                    "`Disapprove`, `No opinion`), but `Kreps_etal_vax_replication_data.dta` stores only binary `1=Approve` "
                    "(815) vs unlabeled `0=Disapprove or No opinion` (1,156), so `Disapprove` cannot be separated from `No opinion`."
                ),
            },
        ),
    ]
)

STATE_LABELS: dict[int, str] = {
    1: "Alabama",
    2: "Alaska",
    3: "Arizona",
    4: "Arkansas",
    5: "California",
    6: "Colorado",
    7: "Connecticut",
    8: "Delaware",
    9: "District of Columbia",
    10: "Florida",
    11: "Georgia",
    12: "Hawaii",
    13: "Idaho",
    14: "Illinois",
    15: "Indiana",
    16: "Iowa",
    17: "Kansas",
    18: "Kentucky",
    19: "Louisiana",
    20: "Maine",
    21: "Maryland",
    22: "Massachusetts",
    23: "Michigan",
    24: "Minnesota",
    25: "Mississippi",
    26: "Missouri",
    27: "Montana",
    28: "Nebraska",
    29: "Nevada",
    30: "New Hampshire",
    31: "New Jersey",
    32: "New Mexico",
    33: "New York",
    34: "North Carolina",
    35: "North Dakota",
    36: "Ohio",
    37: "Oklahoma",
    38: "Oregon",
    39: "Pennsylvania",
    40: "Rhode Island",
    41: "South Carolina",
    42: "South Dakota",
    43: "Tennessee",
    44: "Texas",
    45: "Utah",
    46: "Vermont",
    47: "Virginia",
    48: "Washington",
    49: "West Virginia",
    50: "Wisconsin",
    51: "Wyoming",
    52: "I do not live in the continental United States",
}


def load_raw_conjoint_df(raw_dta: Path = RAW_DTA) -> pd.DataFrame:
    """Load `Kreps_etal_vax_replication_data.dta` (`19,710` rows across `1,971` respondents) with strict validation."""
    if not raw_dta.exists():
        raise FileNotFoundError(f"Kreps et al. (2020) Stata dataset not found: {raw_dta}")
    df = pd.read_stata(raw_dta, convert_categoricals=False)
    if len(df) != 19_710:
        raise ValueError(f"Expected 19,710 profile rows in {raw_dta.name}, got {len(df)}")
    if df["respondent"].isna().any() or df["respondent"].nunique() != 1_971:
        raise ValueError(f"Expected 1,971 unique non-null respondents, got {df['respondent'].nunique()}")
    if df["choice_set"].isna().any():
        raise ValueError("Unexpected missing values in `choice_set`")

    df = df.copy()
    df["respondent"] = df["respondent"].astype(int)
    df["choice_set"] = df["choice_set"].astype(int)
    df["pair_id"] = df["choice_set"] // 10
    df["option_ab"] = df["choice_set"] % 10
    if set(df["pair_id"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Unexpected pair_id set: {sorted(df['pair_id'].unique())}")
    if set(df["option_ab"].unique()) != {1, 2}:
        raise ValueError(f"Unexpected option_ab set: {sorted(df['option_ab'].unique())}")

    if df[ATTR_CODE_COLS + ["vaxbin", "vaxord"]].isna().any().any():
        raise ValueError("Unexpected missing values in conjoint attribute codes or outcomes")

    profile_ids: list[str] = []
    for row_tuple in df[ATTR_CODE_COLS].astype(int).itertuples(index=False, name=None):
        if row_tuple not in TUPLE_TO_PROFILE_ID:
            raise ValueError(f"Unexpected attribute code tuple: {row_tuple!r}")
        profile_ids.append(TUPLE_TO_PROFILE_ID[row_tuple])
    df["profile_id"] = profile_ids
    df["CASEID"] = "covid_" + df["respondent"].astype(str)
    return df.sort_values(["respondent", "choice_set"]).reset_index(drop=True)


def extract_respondent_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract one validated row per respondent (`N = 1,971`) from the 10-row-per-respondent conjoint table."""
    resp_df = raw_df.loc[raw_df["choice_set"] == 11].sort_values("respondent").reset_index(drop=True)
    if len(resp_df) != 1_971 or resp_df["respondent"].nunique() != 1_971:
        raise ValueError(f"Expected 1,971 rows at choice_set==11, got {len(resp_df)}")
    return resp_df


def extract_choice_pairs_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pair Vaccine A (`option_ab == 1`) and Vaccine B (`option_ab == 2`) into `9,855` choice-set rows."""
    df_a = (
        raw_df.loc[raw_df["option_ab"] == 1, ["respondent", "CASEID", "pair_id", "profile_id", "vaxbin"]]
        .rename(columns={"profile_id": "profile_a", "vaxbin": "vaxbin_a"})
        .sort_values(["respondent", "pair_id"])
        .reset_index(drop=True)
    )
    df_b = (
        raw_df.loc[raw_df["option_ab"] == 2, ["respondent", "pair_id", "profile_id", "vaxbin"]]
        .rename(columns={"profile_id": "profile_b", "vaxbin": "vaxbin_b"})
        .sort_values(["respondent", "pair_id"])
        .reset_index(drop=True)
    )
    if len(df_a) != 9_855 or len(df_b) != 9_855:
        raise ValueError(f"Expected 9,855 Vaccine A and B rows, got {len(df_a)} and {len(df_b)}")
    if not (df_a["respondent"].equals(df_b["respondent"]) and df_a["pair_id"].equals(df_b["pair_id"])):
        raise ValueError("Mismatched (respondent, pair_id) alignment between Vaccine A and Vaccine B rows")

    merged = df_a.copy()
    merged["profile_b"] = df_b["profile_b"]
    merged["vaxbin_b"] = df_b["vaxbin_b"]

    choice_codes: list[int] = []
    for va, vb in zip(merged["vaxbin_a"], merged["vaxbin_b"], strict=True):
        ia, ib = int(va), int(vb)
        if ia not in {0, 1} or ib not in {0, 1} or ia + ib > 1:
            raise ValueError(f"Invalid paired (vaxbin_a, vaxbin_b) = ({va!r}, {vb!r})")
        if ia == 1:
            choice_codes.append(1)
        elif ib == 1:
            choice_codes.append(2)
        else:
            choice_codes.append(3)
    merged["choice_code"] = choice_codes
    merged["pair_key"] = merged["profile_a"] + "_" + merged["profile_b"]
    return merged


def _count_codes(series: pd.Series, codes: list[int]) -> list[int]:
    vc = series.dropna().astype(int).value_counts()
    return [int(vc[c]) if c in vc.index else 0 for c in codes]


def build_survey_audit_artifacts(
    raw_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_by_key, details_payload)` covering the complete Kreps et al. (2020) survey flow."""
    if raw_df is None:
        raw_df = load_raw_conjoint_df()
    resp_df = extract_respondent_df(raw_df)
    pairs_df = extract_choice_pairs_df(raw_df)

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
        paired_variants_total: int | None = None,
        tags: list[str],
        q_type_meta: str = "MC",
        selector_meta: str = "SAVR",
        sample_texts: list[str] | None = None,
    ) -> None:
        idx = len(rows)
        row_key = f"{idx}:{qid}"
        opts_preview = (
            f"[{len(options)}] " + " | ".join(str(o["option"]) for o in options)
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

        qtext_variants: list[dict] = []
        db_variants: list[dict] = []
        if paired_variants:
            seen_stems: OrderedDict[str, tuple[str, int]] = OrderedDict()
            seen_dbs: OrderedDict[str, tuple[str, int]] = OrderedDict()
            for pv in paired_variants:
                st = str(pv["stem"])
                n_pv = int(pv["n"])
                lbl_pv = str(pv["in_block_db_qid"])
                if st not in seen_stems:
                    seen_stems[st] = (lbl_pv, n_pv)
                else:
                    lbl_prev, prev_n = seen_stems[st]
                    seen_stems[st] = (lbl_prev, prev_n + n_pv)
                db_t = str(pv["in_block_db_text"])
                if db_t:
                    if db_t not in seen_dbs:
                        seen_dbs[db_t] = (lbl_pv, n_pv)
                    else:
                        lbl_prev, prev_n = seen_dbs[db_t]
                        seen_dbs[db_t] = (lbl_prev, prev_n + n_pv)
            qtext_variants = [
                {"label": lbl, "text": txt, "n": n}
                for txt, (lbl, n) in list(seen_stems.items())[:15]
            ]
            db_variants = [
                {"label": lbl, "text": txt, "n": n}
                for txt, (lbl, n) in list(seen_dbs.items())[:15]
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
            "paired_variants_total": (
                paired_variants_total if paired_variants_total is not None else len(paired_variants)
            ),
            "sample_texts": sample_texts if sample_texts is not None else [],
            "in_block_db_text": in_block_db_text,
        }

        if family == "0_DB":
            study_contexts.append(
                {
                    "qid": qid,
                    "block": block,
                    "is_question": False,
                    "n_variants": 1,
                    "preview": question_text[:140],
                    "full_text": question_text,
                }
            )

    # --- 1. Consent & Conjoint Introduction (survey pp. 1-3) ---
    add_entry(
        qid="consent_info",
        csv_cols=[],
        family="0_DB",
        block="Consent",
        n_answered=len(resp_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Cornell IRB study description shown on opening screen (survey pp. 1-2).",
        question_text=CONSENT_TEXT,
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )

    consent_cnts = _count_codes(pd.to_numeric(resp_df["participate"], errors="raise"), [1, 2])
    add_entry(
        qid="participate",
        csv_cols=["participate"],
        family="0_Consent_Check",
        block="Consent",
        n_answered=int(resp_df["participate"].notna().sum()),
        include=False,
        kev_type="",
        context_source="qid:consent_info",
        review_note="Informed consent screener (survey p. 2; 1,962 `I agree to participate`, 9 `NaN` in `.dta`).",
        question_text="Please indicate below that you agree to participate in this survey.",
        context_preview="",
        in_block_db_text=CONSENT_TEXT,
        options=[
            {"option": "I agree to participate", "count": consent_cnts[0]},
            {"option": "I do not agree to participate", "count": consent_cnts[1]},
        ],
        paired_variants=[],
        tags=[],
    )

    add_entry(
        qid="intro_conjoint",
        csv_cols=[],
        family="0_DB",
        block="Intro Conjoint",
        n_answered=len(resp_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Conjoint instructions preceding the 5 hypothetical vaccine choice sets (survey p. 3).",
        question_text=INTRO_CONJOINT_TEXT,
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )

    # --- 2. Conjoint Experiment: Discrete Choice (`vax_choice`) & 7-pt Likert (`vax_likert`) ---
    choice_codes = [c for c, _ in CHOICE_OPTIONS]
    choice_texts = [t for _, t in CHOICE_OPTIONS]
    pooled_choice_cnts = _count_codes(pairs_df["choice_code"], choice_codes)

    pair_groups: list[tuple[int, str, str, str, list[int]]] = []
    for (pa, pb), grp in pairs_df.groupby(["profile_a", "profile_b"], sort=False):
        cnts = _count_codes(grp["choice_code"], choice_codes)
        pair_groups.append((len(grp), f"{pa}_{pb}", str(pa), str(pb), cnts))
    pair_groups.sort(key=lambda x: (-x[0], x[1]))

    choice_paired_variants: list[dict] = []
    for n_obs, pkey, pa, pb, cnts in pair_groups[:MAX_PAIRED_CHOICE_VARIANTS]:
        desc_a = format_profile_description(pa)
        desc_b = format_profile_description(pb)
        choice_paired_variants.append(
            {
                "n": n_obs,
                "stem": CHOICE_STEM,
                "in_block_db_qid": pkey,
                "in_block_db_text": CONJOINT_PAIR_INTRO,
                "qid_texts": {},
                "options": [
                    {"option": choice_texts[0], "description": desc_a, "count": cnts[0]},
                    {"option": choice_texts[1], "description": desc_b, "count": cnts[1]},
                    {"option": choice_texts[2], "count": cnts[2]},
                ],
                "row_counts": [],
            }
        )

    top_pa, top_pb = pair_groups[0][2], pair_groups[0][3]
    add_entry(
        qid="vax_choice",
        csv_cols=["vaxbin", *CONJOINT_CSV_COLS],
        family="1_Choice_Multi",
        block="Conjoint (conjoint1..conjoint5)",
        n_answered=len(pairs_df),
        include=True,
        kev_type="choice",
        context_source="in_block_db",
        review_note=(
            "Primary discrete choice conjoint outcome across 5 choice sets x 1,971 respondents = 9,855 pairs "
            f"({len(pair_groups):,} unique ordered `(Vaccine A, Vaccine B)` pairs; top {MAX_PAIRED_CHOICE_VARIANTS} "
            "previewed in UI, all 9,855 exported to `covid_US_long.parquet`)."
        ),
        question_text=CHOICE_STEM,
        context_preview=CONJOINT_PAIR_INTRO,
        in_block_db_text=CONJOINT_PAIR_INTRO,
        options=[
            {
                "option": choice_texts[0],
                "description": format_profile_description(top_pa),
                "count": pooled_choice_cnts[0],
            },
            {
                "option": choice_texts[1],
                "description": format_profile_description(top_pb),
                "count": pooled_choice_cnts[1],
            },
            {"option": choice_texts[2], "count": pooled_choice_cnts[2]},
        ],
        paired_variants=choice_paired_variants,
        paired_variants_total=len(pair_groups),
        tags=[],
    )

    likert_codes = [c for c, _ in LIKERT_OPTIONS_DESC]
    likert_texts = [t for _, t in LIKERT_OPTIONS_DESC]
    vaxord_num = pd.to_numeric(raw_df["vaxord"], errors="raise")
    pooled_likert_cnts = _count_codes(vaxord_num, likert_codes)

    prof_groups: list[tuple[int, str, list[int]]] = []
    for pid, grp in raw_df.groupby("profile_id", sort=False):
        cnts = _count_codes(pd.to_numeric(grp["vaxord"], errors="raise"), likert_codes)
        prof_groups.append((len(grp), str(pid), cnts))
    prof_groups.sort(key=lambda x: (-x[0], x[1]))

    likert_paired_variants: list[dict] = []
    for n_obs, pid, cnts in prof_groups:
        likert_paired_variants.append(
            {
                "n": n_obs,
                "stem": LIKERT_STEM,
                "in_block_db_qid": pid,
                "in_block_db_text": format_profile_block_context(pid),
                "qid_texts": {},
                "options": [
                    {"option": opt_txt, "count": cnt}
                    for opt_txt, cnt in zip(likert_texts, cnts, strict=True)
                ],
                "row_counts": [],
            }
        )

    top_pid_ctx = format_profile_block_context(prof_groups[0][1])
    add_entry(
        qid="vax_likert",
        csv_cols=["vaxord", *CONJOINT_CSV_COLS],
        family="2_Ordinal_Scale",
        block="Conjoint (conjoint1..conjoint5)",
        n_answered=len(raw_df),
        include=True,
        kev_type="score",
        context_source="in_block_db",
        review_note=(
            "Primary 7-point willingness-to-vaccinate rating across 10 profiles x 1,971 respondents = 19,710 "
            "evaluations covering all 576 full-factorial vaccine profiles. Options kept in verbatim descending "
            "Qualtrics order (`Extremely likely` [7] -> `Extremely unlikely` [1], `reversed=True`)."
        ),
        question_text=LIKERT_STEM,
        context_preview=top_pid_ctx,
        in_block_db_text=top_pid_ctx,
        options=[
            {"option": opt_txt, "count": cnt}
            for opt_txt, cnt in zip(likert_texts, pooled_likert_cnts, strict=True)
        ],
        paired_variants=likert_paired_variants,
        paired_variants_total=len(prof_groups),
        tags=[],
    )

    # --- 3. Post-Conjoint Employment, COVID-19, and Vaccine Attitude Blocks (survey pp. 11-15) ---
    add_entry(
        qid="employment_intro",
        csv_cols=[],
        family="0_DB",
        block="Employment status",
        n_answered=len(resp_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Section header before employment-impact items (survey p. 11).",
        question_text=EMPLOYMENT_INTRO_TEXT,
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )

    for stem, spec in POST_CONJOINT_SPECS.items():
        if stem == "pres_approval":
            continue
        s_num = pd.to_numeric(resp_df[stem], errors="raise")
        opt_codes = [c for c, _ in spec["options"]]
        cnts = _count_codes(s_num, opt_codes)
        add_entry(
            qid=stem,
            csv_cols=spec["csv_cols"],
            family=spec["family"],
            block=spec["block"],
            n_answered=int(s_num.notna().sum()),
            include=spec["include"],
            kev_type=spec["kev_type"],
            context_source=spec["context_source"],
            review_note=spec["note"],
            question_text=spec["question_stem"],
            context_preview=spec["in_block_db_text"],
            in_block_db_text=spec["in_block_db_text"],
            options=[
                {"option": lbl, "count": cnt}
                for (_, lbl), cnt in zip(spec["options"], cnts, strict=True)
            ],
            paired_variants=[],
            tags=spec["tags"],
        )

    # --- 4. Demographics Block (`Kreps et al JNO 2020 survey.pdf`, pp. 14-17) ---
    add_entry(
        qid="demographics_intro",
        csv_cols=[],
        family="0_DB",
        block="Demographics",
        n_answered=len(resp_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Section header preceding respondent background questions (survey p. 14).",
        question_text=DEMOGRAPHICS_INTRO_TEXT,
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="DB",
        selector_meta="",
    )

    add_entry(
        qid="age",
        csv_cols=["age"],
        family="0_Demographics",
        block="Demographics",
        n_answered=int(resp_df["age"].notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note="Respondent age in years (18-94, median 43); mapped to canonical persona `AGE` (`18-29`, `30-49`, `50-64`, `65+`).",
        question_text="What is your age? [dropdown, 18-97]",
        context_preview="",
        in_block_db_text="",
        options=[],
        paired_variants=[],
        tags=[],
        q_type_meta="TE",
        selector_meta="",
        sample_texts=resp_df["age"].astype(int).astype(str).head(8).tolist(),
    )

    demo_mc_before_pres = [
        (
            "ideology",
            ["ideology", "conservatism"],
            "Politically, I consider myself:",
            [
                (1, "Very Liberal"),
                (2, "Liberal"),
                (3, "Somewhat Liberal"),
                (4, "Moderate"),
                (5, "Somewhat Conservative"),
                (6, "Conservative"),
                (7, "Very Conservative"),
            ],
            [],
            "7-point political ideology; mapped to canonical persona `PARTY_LR` in `persona.py`.",
        ),
        (
            "party3",
            ["party3", "dem3", "gop3"],
            "In politics, as of today, do you consider yourself a Republican, a Democrat, or an Independent?",
            [
                (1, "Republican"),
                (2, "Democrat"),
                (3, "Independent"),
                (4, "Other/don't know"),
            ],
            [],
            "Partisan self-identification; mapped directly to canonical persona `US_PRTY` in `persona.py`.",
        ),
        (
            "party_lean",
            ["party_lean", "dem5", "gop5"],
            "As of today, do you lean more toward the Democratic Party or the Republican Party?",
            [
                (1, "Democratic Party"),
                (2, "Republican Party"),
                (3, "Neither/don't know"),
            ],
            [],
            "Partisan lean item asked to all 1,971 respondents without skip logic; excluded because `US_PRTY` (`party3`) is in persona.",
        ),
    ]
    for qid, cols, q_txt, opt_pairs, d_tags, note in demo_mc_before_pres:
        s_num = pd.to_numeric(resp_df[cols[0]], errors="raise")
        cnts = _count_codes(s_num, [c for c, _ in opt_pairs])
        add_entry(
            qid=qid,
            csv_cols=cols,
            family="0_Demographics",
            block="Demographics",
            n_answered=int(s_num.notna().sum()),
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=q_txt,
            context_preview="",
            in_block_db_text="",
            options=[
                {"option": lbl, "count": cnt}
                for (_, lbl), cnt in zip(opt_pairs, cnts, strict=True)
            ],
            paired_variants=[],
            tags=d_tags,
        )

    pres_spec = POST_CONJOINT_SPECS["pres_approval"]
    pres_num = pd.to_numeric(resp_df["pres_approval"], errors="raise")
    pres_cnts = _count_codes(pres_num, [c for c, _ in pres_spec["options"]])
    add_entry(
        qid="pres_approval",
        csv_cols=pres_spec["csv_cols"],
        family=pres_spec["family"],
        block=pres_spec["block"],
        n_answered=int(pres_num.notna().sum()),
        include=pres_spec["include"],
        kev_type=pres_spec["kev_type"],
        context_source=pres_spec["context_source"],
        review_note=pres_spec["note"],
        question_text=pres_spec["question_stem"],
        context_preview=pres_spec["in_block_db_text"],
        in_block_db_text=pres_spec["in_block_db_text"],
        options=[
            {"option": lbl, "count": cnt}
            for (_, lbl), cnt in zip(pres_spec["options"], pres_cnts, strict=True)
        ],
        paired_variants=[],
        tags=pres_spec["tags"],
    )

    gender_pairs = [(1, "Male"), (2, "Female"), (3, "Prefer not to say")]
    gender_num = pd.to_numeric(resp_df["gender"], errors="raise")
    gender_cnts = _count_codes(gender_num, [c for c, _ in gender_pairs])
    add_entry(
        qid="gender",
        csv_cols=["gender", "female"],
        family="0_Demographics",
        block="Demographics",
        n_answered=int(gender_num.notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note="Gender (`1=Male`, `2=Female`, `3=Prefer not to say` -> `None`); mapped to persona `SEX`.",
        question_text="What is your gender?",
        context_preview="",
        in_block_db_text="",
        options=[
            {"option": lbl, "count": cnt}
            for (_, lbl), cnt in zip(gender_pairs, gender_cnts, strict=True)
        ],
        paired_variants=[],
        tags=[],
    )

    race_pairs = [
        ("race_1", "American Indian"),
        ("race_2", "Asian"),
        ("race_3", "Black or African American"),
        ("race_4", "Hispanic"),
        ("race_5", "White"),
        ("race_6", "Other"),
    ]
    race_opts = [
        {"option": lbl, "count": int((pd.to_numeric(resp_df[col], errors="raise") == 1).sum())}
        for col, lbl in race_pairs
    ]
    add_entry(
        qid="race",
        csv_cols=["race_1", "race_2", "race_3", "race_4", "race_5", "race_6", "white", "black", "latino"],
        family="0_Demographics",
        block="Demographics",
        n_answered=len(resp_df),
        include=False,
        kev_type="",
        context_source="",
        review_note=(
            "Multi-select race/ethnicity checkboxes; mapped to canonical persona `RACE` in `persona.py` "
            "(`len(checked) == 1` -> single category, `len(checked) > 1` or `American Indian` -> `Other`)."
        ),
        question_text="What best describes your race/ethnicity? Check all that apply.",
        context_preview="",
        in_block_db_text="",
        options=race_opts,
        paired_variants=[],
        tags=[],
        q_type_meta="MC_Multi",
        selector_meta="MAVR",
    )

    demo_mc_after_race = [
        (
            "income",
            ["income"],
            "What is your annual income range?",
            [
                (1, "Below $20,000"),
                (2, "$20,000 - $39,999"),
                (3, "$40,000 - $59,999"),
                (4, "$60,000 - $79,999"),
                (5, "$80,000 - $99,999"),
                (6, "$100,000 or more"),
            ],
            [],
            "6-category household income; mapped to canonical persona `US_INC` in `persona.py`.",
        ),
        (
            "education",
            ["education"],
            "What is the highest level of education you have completed?",
            [
                (1, "Less than High School"),
                (2, "High School / GED"),
                (3, "Some College"),
                (4, "2-year College Degree"),
                (5, "4-year College Degree"),
                (6, "Master's Degree"),
                (7, "Doctoral Degree"),
                (8, "Professional Degree"),
            ],
            [],
            "8-level educational attainment; mapped to 4-level canonical persona `EDULEVEL` in `persona.py`.",
        ),
        (
            "relig",
            ["relig", "not_religious"],
            "What best describes your religious beliefs?",
            [
                (1, "Protestant"),
                (2, "Roman Catholic"),
                (3, "Mormon"),
                (4, "Orthodox such as Greek or Russian Orthodox"),
                (5, "Jewish"),
                (6, "Muslim"),
                (7, "Buddhist"),
                (8, "Hindu"),
                (9, "Atheist"),
                (10, "Agnostic"),
                (11, "Something else"),
                (12, "Nothing in particular"),
            ],
            [],
            "12-category religious affiliation; mapped to canonical persona `RELIGGRP` in `persona.py`.",
        ),
        (
            "evangelical",
            ["evangelical", "evangelical2", "not_evangelical"],
            "Would you describe yourself as a born-again or evangelical Christian, or not?",
            [(1, "Yes"), (2, "No")],
            [],
            (
                "Intentionally excluded (`0_Demographics`): conditional follow-up shown only when `relig in {1, 2, 3, 4}` "
                "(n=986 answered; Yes=435, No=551, NaN=985). Kept out of `persona.py` (`RELIGGRP` uses `relig` alone, "
                "matching `twin2k` QID18 and `issp` without overwriting `Protestant`/`Catholic`/`Orthodox`)."
            ),
        ),
    ]
    for qid, cols, q_txt, opt_pairs, d_tags, note in demo_mc_after_race:
        s_num = pd.to_numeric(resp_df[cols[0]], errors="raise")
        cnts = _count_codes(s_num, [c for c, _ in opt_pairs])
        add_entry(
            qid=qid,
            csv_cols=cols,
            family="0_Demographics",
            block="Demographics",
            n_answered=int(s_num.notna().sum()),
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=q_txt,
            context_preview="",
            in_block_db_text="",
            options=[
                {"option": lbl, "count": cnt}
                for (_, lbl), cnt in zip(opt_pairs, cnts, strict=True)
            ],
            paired_variants=[],
            tags=d_tags,
        )

    state_num = pd.to_numeric(resp_df["state"], errors="raise")
    state_codes = sorted(STATE_LABELS.keys())
    state_cnts = _count_codes(state_num, state_codes)
    add_entry(
        qid="state",
        csv_cols=["state"],
        family="0_Demographics",
        block="Demographics",
        n_answered=int(state_num.notna().sum()),
        include=False,
        kev_type="",
        context_source="",
        review_note="Persona attribute US_REG via persona.py (50 states + DC mapped to 4 US Census Bureau regions; survey p. 17).",
        question_text="In which state do you currently reside?",
        context_preview="",
        in_block_db_text="",
        options=[
            {"option": STATE_LABELS[c], "count": cnt}
            for c, cnt in zip(state_codes, state_cnts, strict=True)
            if cnt > 0
        ],
        paired_variants=[],
        tags=[],
    )

    # --- 5. End of Survey (survey p. 17) ---
    add_entry(
        qid="end_of_survey",
        csv_cols=[],
        family="0_DB",
        block="End of survey",
        n_answered=len(resp_df),
        include=False,
        kev_type="",
        context_source="",
        review_note="Closing feedback screen (survey p. 17).",
        question_text=END_OF_SURVEY_TEXT,
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


def write_covid_audit_files(out_dir: Path = COVID_DIR) -> tuple[Path, Path, Path]:
    """Generate `question_audit.csv`, `question_tags.json`, and `.audit_details.json` in `out_dir`."""
    audit_df, tags_by_key, details_payload = build_survey_audit_artifacts()
    csv_path = out_dir / "question_audit.csv"
    tags_path = out_dir / "question_tags.json"
    details_path = out_dir / ".audit_details.json"

    audit_df.to_csv(csv_path, index=False)
    tags_path.write_text(json.dumps(tags_by_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    details_path.write_text(json.dumps(details_payload, ensure_ascii=False), encoding="utf-8")
    return csv_path, tags_path, details_path

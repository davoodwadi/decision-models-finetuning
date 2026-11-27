"""Survey catalog and Audit UI indexers for Kreps et al. (2021), *npj Vaccines* COVID-19 vaccination conjoint.

Source files in `code/data/kreps2021/`:
- `dataverse_files.zip` -> `vax_misinfo_replication_data_raw.dta`: 1,096 US Lucid respondents x 869 columns
  (7 single-profile vaccine conjoint tasks per respondent = 7,672 profile evaluations, plus post-conjoint
  vaccine attitude items, the 8-item WHO COVID-19 treatment misinformation battery, and respondent demographics).
- `s41541-021-00335-2.pdf` (Table 2, Methods) & `41541_2021_335_MOESM1_ESM.pdf` (p. 1, Supp. Fig. 1, Appendix 1):
  verbatim participant-facing conjoint table intro, attribute labels, full FDA Approval / EUA descriptions,
  outcome stems, and WHO misinformation statements.

Raw data quirks handled explicitly (fail-fast on any unexpected code or text):
- `F_{t}_{slot}` stores `'Risk of mild side effects \\r\\n(flu-like symptoms)'`; normalized to a single space.
- `F_{t}_1_{slot}` truncates the FDA EUA text at Stata's 244-byte `str244` limit (`...may be eff`); restored to the
  exact verbatim wording from MOESM1 p. 1 / Supplementary Figure 1.
- `F_{t}_1_{slot}` has double spaces after the first period in FDA Approval/EUA text and a Qualtrics typo
  `'Your receive $10 incentive'`; normalized to clean English matching Table 2.
- Tasks 1-3 store the 7-point ordinal likelihood column as `vaxa_ord_1..3` while tasks 4-7 use `vax_ord_4..7`.
- `true1` codes `Unsure` as `4` instead of `3` in the raw `.dta`; preserved in raw audit counts and normalized to `3`
  when exporting so all 8 misinformation items share consistent `(1, 2, 3)` option codes.
"""

from __future__ import annotations

from collections import OrderedDict
import io
from pathlib import Path
import zipfile

import pandas as pd
from pandas.io.stata import StataReader

KREPS2021_DIR = Path(__file__).resolve().parent
DATAVERSE_ZIP = KREPS2021_DIR / "dataverse_files.zip"
RAW_DTA_MEMBER = "vax_misinfo_replication_data_raw.dta"

N_RESPONDENTS = 1_096
N_TASKS_PER_RESPONDENT = 7
N_TOTAL_TASKS = N_RESPONDENTS * N_TASKS_PER_RESPONDENT  # 7,672

# --- Verbatim on-screen texts (Supplementary Figure 1 & MOESM1 p. 1, Appendix 1) ---
CONJOINT_INTRO_TEXT = (
    "As you may know, scientists around the world are working to develop a vaccine for Covid-19. "
    "Please consider the hypothetical vaccine described in the table below:"
)
VAX_BIN_STEM = (
    "If you had to choose, would you choose to get this vaccine, or would you choose not to be vaccinated?"
)
VAX_ORD_STEM = "How likely or unlikely would you be to get the vaccine described above?"

MISINFO_INTRO_TEXT = (
    "Please indicate whether you think the following statement about the new coronavirus (COVID-19) "
    "is true, false, or if you are unsure."
)

VAX_BIN_OPTIONS: list[tuple[int, str]] = [
    (1, "I would choose to get this vaccine"),
    (2, "I would choose not to be vaccinated"),
]

# Verbatim descending Qualtrics order (1 = Extremely likely .. 7 = Extremely unlikely; reversed=True).
VAX_ORD_OPTIONS: list[tuple[int, str]] = [
    (1, "Extremely likely"),
    (2, "Moderately likely"),
    (3, "Slightly likely"),
    (4, "Neither likely nor unlikely"),
    (5, "Slightly unlikely"),
    (6, "Moderately unlikely"),
    (7, "Extremely unlikely"),
]

FDA_FULL_APPROVAL_RAW = (
    "The vaccine has been approved and licensed by the US Food and Drug Administration.  "
    "This certifies that the vaccine has been shown to be safe and effective in clinical trials."
)
FDA_FULL_APPROVAL_CLEAN = (
    "The vaccine has been approved and licensed by the US Food and Drug Administration. "
    "This certifies that the vaccine has been shown to be safe and effective in clinical trials."
)
FDA_EUA_RAW_TRUNCATED = (
    "The vaccine has received an emergency use authorization from the US Food and Drug Administration.  "
    "This allows the expedited use of promising drugs that have not received full FDA approval, "
    "but that early results suggest are safe and may be eff"
)
FDA_EUA_CLEAN = (
    "The vaccine has received an emergency use authorization from the US Food and Drug Administration. "
    "This allows the expedited use of promising drugs that have not received full FDA approval, "
    "but that early results suggest are safe and may be effective in combatting the virus."
)

# Canonical Table 2 attribute order: `(key, raw_attr_name, display_label, OrderedDict[(raw_val, clean_val)])`.
ATTRIBUTE_SPECS: list[tuple[str, str, str, OrderedDict[str, str]]] = [
    (
        "efficacy",
        "Efficacy \u2013 protection against severe symptoms such as respiratory failure, admission to a hospital ICU, or death",
        "Efficacy \u2013 protection against severe symptoms such as respiratory failure, admission to a hospital ICU, or death",
        OrderedDict([("50%", "50%"), ("70%", "70%"), ("90%", "90%")]),
    ),
    (
        "side_effects",
        "Risk of mild side effects \r\n(flu-like symptoms)",
        "Risk of mild side effects (flu-like symptoms)",
        OrderedDict([("1 in 10", "1 in 10"), ("1 in 4", "1 in 4"), ("1 in 2", "1 in 2")]),
    ),
    (
        "fda",
        "Development and testing procedure",
        "Development and testing procedure",
        OrderedDict(
            [
                (FDA_FULL_APPROVAL_RAW, FDA_FULL_APPROVAL_CLEAN),
                (FDA_EUA_RAW_TRUNCATED, FDA_EUA_CLEAN),
            ]
        ),
    ),
    (
        "manufacturer",
        "Manufacturer",
        "Manufacturer",
        OrderedDict(
            [
                ("Pfizer", "Pfizer"),
                ("Moderna", "Moderna"),
                ("AstraZeneca", "AstraZeneca"),
                ("Johnson & Johnson", "Johnson & Johnson"),
            ]
        ),
    ),
    (
        "cost",
        "Cost or Financial Incentive",
        "Cost or Financial Incentive",
        OrderedDict(
            [
                ("Free", "Free"),
                ("You pay $20 co-pay", "You pay $20 co-pay"),
                ("Your receive $10 incentive", "You receive $10 incentive"),
                ("You receive $100 incentive", "You receive $100 incentive"),
            ]
        ),
    ),
]

RAW_ATTR_TO_SPEC = {raw_attr: (key, label, lvl_map) for key, raw_attr, label, lvl_map in ATTRIBUTE_SPECS}
ATTR_KEYS = [key for key, _, _, _ in ATTRIBUTE_SPECS]
ATTR_LEVEL_LISTS: dict[str, list[str]] = {
    key: list(lvl_map.values()) for key, _, _, lvl_map in ATTRIBUTE_SPECS
}
ATTR_LABELS: dict[str, str] = {key: label for key, _, label, _ in ATTRIBUTE_SPECS}

# Build exhaustive 288-profile lookup (`3 x 3 x 2 x 4 x 4`).
PROFILE_ID_TO_TUPLE: OrderedDict[str, tuple[int, ...]] = OrderedDict()
TUPLE_TO_PROFILE_ID: dict[tuple[int, ...], str] = {}
for e_idx in range(3):
    for s_idx in range(3):
        for f_idx in range(2):
            for m_idx in range(4):
                for c_idx in range(4):
                    tup = (e_idx, s_idx, f_idx, m_idx, c_idx)
                    pid = f"p{e_idx}{s_idx}{f_idx}{m_idx}{c_idx}"
                    PROFILE_ID_TO_TUPLE[pid] = tup
                    TUPLE_TO_PROFILE_ID[tup] = pid


def format_profile_description(profile_id: str) -> str:
    """Format one single-vaccine profile as bullet-free attribute lines in canonical Table 2 order."""
    if profile_id not in PROFILE_ID_TO_TUPLE:
        raise KeyError(f"Unknown kreps2021 profile_id: {profile_id!r}")
    tup = PROFILE_ID_TO_TUPLE[profile_id]
    lines = [
        f"- {ATTR_LABELS[key]}: {ATTR_LEVEL_LISTS[key][idx]}"
        for key, idx in zip(ATTR_KEYS, tup, strict=True)
    ]
    return "\n".join(lines)


def format_profile_block_context(profile_id: str) -> str:
    """Full participant-facing conjoint vignette preceding `vax_bin` and `vax_ord`."""
    return f"{CONJOINT_INTRO_TEXT}\n\n{format_profile_description(profile_id)}"


# Substantive post-conjoint items (non-persona, included by default).
SUBSTANTIVE_ITEM_SPECS: OrderedDict[str, dict] = OrderedDict(
    [
        (
            "vaccine_safety",
            {
                "csv_cols": ["vaccine_safety"],
                "block": "Vaccine & Pharma Attitudes",
                "family": "2_Ordinal_Scale",
                "kev_type": "score",
                "reversed": True,
                "include": True,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": "In general, how safe do you think vaccines are?",
                "options": [
                    (1, "Extremely safe"),
                    (2, "Very safe"),
                    (3, "Somewhat safe"),
                    (4, "Not at all safe"),
                ],
                "raw_code_map": {1: 1, 2: 2, 3: 3, 4: 4},
                "tags": [],
                "note": (
                    "Included as score (reversed=True): 4-point descending general vaccine safety perception "
                    "(n=1,053 valid; 43 Don't know / missing coded NaN in `.dta`)."
                ),
            },
        ),
        (
            "mandatory_vax",
            {
                "csv_cols": ["mandatory_vax"],
                "block": "Vaccine & Pharma Attitudes",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Thinking about childhood diseases, such as measles, mumps, rubella, and polio: "
                    "should all children be required to be vaccinated, or should parents be able to "
                    "decide NOT to vaccinate their children?"
                ),
                "options": [
                    (1, "Require all be vaccinated"),
                    (2, "Parents should decide"),
                    (3, "Don't know"),
                ],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: childhood vaccination mandate attitude (n=1,096).",
            },
        ),
        (
            "pharma",
            {
                "csv_cols": ["pharma"],
                "block": "Vaccine & Pharma Attitudes",
                "family": "2_Ordinal_Scale",
                "kev_type": "score",
                "reversed": True,
                "include": True,
                "context_source": "",
                "in_block_db_text": "",
                "question_stem": (
                    "Is your view of the pharmaceutical industry very positive, somewhat positive, "
                    "neutral, somewhat negative, or very negative?"
                ),
                "options": [
                    (1, "Very positive"),
                    (2, "Somewhat positive"),
                    (3, "Neutral"),
                    (4, "Somewhat negative"),
                    (5, "Very negative"),
                ],
                "raw_code_map": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
                "tags": [],
                "note": "Included as score (reversed=True): 5-point descending pharmaceutical industry favorability (n=1,096).",
            },
        ),
        (
            "false1",
            {
                "csv_cols": ["false1"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": "Spraying alcohol or chlorine all over your body can kill the new coronavirus.",
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: WHO COVID-19 misinformation battery item 1 (false claim; MOESM1 Appendix 1).",
            },
        ),
        (
            "false2",
            {
                "csv_cols": ["false2"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": "Hand dryers are effective in killing the new coronavirus.",
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: WHO COVID-19 misinformation battery item 2 (false claim; MOESM1 Appendix 1).",
            },
        ),
        (
            "false3",
            {
                "csv_cols": ["false3"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": "Antibiotics are effective in preventing and treating the new coronavirus.",
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: WHO COVID-19 misinformation battery item 3 (false claim; MOESM1 Appendix 1).",
            },
        ),
        (
            "false4",
            {
                "csv_cols": ["false4"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": "Cold weather and snow can kill the new coronavirus.",
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: WHO COVID-19 misinformation battery item 4 (false claim; MOESM1 Appendix 1).",
            },
        ),
        (
            "false5",
            {
                "csv_cols": ["false5"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": "Vaccines against pneumonia protect against the new coronavirus.",
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: WHO COVID-19 misinformation battery item 5 (false claim; MOESM1 Appendix 1).",
            },
        ),
        (
            "true1",
            {
                "csv_cols": ["true1"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": "Avoiding shaking hands can help limit the spread of the new coronavirus.",
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 4: 3},
                "tags": ["has-dont-know"],
                "note": (
                    "Included as choice: WHO COVID-19 misinformation battery item 6 (true statement; MOESM1 Appendix 1). "
                    "Raw `.dta` codes Unsure as 4 (normalized to 3 on export)."
                ),
            },
        ),
        (
            "true2",
            {
                "csv_cols": ["true2"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": (
                    "Studies show hydroxychloroquine does not have clinical benefits in treating the new coronavirus."
                ),
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: WHO COVID-19 misinformation battery item 7 (true statement; MOESM1 Appendix 1).",
            },
        ),
        (
            "true3",
            {
                "csv_cols": ["true3"],
                "block": "COVID-19 Misinformation Battery",
                "family": "1_Choice_Multi",
                "kev_type": "choice",
                "reversed": False,
                "include": True,
                "context_source": "in_block_db",
                "in_block_db_text": MISINFO_INTRO_TEXT,
                "question_stem": (
                    "The FDA has authorized on an emergency basis the use of the Ebola drug, remdesivir, "
                    "as well as convalescent plasma for the treatment of the new coronavirus."
                ),
                "options": [(1, "True"), (2, "False"), (3, "Unsure")],
                "raw_code_map": {1: 1, 2: 2, 3: 3},
                "tags": ["has-dont-know"],
                "note": "Included as choice: WHO COVID-19 misinformation battery item 8 (true statement; MOESM1 Appendix 1).",
            },
        ),
    ]
)

# Excluded persona & administrative items (every raw source column in `_STATIC_PERSONA_SOURCE_COLS["kreps2021"]`
# is cataloged explicitly with `include=False` so the Audit UI flags all 27 persona columns with 0 overlap).
EXCLUDED_ITEM_SPECS: list[tuple[str, str, str, str, str]] = [
    (
        "participate",
        "Consent",
        "0_Consent_Check",
        "Please indicate below that you agree to participate in this survey.",
        "Excluded: informed consent screener.",
    ),
    (
        "flu_vaccine",
        "Vaccine & Pharma Attitudes",
        "4_Demographics_Persona",
        "How often in the past have you gotten a flu vaccination?",
        "Excluded: locked respondent persona attribute (`flu_vaccination_frequency`).",
    ),
    (
        "insurance",
        "Vaccine & Pharma Attitudes",
        "4_Demographics_Persona",
        "Do you currently have health insurance?",
        "Excluded: locked respondent persona attribute (`health_insurance`).",
    ),
    (
        "tv_news",
        "Media & Social Media",
        "4_Demographics_Persona",
        "Watching television news programs that report the day's news",
        "Excluded: locked respondent persona attribute (`tv_news_consumption`).",
    ),
    (
        "paper_news",
        "Media & Social Media",
        "4_Demographics_Persona",
        "Reading a newspaper (printed or online version)",
        "Excluded: locked respondent persona attribute (`newspaper_consumption`).",
    ),
    (
        "social_news",
        "Media & Social Media",
        "4_Demographics_Persona",
        "Seeing or reading links to news stories on Facebook or other social media sites",
        "Excluded: locked respondent persona attribute (`social_media_news_consumption`).",
    ),
    (
        "sns1",
        "Media & Social Media",
        "4_Demographics_Persona",
        "I use social media very often every day.",
        "Excluded: locked respondent persona attribute (`social_media_daily_use`).",
    ),
    (
        "sns2",
        "Media & Social Media",
        "4_Demographics_Persona",
        "I share photos and videos on social media.",
        "Excluded: locked respondent persona attribute (`social_media_shares_media`).",
    ),
    (
        "sns3",
        "Media & Social Media",
        "4_Demographics_Persona",
        "There are social media accounts that I follow constantly.",
        "Excluded: locked respondent persona attribute (`social_media_follows_accounts`).",
    ),
    (
        "sns4",
        "Media & Social Media",
        "4_Demographics_Persona",
        "I have more than one social media account that I use actively.",
        "Excluded: locked respondent persona attribute (`social_media_multiple_accounts`).",
    ),
    (
        "sns5",
        "Media & Social Media",
        "4_Demographics_Persona",
        "I think that social media is very useful.",
        "Excluded: locked respondent persona attribute (`social_media_perceived_usefulness`).",
    ),
    (
        "sns6",
        "Media & Social Media",
        "4_Demographics_Persona",
        "I like to follow my friends' shares on social media.",
        "Excluded: locked respondent persona attribute (`social_media_follows_friends_shares`).",
    ),
    (
        "sns7",
        "Media & Social Media",
        "4_Demographics_Persona",
        "I usually check out what my friends do on social media.",
        "Excluded: locked respondent persona attribute (`social_media_checks_friends_activity`).",
    ),
    (
        "age",
        "Demographics",
        "4_Demographics_Persona",
        "What is your age?",
        "Excluded: locked respondent persona attribute (`age`).",
    ),
    (
        "ideology",
        "Demographics",
        "4_Demographics_Persona",
        "Politically, I consider myself:",
        "Excluded: locked respondent persona attribute (`political_ideology`).",
    ),
    (
        "party3",
        "Demographics",
        "4_Demographics_Persona",
        "In politics, as of today, do you consider yourself a Republican, a Democrat, or an Independent?",
        "Excluded: locked respondent persona attribute (`political_party`).",
    ),
    (
        "party_lean",
        "Demographics",
        "4_Demographics_Persona",
        "As of today, do you lean more toward the Democratic Party or the Republican Party?",
        "Excluded: partisan-leaning follow-up.",
    ),
    (
        "pres_approval",
        "Demographics",
        "4_Demographics_Persona",
        "Do you approve or disapprove of the way Donald Trump is handling his job as president?",
        "Excluded: locked respondent persona attribute (`trump_approval`).",
    ),
    (
        "gender",
        "Demographics",
        "4_Demographics_Persona",
        "What is your gender?",
        "Excluded: locked respondent persona attribute (`gender`).",
    ),
    (
        "race_1",
        "Demographics",
        "4_Demographics_Persona",
        "What best describes your race/ethnicity? Check all that apply: American Indian or Alaska Native",
        "Excluded: locked respondent persona attribute (`race`).",
    ),
    (
        "race_2",
        "Demographics",
        "4_Demographics_Persona",
        "What best describes your race/ethnicity? Check all that apply: Asian",
        "Excluded: locked respondent persona attribute (`race`).",
    ),
    (
        "race_3",
        "Demographics",
        "4_Demographics_Persona",
        "What best describes your race/ethnicity? Check all that apply: Black or African American",
        "Excluded: locked respondent persona attribute (`race`).",
    ),
    (
        "race_4",
        "Demographics",
        "4_Demographics_Persona",
        "What best describes your race/ethnicity? Check all that apply: Hispanic or Latino",
        "Excluded: locked respondent persona attribute (`race`).",
    ),
    (
        "race_5",
        "Demographics",
        "4_Demographics_Persona",
        "What best describes your race/ethnicity? Check all that apply: White",
        "Excluded: locked respondent persona attribute (`race`).",
    ),
    (
        "race_6",
        "Demographics",
        "4_Demographics_Persona",
        "What best describes your race/ethnicity? Check all that apply: Other",
        "Excluded: locked respondent persona attribute (`race`).",
    ),
    (
        "income",
        "Demographics",
        "4_Demographics_Persona",
        "What is your annual income range?",
        "Excluded: locked respondent persona attribute (`income`).",
    ),
    (
        "education",
        "Demographics",
        "4_Demographics_Persona",
        "What is the highest level of education you have completed?",
        "Excluded: locked respondent persona attribute (`education`).",
    ),
    (
        "relig",
        "Demographics",
        "4_Demographics_Persona",
        "What best describes your religious beliefs?",
        "Excluded: locked respondent persona attribute (`religion`).",
    ),
    (
        "evangelical",
        "Demographics",
        "4_Demographics_Persona",
        "Would you describe yourself as a born-again or evangelical Christian, or not?",
        "Excluded: locked respondent persona attribute (`evangelical_christian`).",
    ),
    (
        "state",
        "Demographics",
        "4_Demographics_Persona",
        "In which state do you currently reside?",
        "Excluded: locked respondent persona attribute (`state`).",
    ),
    (
        "comments_end",
        "Closing",
        "3_Open_Text",
        "Do you have any comments about this survey?",
        "Excluded: open-ended closing comment box.",
    ),
]


def load_raw_dta(
    zip_path: Path = DATAVERSE_ZIP,
) -> tuple[pd.DataFrame, dict[str, dict[int, str]], dict[str, str]]:
    """Load `vax_misinfo_replication_data_raw.dta` (`1,096` respondents) and its Stata value labels."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Kreps et al. (2021) archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        raw_bytes = zf.read(RAW_DTA_MEMBER)
    with StataReader(io.BytesIO(raw_bytes)) as reader:
        raw_vlbls = reader.value_labels()
        col_to_lbl = dict(zip(reader._varlist, reader._lbllist, strict=True))
    value_labels: dict[str, dict[int, str]] = {
        name: {int(k): str(v) for k, v in mapping.items()} for name, mapping in raw_vlbls.items()
    }
    df = pd.read_stata(io.BytesIO(raw_bytes), convert_categoricals=False).reset_index(drop=True)
    if len(df) != N_RESPONDENTS:
        raise ValueError(f"kreps2021: expected {N_RESPONDENTS} rows, got {len(df)}")
    if df["ResponseId"].isna().any() or df["ResponseId"].nunique() != N_RESPONDENTS:
        raise ValueError("kreps2021: ResponseId must be non-null and unique per respondent")
    return df, value_labels, col_to_lbl


def extract_conjoint_tasks_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Unpivot tasks 1..7 into `7,672` validated single-profile conjoint observations."""
    rows: list[dict[str, object]] = []
    resp_ids = raw_df["ResponseId"].astype(str).tolist()
    for t in range(1, N_TASKS_PER_RESPONDENT + 1):
        bin_col = f"vax_bin_{t}"
        ord_col = f"vaxa_ord_{t}" if t <= 3 else f"vax_ord_{t}"
        if raw_df[bin_col].isna().any() or raw_df[ord_col].isna().any():
            raise ValueError(f"kreps2021: missing outcome values in task {t} ({bin_col}/{ord_col})")
        bin_vals = raw_df[bin_col].astype(int).tolist()
        ord_vals = raw_df[ord_col].astype(int).tolist()
        attr_cols = [f"F_{t}_{slot}" for slot in range(1, 6)]
        val_cols = [f"F_{t}_1_{slot}" for slot in range(1, 6)]
        attr_mat = raw_df[attr_cols].to_numpy()
        val_mat = raw_df[val_cols].to_numpy()

        for i in range(len(raw_df)):
            b_val = bin_vals[i]
            o_val = ord_vals[i]
            if b_val not in (1, 2):
                raise ValueError(f"kreps2021: unexpected {bin_col}={b_val!r} for {resp_ids[i]}")
            if o_val not in (1, 2, 3, 4, 5, 6, 7):
                raise ValueError(f"kreps2021: unexpected {ord_col}={o_val!r} for {resp_ids[i]}")
            level_idx_by_key: dict[str, int] = {}
            for slot in range(5):
                raw_attr = attr_mat[i, slot]
                raw_val = val_mat[i, slot]
                if raw_attr not in RAW_ATTR_TO_SPEC:
                    raise ValueError(f"kreps2021: unexpected attribute {raw_attr!r} at task {t} slot {slot + 1}")
                key, _, lvl_map = RAW_ATTR_TO_SPEC[raw_attr]
                if raw_val not in lvl_map:
                    raise ValueError(
                        f"kreps2021: unexpected level {raw_val!r} for {key!r} at task {t} slot {slot + 1}"
                    )
                level_idx_by_key[key] = list(lvl_map.keys()).index(raw_val)
            if len(level_idx_by_key) != 5:
                raise ValueError(f"kreps2021: duplicate/missing attributes at task {t} for {resp_ids[i]}")
            tup = tuple(level_idx_by_key[k] for k in ATTR_KEYS)
            pid = TUPLE_TO_PROFILE_ID[tup]
            rows.append(
                {
                    "CASEID": resp_ids[i],
                    "task": t,
                    "profile_id": pid,
                    "efficacy_idx": tup[0],
                    "side_effects_idx": tup[1],
                    "fda_idx": tup[2],
                    "manufacturer_idx": tup[3],
                    "cost_idx": tup[4],
                    "vax_bin": b_val,
                    "vax_ord": o_val,
                }
            )
    out = pd.DataFrame(rows)
    if len(out) != N_TOTAL_TASKS:
        raise ValueError(f"kreps2021: expected {N_TOTAL_TASKS} conjoint rows, got {len(out)}")
    return out


def _count_codes(series: pd.Series, codes: list[int]) -> list[int]:
    vc = series.dropna().astype(int).value_counts()
    return [int(vc[c]) if c in vc.index else 0 for c in codes]


def build_survey_audit_artifacts() -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_by_key, details_payload)` covering the Kreps et al. (2021) survey."""
    raw_df, value_labels, col_to_lbl = load_raw_dta()
    tasks_df = extract_conjoint_tasks_df(raw_df)

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
            f"[{len(options)}] " + " | ".join(str(o["option"]) for o in options) if options else ""
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
                {"label": lbl, "text": txt, "n": n} for txt, (lbl, n) in list(seen_stems.items())[:15]
            ]
            db_variants = [
                {"label": lbl, "text": txt, "n": n} for txt, (lbl, n) in list(seen_dbs.items())[:15]
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

    # --- 1. Single-profile conjoint outcomes (`vax_bin`, `vax_ord`) ---
    bin_codes = [c for c, _ in VAX_BIN_OPTIONS]
    bin_texts = [t for _, t in VAX_BIN_OPTIONS]
    pooled_bin_cnts = _count_codes(tasks_df["vax_bin"], bin_codes)

    bin_groups: list[tuple[int, str, list[int]]] = []
    for pid, grp in tasks_df.groupby("profile_id", sort=False):
        bin_groups.append((len(grp), str(pid), _count_codes(grp["vax_bin"], bin_codes)))
    bin_groups.sort(key=lambda x: (-x[0], x[1]))

    bin_variants = [
        {
            "n": n_obs,
            "stem": VAX_BIN_STEM,
            "in_block_db_qid": pid,
            "in_block_db_text": format_profile_block_context(pid),
            "qid_texts": {},
            "options": [
                {"option": opt_txt, "count": cnt} for opt_txt, cnt in zip(bin_texts, cnts, strict=True)
            ],
            "row_counts": [],
        }
        for n_obs, pid, cnts in bin_groups
    ]
    top_bin_ctx = format_profile_block_context(bin_groups[0][1])
    add_entry(
        qid="vax_bin",
        csv_cols=[f"vax_bin_{t}" for t in range(1, 8)],
        family="1_Choice_Multi",
        block="Conjoint (Tasks 1-7)",
        n_answered=len(tasks_df),
        include=True,
        kev_type="choice",
        context_source="in_block_db",
        review_note=(
            f"Included as choice: primary binary vaccination conjoint outcome across 7 single-profile tasks x "
            f"{N_RESPONDENTS:,} respondents = {len(tasks_df):,} evaluations covering {len(bin_groups)}/288 vaccine "
            "profiles (Supplementary Fig. 1 & Table 2; truncated str244 FDA EUA text restored verbatim from MOESM1 p. 1)."
        ),
        question_text=VAX_BIN_STEM,
        context_preview=top_bin_ctx,
        in_block_db_text=top_bin_ctx,
        options=[
            {"option": opt_txt, "count": cnt}
            for opt_txt, cnt in zip(bin_texts, pooled_bin_cnts, strict=True)
        ],
        paired_variants=bin_variants,
        paired_variants_total=len(bin_groups),
        tags=[],
    )

    ord_codes = [c for c, _ in VAX_ORD_OPTIONS]
    ord_texts = [t for _, t in VAX_ORD_OPTIONS]
    pooled_ord_cnts = _count_codes(tasks_df["vax_ord"], ord_codes)

    ord_groups: list[tuple[int, str, list[int]]] = []
    for pid, grp in tasks_df.groupby("profile_id", sort=False):
        ord_groups.append((len(grp), str(pid), _count_codes(grp["vax_ord"], ord_codes)))
    ord_groups.sort(key=lambda x: (-x[0], x[1]))

    ord_variants = [
        {
            "n": n_obs,
            "stem": VAX_ORD_STEM,
            "in_block_db_qid": pid,
            "in_block_db_text": format_profile_block_context(pid),
            "qid_texts": {},
            "options": [
                {"option": opt_txt, "count": cnt} for opt_txt, cnt in zip(ord_texts, cnts, strict=True)
            ],
            "row_counts": [],
        }
        for n_obs, pid, cnts in ord_groups
    ]
    top_ord_ctx = format_profile_block_context(ord_groups[0][1])
    add_entry(
        qid="vax_ord",
        csv_cols=[*(f"vaxa_ord_{t}" for t in range(1, 4)), *(f"vax_ord_{t}" for t in range(4, 8))],
        family="2_Ordinal_Scale",
        block="Conjoint (Tasks 1-7)",
        n_answered=len(tasks_df),
        include=True,
        kev_type="score",
        context_source="in_block_db",
        review_note=(
            f"Included as score (reversed=True): 7-point descending vaccination likelihood across 7 single-profile "
            f"tasks x {N_RESPONDENTS:,} respondents = {len(tasks_df):,} evaluations (`vaxa_ord_1..3` + `vax_ord_4..7`)."
        ),
        question_text=VAX_ORD_STEM,
        context_preview=top_ord_ctx,
        in_block_db_text=top_ord_ctx,
        options=[
            {"option": opt_txt, "count": cnt}
            for opt_txt, cnt in zip(ord_texts, pooled_ord_cnts, strict=True)
        ],
        paired_variants=ord_variants,
        paired_variants_total=len(ord_groups),
        tags=[],
    )

    # --- 2. Post-conjoint substantive items (vaccine/pharma attitudes + 8 WHO misinformation items) ---
    for qid, spec in SUBSTANTIVE_ITEM_SPECS.items():
        raw_map: dict[int, int] = spec["raw_code_map"]
        s = raw_df[qid].dropna().astype(int)
        unexpected = set(s.unique()) - set(raw_map.keys())
        if unexpected:
            raise ValueError(f"kreps2021.{qid}: unexpected raw codes {sorted(unexpected)}")
        norm_s = s.map(raw_map)
        opt_codes = [c for c, _ in spec["options"]]
        opt_texts = [t for _, t in spec["options"]]
        cnts = _count_codes(norm_s, opt_codes)
        add_entry(
            qid=qid,
            csv_cols=spec["csv_cols"],
            family=spec["family"],
            block=spec["block"],
            n_answered=int(len(s)),
            include=bool(spec["include"]),
            kev_type=spec["kev_type"],
            context_source=spec["context_source"],
            review_note=spec["note"],
            question_text=spec["question_stem"],
            context_preview=spec["in_block_db_text"],
            in_block_db_text=spec["in_block_db_text"],
            options=[
                {"option": opt_txt, "count": cnt} for opt_txt, cnt in zip(opt_texts, cnts, strict=True)
            ],
            paired_variants=[],
            tags=list(spec["tags"]),
        )

    # --- 3. Excluded persona & administrative columns ---
    for qid, block, family, qtext, note in EXCLUDED_ITEM_SPECS:
        col_s = raw_df[qid]
        if qid == "comments_end":
            valid_txts = [str(v).strip() for v in col_s.dropna() if str(v).strip()]
            add_entry(
                qid=qid,
                csv_cols=[qid],
                family=family,
                block=block,
                n_answered=len(valid_txts),
                include=False,
                kev_type="",
                context_source="",
                review_note=note,
                question_text=qtext,
                context_preview="",
                in_block_db_text="",
                options=[],
                paired_variants=[],
                tags=[],
                q_type_meta="TE",
                selector_meta="SL",
                sample_texts=valid_txts[:8],
            )
            continue

        if qid.startswith("race_"):
            n_checked = int((col_s.notna() & (col_s == 1)).sum())
            add_entry(
                qid=qid,
                csv_cols=[qid],
                family=family,
                block=block,
                n_answered=n_checked,
                include=False,
                kev_type="",
                context_source="",
                review_note=note,
                question_text=qtext,
                context_preview="",
                in_block_db_text="",
                options=[{"option": "Selected", "count": n_checked}],
                paired_variants=[],
                tags=[],
                selector_meta="MAVR",
            )
            continue

        lbl_name = col_to_lbl[qid] if qid in col_to_lbl else ""
        lbl_map = value_labels[lbl_name] if lbl_name in value_labels else {}
        s_valid = col_s.dropna().astype(int)
        vc = s_valid.value_counts()
        if lbl_map:
            opts = [
                {"option": lbl_map[code], "count": int(vc[code]) if code in vc.index else 0}
                for code in sorted(lbl_map.keys())
                if (code in vc.index or len(lbl_map) <= 15)
            ]
        else:
            opts = [{"option": str(code), "count": int(cnt)} for code, cnt in sorted(vc.items())]
        add_entry(
            qid=qid,
            csv_cols=[qid],
            family=family,
            block=block,
            n_answered=int(len(s_valid)),
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=qtext,
            context_preview="",
            in_block_db_text="",
            options=opts,
            paired_variants=[],
            tags=[],
        )

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 3,
        "schema_kind": "question_audit",
        "study_contexts": {"": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload

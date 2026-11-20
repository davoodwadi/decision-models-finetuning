"""Global persona taxonomy and fail-fast per-dataset mappings across training and evaluation datasets."""

import pandas as pd

AGE_BINS = [17, 29, 49, 64, 120]
AGE_LABELS = ["18-29", "30-49", "50-64", "65+"]

PERSONA_NAMES = {
    "c_alphan": "Country",
    "SEX": "Sex",
    "AGE": "Age",
    "EDULEVEL": "Education",
    "RELIGGRP": "Religion",
    "ATTEND": "Attendance of religious services",
    "PARTY_LR": "Political ideology",
    "VOTE_LE": "Voted in the last general election",
    "URBRURAL": "Place of living",
    "MARITAL": "Marital status",
    "MAINSTAT": "Main activity",
    "UNION": "Trade union membership",
    "US_PRTY": "Political party",
    "US_INC": "Income",
    "US_REG": "Region",
    "US_ETHN1": "Ancestry",
    "RACE": "Race",
    "HISPANIC": "Hispanic or Latino",  # separate Hispanic question (Grummon 2023/2025, wu2025)
    "HHSIZE": "How many people, including yourself, live in your household?",
    "VOTE_FREQ": "Votes in national elections",
    "RESTYPE": "What type of residence do you live in?",  # berke2023 Q_residence_type
    "ELECHEAT": "Does your household use electricity for heat?",  # wu2025 householdheat
    "MILITARY": "Have you ever served on active duty in the US Armed Forces?",  # wu2025 military
    "ORIENTATION": "Which of the following best describes your sexual orientation?",  # wu2025 orientation
    "OWNHOME": "Do you own your primary residence?",  # wu2025 residenceownership
    "RENTHOME": "Do you rent your primary residence?",  # wu2025 residencerental
    "POL_LABEL": "When it comes to politics, would you describe yourself as libertarian, socialist, communist, liberal, conservative, centrist, or something else?",  # wu2025 ideology_original
}

EXTRA_PERSONA_NAMES = {
    "CHILDREN": "How many children (ages 0-18) currently live in your household?",
    "SNAP": "SNAP / Food Stamps recipient",
    "WIC": "Do you or anyone else in your household currently receive WIC benefits?",
    "NFPUSE": "Nutrition Facts label usage",
    "ONLINEAMT": "Share of groceries bought online",
    # Derived in Grummon 2025 (scr_litall = 0..4 sum across 4 NVS ice-cream Nutrition Facts math items, not a single verbatim survey question); may be adapted in the future.
    "HEALTHLIT": "Nutrition literacy score",
    "MEDHELP": (
        "How often do you need to have someone help you when you read instructions, "
        "pamphlets, or other written materials from your doctor or pharmacy?"
    ),
    "DIETQUAL": "In general, how healthy is your overall diet?",
    "VENDING_BEV": "In a typical week, how often do you buy beverages from vending machines?",
    "VENDING_SNACK": "In a typical week, how often do you buy snacks from vending machines?",
    # Derived in Grummon 2023 from self-reported height_feet, height_inches, and weight_lbs matching published Table 1; may be adapted in the future.
    "BMI": "Body mass index (BMI)",
}
ALL_PERSONA_NAMES = {**PERSONA_NAMES, **EXTRA_PERSONA_NAMES}

DATASET_PERSONA_COLS = {
    "issp": [c for c in PERSONA_NAMES if c not in {"RACE", "HISPANIC", "HHSIZE", "VOTE_FREQ", "RESTYPE", "POL_LABEL", "OWNHOME", "RENTHOME", "ORIENTATION", "MILITARY", "ELECHEAT"}],
    "twin2k": [
        "c_alphan",
        "US_REG",
        "SEX",
        "AGE",
        "EDULEVEL",
        "RACE",
        "MARITAL",
        "RELIGGRP",
        "ATTEND",
        "US_PRTY",
        "US_INC",
        "PARTY_LR",
        "HHSIZE",
        "MAINSTAT",
    ],
    "covid": [
        "c_alphan",
        "US_REG",
        "SEX",
        "AGE",
        "EDULEVEL",
        "RELIGGRP",
        "PARTY_LR",
        "US_PRTY",
        "US_INC",
        "RACE",
    ],
    "kriner2026": [
        "c_alphan",
        "US_REG",
        "SEX",
        "AGE",
        "EDULEVEL",
        "RELIGGRP",
        "PARTY_LR",
        "US_PRTY",
        "US_INC",
        "RACE",
        "URBRURAL",
        "VOTE_LE",
    ],
    "berke2023": [  # decisions in berke2023/DECISIONS.md
        "c_alphan",
        "US_REG",
        "SEX",
        "AGE",
        "RACE",
        "US_INC",
        "URBRURAL",
        "RESTYPE",
    ],
    "wu2025": [  # decisions in wu2025/DECISIONS.md; remaining attributes added as decided
        "c_alphan",
        "US_REG",
        "SEX",
        "AGE",
        "RACE",
        "HISPANIC",
        "EDULEVEL",
        "US_INC",
        "US_PRTY",
        "PARTY_LR",
        "POL_LABEL",
        "MARITAL",
        "MAINSTAT",
        "RESTYPE",
        "OWNHOME",
        "RENTHOME",
        "ORIENTATION",
        "MILITARY",
        "ELECHEAT",
    ],
    "kantorowicz2026": [  # decisions in kantorowicz2026/DECISIONS.md
        "c_alphan",
        "SEX",
        "AGE",
        "EDULEVEL",
        "US_INC",
    ],
    "wvs": [
        "c_alphan",
        "SEX",
        "AGE",
        "US_REG",
        "EDULEVEL",
        "RACE",
        "MARITAL",
        "RELIGGRP",
        "HHSIZE",
        "MAINSTAT",
        "URBRURAL",
        "US_PRTY",
        "ATTEND",
        "UNION",
        "PARTY_LR",
        "VOTE_FREQ",
    ],
}

# Note: `party_lean` (asked to all 1,971 respondents without skip logic) and `evangelical`
# (asked only when `relig in {1, 2, 3, 4}`; neither ISSP nor Twin-2K QID18 has an evangelical
# persona key, and folding it into `Other Christian` would overwrite `Protestant`/`Catholic`/`Orthodox`)
# are intentionally excluded from COVID personas.
COVID_COL_TO_PERSONA = {
    "state": "US_REG",
    "gender": "SEX",
    "age": "AGE",
    "education": "EDULEVEL",
    "relig": "RELIGGRP",
    "ideology": "PARTY_LR",
    "party3": "US_PRTY",
    "income": "US_INC",
}

COVID_RACE_COLS = [
    ("race_1", "American Indian"),
    ("race_2", "Asian"),
    ("race_3", "Black or African American"),
    ("race_4", "Hispanic"),
    ("race_5", "White"),
    ("race_6", "Other"),
]

# Kriner2026 (Goldfarb, Kriner et al. 2026, SAF survey); decisions in `Kriner2026/DECISIONS.md`.
# `age_years` is built by the loader from the `.dta` value labels (raw `age` holds codes: 1 = 18 ... 80 = 97).
# `party_lean` and the authors' `party3` are intentionally not used: `US_PRTY` uses `party` alone.
# `vs_votechoice_2024` (Verasight profile 2024 vote choice) is intentionally not used.
KRINER2026_COL_TO_PERSONA = {
    "state": "US_REG",
    "gender": "SEX",
    "age_years": "AGE",
    "education": "EDULEVEL",
    "relig": "RELIGGRP",
    "ideology": "PARTY_LR",
    "party": "US_PRTY",
    "income": "US_INC",
    "urban_rural_best": "URBRURAL",
    "vs_vote_2024": "VOTE_LE",  # Verasight panel profile (2024 presidential election), not asked in this survey
}

KRINER2026_RACE_COLS = [
    ("native_american", "American Indian"),
    ("asian", "Asian"),
    ("black", "Black or African American"),
    ("hispanic", "Hispanic"),
    ("white", "White"),
    ("race_6", "Other"),
]

# berke2023 (Berke et al. 2023, drone delivery); decisions in `berke2023/DECISIONS.md`.
BERKE2023_COL_TO_PERSONA = {
    "Q_gender": "SEX",
    "Q_age": "AGE",
    "Q_income": "US_INC",
    "Q_urban_rural": "URBRURAL",
    "Q_residence_type": "RESTYPE",
    "Q_state": "US_REG",
}

# `Q_race` is select-all-that-apply stored as comma-separated codes ("1,6"); labels from the authors' codebook.py.
BERKE2023_RACE_CODES = {
    1: "White",
    2: "Black or African American",
    3: "American Indian or Alaska Native",
    4: "Asian",
    5: "Native Hawaiian or Pacific Islander",
    6: "Hispanic or Latino",
    7: "Other",
}

# Grummon 2023 / 2025 "What is your race? (Check all that apply)" option labels by checkbox index
# (columns `Race_<i>` in Grummon 2023, `race_<i>` in Grummon 2025).
GRUMMON_RACE_LABELS = {
    1: "American Indian or Alaska Native",
    2: "Asian",
    3: "Black or African American",
    4: "Native Hawaiian or Other Pacific Islander",
    5: "White",
    6: "Another race",
}

# wu2025 (Wu et al. 2025, CCA electricity plans); decisions in `wu2025/DECISIONS.md`. The `.dta` stores text labels,
# with "" where no answer was recorded.
WU2025_COL_TO_PERSONA = {
    "sex": "SEX",
    "age": "AGE",
    "ethnicity": "HISPANIC",
    "education": "EDULEVEL",
    "income": "US_INC",
    "partisan": "US_PRTY",  # party strength and `leaner` follow-ups intentionally not used (as in Kriner2026)
    "ideology": "PARTY_LR",
    "ideology_original": "POL_LABEL",
    "state": "US_REG",
    "marriage": "MARITAL",
    "employment": "MAINSTAT",
    "residence": "RESTYPE",
    "residenceownership": "OWNHOME",
    "orientation": "ORIENTATION",
    "military": "MILITARY",
    "householdheat": "ELECHEAT",
    "residencerental": "RENTHOME",  # asked to everyone; 239 answers contradict OWNHOME (both or neither), kept as given
}

# `race` is select-all stored as comma-separated labels ("White,Multi-racial"), in this survey's option wording.
WU2025_RACE_LABELS = [
    "White",
    "Black or African American",
    "American Indian or Alaska Native",
    "Asian/Pacific Islander",
    "Multi-racial",
    "Other",
]

# kantorowicz2026 (Kantorowicz-Reznichenko et al. 2026, computers & laundry machines).
# The public RDS files store binary demographic splits with R `NA` for missing values.
KANTOROWICZ2026_COL_TO_PERSONA = {
    "gender_dummy": "SEX",
    "age_dummy": "AGE",
    "education_dummy": "EDULEVEL",
    "income_dummy": "US_INC",
}

TWIN2K_QID_TO_PERSONA = {
    "QID11": "US_REG",
    "QID12": "SEX",
    "QID13": "AGE",
    "QID14": "EDULEVEL",
    "QID15": "RACE",
    "QID17": "MARITAL",
    "QID18": "RELIGGRP",
    "QID19": "ATTEND",
    "QID20": "US_PRTY",
    "QID21": "US_INC",
    "QID22": "PARTY_LR",
    "QID23": "HHSIZE",
    "QID24": "MAINSTAT",
}

WVS_COL_TO_PERSONA = {
    "B_COUNTRY": "c_alphan",
    "Q260": "SEX",
    "Q262": "AGE",
    "N_REGION_ISO": "US_REG",
    "Q275": "EDULEVEL",
    "Q290": "RACE",
    "Q273": "MARITAL",
    "Q289": "RELIGGRP",
    "Q270": "HHSIZE",
    "Q279": "MAINSTAT",
    "H_URBRURAL": "URBRURAL",
    "Q223": "US_PRTY",
    "Q171": "ATTEND",
    "Q97": "UNION",
    "Q240": "PARTY_LR",
    "Q222": "VOTE_FREQ",
}

US_ETHN1_LEVELS = [
    "Algeria", "American Indian", "American only", "Arabic", "Argentina", "Australia", "Austria",
    "Bangladesh", "Belgium", "Belize", "Bolivia", "Brazil", "Canada", "Canada (French)", "Chile",
    "China", "Colombia", "Costa Rica", "Cuba", "Czechoslovakia", "Democratic Republic of the Congo",
    "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "England and Wales",
    "Ethiopia", "Finland", "France", "French Guiana", "Germany", "Greece", "Guatemala", "Guyana",
    "Haiti", "Honduras", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Jamaica", "Japan", "Jordan", "Kenya", "Lithuania", "Mexico",
    "Netherlands (Dutch / Holland)", "New Zealand", "Nicaragua", "Nigeria", "Norway", "Other",
    "Other African", "Other Asian", "Other Caribbean", "Other Central American", "Other European",
    "Other Middle Eastern", "Other North American", "Other Oceanean", "Other South American",
    "Other Spanish", "Pakistan", "Panama", "Peru", "Philippines", "Poland", "Portugal",
    "Puerto Rico", "Romania", "Russia (USSR)", "Samoa", "Saudi Arabia", "Scotland", "South Africa",
    "South Korea", "Spain", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tanzania",
    "Thailand", "Turkey", "Uganda", "Uruguay", "Venezuela", "Vietnam", "Yemen", "Yugoslavia",
]

LEFT_RIGHT_LEVELS = [   # WVS Q240 left-right self-placement; each label says what its number means
    "1 (left) on a 1-10 left-right scale",
    *[f"{i} on a 1-10 left-right scale" for i in range(2, 10)],
    "10 (right) on a 1-10 left-right scale",
]

GLOBAL_PERSONA_LEVELS = {
    "c_alphan": ["United States"],
    "SEX": ["Male", "Female", "Non-binary", "Other", "Prefer not to say"],  # last two: answers offered by some surveys
    "AGE": [
        "18-29", "30-49", "50-64", "65+",
        # berke2023 Q_age brackets (25-34 and 45-54 cross AGE_BINS, so they cannot be bucketed)
        "18 - 24 years", "25 - 34 years", "35 - 44 years", "45 - 54 years", "55 - 64 years", "65 or older",
        # wu2025 `age` (its first five brackets share berke2023's levels)
        "65 - 74 years", "75 - 84 years", "85 or older",
        # kantorowicz2026 `age_dummy` (binary age split; numeric cutoff not documented)
        "Older", "Younger",
    ],
    "US_REG": ["Northeast", "Midwest", "South", "West"],
    "EDULEVEL": [
        "Less than high school",
        "High school or some college",
        "Associate's degree",
        "College graduate/some postgrad",
        # kantorowicz2026 `education_dummy` (binary education split; exact degree cutoff not documented)
        "Below higher education",
        "Higher education",
    ],
    "RACE": [
        "White", "Black", "Asian", "Hispanic", "Other", "Other, Non-Hispanic", "Two plus, non-Hispanic",
        # race checkbox questions (berke2023, Grummon 2023/2025, Covid, Kriner2026): single-box answers without a match,
        # and 2+ boxes. NHOPI uses the official OMB wording (berke2023's "Native Hawaiian or Pacific Islander" = same group).
        "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander", "Two or more races",
        "American Indian",  # Covid / Kriner2026 wording (no "or Alaska Native")
        "Asian/Pacific Islander",  # wu2025 (one combined option)
    ],
    "MARITAL": [
        "Married",
        "Living with a partner",
        "Divorced",
        "Separated",
        "Widowed",
        "Never been married",
    ],
    # Union of all datasets' religion answers: one level per meaning (same-meaning wordings share a level).
    "RELIGGRP": [
        "Protestant",
        "Catholic",
        "Orthodox",
        "Mormon",                 # Twin-2K, Covid
        "Other Christian",
        "Jewish",
        "Muslim",
        "Buddhist",
        "Hindu",
        "Other Asian Religions",  # ISSP
        "Other Religions",        # ISSP
        "Other",                  # each list's catch-all ("Other", "Something else")
        "Atheist",
        "Agnostic",
        "Nothing in particular",
        "No religion",            # ISSP "No religion", Kriner2026 "None", WVS "Do not belong to a denomination"
    ],
    "ATTEND": [
        "More than once a week",
        "Once a week",
        "Once or twice a month",
        "A few times a year",
        "Seldom",
        "Never",
    ],
    "US_PRTY": [
        "Democrat",
        "Strong Democrat",
        "Moderate Democrat",
        "Lean Democrat",
        "Republican",
        "Lean Republican",
        "Moderate Republican",
        "Strong Republican",
        "Independent",
        "Something else",
        "Other candidate",
        "Libertarian",
        "Green",
    ],
    # Union of all datasets' income brackets, ordered by amount: one level per distinct range (no merging). Identical
    # ranges share a level ("$50,000 to $74,999" = "$50,000-$75,000"); one format "$X - $Y". Levels overlap across datasets.
    "US_INC": [
        "Less than $10,000",       # Grummon, wu2025
        "$10,000 - $14,999",       # Grummon
        "$10,000 - $19,999",       # wu2025
        "$15,000 - $24,999",       # Grummon
        "Below $20,000",           # Covid, Kriner2026
        "Less than $25,000",       # berke2023
        "Less than $30,000",       # ISSP, Twin-2K, SubPOP
        "$20,000 - $29,999",       # wu2025
        "$20,000 - $39,999",       # Covid, Kriner2026
        "$25,000 - $34,999",       # Grummon
        "$25,000 - $49,999",       # berke2023
        "$30,000 - $34,999",       # ISSP
        "$30,000 - $39,999",       # wu2025
        "$30,000 - $50,000",       # Twin-2K
        "$35,000 - $39,999",       # ISSP
        "$35,000 - $49,999",       # Grummon
        "$40,000 - $49,999",       # ISSP, wu2025
        "$40,000 - $59,999",       # Covid, Kriner2026
        "$50,000 - $59,999",       # ISSP, wu2025
        "$50,000 - $75,000",       # Twin-2K, Grummon, berke2023
        "$60,000 - $69,999",       # wu2025
        "$60,000 - $74,999",       # ISSP
        "$60,000 - $79,999",       # Covid, Kriner2026
        "$70,000 - $79,999",       # wu2025
        "$75,000 - $89,999",       # ISSP
        "$75,000 - $100,000",      # Twin-2K, Grummon, berke2023
        "$80,000 - $89,999",       # wu2025
        "$80,000 - $99,999",       # Covid, Kriner2026
        "$90,000 - $99,999",       # wu2025
        "$90,000 - $109,999",      # ISSP
        "$100,000 or more",        # ISSP, Twin-2K, SubPOP, Covid, Kriner2026
        "$100,000 - $149,999",     # Grummon, berke2023, wu2025
        "$150,000 - $199,999",     # Grummon, berke2023
        "$150,000 or more",        # wu2025
        "$200,000 or more",        # Grummon, berke2023
        # kantorowicz2026 `income_dummy` (binary income split; dollar cutoff not documented)
        "Lower income",
        "Higher income",
    ],
    "PARTY_LR": [
        "Very liberal",
        "Liberal",
        "Somewhat liberal",
        "Moderate",
        "Somewhat conservative",
        "Conservative",
        "Very conservative",
        "Other",
        *LEFT_RIGHT_LEVELS,
        # wu2025 7-point wording (Very / Somewhat / Slightly): mapped by wording, so its "Somewhat liberal" (2nd step)
        # shares the level with other surveys' "Somewhat liberal" (their 3rd step)
        "Slightly liberal",
        "Slightly conservative",
    ],
    "HHSIZE": ["1", "2", "3", "4", "More than 4"],
    "MAINSTAT": [
        "Full-time employment",
        "Part-time employment",
        "Self-employed",
        "In paid work",
        "Unemployed",
        "Home-maker",
        "Student",
        "Retired",
        "Permanently sick or disabled",
        "Other",
        # wu2025 `employment` answers without an equivalent level (verbatim)
        "Working (paid employee)",
        "Not working (disabled)",
        "Not working (temporary layoff from a job)",
        "Not working (other)",
        "Prefer not to say",
    ],
    "VOTE_LE": ["Yes", "No", "I was too young to vote", "I was not registered to vote"],  # last two: Kriner2026 `vs_vote_2024`
    "VOTE_FREQ": ["Always", "Usually", "Never"],
    "URBRURAL": [
        "A big city",
        "The suburbs or outskirts of a big city",
        "A town or a small city",
        "A country village",
        "A farm or home in the country",
        "Urban",
        "Rural",
        "Suburban",  # Kriner2026 `urban_rural_best` (Urban / Suburban / Rural)
        "I don't know",  # berke2023 `Q_urban_rural`
    ],
    "UNION": ["Yes", "No", "Active member", "Inactive member"],
    "RESTYPE": [
        "Private home", "Apartment", "Other",  # berke2023
        "Single family home", "Townhouse", "Mobile home", "Condominium", "Something else",  # wu2025 `residence`
    ],
    "HISPANIC": ["Yes", "No"],
    "POL_LABEL": ["Libertarian", "Socialist", "Communist", "Liberal", "Conservative", "Centrist", "Other"],  # wu2025
    "OWNHOME": ["Yes", "No"],  # wu2025
    "RENTHOME": ["Yes", "No"],  # wu2025
    "ORIENTATION": ["Heterosexual (straight)", "Bisexual", "Homosexual (gay)", "Other", "Prefer not to say"],  # wu2025
    "MILITARY": ["Yes", "No"],  # wu2025
    "ELECHEAT": ["Yes", "No"],  # wu2025
    "US_ETHN1": US_ETHN1_LEVELS,
}

STATE_CODE_TO_REGION = {  # Covid / Kriner2026 `state` codes (1..51, alphabetical incl. DC) -> US Census Bureau region
        1: "South",  # Alabama
        2: "West",  # Alaska
        3: "West",  # Arizona
        4: "South",  # Arkansas
        5: "West",  # California
        6: "West",  # Colorado
        7: "Northeast",  # Connecticut
        8: "South",  # Delaware
        9: "South",  # District of Columbia
        10: "South",  # Florida
        11: "South",  # Georgia
        12: "West",  # Hawaii
        13: "West",  # Idaho
        14: "Midwest",  # Illinois
        15: "Midwest",  # Indiana
        16: "Midwest",  # Iowa
        17: "Midwest",  # Kansas
        18: "South",  # Kentucky
        19: "South",  # Louisiana
        20: "Northeast",  # Maine
        21: "South",  # Maryland
        22: "Northeast",  # Massachusetts
        23: "Midwest",  # Michigan
        24: "Midwest",  # Minnesota
        25: "South",  # Mississippi
        26: "Midwest",  # Missouri
        27: "West",  # Montana
        28: "Midwest",  # Nebraska
        29: "West",  # Nevada
        30: "Northeast",  # New Hampshire
        31: "Northeast",  # New Jersey
        32: "West",  # New Mexico
        33: "Northeast",  # New York
        34: "South",  # North Carolina
        35: "Midwest",  # North Dakota
        36: "Midwest",  # Ohio
        37: "South",  # Oklahoma
        38: "West",  # Oregon
        39: "Northeast",  # Pennsylvania
        40: "Northeast",  # Rhode Island
        41: "South",  # South Carolina
        42: "Midwest",  # South Dakota
        43: "South",  # Tennessee
        44: "South",  # Texas
        45: "West",  # Utah
        46: "Northeast",  # Vermont
        47: "South",  # Virginia
        48: "West",  # Washington
        49: "South",  # West Virginia
        50: "Midwest",  # Wisconsin
        51: "West",  # Wyoming
}

PERSONA_MAPS = {
    "c_alphan": {
        "issp": {"US": "United States"},
        "twin2k": {"United States": "United States"},
        "subpop": {"United States": "United States"},
        "covid": {"United States": "United States"},
        "grummon2025": {"United States": "United States"},
        "grummon2023": {"United States": "United States"},
        "wvs": {"United States": "United States"},
        "kriner2026": {"United States": "United States"},
        "berke2023": {"United States": "United States"},  # Prolific US residents; nobody chose "not in the US"
        "wu2025": {"United States": "United States"},  # non-US respondents removed by the authors
        "kantorowicz2026": {"United States": "United States"},  # quota-representative US sample (MSI/Savanta)
    },
    "SEX": {
        "issp": {"Male": "Male", "Female": "Female"},
        "twin2k": {"Male": "Male", "Female": "Female"},
        "subpop": {"Male": "Male", "Female": "Female"},
        "covid": {1: "Male", 2: "Female", 3: "Prefer not to say"},
        "grummon2025": {1.0: "Female", 2.0: "Male", 3.0: "Non-binary", 4.0: "Other"},  # 4 = Prefer to self-describe
        "grummon2023": {1.0: "Female", 2.0: "Male", 3.0: "Non-binary", 4.0: "Other"},  # 4 = Another gender (specify)
        "wvs": {"Male": "Male", "Female": "Female"},
        "berke2023": {1: "Male", 2: "Female", 3: "Other", 4: "Prefer not to say"},  # 4 = Prefer not to answer
        "kriner2026": {1: "Male", 2: "Female", 3: "Prefer not to say"},
        "wu2025": {"Female": "Female", "Male": "Male"},  # `sex`; "" (no answer) -> NaN in the loader
        "kantorowicz2026": {"Female": "Female", "Male": "Male"},  # `gender_dummy`; NA -> NaN
    },
    "AGE": {
        "twin2k": {
            "18-29": "18-29",
            "30-49": "30-49",
            "50-64": "50-64",
            "65+": "65+",
        },
        "berke2023": {
            1: "18 - 24 years",
            2: "25 - 34 years",
            3: "35 - 44 years",
            4: "45 - 54 years",
            5: "55 - 64 years",
            6: "65 or older",
        },
        "wu2025": {  # `age`; first five share berke2023's levels (same ranges)
            "18 - 24": "18 - 24 years",
            "25 - 34": "25 - 34 years",
            "35 - 44": "35 - 44 years",
            "45 - 54": "45 - 54 years",
            "55 - 64": "55 - 64 years",
            "65 - 74": "65 - 74 years",
            "75 - 84": "75 - 84 years",
            "85 or older": "85 or older",
        },
        "kantorowicz2026": {  # `age_dummy`; NA -> NaN
            "Older": "Older",
            "Younger": "Younger",
        },
    },
    "US_REG": {
        "issp": {
            "Northeast": "Northeast",
            "East North Central": "Midwest",
            "West North Central": "Midwest",
            "South": "South",
            "Mountain": "West",
            "Pacific": "West",
        },
        "twin2k": {
            "Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME)": "Northeast",
            "Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH)": "Midwest",
            "South (TX, OK, AR, LA, KY, TN, MS, AL, WV, DC, MD, DE, VA, NC, SC, GA, FL)": "South",
            "West (WA, OR, ID, MT, WY, CA, NV, UT, CO, AZ, NM)": "West",
            "Pacific (HI, AK)": "West",
        },
        "subpop": {
            "Northeast": "Northeast",
            "South": "South",
        },
        "covid": STATE_CODE_TO_REGION,
        "wvs": {  # state -> US Census Bureau region
            "US-AL Alabama": "South",
            "US-AR Arkansas": "South",
            "US-AZ Arizona": "West",
            "US-CA California": "West",
            "US-CO Colorado": "West",
            "US-CT Connecticut": "Northeast",
            "US-DC District of Columbia": "South",
            "US-DE Delaware": "South",
            "US-FL Florida": "South",
            "US-GA Georgia": "South",
            "US-HI Hawaii": "West",
            "US-IA Iowa": "Midwest",
            "US-ID Idaho": "West",
            "US-IL Illinois": "Midwest",
            "US-IN Indiana": "Midwest",
            "US-KS Kansas": "Midwest",
            "US-KY Kentucky": "South",
            "US-LA Louisiana": "South",
            "US-MA Massachusetts": "Northeast",
            "US-MD Maryland": "South",
            "US-ME Maine": "Northeast",
            "US-MI Michigan": "Midwest",
            "US-MN Minnesota": "Midwest",
            "US-MO Missouri": "Midwest",
            "US-MS Mississippi": "South",
            "US-MT Montana": "West",
            "US-NC North Carolina": "South",
            "US-ND North Dakota": "Midwest",
            "US-NE Nebraska": "Midwest",
            "US-NH New Hampshire": "Northeast",
            "US-NJ New Jersey": "Northeast",
            "US-NM New Mexico": "West",
            "US-NV Nevada": "West",
            "US-NY New York": "Northeast",
            "US-OH Ohio": "Midwest",
            "US-OK Oklahoma": "South",
            "US-OR Oregon": "West",
            "US-PA Pennsylvania": "Northeast",
            "US-RI Rhode Island": "Northeast",
            "US-SC South Carolina": "South",
            "US-SD South Dakota": "Midwest",
            "US-TN Tennessee": "South",
            "US-TX Texas": "South",
            "US-UT Utah": "West",
            "US-VA Virginia": "South",
            "US-VT Vermont": "Northeast",
            "US-WA Washington": "West",
            "US-WI Wisconsin": "Midwest",
            "US-WV West Virginia": "South",
            "US-WY Wyoming": "West",
        },
        "kriner2026": STATE_CODE_TO_REGION,  # same state codes as Covid (52 has no region)
        # berke2023 `Q_state` codes: alphabetical incl. DC (9) and Puerto Rico (40), so codes from 41 on are Covid's + 1.
        # Puerto Rico (40) and 53 ("I do not reside in the United States") were never chosen and have no region.
        "berke2023": {
            1: "South",  # Alabama
            2: "West",  # Alaska
            3: "West",  # Arizona
            4: "South",  # Arkansas
            5: "West",  # California
            6: "West",  # Colorado
            7: "Northeast",  # Connecticut
            8: "South",  # Delaware
            9: "South",  # District of Columbia
            10: "South",  # Florida
            11: "South",  # Georgia
            12: "West",  # Hawaii
            13: "West",  # Idaho
            14: "Midwest",  # Illinois
            15: "Midwest",  # Indiana
            16: "Midwest",  # Iowa
            17: "Midwest",  # Kansas
            18: "South",  # Kentucky
            19: "South",  # Louisiana
            20: "Northeast",  # Maine
            21: "South",  # Maryland
            22: "Northeast",  # Massachusetts
            23: "Midwest",  # Michigan
            24: "Midwest",  # Minnesota
            25: "South",  # Mississippi
            26: "Midwest",  # Missouri
            27: "West",  # Montana
            28: "Midwest",  # Nebraska
            29: "West",  # Nevada
            30: "Northeast",  # New Hampshire
            31: "Northeast",  # New Jersey
            32: "West",  # New Mexico
            33: "Northeast",  # New York
            34: "South",  # North Carolina
            35: "Midwest",  # North Dakota
            36: "Midwest",  # Ohio
            37: "South",  # Oklahoma
            38: "West",  # Oregon
            39: "Northeast",  # Pennsylvania
            41: "Northeast",  # Rhode Island
            42: "South",  # South Carolina
            43: "Midwest",  # South Dakota
            44: "South",  # Tennessee
            45: "South",  # Texas
            46: "West",  # Utah
            47: "Northeast",  # Vermont
            48: "South",  # Virginia
            49: "West",  # Washington
            50: "South",  # West Virginia
            51: "Midwest",  # Wisconsin
            52: "West",  # Wyoming
        },
        # wu2025 `state` holds state names (not codes); regions as in Covid's STATE_CODE_TO_REGION.
        "wu2025": {
            "Alabama": "South",
            "Alaska": "West",
            "Arizona": "West",
            "Arkansas": "South",
            "California": "West",
            "Colorado": "West",
            "Connecticut": "Northeast",
            "Delaware": "South",
            "District of Columbia": "South",
            "Florida": "South",
            "Georgia": "South",
            "Hawaii": "West",
            "Idaho": "West",
            "Illinois": "Midwest",
            "Indiana": "Midwest",
            "Iowa": "Midwest",
            "Kansas": "Midwest",
            "Kentucky": "South",
            "Louisiana": "South",
            "Maine": "Northeast",
            "Maryland": "South",
            "Massachusetts": "Northeast",
            "Michigan": "Midwest",
            "Minnesota": "Midwest",
            "Mississippi": "South",
            "Missouri": "Midwest",
            "Montana": "West",
            "Nebraska": "Midwest",
            "Nevada": "West",
            "New Hampshire": "Northeast",
            "New Jersey": "Northeast",
            "New Mexico": "West",
            "New York": "Northeast",
            "North Carolina": "South",
            "North Dakota": "Midwest",
            "Ohio": "Midwest",
            "Oklahoma": "South",
            "Oregon": "West",
            "Pennsylvania": "Northeast",
            "Rhode Island": "Northeast",
            "South Carolina": "South",
            "South Dakota": "Midwest",
            "Tennessee": "South",
            "Texas": "South",
            "Utah": "West",
            "Vermont": "Northeast",
            "Virginia": "South",
            "Washington": "West",
            "West Virginia": "South",
            "Wisconsin": "Midwest",
            "Wyoming": "West",
        },
    },
    "EDULEVEL": {
        "issp": {
            "Less than high school": "Less than high school",
            "Upper secondary": "High school or some college",
            "Short-cycle tertiary": "Associate's degree",
            "College graduate/some postgrad": "College graduate/some postgrad",
        },
        "twin2k": {
            "Less than high school": "Less than high school",
            "High school graduate": "High school or some college",
            "Some college, no degree": "High school or some college",
            "Associate's degree": "Associate's degree",
            "College graduate/some postgrad": "College graduate/some postgrad",
            "Postgraduate": "College graduate/some postgrad",
        },
        "subpop": {
            "Less than high school": "Less than high school",
            "College graduate/some postgrad": "College graduate/some postgrad",
        },
        "covid": {
            1: "Less than high school",
            2: "High school or some college",
            3: "High school or some college",
            4: "Associate's degree",
            5: "College graduate/some postgrad",
            6: "College graduate/some postgrad",
            7: "College graduate/some postgrad",
            8: "College graduate/some postgrad",
        },
        "grummon2025": {
            1.0: "Less than high school",
            2.0: "High school or some college",
            3.0: "High school or some college",
            4.0: "Associate's degree",
            5.0: "College graduate/some postgrad",
            6.0: "College graduate/some postgrad",
        },
        "grummon2023": {
            1.0: "Less than high school",
            2.0: "High school or some college",
            3.0: "High school or some college",
            4.0: "Associate's degree",
            5.0: "College graduate/some postgrad",
            6.0: "College graduate/some postgrad",
        },
        "wvs": {  # ISCED 2011
            "Early childhood education (ISCED 0) / no education": "Less than high school",
            "Primary education (ISCED 1)": "Less than high school",
            "Lower secondary education (ISCED 2)": "Less than high school",
            "Upper secondary education (ISCED 3)": "High school or some college",
            "Post-secondary non-tertiary education (ISCED 4)": "High school or some college",
            "Short-cycle tertiary education (ISCED 5)": "Associate's degree",
            "Bachelor or equivalent (ISCED 6)": "College graduate/some postgrad",
            "Master or equivalent (ISCED 7)": "College graduate/some postgrad",
            "Doctoral or equivalent (ISCED 8)": "College graduate/some postgrad",
        },
        "kriner2026": {
            1: "Less than high school",           # Less than High School
            2: "High school or some college",     # High School / GED
            3: "High school or some college",     # Some College
            4: "Associate's degree",              # 2-year College Degree
            5: "College graduate/some postgrad",  # 4-year College Degree
            6: "College graduate/some postgrad",  # Master's Degree
            7: "College graduate/some postgrad",  # Doctoral Degree
            8: "College graduate/some postgrad",  # Professional Degree
        },
        "wu2025": {  # `education`; same 8-level -> 4-level scheme as Covid
            "Less than high school degree": "Less than high school",
            "High school graduate (high school diploma or equivalent including GED)": "High school or some college",
            "Some college but no degree": "High school or some college",
            "Associate degree in college (2-year)": "Associate's degree",
            "Bachelor's degree in college (4-year)": "College graduate/some postgrad",
            "Master's degree": "College graduate/some postgrad",
            "Professional degree (JD, MD)": "College graduate/some postgrad",
            "Doctoral degree": "College graduate/some postgrad",
        },
        "kantorowicz2026": {  # `education_dummy`; NA -> NaN
            "Below higher edu.": "Below higher education",
            "Higher edu.": "Higher education",
        },
    },
    # Select-all race questions (Covid, Kriner2026, berke2023, Grummon 2023/2025, wu2025) are first combined by
    # `race_from_checkboxes` (one box -> its option label, 2+ boxes -> "Two or more races"); the option labels it uses
    # are in COVID_RACE_COLS, KRINER2026_RACE_COLS, BERKE2023_RACE_CODES, GRUMMON_RACE_LABELS and WU2025_RACE_LABELS. The maps below then
    # turn those labels into global levels.
    "RACE": {
        "twin2k": {
            "White": "White",
            "Black": "Black",
            "Asian": "Asian",
            "Hispanic": "Hispanic",
            "Other": "Other",
        },
        "subpop": {
            "White": "White",
            "Black": "Black",
            "Asian": "Asian",
            "Hispanic": "Hispanic",
        },
        "covid": {  # applied after the multi-box rule (2+ boxes -> "Two or more races")
            "White": "White",
            "Black or African American": "Black",
            "Asian": "Asian",
            "Hispanic": "Hispanic",
            "American Indian": "American Indian",
            "Other": "Other",
            "Two or more races": "Two or more races",
        },
        # Grummon 2023 / 2025: "What is your race? (Check all that apply)" checkboxes (Hispanic is the separate
        # HISPANIC attribute); single box -> its label below, 2+ boxes -> "Two or more races" (derived in the loaders).
        "grummon2025": {
            "White": "White",
            "Black or African American": "Black",
            "Asian": "Asian",
            "American Indian or Alaska Native": "American Indian or Alaska Native",
            "Native Hawaiian or Other Pacific Islander": "Native Hawaiian or Other Pacific Islander",
            "Another race": "Other",
            "Two or more races": "Two or more races",
        },
        "grummon2023": {
            "White": "White",
            "Black or African American": "Black",
            "Asian": "Asian",
            "American Indian or Alaska Native": "American Indian or Alaska Native",
            "Native Hawaiian or Other Pacific Islander": "Native Hawaiian or Other Pacific Islander",
            "Another race": "Other",
            "Two or more races": "Two or more races",
        },
        "wvs": {  # WVS US has no Asian group (Asians are in "Other, Non-Hispanic"): its own levels, so "Other" keeps its meaning across datasets
            "US: White, non-Hispanic": "White",
            "US: Black, Non-Hispanic": "Black",
            "US: Hispanic": "Hispanic",
            "US: Other, Non-Hispanic": "Other, Non-Hispanic",
            "US: Two plus, non-Hispanic": "Two plus, non-Hispanic",
        },
        "berke2023": {  # applied after the multi-box rule (2+ boxes -> "Two or more races")
            "White": "White",
            "Black or African American": "Black",
            "American Indian or Alaska Native": "American Indian or Alaska Native",
            "Asian": "Asian",
            "Native Hawaiian or Pacific Islander": "Native Hawaiian or Other Pacific Islander",  # same group, official wording
            "Hispanic or Latino": "Hispanic",
            "Other": "Other",
            "Two or more races": "Two or more races",
        },
        "kriner2026": {  # applied after the multi-box rule (2+ boxes -> "Two or more races")
            "White": "White",
            "Black or African American": "Black",
            "Asian": "Asian",
            "Hispanic": "Hispanic",
            "American Indian": "American Indian",
            "Other": "Other",
            "Two or more races": "Two or more races",
        },
        "wu2025": {  # applied after the multi-box rule (2+ boxes -> "Two or more races")
            "White": "White",
            "Black or African American": "Black",
            "American Indian or Alaska Native": "American Indian or Alaska Native",
            "Asian/Pacific Islander": "Asian/Pacific Islander",
            "Multi-racial": "Two or more races",  # an answer option in this survey; same meaning
            "Other": "Other",
            "Two or more races": "Two or more races",
        },
    },
    "MARITAL": {
        "issp": {
            "Married": "Married",
            "Divorced from spouse/legally separated from civil partner": "Divorced",
            "Separated from spouse/civil partner (still legally married/still legally in a civil partnership)": "Separated",
            "Widowed/civil partner died": "Widowed",
            "Never married/never in a civil partnership": "Never been married",
        },
        "twin2k": {
            "Married": "Married",
            "Living with a partner": "Living with a partner",
            "Divorced": "Divorced",
            "Separated": "Separated",
            "Widowed": "Widowed",
            "Never been married": "Never been married",
        },
        "wvs": {
            "Married": "Married",
            "Living together as married": "Living with a partner",
            "Divorced": "Divorced",
            "Separated": "Separated",
            "Widowed": "Widowed",
            "Single": "Never been married",
        },
        "wu2025": {  # `marriage`
            "Married": "Married",
            "Never married": "Never been married",
            "Divorced": "Divorced",
            "Widowed": "Widowed",
            "Separated": "Separated",
        },
    },
    "RELIGGRP": {
        "issp": {
            "Protestant": "Protestant",
            "Catholic": "Catholic",
            "Orthodox": "Orthodox",
            "Other Christian": "Other Christian",
            "Jewish": "Jewish",
            "Muslim": "Muslim",
            "Buddhist": "Buddhist",
            "Hindu": "Hindu",
            "Atheist": "No religion",  # ISSP "No religion": the build notebooks store it as "Atheist" (ISSP has no atheist answer)
            "Other Religions": "Other Religions",
            "Other Asian Religions": "Other Asian Religions",
        },
        "twin2k": {
            "Protestant": "Protestant",
            "Roman Catholic": "Catholic",
            "Mormon": "Mormon",
            "Orthodox": "Orthodox",
            "Jewish": "Jewish",
            "Muslim": "Muslim",
            "Buddhist": "Buddhist",
            "Hindu": "Hindu",
            "Atheist": "Atheist",
            "Agnostic": "Agnostic",
            "Nothing in particular": "Nothing in particular",
            "Other": "Other",
        },
        "subpop": {
            "Protestant": "Protestant",
            "Jewish": "Jewish",
            "Muslim": "Muslim",
            "Hindu": "Hindu",
            "Atheist": "Atheist",
        },
        "covid": {
            1: "Protestant",
            2: "Catholic",
            3: "Mormon",
            4: "Orthodox",
            5: "Jewish",
            6: "Muslim",
            7: "Buddhist",
            8: "Hindu",
            9: "Atheist",
            10: "Agnostic",
            11: "Other",
            12: "Nothing in particular",
        },
        "wvs": {
            "Do not belong to a denomination": "No religion",
            "Roman Catholic": "Catholic",
            "Protestant": "Protestant",
            "Orthodox (Russian/Greek/etc.)": "Orthodox",
            "Other Christian (Evangelical/Pentecostal/Fee church/etc.)": "Other Christian",
            "Jew": "Jewish",
            "Muslim": "Muslim",
            "Hindu": "Hindu",
            "Buddhist": "Buddhist",
            "Other": "Other",
        },
        "kriner2026": {
            1: "Protestant",
            2: "Catholic",
            3: "Jewish",
            4: "Muslim",
            5: "Other",
            6: "No religion",  # answer "None"
        },
    },
    "ATTEND": {
        "issp": {
            "Several times a week or more often": "More than once a week",
            "Once a week": "Once a week",
            "2 or 3 times a month": "Once or twice a month",
            "Once a month": "Once or twice a month",
            "Several times a year": "A few times a year",
            "Once or twice a year": "A few times a year",
            "Less frequently than once a year": "Seldom",
            "Never": "Never",
        },
        "twin2k": {
            "More than once a week": "More than once a week",
            "Once a week": "Once a week",
            "Once or twice a month": "Once or twice a month",
            "A few times a year": "A few times a year",
            "Seldom": "Seldom",
            "Never": "Never",
        },
        "wvs": {  # Q171
            "More than once a week": "More than once a week",
            "Once a week": "Once a week",
            "Once a month": "Once or twice a month",
            "Only on special holy days": "A few times a year",
            "Once a year": "Seldom",
            "Less often": "Seldom",
            "Never, practically never": "Never",
        },
    },
    "US_PRTY": {
        "issp": {
            "Democrat": "Democrat",
            "Republican": "Republican",
            "Other candidate": "Other candidate",
        },
        "twin2k": {
            "Democrat": "Democrat",
            "Republican": "Republican",
            "Independent": "Independent",
            "Something else": "Something else",
        },
        "subpop": {
            "Democrat": "Democrat",
            "Republican": "Republican",
        },
        "covid": {
            1: "Republican",
            2: "Democrat",
            3: "Independent",
            4: "Something else",
        },
        "wvs": {  # Q223: party the respondent would vote for in a national election tomorrow
            "USA: Democratic Party": "Democrat",
            "USA: Republican Party": "Republican",
            "USA: Libertarian": "Libertarian",
            "USA: Green Party of the United States": "Green",
            "Other": "Other candidate",
        },
        "grummon2023": {
            1.0: "Strong Democrat",
            2.0: "Moderate Democrat",
            3.0: "Lean Democrat",
            4.0: "Independent",
            5.0: "Lean Republican",
            6.0: "Moderate Republican",
            7.0: "Strong Republican",
            8.0: "Something else",
        },
        "kriner2026": {
            1: "Republican",
            2: "Democrat",
            3: "Independent",
            4: "Something else",  # Other/don't know
        },
        "wu2025": {  # `partisan`
            "Democrat": "Democrat",
            "Republican": "Republican",
            "Independent": "Independent",
            "Something else": "Something else",
        },
    },
    "US_INC": {
        "issp": {
            "Less than $30,000": "Less than $30,000",
            "$30,000 - $34,999": "$30,000 - $34,999",
            "$35,000 - $39,999": "$35,000 - $39,999",
            "$40,000 - $49,999": "$40,000 - $49,999",
            "$50,000 - $59,999": "$50,000 - $59,999",
            "$60,000 - $74,999": "$60,000 - $74,999",
            "$75,000 - $89,999": "$75,000 - $89,999",
            "$90,000 - $109,999": "$90,000 - $109,999",
            "$100,000 or more": "$100,000 or more",
        },
        "twin2k": {
            "Less than $30,000": "Less than $30,000",
            "$30,000-$50,000": "$30,000 - $50,000",
            "$50,000-$75,000": "$50,000 - $75,000",
            "$75,000-$100,000": "$75,000 - $100,000",
            "$100,000 or more": "$100,000 or more",
        },
        "subpop": {
            "Less than $30,000": "Less than $30,000",
            "$100,000 or more": "$100,000 or more",
        },
        "covid": {
            1: "Below $20,000",
            2: "$20,000 - $39,999",
            3: "$40,000 - $59,999",
            4: "$60,000 - $79,999",
            5: "$80,000 - $99,999",
            6: "$100,000 or more",
        },
        "grummon2025": {
            1.0: "Less than $10,000",
            2.0: "$10,000 - $14,999",
            3.0: "$15,000 - $24,999",
            4.0: "$25,000 - $34,999",
            5.0: "$35,000 - $49,999",
            6.0: "$50,000 - $75,000",    # $50,000 to $74,999
            7.0: "$75,000 - $100,000",   # $75,000 to $99,999
            8.0: "$100,000 - $149,999",
            9.0: "$150,000 - $199,999",
            10.0: "$200,000 or more",
        },
        "grummon2023": {
            1.0: "Less than $10,000",
            2.0: "$10,000 - $14,999",
            3.0: "$15,000 - $24,999",
            4.0: "$25,000 - $34,999",
            5.0: "$35,000 - $49,999",
            6.0: "$50,000 - $75,000",    # $50,000 to $74,999
            7.0: "$75,000 - $100,000",   # $75,000 to $99,999
            8.0: "$100,000 - $149,999",
            9.0: "$150,000 - $199,999",
            10.0: "$200,000 or more",
        },
        "kriner2026": {
            1: "Below $20,000",
            2: "$20,000 - $39,999",
            3: "$40,000 - $59,999",
            4: "$60,000 - $79,999",
            5: "$80,000 - $99,999",
            6: "$100,000 or more",
        },
        "berke2023": {
            1: "Less than $25,000",
            2: "$25,000 - $49,999",      # $25,000 to $49,999
            3: "$50,000 - $75,000",      # $50,000 to $74,999
            4: "$75,000 - $100,000",     # $75,000 to $99,999
            5: "$100,000 - $149,999",
            6: "$150,000 - $199,999",
            7: "$200,000 or more",
        },
        "wu2025": {  # `income`; one level per bracket (income union)
            "Less than $10,000": "Less than $10,000",
            "$10,000 to $19,999": "$10,000 - $19,999",
            "$20,000 to $29,999": "$20,000 - $29,999",
            "$30,000 to $39,999": "$30,000 - $39,999",
            "$40,000 to $49,999": "$40,000 - $49,999",
            "$50,000 to $59,999": "$50,000 - $59,999",
            "$60,000 to $69,999": "$60,000 - $69,999",
            "$70,000 to $79,999": "$70,000 - $79,999",
            "$80,000 to $89,999": "$80,000 - $89,999",
            "$90,000 to $99,999": "$90,000 - $99,999",
            "$100,000 to $149,999": "$100,000 - $149,999",
            "$150,000 or more": "$150,000 or more",
        },
        "kantorowicz2026": {  # `income_dummy`; NA -> NaN
            "Lower income": "Lower income",
            "Higher income": "Higher income",
        },
    },
    "PARTY_LR": {
        "issp": {
            "Liberal": "Liberal",
            "Conservative": "Conservative",
            "Other": "Other",
        },
        "twin2k": {
            "Very liberal": "Very liberal",
            "Liberal": "Liberal",
            "Moderate": "Moderate",
            "Conservative": "Conservative",
            "Very conservative": "Very conservative",
        },
        "subpop": {
            "Liberal": "Liberal",
            "Moderate": "Moderate",
            "Conservative": "Conservative",
        },
        "covid": {
            1: "Very liberal",
            2: "Liberal",
            3: "Somewhat liberal",
            4: "Moderate",
            5: "Somewhat conservative",
            6: "Conservative",
            7: "Very conservative",
        },
        "wvs": dict(zip(["Left", *[str(i) for i in range(2, 10)], "Right"], LEFT_RIGHT_LEVELS, strict=True)),   # Q240
        "kriner2026": {
            1: "Very liberal",
            2: "Liberal",
            3: "Somewhat liberal",
            4: "Moderate",
            5: "Somewhat conservative",
            6: "Conservative",
            7: "Very conservative",
        },
        "wu2025": {  # `ideology`, mapped by wording (not scale position)
            "Very liberal": "Very liberal",
            "Somewhat liberal": "Somewhat liberal",
            "Slightly liberal": "Slightly liberal",
            "Moderate; middle of the road": "Moderate",
            "Slightly conservative": "Slightly conservative",
            "Somewhat conservative": "Somewhat conservative",
            "Very conservative": "Very conservative",
        },
    },
    "HHSIZE": {
        "twin2k": {
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "More than 4": "More than 4",
        },
        "wvs": {"1": "1", "2": "2", "3": "3", "4": "4", "5": "More than 4", "6": "More than 4"},
        "grummon2025": {float(k): (str(k) if k <= 4 else "More than 4") for k in range(1, 21)},
        "grummon2023": {float(k): (str(k) if k <= 4 else "More than 4") for k in range(1, 21)},
    },
    "MAINSTAT": {
        "issp": {
            "In paid work": "In paid work",
            "Unemployed and looking for a job": "Unemployed",
            "Domestic work": "Home-maker",
            "In education": "Student",
            "Retired": "Retired",
            "Permanently sick or disabled": "Permanently sick or disabled",
            "Other": "Other",
        },
        "twin2k": {
            "Full-time employment": "Full-time employment",
            "Part-time employment": "Part-time employment",
            "Self-employed": "Self-employed",
            "Unemployed": "Unemployed",
            "Home-maker": "Home-maker",
            "Student": "Student",
            "Retired": "Retired",
        },
        "wvs": {
            "Full time (30 hours a week or more)": "Full-time employment",
            "Part time (less than 30 hours a week)": "Part-time employment",
            "Self employed": "Self-employed",
            "Retired/pensioned": "Retired",
            "Housewife not otherwise employed": "Home-maker",
            "Student": "Student",
            "Unemployed": "Unemployed",
            "Other": "Other",
        },
        "wu2025": {  # `employment`
            "Working (paid employee)": "Working (paid employee)",
            "Working (self-employed)": "Self-employed",
            "Not working (retired)": "Retired",
            "Not working (disabled)": "Not working (disabled)",
            "Not working (looking for work)": "Unemployed",
            "Not working (temporary layoff from a job)": "Not working (temporary layoff from a job)",
            "Not working (other)": "Not working (other)",
            "Prefer not to answer": "Prefer not to say",
        },
    },
    "VOTE_LE": {
        "issp": {"Yes": "Yes", "No": "No"},
        "kriner2026": {  # `vs_vote_2024`
            1: "Yes",                             # I voted
            2: "I was too young to vote",
            3: "I was not registered to vote",
            4: "No",                              # I did not vote
        },
    },
    "VOTE_FREQ": {
        "wvs": {"Always": "Always", "Usually": "Usually", "Never": "Never", "Not allowed to vote": None},   # Q222
    },
    "URBRURAL": {
        "issp": {
            "A big city": "A big city",
            "The suburbs or outskirts of a big city": "The suburbs or outskirts of a big city",
            "A town or a small city": "A town or a small city",
            "A country village": "A country village",
            "A farm or home in the country": "A farm or home in the country",
        },
        "wvs": {"Urban": "Urban", "Rural": "Rural"},
        "kriner2026": {1: "Urban", 2: "Suburban", 3: "Rural"},  # `urban_rural_best`
        "berke2023": {1: "Urban", 2: "Suburban", 3: "Rural", 4: "I don't know"},  # `Q_urban_rural`
    },
    "UNION": {
        "issp": {"Yes": "Yes", "No": "No"},
        "wvs": {"Not a member": "No", "Inactive member": "Inactive member", "Active member": "Active member"},   # Q97
    },
    "US_ETHN1": {
        "issp": {v: v for v in US_ETHN1_LEVELS},
    },
    "RESTYPE": {
        "berke2023": {1: "Private home", 2: "Apartment", 3: "Other"},  # `Q_residence_type`
        "wu2025": {v: v for v in ["Single family home", "Apartment", "Townhouse", "Mobile home", "Condominium", "Something else"]},  # `residence`
    },
    # Separate "Hispanic / Latino" question (not a race option in these surveys).
    "HISPANIC": {
        "grummon2023": {0.0: "No", 1.0: "Yes"},  # `latino`: Are you of Hispanic, Latino, or Spanish origin?
        "grummon2025": {0.0: "No", 1.0: "Yes"},  # `latino`: same question
        "wu2025": {  # `ethnicity`: Are you Spanish, Hispanic, or Latino or none of these?
            "Spanish, Hispanic, or Latino": "Yes",
            "None of these": "No",
        },
    },
    "POL_LABEL": {
        "wu2025": {v: v for v in ["Libertarian", "Socialist", "Communist", "Liberal", "Conservative", "Centrist", "Other"]},
    },
    "OWNHOME": {
        "wu2025": {"Yes": "Yes", "No": "No"},  # `residenceownership`
    },
    "RENTHOME": {
        "wu2025": {"Yes": "Yes", "No": "No"},  # `residencerental`
    },
    "ORIENTATION": {
        "wu2025": {v: v for v in GLOBAL_PERSONA_LEVELS["ORIENTATION"]},  # `orientation`, verbatim
    },
    "MILITARY": {
        "wu2025": {"Yes": "Yes", "No": "No"},  # `military`
    },
    "ELECHEAT": {
        "wu2025": {"Yes": "Yes", "No": "No"},  # `householdheat`
    },
}



def bucket_age_series(series: pd.Series, dataset: str) -> pd.Series:
    """Bucket numeric respondent ages into canonical AGE_LABELS with strict range validation."""
    age_num = pd.to_numeric(series, errors="raise")
    age_bucketed = pd.cut(age_num, bins=AGE_BINS, labels=AGE_LABELS).astype(object)
    if int(age_bucketed.isna().sum()) != int(series.isna().sum()):
        raise ValueError(f"{dataset}: some non-missing AGE values fell outside AGE_BINS {AGE_BINS}")
    return age_bucketed


MULTI_RACE_LEVEL = "Two or more races"


def race_from_checkboxes(
    flags: pd.DataFrame, labels: list[tuple[str, str]], dataset: str, *, allow_all_missing: bool
) -> pd.Series:
    """Raw race label per row from select-all race checkboxes (0/1 columns).

    `labels` lists `(column, option label)` in survey order. One box checked -> its label; two or more -> `MULTI_RACE_LEVEL`.
    A row with every box missing -> NaN if `allow_all_missing`, else an error. Partly missing rows, values other than 0/1,
    and rows with no box checked raise. Map the result with `map_persona_series(..., "RACE", dataset)`.
    """
    cols = [c for c, _ in labels]
    missing_cols = [c for c in cols if c not in flags.columns]
    if missing_cols:
        raise KeyError(f"{dataset}: missing race columns {missing_cols}")
    mat = flags[cols].apply(pd.to_numeric, errors="raise")
    unexpected = set(mat.stack().dropna().unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"{dataset}: race indicators must be 0/1, got {sorted(unexpected)}")
    out: list[object] = []
    for _, row in mat.iterrows():
        if row.isna().all():
            if not allow_all_missing:
                raise ValueError(f"{dataset}: respondent with every race box missing")
            out.append(float("nan"))
            continue
        if row.isna().any():
            raise ValueError(f"{dataset}: partly missing race boxes")
        checked = [label for col, label in labels if row[col] == 1]
        if not checked:
            raise ValueError(f"{dataset}: respondent with no race box checked")
        out.append(checked[0] if len(checked) == 1 else MULTI_RACE_LEVEL)
    return pd.Series(out, index=flags.index, name="RACE", dtype=object)


def map_persona_series(series: pd.Series, attr: str, dataset: str) -> pd.Series:
    """Map a dataset persona column into its canonical global levels with fail-fast validation."""
    if attr == "AGE" and dataset in {"issp", "covid", "grummon2025", "grummon2023", "wvs", "kriner2026"}:
        return bucket_age_series(series, dataset=dataset)

    mapping = PERSONA_MAPS[attr][dataset]
    observed = set(series.dropna().unique())
    unexpected = observed - set(mapping.keys())
    if unexpected:
        raise ValueError(f"{dataset}.{attr}: unexpected values not in PERSONA_MAPS: {unexpected}")

    allowed = set(GLOBAL_PERSONA_LEVELS[attr])
    out = [mapping[v] if pd.notna(v) else float("nan") for v in series]
    mapped_vals = {v for v in out if pd.notna(v)}
    invalid_targets = mapped_vals - allowed
    if invalid_targets:
        raise ValueError(f"{dataset}.{attr}: mapped values outside GLOBAL_PERSONA_LEVELS: {invalid_targets}")
    return pd.Series(out, index=series.index, name=attr, dtype=object)


def apply_persona_mapping(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Validate dataset persona columns and map all configured persona attributes into global levels."""
    expected_cols = DATASET_PERSONA_COLS[dataset]
    df = df.copy()
    if dataset == "issp":
        persona_cols = list(df.columns[1:df.columns.get_loc("v")])
        if persona_cols != expected_cols:
            raise ValueError(f"{dataset}: persona columns mismatch: {persona_cols} != {expected_cols}")
    elif dataset == "twin2k":
        if "c_alphan" not in df.columns:
            df["c_alphan"] = "United States"
        for qid, attr in TWIN2K_QID_TO_PERSONA.items():
            if attr not in df.columns:
                if qid not in df.columns:
                    raise ValueError(f"twin2k: missing column {qid} for persona attribute {attr}")
                df[attr] = df[qid]
    elif dataset == "covid":
        if "c_alphan" not in df.columns:
            df["c_alphan"] = "United States"
        for src_col, attr in COVID_COL_TO_PERSONA.items():
            if attr not in df.columns:
                if src_col not in df.columns:
                    raise ValueError(f"covid: missing column {src_col} for persona attribute {attr}")
                df[attr] = df[src_col]
        if "RACE" not in df.columns:
            df["RACE"] = race_from_checkboxes(df, COVID_RACE_COLS, "covid", allow_all_missing=False)
    elif dataset == "berke2023":
        df["c_alphan"] = "United States"
        for src_col, attr in BERKE2023_COL_TO_PERSONA.items():
            if src_col not in df.columns:
                raise ValueError(f"berke2023: missing column {src_col} for persona attribute {attr}")
            df[attr] = df[src_col]
        if "Q_race" not in df.columns:
            raise ValueError("berke2023: missing column Q_race")
        # `Q_race` stores the checked boxes as comma-separated codes ("1,6"): expand to one 0/1 column per code.
        flag_rows = []
        for val in df["Q_race"]:
            if pd.isna(val):
                raise ValueError("berke2023: missing Q_race")
            codes = [int(x) for x in str(val).split(",")]
            unknown = [c for c in codes if c not in BERKE2023_RACE_CODES]
            if unknown or len(set(codes)) != len(codes):
                raise ValueError(f"berke2023: invalid Q_race value {val!r}")
            flag_rows.append({f"Q_race_{c}": int(c in codes) for c in BERKE2023_RACE_CODES})
        flags = pd.DataFrame(flag_rows, index=df.index)
        race_labels = [(f"Q_race_{c}", label) for c, label in BERKE2023_RACE_CODES.items()]
        df["RACE"] = race_from_checkboxes(flags, race_labels, "berke2023", allow_all_missing=False)
    elif dataset == "wu2025":
        df["c_alphan"] = "United States"  # the authors dropped "I do not reside in the United States"
        source_cols = [*WU2025_COL_TO_PERSONA, "race"]
        missing_cols = [c for c in source_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"wu2025: missing columns {missing_cols}")
        for src_col, attr in WU2025_COL_TO_PERSONA.items():
            df[attr] = df[src_col].replace("", float("nan"))  # "" = no answer recorded
        flag_rows = []
        for val in df["race"]:
            if val == "":
                flag_rows.append({f"race_{i}": float("nan") for i in range(len(WU2025_RACE_LABELS))})
                continue
            parts = str(val).split(",")
            unknown = [p for p in parts if p not in WU2025_RACE_LABELS]
            if unknown or len(set(parts)) != len(parts):
                raise ValueError(f"wu2025: invalid race value {val!r}")
            flag_rows.append({f"race_{i}": int(lab in parts) for i, lab in enumerate(WU2025_RACE_LABELS)})
        flags = pd.DataFrame(flag_rows, index=df.index)
        race_labels = [(f"race_{i}", lab) for i, lab in enumerate(WU2025_RACE_LABELS)]
        df["RACE"] = race_from_checkboxes(flags, race_labels, "wu2025", allow_all_missing=True)
    elif dataset == "kriner2026":
        df["c_alphan"] = "United States"
        for src_col, attr in KRINER2026_COL_TO_PERSONA.items():
            if src_col not in df.columns:
                raise ValueError(f"kriner2026: missing column {src_col} for persona attribute {attr}")
            df[attr] = df[src_col]
        df["RACE"] = race_from_checkboxes(df, KRINER2026_RACE_COLS, "kriner2026", allow_all_missing=False)
    elif dataset == "kantorowicz2026":
        df["c_alphan"] = "United States"
        missing_cols = [c for c in KANTOROWICZ2026_COL_TO_PERSONA if c not in df.columns]
        if missing_cols:
            raise ValueError(f"kantorowicz2026: missing columns {missing_cols}")
        for src_col, attr in KANTOROWICZ2026_COL_TO_PERSONA.items():
            df[attr] = df[src_col]
    elif dataset == "wvs":
        for src_col, attr in WVS_COL_TO_PERSONA.items():
            if src_col not in df.columns:
                raise ValueError(f"wvs: missing column {src_col} for persona attribute {attr}")
            df[attr] = df[src_col]
    else:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{dataset}: missing expected persona columns {missing}")

    for col in expected_cols:
        df[col] = map_persona_series(df[col], col, dataset)
    return df


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
    "HHSIZE": "How many people, including yourself, live in your household?",
    "VOTE_FREQ": "Votes in national elections",
}

DATASET_PERSONA_COLS = {
    "issp": [c for c in PERSONA_NAMES if c not in {"RACE", "HHSIZE", "VOTE_FREQ"}],
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
    "SEX": ["Male", "Female", "Non-binary"],
    "AGE": ["18-29", "30-49", "50-64", "65+"],
    "US_REG": ["Northeast", "Midwest", "South", "West"],
    "EDULEVEL": [
        "Less than high school",
        "High school or some college",
        "Associate's degree",
        "College graduate/some postgrad",
    ],
    "RACE": ["White", "Black", "Asian", "Hispanic", "Other", "Other, Non-Hispanic", "Two plus, non-Hispanic"],
    "MARITAL": [
        "Married",
        "Living with a partner",
        "Divorced",
        "Separated",
        "Widowed",
        "Never been married",
    ],
    "RELIGGRP": [
        "Protestant",
        "Catholic",
        "Orthodox",
        "Other Christian",
        "Jewish",
        "Muslim",
        "Buddhist",
        "Hindu",
        "Atheist",
        "Agnostic",
        "Nothing in particular",
        "Other",
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
    "US_INC": [
        "Less than $30,000",
        "$30,000 - $50,000",
        "$50,000 - $75,000",
        "$75,000 - $89,999",
        "$75,000 - $100,000",
        "$90,000 - $109,999",
        "$100,000 or more",
        "Below $20,000",
        "$20,000 - $39,999",
        "$40,000 - $59,999",
        "$60,000 - $79,999",
        "$80,000 - $99,999",
        "Less than $25,000",
        "$25,000 - $34,999",
        "$35,000 - $49,999",
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
    ],
    "VOTE_LE": ["Yes", "No"],
    "VOTE_FREQ": ["Always", "Usually", "Never"],
    "URBRURAL": [
        "A big city",
        "The suburbs or outskirts of a big city",
        "A town or a small city",
        "A country village",
        "A farm or home in the country",
        "Urban",
        "Rural",
    ],
    "UNION": ["Yes", "No", "Active member", "Inactive member"],
    "US_ETHN1": US_ETHN1_LEVELS,
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
    },
    "SEX": {
        "issp": {"Male": "Male", "Female": "Female"},
        "twin2k": {"Male": "Male", "Female": "Female"},
        "subpop": {"Male": "Male", "Female": "Female"},
        "covid": {1: "Male", 2: "Female", 3: None},
        "grummon2025": {1.0: "Female", 2.0: "Male", 3.0: "Non-binary", 4.0: None},
        "grummon2023": {1.0: "Female", 2.0: "Male", 3.0: "Non-binary", 4.0: None},
        "wvs": {"Male": "Male", "Female": "Female"},
    },
    "AGE": {
        "twin2k": {
            "18-29": "18-29",
            "30-49": "30-49",
            "50-64": "50-64",
            "65+": "65+",
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
        "covid": {  # state (1..51) -> US Census Bureau region
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
        },
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
    },
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
        "covid": {
            "White": "White",
            "Black or African American": "Black",
            "Asian": "Asian",
            "Hispanic": "Hispanic",
            "American Indian": "Other",
            "Other": "Other",
        },
        "grummon2025": {
            "White": "White",
            "Black": "Black",
            "Asian": "Asian",
            "Hispanic": "Hispanic",
            "Amer Ind AK Native": "Other",
            "Native Hawaiian or Pacific Islander": "Other",
            "Other or multi-racial": "Other",
        },
        "grummon2023": {
            "White": "White",
            "Black": "Black",
            "Asian": "Asian",
            "Other": "Other",
        },
        "wvs": {  # WVS US has no Asian group (Asians are in "Other, Non-Hispanic"): its own levels, so "Other" keeps its meaning across datasets
            "US: White, non-Hispanic": "White",
            "US: Black, Non-Hispanic": "Black",
            "US: Hispanic": "Hispanic",
            "US: Other, Non-Hispanic": "Other, Non-Hispanic",
            "US: Two plus, non-Hispanic": "Two plus, non-Hispanic",
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
            "Atheist": "Atheist",
            "Other Religions": "Other",
            "Other Asian Religions": "Other",
        },
        "twin2k": {
            "Protestant": "Protestant",
            "Roman Catholic": "Catholic",
            "Mormon": "Other Christian",
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
            3: "Other Christian",
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
            "Do not belong to a denomination": "Nothing in particular",
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
    },
    "US_INC": {
        "issp": {
            "Less than $30,000": "Less than $30,000",
            "$30,000 - $34,999": "$30,000 - $50,000",
            "$35,000 - $39,999": "$30,000 - $50,000",
            "$40,000 - $49,999": "$30,000 - $50,000",
            "$50,000 - $59,999": "$50,000 - $75,000",
            "$60,000 - $74,999": "$50,000 - $75,000",
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
            1.0: "Less than $25,000",
            2.0: "Less than $25,000",
            3.0: "Less than $25,000",
            4.0: "$25,000 - $34,999",
            5.0: "$35,000 - $49,999",
            6.0: "$50,000 - $75,000",
            7.0: "$75,000 - $100,000",
            8.0: "$100,000 or more",
            9.0: "$100,000 or more",
            10.0: "$100,000 or more",
        },
        "grummon2023": {
            1.0: "Less than $25,000",
            2.0: "Less than $25,000",
            3.0: "Less than $25,000",
            4.0: "$25,000 - $34,999",
            5.0: "$35,000 - $49,999",
            6.0: "$50,000 - $75,000",
            7.0: "$75,000 - $100,000",
            8.0: "$100,000 or more",
            9.0: "$100,000 or more",
            10.0: "$100,000 or more",
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
    },
    "VOTE_LE": {
        "issp": {"Yes": "Yes", "No": "No"},
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
    },
    "UNION": {
        "issp": {"Yes": "Yes", "No": "No"},
        "wvs": {"Not a member": "No", "Inactive member": "Inactive member", "Active member": "Active member"},   # Q97
    },
    "US_ETHN1": {
        "issp": {v: v for v in US_ETHN1_LEVELS},
    },
}


def bucket_age_series(series: pd.Series, dataset: str) -> pd.Series:
    """Bucket numeric respondent ages into canonical AGE_LABELS with strict range validation."""
    age_num = pd.to_numeric(series, errors="raise")
    age_bucketed = pd.cut(age_num, bins=AGE_BINS, labels=AGE_LABELS).astype(object)
    if int(age_bucketed.isna().sum()) != int(series.isna().sum()):
        raise ValueError(f"{dataset}: some non-missing AGE values fell outside AGE_BINS {AGE_BINS}")
    return age_bucketed


def map_persona_series(series: pd.Series, attr: str, dataset: str) -> pd.Series:
    """Map a dataset persona column into its canonical global levels with fail-fast validation."""
    if attr == "AGE" and dataset in {"issp", "covid", "grummon2025", "grummon2023", "wvs"}:
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
            missing_race = [col for col, _ in COVID_RACE_COLS if col not in df.columns]
            if missing_race:
                raise ValueError(f"covid: missing race columns {missing_race}")
            raw_races = []
            for _, r in df.iterrows():
                checked = [label for col, label in COVID_RACE_COLS if r[col] == 1]
                if not checked:
                    raise ValueError("covid: respondent has no race indicator checked in race_1..race_6")
                raw_races.append(checked[0] if len(checked) == 1 else "Other")
            df["RACE"] = pd.Series(raw_races, index=df.index, dtype=object)
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


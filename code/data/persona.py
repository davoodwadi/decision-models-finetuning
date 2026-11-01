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
}

DATASET_PERSONA_COLS = {
    "issp": [c for c in PERSONA_NAMES if c != "RACE"],
}

GLOBAL_PERSONA_LEVELS = {
    "c_alphan": ["United States"],
    "SEX": ["Male", "Female"],
    "AGE": ["18-29", "30-49", "50-64", "65+"],
    "US_REG": ["Northeast", "Midwest", "South", "West"],
    "EDULEVEL": [
        "Less than high school",
        "High school or some college",
        "Associate's degree",
        "College graduate/some postgrad",
    ],
    "RACE": ["White", "Black", "Asian", "Hispanic", "Other"],
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
        "Republican",
        "Independent",
        "Something else",
        "Other candidate",
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
    ],
}

PERSONA_MAPS = {
    "c_alphan": {
        "issp": {"US": "United States"},
        "twin2k": {"United States": "United States"},
        "subpop": {"United States": "United States"},
        "covid": {"United States": "United States"},
        "grummon": {"United States": "United States"},
    },
    "SEX": {
        "issp": {"Male": "Male", "Female": "Female"},
        "twin2k": {"Male": "Male", "Female": "Female"},
        "subpop": {"Male": "Male", "Female": "Female"},
        "covid": {1: "Male", 2: "Female", 3: None},
        "grummon": {1.0: "Female", 2.0: "Male", 3.0: None, 4.0: None},
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
        "grummon": {
            1.0: "Less than high school",
            2.0: "High school or some college",
            3.0: "High school or some college",
            4.0: "Associate's degree",
            5.0: "College graduate/some postgrad",
            6.0: "College graduate/some postgrad",
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
        "grummon": {
            "White": "White",
            "Black": "Black",
            "Asian": "Asian",
            "Hispanic": "Hispanic",
            "Amer Ind AK Native": "Other",
            "Native Hawaiian or Pacific Islander": "Other",
            "Other or multi-racial": "Other",
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
            "Democrat": "Democrat",
            "Republican": "Republican",
            "Independent": "Independent",
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
        "grummon": {
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
    if attr == "AGE" and dataset in {"issp", "covid", "grummon"}:
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
    if dataset == "issp":
        persona_cols = list(df.columns[1:df.columns.get_loc("v")])
        if persona_cols != expected_cols:
            raise ValueError(f"{dataset}: persona columns mismatch: {persona_cols} != {expected_cols}")
    else:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{dataset}: missing expected persona columns {missing}")

    df = df.copy()
    for col in expected_cols:
        if (col == "AGE" and dataset in {"issp", "covid", "grummon"}) or (
            col in PERSONA_MAPS and dataset in PERSONA_MAPS[col]
        ):
            df[col] = map_persona_series(df[col], col, dataset)
    return df


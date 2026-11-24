"""Clean per-dataset raw persona loaders using `persona_spec.yaml`."""

import json
from pathlib import Path

import pandas as pd
from pandas.io.stata import StataReader

from data import grummon2023 as g23
from data import grummon2025 as g25
from data.berke2023 import stimulus as berke_s
from data.covid import stimulus as covid_s
from data.kantorowicz2026 import stimulus as kant_s
from data.Kriner2026 import stimulus as kriner_s
from data.magistro2025 import stimulus as mag_s
from data.persona_spec import PersonaAttribute, load_persona_spec, standardize_dataframe
from data.wu2025 import stimulus as wu_s

DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Strict helpers (zero silent fallbacks)
# ---------------------------------------------------------------------------


def _strict_map(series: pd.Series, mapping: dict, name: str) -> pd.Series:
    """Map non-null codes strictly through `mapping`, raising on any unexpected value."""
    out = []
    for val in series:
        if pd.isna(val):
            out.append(float("nan"))
        elif val in mapping:
            out.append(mapping[val])
        else:
            raise ValueError(f"{name}: unexpected raw code {val!r}")
    return pd.Series(out, index=series.index, dtype=object)


def _checkbox_race(df: pd.DataFrame, col_labels: list[tuple[str, str]], dataset: str) -> pd.Series:
    """Combine raw select-all-that-apply 0/1 race checkboxes into a comma-separated raw string."""
    out = []
    for _, row in df.iterrows():
        vals = [row[col] for col, _ in col_labels]
        if all(pd.isna(v) for v in vals):
            out.append(float("nan"))
            continue
        chosen = []
        for (col, label), val in zip(col_labels, vals, strict=True):
            if pd.isna(val) or val == 0:
                continue
            if val == 1:
                chosen.append(label)
            else:
                raise ValueError(f"{dataset}.{col}: expected 0/1/NaN, got {val!r}")
        out.append("; ".join(chosen) if chosen else float("nan"))
    return pd.Series(out, index=df.index, dtype=object)


def _combine_multivalue_cols(df: pd.DataFrame, cols: list[str], name: str) -> pd.Series:
    """Join non-null string values across multiple mention columns using `"; "`."""
    out = []
    for _, row in df.iterrows():
        parts = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                continue
            text = str(val)
            if "; " in text:
                raise ValueError(f"{name}.{col}: raw value contains delimiter '; ': {text!r}")
            parts.append(text)
        out.append("; ".join(parts) if parts else float("nan"))
    return pd.Series(out, index=df.index, dtype=object)


def _read_stata_value_labels(dta_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    with StataReader(dta_path) as reader:
        value_labels = reader.value_labels()
        col_to_lblname = dict(zip(reader._varlist, reader._lbllist, strict=True))
    return value_labels, col_to_lblname


def _decode_stata_col(
    series: pd.Series,
    col: str,
    value_labels: dict[str, dict],
    col_to_lblname: dict[str, str],
) -> pd.Series:
    lbl_name = col_to_lblname[col]
    if lbl_name and lbl_name in value_labels:
        mapping = value_labels[lbl_name]
        return series.map(lambda x: mapping[x] if (pd.notna(x) and x in mapping) else x)
    return series


# ---------------------------------------------------------------------------
# 1. Surveys & benchmark (raw)
# ---------------------------------------------------------------------------

ISSP_SHARED_RENAME = {
    "c_alphan": "country",
    "SEX": "gender",
    "AGE": "age",
    "US_REG": "region",
    "EDULEVEL": "education",
    "US_INC": "family_income",
    "RELIGGRP": "religion",
    "ATTEND": "religious_attendance",
    "US_PRTY": "political_party",
    "PARTY_LR": "political_ideology",
    "MARITAL": "marital_status",
    "MAINSTAT": "employment_status",
    "HOMPOP": "household_size",
    "URBRURAL": "urban_rural",
    "VOTE_LE": "voted_last_election",
    "UNION": "union_member",
    "INTLANG": "interview_language",
    "TYPORG2": "employer_sector",
    "BIRTH": "birth_year",
    "SPEDULEV": "spouse_education",
    "SPMAINST": "spouse_employment",
    "ISCO08": "occupation",
}

ISSP_SINGLE_RENAME = {
    "US_RINC": "personal_income",
}

ISSP_RAW_CANDIDATES = [
    *ISSP_SHARED_RENAME.keys(),
    "DEGREE",
    "EDUCYRS",
    "US_DEGR",
    "US_ETHN1",
    "US_ETHN2",
    "US_RINC",
    "WORK",
    "WRKHRS",
    "EMPREL",
    "WRKSUP",
    "NSUP",
    "TYPORG1",
    "PARTLIV",
    "SPWORK",
    "SPWRKHRS",
    "SPEMPREL",
    "SPWRKSUP",
    "SPISCO08",
    "HHCHILDR",
    "HHTODD",
    "HHADULT",
    "TOPBOT",
    "RELIG",
]


def load_raw_issp_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw ISSP (2015-2024 US) persona columns from `.dta` files with Stata value labels."""
    candidates = [*ISSP_SHARED_RENAME.keys(), "DEGREE"] if shared_only else ISSP_RAW_CANDIDATES
    frames = []

    for year in range(2015, 2025):
        dta_files = list((data_dir / f"issp{year}" / "raw").glob("*.dta"))
        if len(dta_files) != 1:
            raise FileNotFoundError(f"Expected 1 .dta file for ISSP {year}, found {dta_files}")
        dta_path = dta_files[0]

        value_labels, col_to_lblname = _read_stata_value_labels(dta_path)
        present = [c for c in candidates if c in col_to_lblname]
        raw_cols = sorted({"CASEID", "c_alphan", *present})

        df = pd.read_stata(dta_path, columns=raw_cols, convert_categoricals=False)
        us = df[df["c_alphan"] == "US"].reset_index(drop=True)

        wave = pd.DataFrame({"CASEID": [f"{year}:{int(cid)}" for cid in us["CASEID"]]})
        for col in present:
            wave[col] = _decode_stata_col(us[col], col, value_labels, col_to_lblname)

        # 2015-2019 waves name EDULEVEL as DEGREE
        if "EDULEVEL" not in wave.columns and "DEGREE" in wave.columns:
            wave = wave.rename(columns={"DEGREE": "EDULEVEL"})
        elif shared_only and "DEGREE" in wave.columns:
            wave = wave.drop(columns=["DEGREE"])

        # Combine first and second US ancestry mentions into one multi-select column
        if "US_ETHN1" in wave.columns or "US_ETHN2" in wave.columns:
            ethn_cols = [c for c in ["US_ETHN1", "US_ETHN2"] if c in wave.columns]
            wave["ancestry"] = _combine_multivalue_cols(wave, ethn_cols, f"issp{year}.ancestry")
            wave = wave.drop(columns=ethn_cols)

        rename_map = ISSP_SHARED_RENAME if shared_only else {**ISSP_SHARED_RENAME, **ISSP_SINGLE_RENAME}
        wave = wave.rename(columns={k: v for k, v in rename_map.items() if k in wave.columns})
        frames.append(wave)

    return pd.concat(frames, ignore_index=True)


TWIN2K_QID_TO_COL = {
    "QID11": "region",
    "QID12": "gender",
    "QID13": "age",
    "QID14": "education",
    "QID15": "race",
    "QID16": "citizen",
    "QID17": "marital_status",
    "QID18": "religion",
    "QID19": "religious_attendance",
    "QID20": "political_party",
    "QID21": "family_income",
    "QID22": "political_ideology",
    "QID23": "household_size",
    "QID24": "employment_status",
}


def load_raw_twin2k_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw Twin-2K-500 Demographics block from HuggingFace chunk parquets."""
    chunk_paths = sorted(
        (data_dir / "Twin-2K-500" / "hf" / "wave_split" / "chunks").glob("wave_persona_chunk_*.parquet")
    )
    if not chunk_paths:
        raise FileNotFoundError("No Twin-2K persona chunk parquets found")

    rows = []
    for path in chunk_paths:
        chunk = pd.read_parquet(path, columns=["pid", "wave1_3_persona_json"])
        for _, rec in chunk.iterrows():
            demo_block = json.loads(rec["wave1_3_persona_json"])[0]
            if demo_block["BlockName"] != "Demographics":
                raise ValueError(f"Expected first Twin-2K block 'Demographics', got {demo_block['BlockName']!r}")
            row = {"CASEID": str(rec["pid"]), "country": "United States"}
            for q in demo_block["Questions"]:
                qid = q["QuestionID"]
                if qid not in TWIN2K_QID_TO_COL:
                    raise KeyError(f"Unexpected Twin-2K Demographics QID: {qid!r}")
                row[TWIN2K_QID_TO_COL[qid]] = q["Answers"]["SelectedText"]
            rows.append(row)

    return pd.DataFrame(rows)


WVS_DROP_COLS = [
    "A_YEAR",
    "J_INTDATE",
    "N_TOWN",
    "v",
    "label",
    "question",
    "type",
    "reversed",
    "option_codes",
    "option_texts",
    "answer_code",
    "answer_text",
]

WVS_SHARED_RENAME = {
    "B_COUNTRY": "country",
    "N_REGION_ISO": "state",
    "H_URBRURAL": "urban_rural",
    "S_INTLANGUAGE": "interview_language",
    "Q260": "gender",
    "Q261": "birth_year",
    "Q262": "age",
    "Q269": "citizen",
    "Q270": "household_size",
    "Q273": "marital_status",
    "Q274": "children_ever_had",
    "Q275": "education",
    "Q276": "spouse_education",
    "Q279": "employment_status",
    "Q280": "spouse_employment",
    "Q281": "occupation",
    "Q284": "employer_sector",
    "Q289": "religion",
    "Q290": "race",
    "Q223": "political_party",
    "Q171": "religious_attendance",
    "Q97": "union_member",
    "Q240": "political_ideology",
    "Q222": "voting_frequency",
}

WVS_SINGLE_RENAME = {
    "G_TOWNSIZE": "town_size",
    "Q263": "immigrant_status",
    "Q271": "lives_with_parents",
    "Q272": "home_language",
    "Q285": "chief_wage_earner",
}


def load_raw_wvs7_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw WVS Wave 7 US persona columns."""
    pq_path = data_dir / "not-clean-WVS-Wave7-unclean" / "wvs7_US_long.parquet"
    if shared_only:
        raw_df = pd.read_parquet(pq_path, columns=["CASEID", *WVS_SHARED_RENAME.keys()])
        rename_map = WVS_SHARED_RENAME
    else:
        raw_df = pd.read_parquet(pq_path).drop(columns=WVS_DROP_COLS)
        rename_map = {**WVS_SHARED_RENAME, **WVS_SINGLE_RENAME}
    df = raw_df.drop_duplicates("CASEID").rename(columns=rename_map).reset_index(drop=True)
    df["CASEID"] = df["CASEID"].astype(str)
    return df


SUBPOP_KEY_TO_COL = {
    "Country": "country",
    "Region": "region",
    "Education": "education",
    "Income": "income",
    "Political ideology": "political_ideology",
    "Political party": "political_party",
    "Race": "race",
    "Religion": "religion",
    "Sex": "gender",
}


def load_raw_subpop_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw SubPOP target subgroup profiles."""
    rows_by_group = {}
    for filename in ["subpop_eval.jsonl", "subpop100.jsonl"]:
        with open(data_dir / "subpop" / filename, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                group = record["group"]
                if group not in rows_by_group:
                    row = {"CASEID": group}
                    for key, value in record["persona"].items():
                        row[SUBPOP_KEY_TO_COL[key]] = value
                    rows_by_group[group] = row
    return pd.DataFrame(rows_by_group.values())


# ---------------------------------------------------------------------------
# 2. Choice datasets (all loaded from raw source files with verbatim labels)
# ---------------------------------------------------------------------------

COVID_RACE_COLS = [
    ("race_1", "American Indian"),
    ("race_2", "Asian"),
    ("race_3", "Black or African American"),
    ("race_4", "Hispanic"),
    ("race_5", "White"),
    ("race_6", "Other"),
]


def load_raw_covid_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw Kreps et al. (2020) COVID vaccine conjoint personas from Stata `.dta`."""
    dta_path = data_dir / "covid" / "dataverse_files" / "Kreps_etal_vax_replication_data.dta"
    value_labels, col_to_lblname = _read_stata_value_labels(dta_path)
    resp = covid_s.extract_respondent_df(covid_s.load_raw_conjoint_df(dta_path)).reset_index(drop=True)

    return pd.DataFrame({
        "CASEID": resp["CASEID"].astype(str),
        "country": "United States",
        "state": _decode_stata_col(resp["state"], "state", value_labels, col_to_lblname),
        "gender": _decode_stata_col(resp["gender"], "gender", value_labels, col_to_lblname),
        "age": resp["age"],  # raw numeric age in years (18-94)
        "education": _decode_stata_col(resp["education"], "education", value_labels, col_to_lblname),
        "religion": _decode_stata_col(resp["relig"], "relig", value_labels, col_to_lblname),
        "political_ideology": _decode_stata_col(resp["ideology"], "ideology", value_labels, col_to_lblname),
        "political_party": _decode_stata_col(resp["party3"], "party3", value_labels, col_to_lblname),
        "income": _decode_stata_col(resp["income"], "income", value_labels, col_to_lblname),
        "race": _checkbox_race(resp, COVID_RACE_COLS, "covid"),
    })


GRUMMON_GENDER_LABELS = {
    1.0: "Woman",
    2.0: "Man",
    3.0: "Non-binary",
    4.0: "Prefer to self-describe",
}

GRUMMON_EDUC_LABELS = {
    1.0: "Less than high school",
    2.0: "High school graduate (or GED)",
    3.0: "Some college or technical school",
    4.0: "Associate's degree",
    5.0: "Bachelor's degree",
    6.0: "Graduate or professional degree",
}

GRUMMON_INCOME_LABELS = {
    1.0: "Less than $10,000",
    2.0: "$10,000 to $14,999",
    3.0: "$15,000 to $24,999",
    4.0: "$25,000 to $34,999",
    5.0: "$35,000 to $49,999",
    6.0: "$50,000 to $74,999",
    7.0: "$75,000 to $99,999",
    8.0: "$100,000 to $149,999",
    9.0: "$150,000 to $199,999",
    10.0: "$200,000 or more",
}

GRUMMON_HISPANIC_LABELS = {0.0: "No", 1.0: "Yes"}

GRUMMON2023_PARTY_LABELS = {
    1.0: "Strong Democrat",
    2.0: "Moderate Democrat",
    3.0: "Lean Democrat",
    4.0: "Independent",
    5.0: "Lean Republican",
    6.0: "Moderate Republican",
    7.0: "Strong Republican",
    8.0: "Something else",
}

GRUMMON2023_RACE_COLS = [
    ("Race_1", "American Indian or Alaska Native"),
    ("Race_2", "Asian"),
    ("Race_3", "Black or African American"),
    ("Race_4", "Native Hawaiian or Other Pacific Islander"),
    ("Race_5", "White"),
    ("Race_6", "Another race"),
]

GRUMMON2025_RACE_COLS = [
    ("race_1", "American Indian or Alaska Native"),
    ("race_2", "Asian"),
    ("race_3", "Black or African American"),
    ("race_4", "Native Hawaiian or Other Pacific Islander"),
    ("race_5", "White"),
    ("race_6", "Another race"),
]


def load_raw_grummon2023_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Grummon et al. (2023) vending-machine RCT personas with verbatim survey labels."""
    clean_df, _, _ = g23.load_analytic_cohort()
    clean_df = clean_df.reset_index(drop=True)

    cols = {
        "CASEID": clean_df["CASEID"].astype(str),
        "country": "United States",
        "gender": _strict_map(clean_df["gender"], GRUMMON_GENDER_LABELS, "grummon2023.gender"),
        "age": clean_df["age"],
        "education": _strict_map(clean_df["educ"], GRUMMON_EDUC_LABELS, "grummon2023.educ"),
        "family_income": _strict_map(clean_df["income_10cat"], GRUMMON_INCOME_LABELS, "grummon2023.income_10cat"),
        "hispanic": _strict_map(clean_df["latino"], GRUMMON_HISPANIC_LABELS, "grummon2023.latino"),
        "race": _checkbox_race(clean_df, GRUMMON2023_RACE_COLS, "grummon2023"),
        "household_size": clean_df["hhsize_num"],
        "children_in_household": clean_df["children_num"],
        "political_party": _strict_map(clean_df["partyid"], GRUMMON2023_PARTY_LABELS, "grummon2023.partyid"),
        "nutrition_label_use": _strict_map(clean_df["NFPuse"], g23.NFPUSE_MAP, "grummon2023.NFPuse"),
    }
    if not shared_only:
        cols.update({
            "DIETQUAL": _strict_map(clean_df["dietqual"], g23.DIETQUAL_MAP, "grummon2023.dietqual"),
            "MEDHELP": _strict_map(clean_df["healthlit"], g23.MEDHELP_MAP, "grummon2023.healthlit"),
            "VENDING_BEV": _strict_map(clean_df["vendingfreq_bev"], g23.VENDING_FREQ_MAP, "grummon2023.vending_bev"),
            "VENDING_SNACK": _strict_map(
                clean_df["vendingfreq_snack"], g23.VENDING_FREQ_MAP, "grummon2023.vending_snack"
            ),
            "BMI": g23.derive_grummon2023_bmi(clean_df),
        })
    return pd.DataFrame(cols)


def load_raw_grummon2025_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Grummon et al. (2025) online grocery RCT personas with verbatim survey labels."""
    clean_df = g25.load_analytic_cohort().reset_index(drop=True)

    cols = {
        "CASEID": clean_df["CASEID"].astype(str),
        "country": "United States",
        "gender": _strict_map(clean_df["gender"], GRUMMON_GENDER_LABELS, "grummon2025.gender"),
        "age": clean_df["age"],
        "education": _strict_map(clean_df["educ"], GRUMMON_EDUC_LABELS, "grummon2025.educ"),
        "family_income": _strict_map(clean_df["income"], GRUMMON_INCOME_LABELS, "grummon2025.income"),
        "hispanic": _strict_map(clean_df["latino"], GRUMMON_HISPANIC_LABELS, "grummon2025.latino"),
        "race": _checkbox_race(clean_df, GRUMMON2025_RACE_COLS, "grummon2025"),
        "household_size": clean_df["hhsize_num"],
        "children_in_household": clean_df["children_num"],
        "nutrition_label_use": _strict_map(clean_df["nfpuse"], g25.NFPUSE_MAP, "grummon2025.nfpuse"),
    }
    if not shared_only:
        cols.update({
            "SNAP": _strict_map(clean_df["snap"], g25.SNAP_MAP, "grummon2025.snap"),
            "WIC": _strict_map(clean_df["wic"], g25.WIC_MAP, "grummon2025.wic"),
            "ONLINEAMT": _strict_map(clean_df["onlineamt"], g25.ONLINE_MAP, "grummon2025.onlineamt"),
            "HEALTHLIT": _strict_map(clean_df["scr_litall"], g25.HEALTHLIT_MAP, "grummon2025.healthlit"),
        })
    return pd.DataFrame(cols)


BERKE_STATE_LABELS = {i: name for i, name in enumerate(berke_s.STATE_NAMES, start=1)}
BERKE_GENDER_LABELS = {1: "Male", 2: "Female", 3: "Other", 4: "Prefer not to answer"}
BERKE_AGE_LABELS = {
    1: "18 - 24 years",
    2: "25 - 34 years",
    3: "35 - 44 years",
    4: "45 - 54 years",
    5: "55 - 64 years",
    6: "65 or older",
}
BERKE_INCOME_LABELS = {
    1: "Less than $25,000",
    2: "$25,000 to $49,999",
    3: "$50,000 to $74,999",
    4: "$75,000 to $99,999",
    5: "$100,000 to $149,999",
    6: "$150,000 to $199,999",
    7: "$200,000 or more",
}
BERKE_URBAN_LABELS = {1: "Urban", 2: "Suburban", 3: "Rural", 4: "I don't know"}
BERKE_RESTYPE_LABELS = {1: "Private home", 2: "Apartment", 3: "Other"}
BERKE_RACE_CODES = {
    1: "White",
    2: "Black or African American",
    3: "American Indian or Alaska Native",
    4: "Asian",
    5: "Native Hawaiian or Pacific Islander",
    6: "Hispanic or Latino",
    7: "Other",
}


def _decode_berke_race(series: pd.Series) -> pd.Series:
    out = []
    for val in series:
        if pd.isna(val):
            out.append(float("nan"))
            continue
        codes = [int(x) for x in str(val).split(",")]
        for c in codes:
            if c not in BERKE_RACE_CODES:
                raise ValueError(f"berke2023.Q_race: unexpected code {c!r} in {val!r}")
        out.append("; ".join(BERKE_RACE_CODES[c] for c in codes))
    return pd.Series(out, index=series.index, dtype=object)


def load_raw_berke2023_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw Berke et al. (2023) drone-delivery conjoint personas with verbatim codebook labels."""
    raw_df = berke_s.load_raw_df().reset_index(drop=True)
    return pd.DataFrame({
        "CASEID": "berke2023_" + raw_df["ResponseId"].astype(str),
        "country": "United States",
        "state": _strict_map(raw_df["Q_state"], BERKE_STATE_LABELS, "berke2023.Q_state"),
        "gender": _strict_map(raw_df["Q_gender"], BERKE_GENDER_LABELS, "berke2023.Q_gender"),
        "age": _strict_map(raw_df["Q_age"], BERKE_AGE_LABELS, "berke2023.Q_age"),
        "race": _decode_berke_race(raw_df["Q_race"]),
        "family_income": _strict_map(raw_df["Q_income"], BERKE_INCOME_LABELS, "berke2023.Q_income"),
        "urban_rural": _strict_map(raw_df["Q_urban_rural"], BERKE_URBAN_LABELS, "berke2023.Q_urban_rural"),
        "residence_type": _strict_map(raw_df["Q_residence_type"], BERKE_RESTYPE_LABELS, "berke2023.Q_residence_type"),
    })


KRINER_RACE_COLS = [
    ("native_american", "American Indian"),
    ("asian", "Asian"),
    ("black", "Black or African American"),
    ("hispanic", "Hispanic"),
    ("white", "White"),
    ("race_6", "Other"),
]


def load_raw_kriner2026_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw Goldfarb/Kriner et al. (2026) SAF conjoint personas from Stata `.dta`."""
    raw_df, _, value_labels = kriner_s.load_raw_df(data_dir / "Kriner2026" / "Goldfarb_etal_SAF_JCP_replication.dta")
    resp = kriner_s.build_persona_input_df(raw_df, value_labels).reset_index(drop=True)

    return pd.DataFrame({
        "CASEID": "kriner2026_" + resp["respondent"].astype(int).astype(str),
        "country": "United States",
        "state": _strict_map(resp["state"], value_labels["state"], "kriner2026.state"),
        "region": _strict_map(resp["vs_region"], value_labels["vs_region"], "kriner2026.vs_region"),
        "gender": _strict_map(resp["gender"], value_labels["gender"], "kriner2026.gender"),
        "age": resp["age_years"],  # exact numeric age in years (18-85)
        "education": _strict_map(resp["education"], value_labels["education"], "kriner2026.education"),
        "religion": _strict_map(resp["relig"], value_labels["relig"], "kriner2026.relig"),
        "political_ideology": _strict_map(resp["ideology"], value_labels["ideology"], "kriner2026.ideology"),
        "political_party": _strict_map(resp["party"], value_labels["party"], "kriner2026.party"),
        "income": _strict_map(resp["income"], value_labels["income"], "kriner2026.income"),
        "urban_rural": _strict_map(
            resp["urban_rural_best"], value_labels["urban_rural_best"], "kriner2026.urban_rural_best"
        ),
        "voted_last_election": _strict_map(
            resp["vs_vote_2024"], value_labels["vs_vote_2024"], "kriner2026.vs_vote_2024"
        ),
        "race": _checkbox_race(resp, KRINER_RACE_COLS, "kriner2026"),
    })


def load_raw_wu2025_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Wu et al. (2025) electricity-plan conjoint personas from Stata `.dta`."""
    resp = wu_s.extract_respondent_df(wu_s.load_raw_df()).reset_index(drop=True)

    def _clean_str(col: str) -> pd.Series:
        return resp[col].replace("", float("nan"))

    cols = {
        "CASEID": "wu2025_" + resp["CaseID"].astype(int).astype(str),
        "country": "United States",
        "state": _clean_str("state"),
        "gender": _clean_str("sex"),
        "age": _clean_str("age"),
        "race": _clean_str("race").map(lambda x: "; ".join(x.split(",")) if pd.notna(x) else x),
        "hispanic": _clean_str("ethnicity"),
        "education": _clean_str("education"),
        "family_income": _clean_str("income"),
        "political_party": _clean_str("partisan"),
        "political_ideology": _clean_str("ideology"),
        "marital_status": _clean_str("marriage"),
        "employment_status": _clean_str("employment"),
        "residence_type": _clean_str("residence"),
    }
    if not shared_only:
        cols.update({
            "POL_LABEL": _clean_str("ideology_original"),
            "OWNHOME": _clean_str("residenceownership"),
            "RENTHOME": _clean_str("residencerental"),
            "ORIENTATION": _clean_str("orientation"),
            "MILITARY": _clean_str("military"),
            "ELECHEAT": _clean_str("householdheat"),
        })
    return pd.DataFrame(cols)


def load_raw_kantorowicz2026_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw Kantorowicz-Reznichenko et al. (2026) conjoint personas from `.RDS` files."""
    comp_df, laundry_df = kant_s.load_raw_dfs()
    resp = kant_s.extract_respondent_df(comp_df, laundry_df).reset_index(drop=True)
    return pd.DataFrame({
        "CASEID": "kantorowicz2026_" + resp["Response.ID"].astype(str),
        "country": "United States",
        "gender": resp["gender_dummy"],
        "age": resp["age_dummy"],
        "education": resp["education_dummy"],
        "income": resp["income_dummy"],
    })


MAGISTRO_US_REGION_LABELS = {1.0: "Northeast", 2.0: "Midwest", 3.0: "South", 4.0: "West"}
MAGISTRO_CA_PROVINCE_LABELS = {
    1.0: "Alberta",
    2.0: "British Columbia",
    3.0: "Manitoba",
    4.0: "New Brunswick",
    5.0: "Newfoundland and Labrador",
    6.0: "Northwest Territories",
    7.0: "Nova Scotia",
    9.0: "Ontario",
    10.0: "Prince Edward Island",
    11.0: "Quebec",
    12.0: "Saskatchewan",
}


def load_raw_magistro2025_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw Magistro et al. (2025) US + Canada conjoint personas from `clean_AJPS.csv`."""
    resp = mag_s.load_raw_df(data_dir / "magistro2025" / "dataverse_files" / "clean_AJPS.csv").reset_index(drop=True)

    us_mask = resp["countr"] == "US"
    ca_mask = resp["countr"] == "Canada"
    if not (us_mask | ca_mask).all():
        raise ValueError(f"magistro2025: unexpected countr values {resp.loc[~(us_mask | ca_mask), 'countr'].unique()}")

    region_series = _strict_map(resp["region"].where(us_mask), MAGISTRO_US_REGION_LABELS, "magistro2025.region")
    state_series = _strict_map(
        resp["GeoPC_Region1"].where(ca_mask), MAGISTRO_CA_PROVINCE_LABELS, "magistro2025.GeoPC_Region1"
    )

    return pd.DataFrame({
        "CASEID": "magistro2025_" + resp["ID"].astype(int).astype(str),
        "country": resp["countr"],
        "region": region_series,
        "state": state_series,
        "gender": resp["gender"],
        "age": resp["age_group"],
        "education": resp["education"],
        "income": resp["income"],
        "political_party": resp["party"],
    })


# ---------------------------------------------------------------------------
# Combined raw + standardized entry points
# ---------------------------------------------------------------------------

RAW_LOADERS = {
    "issp": load_raw_issp_personas,
    "twin2k": load_raw_twin2k_personas,
    "wvs7": load_raw_wvs7_personas,
    "subpop": load_raw_subpop_personas,
    "covid": load_raw_covid_personas,
    "grummon2023": load_raw_grummon2023_personas,
    "grummon2025": load_raw_grummon2025_personas,
    "berke2023": load_raw_berke2023_personas,
    "kriner2026": load_raw_kriner2026_personas,
    "wu2025": load_raw_wu2025_personas,
    "kantorowicz2026": load_raw_kantorowicz2026_personas,
    "magistro2025": load_raw_magistro2025_personas,
}


def load_raw_all_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and stack all 12 datasets' raw (unstandardized) personas into one DataFrame."""
    frames = [loader(data_dir=data_dir).assign(dataset=name) for name, loader in RAW_LOADERS.items()]
    combined = pd.concat(frames, ignore_index=True)
    other_cols = [c for c in combined.columns if c not in {"dataset", "CASEID"}]
    return combined[["dataset", "CASEID", *other_cols]]


def load_all_personas(
    specs: dict[str, PersonaAttribute] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load all 12 datasets from raw files, standardize active attributes in `specs`, and stack."""
    active_specs = specs if specs is not None else load_persona_spec()
    frames = []

    for name, loader in RAW_LOADERS.items():
        raw_df = loader(data_dir=data_dir)
        keep_cols = ["CASEID", *[a for a in active_specs if a in raw_df.columns]]
        part = raw_df[keep_cols].copy()
        std = standardize_dataframe(part, dataset=name, specs=active_specs)
        frames.append(std.assign(dataset=name))

    combined = pd.concat(frames, ignore_index=True)
    ordered_cols = ["dataset", "CASEID", *[a for a in active_specs if a in combined.columns]]
    return combined[ordered_cols]

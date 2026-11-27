"""Clean per-dataset raw persona loaders using `persona_spec.yaml`."""

import io
import json
from pathlib import Path
import zipfile

import pandas as pd
from pandas.io.stata import StataReader

from data.persona_spec import PersonaAttribute, load_persona_spec, standardize_dataframe

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

# Only ISSP 2023 (v46) and 2024 (v59) ask respondent Left-Right self-placement (0-10).
# GESIS PARTY_LR is excluded because in the US it is a deterministic recode of US_PRTY.
ISSP_IDEOLOGY_VAR = {
    2023: "v46",
    2024: "v59",
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


def load_raw_issp_personas(
    data_dir: Path = DATA_DIR,
    shared_only: bool = False,
    years: tuple[int, ...] | list[int] | None = None,
) -> pd.DataFrame:
    """Load raw ISSP (2015-2024 US) persona columns from `.dta` files with Stata value labels."""
    candidates = [*ISSP_SHARED_RENAME.keys(), "DEGREE"] if shared_only else ISSP_RAW_CANDIDATES
    target_years = tuple(range(2015, 2025)) if years is None else tuple(int(y) for y in years)
    frames = []

    for year in target_years:
        dta_files = list((data_dir / f"issp{year}" / "raw").glob("*.dta"))
        if len(dta_files) != 1:
            raise FileNotFoundError(f"Expected 1 .dta file for ISSP {year}, found {dta_files}")
        dta_path = dta_files[0]

        value_labels, col_to_lblname = _read_stata_value_labels(dta_path)
        present = [c for c in candidates if c in col_to_lblname]
        ideology_var = ISSP_IDEOLOGY_VAR.get(year)
        if ideology_var is not None:
            if ideology_var not in col_to_lblname:
                raise KeyError(f"ISSP {year}: expected Left-Right ideology variable {ideology_var!r} in {dta_path}")
            present.append(ideology_var)
        raw_cols = sorted({"CASEID", "c_alphan", *present})

        df = pd.read_stata(dta_path, columns=raw_cols, convert_categoricals=False)
        us = df[df["c_alphan"] == "US"].reset_index(drop=True)

        wave = pd.DataFrame({"CASEID": [f"{year}:{int(cid)}" for cid in us["CASEID"]]})
        for col in present:
            wave[col] = _decode_stata_col(us[col], col, value_labels, col_to_lblname)

        if ideology_var is not None:
            wave = wave.rename(columns={ideology_var: "political_ideology"})

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


WVS_PERSONA_RAW_COLS = [
    "B_COUNTRY",
    "G_TOWNSIZE",
    "H_SETTLEMENT",
    "H_URBRURAL",
    "S_INTLANGUAGE",
    "Q260",
    "Q261",
    "Q262",
    "X003R",
    "Q263",
    "Q264",
    "Q265",
    "Q266",
    "Q267",
    "Q268",
    "Q269",
    "Q270",
    "Q271",
    "Q272",
    "Q273",
    "Q274",
    "Q275",
    "Q276",
    "Q277",
    "Q278",
    "Q279",
    "Q280",
    "Q281",
    "Q282",
    "Q283",
    "Q284",
    "Q285",
    "Q286",
    "Q287",
    "Q288",
    "Q289",
    "Q289CS9",
    "Q290",
    "N_REGION_ISO",
    "Q223",
    "Q171",
    "Q97",
    "Q240",
    "Q222",
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
}

WVS_SINGLE_RENAME = {
    "G_TOWNSIZE": "town_size",
    "Q263": "immigrant_status",
    "Q271": "lives_with_parents",
    "Q272": "home_language",
    "Q285": "chief_wage_earner",
    "Q286": "family_savings",
    "Q287": "social_class",
    "Q222": "voting_frequency",
}


def load_raw_wvs7_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw WVS Wave 7 US persona columns directly from the official WVSA Stata file."""
    dta_path = (
        data_dir
        / "WVS-Wave7"
        / "WVS_Wave_7_United_States_Stata_v5.1.dta"
    )
    if not dta_path.exists():
        raise FileNotFoundError(f"Official WVS Wave 7 US Stata file not found: {dta_path}")

    val_labels, col_to_lbl = _read_stata_value_labels(dta_path)
    cols = list(WVS_SHARED_RENAME.keys()) if shared_only else list(WVS_PERSONA_RAW_COLS)
    rename_map = WVS_SHARED_RENAME if shared_only else {**WVS_SHARED_RENAME, **WVS_SINGLE_RENAME}

    raw_df = pd.read_stata(dta_path, columns=["D_INTERVIEW", *cols], convert_categoricals=False)
    if raw_df["D_INTERVIEW"].isna().any():
        raise ValueError(f"Missing D_INTERVIEW values in {dta_path}")

    out: dict[str, pd.Series] = {"CASEID": raw_df["D_INTERVIEW"].astype("int64").astype(str)}
    for col in cols:
        out[rename_map.get(col, col)] = _decode_stata_col(raw_df[col], col, val_labels, col_to_lbl)
    return pd.DataFrame(out)


SUBPOP_ATTR_TO_COL = {
    "CREGION": "region",
    "EDUCATION": "education",
    "INCOME": "income",
    "POLIDEOLOGY": "political_ideology",
    "POLPARTY": "political_party",
    "RACE": "race",
    "RELIG": "religion",
    "SEX": "gender",
}


def load_raw_subpop_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw SubPOP target subgroup profiles from the two official upstream splits."""
    rows_by_group: dict[str, dict[str, object]] = {}
    for subdir, filename in [
        ("subpop_eval", "raw_subpop_eval.jsonl"),
        ("subpop_train", "raw_subpop_train.jsonl"),
    ]:
        path = data_dir / subdir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing official SubPOP raw file: {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                attr = record["attribute"]
                grp = record["group"]
                col = SUBPOP_ATTR_TO_COL[attr]
                case_id = f"SUBPOP/{attr}/{grp}"
                if case_id not in rows_by_group:
                    rows_by_group[case_id] = {
                        "CASEID": case_id,
                        "country": "United States",
                        col: grp,
                    }
    return pd.DataFrame(rows_by_group.values())


HINTS7_ZIP_REL = Path("HINTS") / "HINTS7_STATA_20250731.zip"
HINTS7_DTA_MEMBER = "HINTS7_STATA_20250731/hints7_public.dta"
HINTS7_MISSING_CODES = frozenset({-9, -7, -6, -5, -4, -2, -1})

HINTS7_NUMERIC_RENAME = {
    "Age": "age",
    "TotalHousehold": "household_size",
    "ChildrenInHH": "children_in_household",
}

HINTS7_CATEGORICAL_RENAME = {
    "CENSDIV": "region",
    "NCHSURCODE2013": "urban_rural",
    "BirthSex": "gender",
    "Education": "education",
    "IncomeRanges": "family_income",
    # Kept over FullTimeOcc2_Cat (+434 valid rows; stays purely R5-derived);
    # downside: collapses full/part-time into 'In paid work' and puts 687 rows in 'Multiple employment statuses'.
    "Occupation2_Cat": "employment_status",
    "Hisp_Cat": "hispanic",
    "PoliticalViewpoint": "political_ideology",
    "MaritalStatus": "marital_status",
    "SexualOrientation_Recode": "sexual_orientation",
}

HINTS7_R10_RACE_GROUPS: list[tuple[tuple[str, ...], str]] = [
    (("White",), "White"),
    (("Black",), "Black or African American"),
    (("AmerInd",), "American Indian or Alaska Native"),
    (("AsInd", "Chinese", "Filipino", "Japanese", "Korean", "Vietnamese", "OthAsian"), "Asian"),
    (("OthPacIsl",), "Native Hawaiian or Other Pacific Islander"),
]

# Complete set of HINTS 7 Section R, geographic, and derived demographic columns so
# `hints7_export.py` refuses to export any persona variable as a survey question.
HINTS7_PERSONA_RAW_COLS: frozenset[str] = frozenset({
    *HINTS7_NUMERIC_RENAME.keys(),
    *HINTS7_CATEGORICAL_RENAME.keys(),
    *(c for cols, _ in HINTS7_R10_RACE_GROUPS for c in cols),
    # Excluded / alternative Section R, geographic, and derived demographic variables
    "APP_REGION",
    "DRA",
    "RUC2003",
    "RUC2013",
    "RUC2023",
    "PR_RUCA_2010",
    "CENSREG",
    "WorkHrsPerWeek",
    "Occupation2_Employed",
    "Occupation2_OOW1Y",
    "Occupation2_OOWLess1Y",
    "Occupation2_Homemaker",
    "Occupation2_Student",
    "Occupation2_Retired",
    "Occupation2_Disabled",
    "Occupation2_Other",
    "Occupation2_Other_OS",
    "FullTimeOcc2_Cat",
    "WorkHrs_DidNotWork",
    "WorkHrs_EarlyAM",
    "WorkHrs_AM",
    "WorkHrs_Afternoon",
    "WorkHrs_Evening",
    "WorkHrs_Night",
    "WorkHrs_Cat",
    "NotHisp",
    "Mexican",
    "PuertoRican",
    "Cuban",
    "OthHisp",
    "Race_Cat2",
    "EthnicGroupBelonging",
    "SexualOrientation2",
    "SexualOrientation2_OS",
    "R_HHADULTS",
    "HHADULTS_NUM",
    "IncomeFeelings",
    "AgeGrpA",
    "AgeGrpB",
    "EducA",
    "EducB",
    "RaceEthn",
    "RaceEthn5",
    "HHInc",
    "IncomeRanges_IMP",
})


def _mask_hints7_negatives(series: pd.Series, col: str) -> pd.Series:
    """Validate HINTS 7 negative missing codes and replace them with NaN."""
    neg = series.dropna()
    neg = neg[neg < 0]
    unexpected = set(neg.unique()) - HINTS7_MISSING_CODES
    if unexpected:
        raise ValueError(f"hints7.{col}: unexpected negative code(s) {sorted(unexpected)}")
    return series.where(series.isna() | (series >= 0), float("nan"))


def _hints7_multiselect(
    df: pd.DataFrame,
    groups: list[tuple[tuple[str, ...], str]],
    name: str,
) -> pd.Series:
    """Combine HINTS 7 1=Selected / 2=Not selected checkbox columns into a '; '-joined string."""
    all_cols = [col for cols, _ in groups for col in cols]
    masked = pd.DataFrame({col: _mask_hints7_negatives(df[col], col) for col in all_cols}, index=df.index)
    for col in all_cols:
        non_null = set(masked[col].dropna().unique())
        bad = non_null - {1, 2}
        if bad:
            raise ValueError(f"{name}.{col}: expected 1/2/NaN, got {sorted(bad)}")

    out: list[object] = []
    for _, row in masked.iterrows():
        is_na = [pd.isna(row[c]) for c in all_cols]
        if all(is_na):
            out.append(float("nan"))
            continue
        if any(is_na):
            raise ValueError(f"{name}: mixed missing and non-missing checkbox values in row")
        chosen = [label for cols, label in groups if any(row[c] == 1 for c in cols)]
        if not chosen:
            raise ValueError(f"{name}: non-missing checkbox row has zero selected options")
        out.append("; ".join(chosen))
    return pd.Series(out, index=df.index, dtype=object)


def _strict_decode_stata_col(
    series: pd.Series,
    col: str,
    value_labels: dict[str, dict],
    col_to_lblname: dict[str, str],
    dataset: str,
) -> pd.Series:
    """Decode non-null Stata codes strictly through embedded value labels."""
    lbl_name = col_to_lblname[col]
    if not lbl_name or lbl_name not in value_labels:
        raise KeyError(f"{dataset}.{col}: missing embedded Stata value label set ({lbl_name!r})")
    mapping = value_labels[lbl_name]
    return _strict_map(series, mapping, f"{dataset}.{col}")


def load_raw_hints7_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw NCI HINTS 7 (2024) persona columns directly from the official Stata zip."""
    zip_path = data_dir / HINTS7_ZIP_REL
    if not zip_path.is_file():
        raise FileNotFoundError(f"Official HINTS 7 Stata archive not found: {zip_path}")

    r10_cols = [c for cols, _ in HINTS7_R10_RACE_GROUPS for c in cols]
    raw_cols = list(
        dict.fromkeys([
            "HHID",
            *HINTS7_NUMERIC_RENAME.keys(),
            *HINTS7_CATEGORICAL_RENAME.keys(),
            *r10_cols,
        ])
    )

    with zipfile.ZipFile(zip_path) as zf:
        dta_bytes = zf.read(HINTS7_DTA_MEMBER)

    with StataReader(io.BytesIO(dta_bytes)) as reader:
        value_labels = reader.value_labels()
        col_to_lblname = dict(zip(reader._varlist, reader._lbllist, strict=True))

    missing_cols = [c for c in raw_cols if c not in col_to_lblname]
    if missing_cols:
        raise KeyError(f"hints7: expected columns missing from {HINTS7_DTA_MEMBER}: {missing_cols}")

    raw_df = pd.read_stata(io.BytesIO(dta_bytes), columns=raw_cols, convert_categoricals=False)
    if raw_df["HHID"].isna().any() or not raw_df["HHID"].is_unique:
        raise ValueError(f"hints7: HHID must be non-null and unique in {HINTS7_DTA_MEMBER}")

    out: dict[str, object] = {
        "CASEID": raw_df["HHID"].astype(str),
        "country": "United States",
    }
    for src_col, dst_col in HINTS7_NUMERIC_RENAME.items():
        out[dst_col] = _mask_hints7_negatives(raw_df[src_col], src_col)

    for src_col, dst_col in HINTS7_CATEGORICAL_RENAME.items():
        masked = _mask_hints7_negatives(raw_df[src_col], src_col)
        out[dst_col] = _strict_decode_stata_col(masked, src_col, value_labels, col_to_lblname, "hints7")

    out["race"] = _hints7_multiselect(raw_df, HINTS7_R10_RACE_GROUPS, "hints7.race")

    return pd.DataFrame(out)


SHED2025_ZIP_REL = Path("SHED") / "SHED_public_use_data_2025_(STATA).zip"
SHED2025_DTA_MEMBER = "public2025.dta"
SHED2025_MISSING_CODES = frozenset({-9, -2, -1})

SHED2025_NUMERIC_RENAME = {
    "ppage": "age",
    "pphhsize": "household_size",
    "ppkid017": "children_in_household",
}

SHED2025_CATEGORICAL_RENAME = {
    "ppgender": "gender",
    "ppreg9": "region",
    "ppstaten": "state",
    "ppmsacat": "urban_rural",
    "ED0": "education",
    "I40": "family_income",
    "ppracem": "race",
    "pphispan": "hispanic",
    "ppmarit5": "marital_status",
    "ppemploy": "employment_status",
    "pphouse4": "residence_type",
    "ppp20197": "citizen",
    "pppa_lgb": "sexual_orientation",
    "UNION100": "union_member",
    "ppcm1301": "employer_sector",
}

# Survey-asked persona columns that carry hot-deck imputation flags (`*_iflag == 1`).
# Imputed values are masked to NaN so personas contain only respondent-reported demographics.
SHED2025_IMPUTED_PERSONA_COLS = frozenset({"ED0", "I40"})

# Complete set of SHED 2025 demographic, derived-demographic, and alternative profile columns
# so `shed2025_export.py` strictly refuses to export any persona variable as a survey question.
SHED2025_PERSONA_RAW_COLS: frozenset[str] = frozenset({
    *SHED2025_NUMERIC_RENAME.keys(),
    *SHED2025_CATEGORICAL_RENAME.keys(),
    "ppagecat",
    "ppagect4",
    "ppeduc5",
    "ppeducat",
    "educ_4cat",
    "ppethm",
    "race_5cat",
    "ppinc7",
    "inc_4cat_50k",
    "ppreg4",
    "pphhsize5",
    "ppt18ov",
    "pprent",
    "pppa1634",
    "malefemale",
    "p_gender",
    "ppcm0160",
    "ind1",
    "pphi0001",
    "CH2A",
    "ED0B",
    "ED0_iflag",
    "I40_iflag",
})


def _mask_shed2025_negatives(series: pd.Series, col: str) -> pd.Series:
    """Validate SHED 2025 negative missing codes strictly and replace them with NaN."""
    neg = series.dropna()
    neg = neg[neg < 0]
    unexpected = set(int(x) for x in neg.unique()) - SHED2025_MISSING_CODES
    if unexpected:
        raise ValueError(f"shed2025.{col}: unexpected negative code(s) {sorted(unexpected)}")
    return series.where(series.isna() | (series >= 0), float("nan"))


def load_raw_shed2025_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load raw Federal Reserve SHED 2025 persona columns directly from the official Stata zip."""
    zip_path = data_dir / SHED2025_ZIP_REL
    if not zip_path.is_file():
        raise FileNotFoundError(f"Official SHED 2025 Stata archive not found: {zip_path}")

    iflag_cols = [f"{c}_iflag" for c in sorted(SHED2025_IMPUTED_PERSONA_COLS)]
    raw_cols = list(
        dict.fromkeys([
            "shedid",
            *SHED2025_NUMERIC_RENAME.keys(),
            *SHED2025_CATEGORICAL_RENAME.keys(),
            *iflag_cols,
        ])
    )

    with zipfile.ZipFile(zip_path) as zf:
        dta_bytes = zf.read(SHED2025_DTA_MEMBER)

    with StataReader(io.BytesIO(dta_bytes)) as reader:
        value_labels = reader.value_labels()
        col_to_lblname = dict(zip(reader._varlist, reader._lbllist, strict=True))

    missing_cols = [c for c in raw_cols if c not in col_to_lblname]
    if missing_cols:
        raise KeyError(f"shed2025: expected columns missing from {SHED2025_DTA_MEMBER}: {missing_cols}")

    raw_df = pd.read_stata(io.BytesIO(dta_bytes), columns=raw_cols, convert_categoricals=False)
    if raw_df["shedid"].isna().any() or not raw_df["shedid"].is_unique:
        raise ValueError(f"shed2025: shedid must be non-null and unique in {SHED2025_DTA_MEMBER}")

    out: dict[str, object] = {
        "CASEID": raw_df["shedid"].astype(str),
        "country": "United States",
    }
    for src_col, dst_col in SHED2025_NUMERIC_RENAME.items():
        out[dst_col] = _mask_shed2025_negatives(raw_df[src_col], src_col)

    for src_col, dst_col in SHED2025_CATEGORICAL_RENAME.items():
        series = raw_df[src_col]
        if src_col in SHED2025_IMPUTED_PERSONA_COLS:
            flag = raw_df[f"{src_col}_iflag"]
            bad_flags = set(flag.dropna().unique()) - {0, 1}
            if bad_flags:
                raise ValueError(f"shed2025.{src_col}_iflag: unexpected values {sorted(bad_flags)}")
            series = series.where(flag != 1, float("nan"))
        masked = _mask_shed2025_negatives(series, src_col)
        out[dst_col] = _strict_decode_stata_col(masked, src_col, value_labels, col_to_lblname, "shed2025")

    return pd.DataFrame(out)


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


def load_raw_covid_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Kreps et al. (2020) COVID vaccine conjoint personas from Stata `.dta`."""
    from data.covid import stimulus as covid_s

    dta_path = data_dir / "covid" / "dataverse_files" / "Kreps_etal_vax_replication_data.dta"
    value_labels, col_to_lblname = _read_stata_value_labels(dta_path)
    resp = covid_s.extract_respondent_df(covid_s.load_raw_conjoint_df(dta_path)).reset_index(drop=True)

    cols = {
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
    }
    if not shared_only:
        cols["health_insurance"] = _decode_stata_col(resp["insurance"], "insurance", value_labels, col_to_lblname)
        cols["evangelical_christian"] = _decode_stata_col(
            resp["evangelical"], "evangelical", value_labels, col_to_lblname
        )
        cols["recent_employment_change"] = _decode_stata_col(
            resp["work_status"], "work_status", value_labels, col_to_lblname
        )
        cols["able_to_work_from_home"] = _decode_stata_col(resp["work_home"], "work_home", value_labels, col_to_lblname)
        cols["flu_vaccination_frequency"] = _decode_stata_col(
            resp["flu_vaccine"], "flu_vaccine", value_labels, col_to_lblname
        )
        cols["knows_someone_tested_positive_covid"] = _decode_stata_col(
            resp["personal_contact1"], "personal_contact1", value_labels, col_to_lblname
        )
        cols["knows_someone_hospitalized_or_died_covid"] = _decode_stata_col(
            resp["personal_contact2"], "personal_contact2", value_labels, col_to_lblname
        )
    return pd.DataFrame(cols)


def load_raw_kreps2021_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Kreps & Kriner (2021) COVID vaccine misinformation/conjoint personas from Stata `.dta`."""
    zip_path = data_dir / "kreps2021" / "dataverse_files.zip"
    with zipfile.ZipFile(zip_path) as zf:
        raw_bytes = zf.read("vax_misinfo_replication_data_raw.dta")

    with StataReader(io.BytesIO(raw_bytes)) as reader:
        value_labels = reader.value_labels()
        col_to_lblname = dict(zip(reader._varlist, reader._lbllist, strict=True))

    resp = pd.read_stata(io.BytesIO(raw_bytes), convert_categoricals=False).reset_index(drop=True)
    if resp["ResponseId"].isna().any() or resp["ResponseId"].nunique() != len(resp):
        raise ValueError("kreps2021: ResponseId must be non-null and unique per respondent")

    age_decoded = _decode_stata_col(resp["age"], "age", value_labels, col_to_lblname)
    age_numeric = age_decoded.astype(int)

    cols = {
        "CASEID": resp["ResponseId"].astype(str),
        "country": "United States",
        "state": _decode_stata_col(resp["state"], "state", value_labels, col_to_lblname),
        "gender": _decode_stata_col(resp["gender"], "gender", value_labels, col_to_lblname),
        "age": age_numeric,
        "education": _decode_stata_col(resp["education"], "education", value_labels, col_to_lblname),
        "religion": _decode_stata_col(resp["relig"], "relig", value_labels, col_to_lblname),
        "political_ideology": _decode_stata_col(resp["ideology"], "ideology", value_labels, col_to_lblname),
        "political_party": _decode_stata_col(resp["party3"], "party3", value_labels, col_to_lblname),
        "income": _decode_stata_col(resp["income"], "income", value_labels, col_to_lblname),
        "race": _checkbox_race(resp, COVID_RACE_COLS, "kreps2021"),
    }
    if not shared_only:
        cols["health_insurance"] = _decode_stata_col(resp["insurance"], "insurance", value_labels, col_to_lblname)
        cols["evangelical_christian"] = _decode_stata_col(
            resp["evangelical"], "evangelical", value_labels, col_to_lblname
        )
        cols["flu_vaccination_frequency"] = _decode_stata_col(
            resp["flu_vaccine"], "flu_vaccine", value_labels, col_to_lblname
        )
        cols["trump_approval"] = _decode_stata_col(
            resp["pres_approval"], "pres_approval", value_labels, col_to_lblname
        )
        cols["tv_news_consumption"] = _decode_stata_col(resp["tv_news"], "tv_news", value_labels, col_to_lblname)
        cols["newspaper_consumption"] = _decode_stata_col(
            resp["paper_news"], "paper_news", value_labels, col_to_lblname
        )
        cols["social_media_news_consumption"] = _decode_stata_col(
            resp["social_news"], "social_news", value_labels, col_to_lblname
        )
        cols["social_media_daily_use"] = _decode_stata_col(resp["sns1"], "sns1", value_labels, col_to_lblname)
        cols["social_media_shares_media"] = _decode_stata_col(resp["sns2"], "sns2", value_labels, col_to_lblname)
        cols["social_media_follows_accounts"] = _decode_stata_col(resp["sns3"], "sns3", value_labels, col_to_lblname)
        cols["social_media_multiple_accounts"] = _decode_stata_col(resp["sns4"], "sns4", value_labels, col_to_lblname)
        cols["social_media_perceived_usefulness"] = _decode_stata_col(resp["sns5"], "sns5", value_labels, col_to_lblname)
        cols["social_media_follows_friends_shares"] = _decode_stata_col(
            resp["sns6"], "sns6", value_labels, col_to_lblname
        )
        cols["social_media_checks_friends_activity"] = _decode_stata_col(
            resp["sns7"], "sns7", value_labels, col_to_lblname
        )
    return pd.DataFrame(cols)


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


GRUMMON_NFPUSE_LABELS = {
    1.0: "Never",
    2.0: "Rarely",
    3.0: "Sometimes",
    4.0: "Most of the time",
    5.0: "Always",
}

GRUMMON2023_DIETQUAL_LABELS = {
    1.0: "Poor",
    2.0: "Fair",
    3.0: "Good",
    4.0: "Very good",
    5.0: "Excellent",
}

GRUMMON2023_MEDHELP_LABELS = {
    1.0: "Never",
    2.0: "Rarely",
    3.0: "Sometimes",
    4.0: "Most of the time",
    5.0: "Always",
}

GRUMMON2023_BMI_LEVELS = (
    "Underweight (<18.5)",
    "Healthy weight (18.5 to <25.0)",
    "Overweight (25.0 to <30.0)",
    "Obese (>=30.0)",
)

GRUMMON2025_SNAP_LABELS = {0: "No", 1: "Yes", 2: "Don't know"}
GRUMMON2025_WIC_LABELS = {0: "No", 1: "Yes", 2: "Don't know"}
GRUMMON2025_ONLINE_LABELS = {1: "Less than 25%", 2: "25-49%", 3: "50-74%", 4: "75% or more"}
GRUMMON2025_HEALTHLIT_LABELS = {i: f"{i} out of 4" for i in range(5)}


def _derive_grummon2023_bmi(clean_df: pd.DataFrame) -> pd.Series:
    """Derive 4-level CDC BMI categories matching Grummon 2023 `01_DataPrep.do` and published Table 1."""
    ft = pd.to_numeric(clean_df["height_feet"], errors="raise")
    inch = pd.to_numeric(clean_df["height_inches"], errors="raise")
    wt = pd.to_numeric(clean_df["weight_lbs"], errors="raise")
    if (ft.dropna() <= 0).any() or (inch.dropna() < 0).any() or (inch.dropna() >= 12).any() or (wt.dropna() <= 0).any():
        raise ValueError("grummon2023.BMI: invalid negative/out-of-range height or weight values")
    height_m = ((ft * 12.0) + inch) * 0.0254
    bmi = (wt * 0.453592) / (height_m * height_m)
    out: list[object] = []
    for val in bmi:
        if pd.isna(val):
            out.append(float("nan"))
        elif val < 18.5:
            out.append(GRUMMON2023_BMI_LEVELS[0])
        elif val < 25.0:
            out.append(GRUMMON2023_BMI_LEVELS[1])
        elif val < 30.0:
            out.append(GRUMMON2023_BMI_LEVELS[2])
        else:
            out.append(GRUMMON2023_BMI_LEVELS[3])
    return pd.Series(out, index=clean_df.index, name="BMI", dtype=object)


def load_raw_grummon2023_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Grummon et al. (2023) vending-machine RCT personas with verbatim survey labels."""
    from data.nutrition.grummon2023 import stimulus as g23_s

    clean_df, _, _ = g23_s.load_analytic_cohort()
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
        "nutrition_label_use": _strict_map(clean_df["NFPuse"], GRUMMON_NFPUSE_LABELS, "grummon2023.NFPuse"),
    }
    if not shared_only:
        cols.update({
            "diet_quality": _strict_map(clean_df["dietqual"], GRUMMON2023_DIETQUAL_LABELS, "grummon2023.dietqual"),
            "medical_reading_help": _strict_map(clean_df["healthlit"], GRUMMON2023_MEDHELP_LABELS, "grummon2023.healthlit"),
            "bmi": _derive_grummon2023_bmi(clean_df),
        })
    return pd.DataFrame(cols)


def load_raw_grummon2025_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Grummon et al. (2025) online grocery RCT personas with verbatim survey labels."""
    from data.nutrition.grummon2025 import stimulus as g25_s

    clean_df = g25_s.load_analytic_cohort().reset_index(drop=True)

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
        "nutrition_label_use": _strict_map(clean_df["nfpuse"], GRUMMON_NFPUSE_LABELS, "grummon2025.nfpuse"),
    }
    if not shared_only:
        cols.update({
            "snap_benefits": _strict_map(clean_df["snap"], GRUMMON2025_SNAP_LABELS, "grummon2025.snap"),
            "wic_benefits": _strict_map(clean_df["wic"], GRUMMON2025_WIC_LABELS, "grummon2025.wic"),
            "online_grocery_share": _strict_map(clean_df["onlineamt"], GRUMMON2025_ONLINE_LABELS, "grummon2025.onlineamt"),
            "nutrition_literacy_score": _strict_map(clean_df["scr_litall"], GRUMMON2025_HEALTHLIT_LABELS, "grummon2025.healthlit"),
        })
    return pd.DataFrame(cols)


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


def load_raw_berke2023_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Berke et al. (2023) drone-delivery conjoint personas with verbatim codebook labels."""
    from data.berke2023 import stimulus as berke_s

    state_labels = {i: name for i, name in enumerate(berke_s.STATE_NAMES, start=1)}
    raw_df = berke_s.load_raw_df().reset_index(drop=True)
    cols = {
        "CASEID": "berke2023_" + raw_df["ResponseId"].astype(str),
        "country": "United States",
        "state": _strict_map(raw_df["Q_state"], state_labels, "berke2023.Q_state"),
        "gender": _strict_map(raw_df["Q_gender"], BERKE_GENDER_LABELS, "berke2023.Q_gender"),
        "age": _strict_map(raw_df["Q_age"], BERKE_AGE_LABELS, "berke2023.Q_age"),
        "race": _decode_berke_race(raw_df["Q_race"]),
        "family_income": _strict_map(raw_df["Q_income"], BERKE_INCOME_LABELS, "berke2023.Q_income"),
        "urban_rural": _strict_map(raw_df["Q_urban_rural"], BERKE_URBAN_LABELS, "berke2023.Q_urban_rural"),
        "residence_type": _strict_map(raw_df["Q_residence_type"], BERKE_RESTYPE_LABELS, "berke2023.Q_residence_type"),
    }
    if not shared_only:
        freq_map = dict(berke_s.FREQ_OPTIONS)
        cols.update({
            "online_shopping_frequency": _strict_map(
                raw_df["Q_online_shopping"], freq_map, "berke2023.Q_online_shopping"
            ),
            "takeout_delivery_frequency": _strict_map(
                raw_df["Q_takeout_food"], freq_map, "berke2023.Q_takeout_food"
            ),
            "online_grocery_frequency": _strict_map(
                raw_df["Q_online_groceries"], freq_map, "berke2023.Q_online_groceries"
            ),
        })
    return pd.DataFrame(cols)


KRINER_RACE_COLS = [
    ("native_american", "American Indian"),
    ("asian", "Asian"),
    ("black", "Black or African American"),
    ("hispanic", "Hispanic"),
    ("white", "White"),
    ("race_6", "Other"),
]


def _build_kriner2026_political_party(resp: pd.DataFrame, value_labels: dict) -> pd.Series:
    """Combine kriner2026 party with Independents-only party_lean follow-up (fail-fast on unexpected combos)."""
    party = _strict_map(resp["party"], value_labels["party"], "kriner2026.party")
    lean = _strict_map(resp["party_lean"], value_labels["party_lean"], "kriner2026.party_lean")

    out = []
    for p, l in zip(party, lean):
        if pd.isna(p):
            if pd.notna(l):
                raise ValueError(f"kriner2026.political_party: missing party with party_lean={l!r}")
            out.append(float("nan"))
        elif p in ("Democrat", "Republican", "Other/don't know"):
            if pd.notna(l):
                raise ValueError(f"kriner2026.political_party: {p!r} with unexpected party_lean={l!r}")
            out.append(p)
        elif p == "Independent":
            if l == "Democratic Party":
                out.append("Lean Democrat")
            elif l == "Republican Party":
                out.append("Lean Republican")
            elif pd.isna(l) or l == "Neither/don't know":
                out.append("Independent")
            else:
                raise ValueError(f"kriner2026.political_party: unexpected Independent party_lean={l!r}")
        else:
            raise ValueError(f"kriner2026.political_party: unexpected party={p!r}")
    return pd.Series(out, index=resp.index, dtype="object")


def load_raw_kriner2026_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Goldfarb/Kriner et al. (2026) SAF conjoint personas from Stata `.dta`."""
    from data.Kriner2026 import stimulus as kriner_s

    raw_df, _, value_labels = kriner_s.load_raw_df(data_dir / "Kriner2026" / "Goldfarb_etal_SAF_JCP_replication.dta")
    resp = kriner_s.build_persona_input_df(raw_df, value_labels).reset_index(drop=True)

    cols = {
        "CASEID": "kriner2026_" + resp["respondent"].astype(int).astype(str),
        "country": "United States",
        "state": _strict_map(resp["state"], value_labels["state"], "kriner2026.state"),
        "region": _strict_map(resp["vs_region"], value_labels["vs_region"], "kriner2026.vs_region"),
        "gender": _strict_map(resp["gender"], value_labels["gender"], "kriner2026.gender"),
        "age": resp["age_years"],  # exact numeric age in years (18-85)
        "education": _strict_map(resp["education"], value_labels["education"], "kriner2026.education"),
        "religion": _strict_map(resp["relig"], value_labels["relig"], "kriner2026.relig"),
        "political_ideology": _strict_map(resp["ideology"], value_labels["ideology"], "kriner2026.ideology"),
        "political_party": _build_kriner2026_political_party(resp, value_labels),
        "income": _strict_map(resp["income"], value_labels["income"], "kriner2026.income"),
        "urban_rural": _strict_map(
            resp["urban_rural_best"], value_labels["urban_rural_best"], "kriner2026.urban_rural_best"
        ),
        "voted_last_election": _strict_map(
            resp["vs_vote_2024"], value_labels["vs_vote_2024"], "kriner2026.vs_vote_2024"
        ),
        "race": _checkbox_race(resp, KRINER_RACE_COLS, "kriner2026"),
    }
    if not shared_only:
        cols["vote_choice_2024"] = _strict_map(
            resp["vs_votechoice_2024"], value_labels["vs_votechoice_2024"], "kriner2026.vs_votechoice_2024"
        )
        cols["trump_approval"] = _strict_map(
            resp["approve"], value_labels["approve"], "kriner2026.approve"
        )
    return pd.DataFrame(cols)


def _build_wu2025_political_party(resp: pd.DataFrame) -> pd.Series:
    """Combine wu2025 partisan with ANES 7-point follow-up branches (fail-fast on unexpected combos)."""
    partisan = resp["partisan"].replace("", float("nan"))
    rep_str = resp["republicanstrength"].replace("", float("nan"))
    dem_str = resp["democratstrength"].replace("", float("nan"))
    leaner = resp["leaner"].replace("", float("nan"))

    out = []
    for p, r, d, l in zip(partisan, rep_str, dem_str, leaner):
        if pd.isna(p):
            if pd.notna(r) or pd.notna(d) or pd.notna(l):
                raise ValueError(f"wu2025.political_party: missing partisan with branch ({r!r}, {d!r}, {l!r})")
            out.append(float("nan"))
        elif p == "Democrat":
            if pd.notna(r) or pd.notna(l):
                raise ValueError(f"wu2025.political_party: Democrat with unexpected branch ({r!r}, {l!r})")
            if d == "Strong":
                out.append("Strong Democrat")
            elif d == "Not very strong":
                out.append("Not very strong Democrat")
            else:
                raise ValueError(f"wu2025.political_party: unexpected democratstrength {d!r}")
        elif p == "Republican":
            if pd.notna(d) or pd.notna(l):
                raise ValueError(f"wu2025.political_party: Republican with unexpected branch ({d!r}, {l!r})")
            if r == "Strong":
                out.append("Strong Republican")
            elif r == "Not very strong":
                out.append("Not very strong Republican")
            else:
                raise ValueError(f"wu2025.political_party: unexpected republicanstrength {r!r}")
        elif p in ("Independent", "Something else"):
            if pd.notna(r) or pd.notna(d):
                raise ValueError(f"wu2025.political_party: {p!r} with unexpected strength ({r!r}, {d!r})")
            if l == "Closer to the Democratic Party":
                out.append("Lean Democrat")
            elif l == "Closer to the Republican Party":
                out.append("Lean Republican")
            elif l == "Neither":
                out.append(p)
            else:
                raise ValueError(f"wu2025.political_party: unexpected leaner {l!r} for {p!r}")
        else:
            raise ValueError(f"wu2025.political_party: unexpected partisan {p!r}")
    return pd.Series(out, index=resp.index, dtype=object)


def load_raw_wu2025_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Wu et al. (2025) electricity-plan conjoint personas from Stata `.dta`."""
    from data.wu2025 import stimulus as wu_s

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
        "political_party": _build_wu2025_political_party(resp),
        "political_ideology": _clean_str("ideology"),
        "marital_status": _clean_str("marriage"),
        "employment_status": _clean_str("employment"),
        "residence_type": _clean_str("residence"),
        "sexual_orientation": _clean_str("orientation"),
    }
    if not shared_only:
        cols.update({
            "political_label": _clean_str("ideology_original"),
            "owns_home": _clean_str("residenceownership"),
            "rents_home": _clean_str("residencerental"),
            "military_service": _clean_str("military"),
            "electric_heating": _clean_str("householdheat"),
            "political_interest": _clean_str("polint"),
            "electricity_included_in_rent": _clean_str("electricityinrent"),
            "knows_electricity_bill": _clean_str("billknowledge"),
            "uses_investor_owned_utility": _clean_str("currentiou"),
            "uses_community_choice_aggregation": _clean_str("currentcca"),
            "knows_community_choice_aggregation": _clean_str("ccaknowledge"),
        })
    return pd.DataFrame(cols)


def load_raw_kantorowicz2026_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Kantorowicz-Reznichenko et al. (2026) conjoint personas from `.RDS` files."""
    from data.kantorowicz2026 import stimulus as kant_s

    comp_df, laundry_df = kant_s.load_raw_dfs()
    resp = kant_s.extract_respondent_df(comp_df, laundry_df).reset_index(drop=True)
    cols = {
        "CASEID": "kantorowicz2026_" + resp["Response.ID"].astype(str),
        "country": "United States",
        "gender": resp["gender_dummy"],
        "age": resp["age_dummy"],
        "education": resp["education_dummy"],
        "income": resp["income_dummy"],
    }
    return pd.DataFrame(cols)


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


def _build_magistro2025_political_party(resp: pd.DataFrame) -> pd.Series:
    """Combine magistro2025 party and party2 to recover US Independent leaners (fail-fast)."""
    out = []
    for country, p, p2 in zip(resp["countr"], resp["party"], resp["party2"], strict=True):
        if pd.isna(p) or pd.isna(p2):
            raise ValueError(f"magistro2025.political_party: unexpected missing value ({country!r}, {p!r}, {p2!r})")
        if country == "Canada":
            if p != p2 or p not in ("CPC", "LPC", "NDP", "Other"):
                raise ValueError(f"magistro2025.political_party: unexpected Canada combo ({p!r}, {p2!r})")
            out.append(p)
        elif country == "US":
            if p == p2 and p in ("Democratic", "Republican", "Other"):
                out.append(p)
            elif p == "Other" and p2 == "Democratic":
                out.append("Lean Democrat")
            elif p == "Other" and p2 == "Republican":
                out.append("Lean Republican")
            else:
                raise ValueError(f"magistro2025.political_party: unexpected US combo ({p!r}, {p2!r})")
        else:
            raise ValueError(f"magistro2025.political_party: unexpected country {country!r}")
    return pd.Series(out, index=resp.index, dtype=object)


def load_raw_magistro2025_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Magistro et al. (2025) US + Canada conjoint personas from `clean_AJPS.csv`."""
    from data.magistro2025 import stimulus as mag_s

    resp = mag_s.load_raw_df(data_dir / "magistro2025" / "dataverse_files" / "clean_AJPS.csv").reset_index(drop=True)

    us_mask = resp["countr"] == "US"
    ca_mask = resp["countr"] == "Canada"
    if not (us_mask | ca_mask).all():
        raise ValueError(f"magistro2025: unexpected countr values {resp.loc[~(us_mask | ca_mask), 'countr'].unique()}")

    region_series = _strict_map(resp["region"].where(us_mask), MAGISTRO_US_REGION_LABELS, "magistro2025.region")
    state_series = _strict_map(
        resp["GeoPC_Region1"].where(ca_mask), MAGISTRO_CA_PROVINCE_LABELS, "magistro2025.GeoPC_Region1"
    )

    cols = {
        "CASEID": "magistro2025_" + resp["ID"].astype(int).astype(str),
        "country": resp["countr"],
        "region": region_series,
        "state": state_series,
        "gender": resp["gender"],
        "age": resp["age_group"],
        "education": resp["education"],
        "income": resp["income"],
        "political_party": _build_magistro2025_political_party(resp),
    }
    return pd.DataFrame(cols)


US_ZIP_STATE_RANGES: tuple[tuple[int, int, str], ...] = (
    (1000, 2799, "Massachusetts"),
    (2800, 2999, "Rhode Island"),
    (3000, 3899, "New Hampshire"),
    (3900, 4999, "Maine"),
    (5000, 5499, "Vermont"),
    (5500, 5599, "Massachusetts"),
    (5600, 5999, "Vermont"),
    (6000, 6389, "Connecticut"),
    (6390, 6390, "New York"),
    (6400, 6999, "Connecticut"),
    (7000, 8999, "New Jersey"),
    (10000, 14999, "New York"),
    (15000, 19699, "Pennsylvania"),
    (19700, 19999, "Delaware"),
    (20000, 20099, "District of Columbia"),
    (20100, 20199, "Virginia"),
    (20200, 20599, "District of Columbia"),
    (20600, 21999, "Maryland"),
    (22000, 24699, "Virginia"),
    (24700, 26899, "West Virginia"),
    (27000, 28999, "North Carolina"),
    (29000, 29999, "South Carolina"),
    (30000, 31999, "Georgia"),
    (32000, 34999, "Florida"),
    (35000, 36999, "Alabama"),
    (37000, 38599, "Tennessee"),
    (38600, 39799, "Mississippi"),
    (39800, 39999, "Georgia"),
    (40000, 42799, "Kentucky"),
    (43000, 45999, "Ohio"),
    (46000, 47999, "Indiana"),
    (48000, 49999, "Michigan"),
    (50000, 52899, "Iowa"),
    (53000, 54999, "Wisconsin"),
    (55000, 56799, "Minnesota"),
    (56900, 56999, "District of Columbia"),
    (57000, 57799, "South Dakota"),
    (58000, 58899, "North Dakota"),
    (59000, 59999, "Montana"),
    (60000, 62999, "Illinois"),
    (63000, 65899, "Missouri"),
    (66000, 67999, "Kansas"),
    (68000, 69399, "Nebraska"),
    (70000, 71499, "Louisiana"),
    (71600, 72999, "Arkansas"),
    (73000, 73199, "Oklahoma"),
    (73300, 73399, "Texas"),
    (73400, 74999, "Oklahoma"),
    (75000, 79999, "Texas"),
    (80000, 81699, "Colorado"),
    (82000, 83199, "Wyoming"),
    (83200, 83899, "Idaho"),
    (84000, 84799, "Utah"),
    (85000, 86599, "Arizona"),
    (87000, 88499, "New Mexico"),
    (88500, 88599, "Texas"),
    (88900, 89899, "Nevada"),
    (90000, 96199, "California"),
    (96700, 96899, "Hawaii"),
    (97000, 97999, "Oregon"),
    (98000, 99499, "Washington"),
    (99500, 99999, "Alaska"),
)

# Known invalid/unassigned ZIP entries in Forsythe2023 (00000 placeholder, 00746 unassigned)
FORSYTHE_INVALID_ZIPS = frozenset({0, 746})


def _us_zip_to_state(series: pd.Series, invalid_zips: frozenset[int], name: str) -> pd.Series:
    """Map numeric US 5-digit ZIP codes strictly to US state/DC names."""
    out = []
    for val in series:
        if pd.isna(val):
            out.append(float("nan"))
            continue
        z = int(val)
        if z != val:
            raise ValueError(f"{name}: non-integer ZIP value {val!r}")
        if z in invalid_zips:
            out.append(float("nan"))
            continue
        matched = None
        for lo, hi, st in US_ZIP_STATE_RANGES:
            if lo <= z <= hi:
                matched = st
                break
        if matched is None:
            raise ValueError(f"{name}: unmapped US ZIP code {z:05d}")
        out.append(matched)
    return pd.Series(out, index=series.index, dtype=object)


def _normalize_semicolon_multiselect(series: pd.Series) -> pd.Series:
    """Normalize ';' multi-select strings to canonical '; '-delimited strings."""
    return series.map(
        lambda x: "; ".join(part.strip() for part in str(x).split(";")) if pd.notna(x) else x
    )


def load_raw_forsythe2023_personas(data_dir: Path = DATA_DIR, shared_only: bool = False) -> pd.DataFrame:
    """Load raw Forsythe et al. (2023) EV conjoint personas from `Pooled.xlsx` inside the replication zip."""
    zip_path = data_dir / "Forsythe2023" / "DrivingElectrificationReplication-v1.0.2.zip"
    member = "crforsythe-DrivingElectrificationReplication-72b0e47/Estimation/Data/ChoiceData/Pooled/Pooled.xlsx"
    with zipfile.ZipFile(zip_path) as zf:
        xl = pd.ExcelFile(io.BytesIO(zf.read(member)))
    demo = pd.concat([xl.parse("demo-car"), xl.parse("demo-suv")], ignore_index=True)
    if demo["id"].isna().any() or demo["id"].duplicated().any():
        raise ValueError("forsythe2023: expected unique non-null respondent ids across demo-car and demo-suv")

    zip_combined = demo["ZIP"].fillna(demo["Clean ZIP"])
    cols: dict[str, object] = {
        "CASEID": "forsythe2023_" + demo["id"].astype(int).astype(str),
        "country": "United States",
        "state": _us_zip_to_state(zip_combined, FORSYTHE_INVALID_ZIPS, "forsythe2023.ZIP_combined"),
        "gender": demo["Clean gender"],
        "age": demo["Clean Age"],
        "education": demo["Clean education"],
        "family_income": demo["Clean Income Entry"],
        "marital_status": demo["Clean livingSit"],
        "household_size": demo["Clean householdSize"].map(
            lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (int, float)) else x
        ),
    }
    if not shared_only:
        cols.update({
            "vehicle_powertrain_owned": _normalize_semicolon_multiselect(demo["Clean elecVehTypeOwned"]),
            "covid_employment_impact": _normalize_semicolon_multiselect(demo["Clean covid"]),
        })
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Combined raw + standardized entry points
# ---------------------------------------------------------------------------

RAW_LOADERS = {
    "issp": load_raw_issp_personas,
    "twin2k": load_raw_twin2k_personas,
    "wvs7": load_raw_wvs7_personas,
    "subpop": load_raw_subpop_personas,
    "hints7": load_raw_hints7_personas,
    "shed2025": load_raw_shed2025_personas,
    "covid": load_raw_covid_personas,
    "kreps2021": load_raw_kreps2021_personas,
    "grummon2023": load_raw_grummon2023_personas,
    "grummon2025": load_raw_grummon2025_personas,
    "berke2023": load_raw_berke2023_personas,
    "kriner2026": load_raw_kriner2026_personas,
    "wu2025": load_raw_wu2025_personas,
    "kantorowicz2026": load_raw_kantorowicz2026_personas,
    "magistro2025": load_raw_magistro2025_personas,
    "forsythe2023": load_raw_forsythe2023_personas,
}


# ---------------------------------------------------------------------------
# Authoritative raw source column -> canonical persona attribute registry
# ---------------------------------------------------------------------------

# Maps each dataset directory name under `code/data/` to `{raw_col: canonical_attr}`
# for every raw column read by `load_raw_*_personas` (ISSP waves handled dynamically).
_STATIC_PERSONA_SOURCE_COLS: dict[str, dict[str, str]] = {
    "Twin-2K-500": dict(TWIN2K_QID_TO_COL),
    "Twin-2K-500-Mega-Study": {},
    "subpop_train": {},
    "subpop_eval": {},
    "HINTS": {
        **HINTS7_NUMERIC_RENAME,
        **HINTS7_CATEGORICAL_RENAME,
        **{c: "race" for cols, _ in HINTS7_R10_RACE_GROUPS for c in cols},
    },
    "SHED": {
        **SHED2025_NUMERIC_RENAME,
        **SHED2025_CATEGORICAL_RENAME,
    },
    "covid": {
        "state": "state",
        "gender": "gender",
        "age": "age",
        "education": "education",
        "relig": "religion",
        "ideology": "political_ideology",
        "party3": "political_party",
        "income": "income",
        **{c: "race" for c, _ in COVID_RACE_COLS},
        "insurance": "health_insurance",
        "evangelical": "evangelical_christian",
        "work_status": "recent_employment_change",
        "work_home": "able_to_work_from_home",
        "flu_vaccine": "flu_vaccination_frequency",
        "personal_contact1": "knows_someone_tested_positive_covid",
        "personal_contact2": "knows_someone_hospitalized_or_died_covid",
    },
    "kreps2021": {
        "state": "state",
        "gender": "gender",
        "age": "age",
        "education": "education",
        "relig": "religion",
        "ideology": "political_ideology",
        "party3": "political_party",
        "income": "income",
        **{c: "race" for c, _ in COVID_RACE_COLS},
        "insurance": "health_insurance",
        "evangelical": "evangelical_christian",
        "flu_vaccine": "flu_vaccination_frequency",
        "pres_approval": "trump_approval",
        "tv_news": "tv_news_consumption",
        "paper_news": "newspaper_consumption",
        "social_news": "social_media_news_consumption",
        "sns1": "social_media_daily_use",
        "sns2": "social_media_shares_media",
        "sns3": "social_media_follows_accounts",
        "sns4": "social_media_multiple_accounts",
        "sns5": "social_media_perceived_usefulness",
        "sns6": "social_media_follows_friends_shares",
        "sns7": "social_media_checks_friends_activity",
    },
    "grummon2023": {
        "gender": "gender",
        "age": "age",
        "educ": "education",
        "income_10cat": "family_income",
        "latino": "hispanic",
        **{c: "race" for c, _ in GRUMMON2023_RACE_COLS},
        "hhsize_num": "household_size",
        "children_num": "children_in_household",
        "partyid": "political_party",
        "NFPuse": "nutrition_label_use",
        "dietqual": "diet_quality",
        "healthlit": "medical_reading_help",
        "height_feet": "bmi",
        "height_inches": "bmi",
        "weight_lbs": "bmi",
    },
    "grummon2025": {
        "gender": "gender",
        "age": "age",
        "educ": "education",
        "income": "family_income",
        "latino": "hispanic",
        **{c: "race" for c, _ in GRUMMON2025_RACE_COLS},
        "hhsize_num": "household_size",
        "children_num": "children_in_household",
        "nfpuse": "nutrition_label_use",
        "snap": "snap_benefits",
        "wic": "wic_benefits",
        "onlineamt": "online_grocery_share",
        "scr_litall": "nutrition_literacy_score",
    },
    "berke2023": {
        "Q_state": "state",
        "Q_gender": "gender",
        "Q_age": "age",
        "Q_race": "race",
        "Q_income": "family_income",
        "Q_urban_rural": "urban_rural",
        "Q_residence_type": "residence_type",
        "Q_online_shopping": "online_shopping_frequency",
        "Q_takeout_food": "takeout_delivery_frequency",
        "Q_online_groceries": "online_grocery_frequency",
    },
    "Kriner2026": {
        "state": "state",
        "vs_region": "region",
        "gender": "gender",
        "age_years": "age",
        "education": "education",
        "relig": "religion",
        "ideology": "political_ideology",
        "party": "political_party",
        "party_lean": "political_party",
        "income": "income",
        "urban_rural_best": "urban_rural",
        "vs_vote_2024": "voted_last_election",
        **{c: "race" for c, _ in KRINER_RACE_COLS},
        "vs_votechoice_2024": "vote_choice_2024",
        "approve": "trump_approval",
    },
    "wu2025": {
        "state": "state",
        "sex": "gender",
        "age": "age",
        "race": "race",
        "ethnicity": "hispanic",
        "education": "education",
        "income": "family_income",
        "partisan": "political_party",
        "republicanstrength": "political_party",
        "democratstrength": "political_party",
        "leaner": "political_party",
        "ideology": "political_ideology",
        "marriage": "marital_status",
        "employment": "employment_status",
        "residence": "residence_type",
        "orientation": "sexual_orientation",
        "ideology_original": "political_label",
        "residenceownership": "owns_home",
        "residencerental": "rents_home",
        "military": "military_service",
        "householdheat": "electric_heating",
        "polint": "political_interest",
        "electricityinrent": "electricity_included_in_rent",
        "billknowledge": "knows_electricity_bill",
        "currentiou": "uses_investor_owned_utility",
        "currentcca": "uses_community_choice_aggregation",
        "ccaknowledge": "knows_community_choice_aggregation",
    },
    "kantorowicz2026": {
        "gender_dummy": "gender",
        "age_dummy": "age",
        "education_dummy": "education",
        "income_dummy": "income",
    },
    "magistro2025": {
        "countr": "country",
        "region": "region",
        "GeoPC_Region1": "state",
        "gender": "gender",
        "age_group": "age",
        "education": "education",
        "income": "income",
        "party": "political_party",
        "party2": "political_party",
    },
    "Forsythe2023": {
        "ZIP": "state",
        "Clean ZIP": "state",
        "Clean gender": "gender",
        "Clean Age": "age",
        "Clean education": "education",
        "Clean Income Entry": "family_income",
        "Clean livingSit": "marital_status",
        "Clean householdSize": "household_size",
        "Clean elecVehTypeOwned": "vehicle_powertrain_owned",
        "Clean covid": "covid_employment_impact",
    },
}


def get_persona_source_columns(
    dataset_dir: str,
    specs: dict[str, PersonaAttribute] | None = None,
) -> dict[str, str]:
    """Return `{raw_col: canonical_persona_attr}` for active `persona_spec.yaml` attributes in `dataset_dir`.

    Fails fast on unknown dataset directory names or any mapped canonical attribute missing from `specs`.
    """
    active_specs = specs if specs is not None else load_persona_spec()

    if dataset_dir.startswith("issp") and len(dataset_dir) == 8 and dataset_dir[4:].isdigit():
        year = int(dataset_dir[4:])
        if not (2015 <= year <= 2024):
            raise ValueError(f"Unsupported ISSP year in dataset_dir: {dataset_dir!r}")
        raw_map: dict[str, str] = {
            raw_col: attr
            for raw_col, attr in {**ISSP_SHARED_RENAME, **ISSP_SINGLE_RENAME}.items()
            if attr in active_specs
        }
        raw_map["DEGREE"] = "education"
        raw_map["US_ETHN1"] = "ancestry"
        raw_map["US_ETHN2"] = "ancestry"
        if year in ISSP_IDEOLOGY_VAR:
            raw_map[ISSP_IDEOLOGY_VAR[year]] = "political_ideology"
    elif dataset_dir == "WVS-Wave7":
        raw_map = {
            raw_col: attr
            for raw_col, attr in {**WVS_SHARED_RENAME, **WVS_SINGLE_RENAME}.items()
            if attr in active_specs
        }
    elif dataset_dir in _STATIC_PERSONA_SOURCE_COLS:
        raw_map = dict(_STATIC_PERSONA_SOURCE_COLS[dataset_dir])
    else:
        raise KeyError(f"Unknown dataset directory for persona source columns: {dataset_dir!r}")

    invalid = {raw_col: attr for raw_col, attr in raw_map.items() if attr not in active_specs}
    if invalid:
        raise KeyError(
            f"get_persona_source_columns({dataset_dir!r}): mapped attributes not in persona_spec.yaml: {invalid}"
        )
    return raw_map


def load_raw_all_personas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and stack all 16 datasets' raw (unstandardized) personas into one DataFrame."""
    frames = [loader(data_dir=data_dir).assign(dataset=name) for name, loader in RAW_LOADERS.items()]
    combined = pd.concat(frames, ignore_index=True)
    other_cols = [c for c in combined.columns if c not in {"dataset", "CASEID"}]
    return combined[["dataset", "CASEID", *other_cols]]


def load_all_personas(
    specs: dict[str, PersonaAttribute] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load all 16 datasets from raw files, standardize active attributes in `specs`, and stack."""
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

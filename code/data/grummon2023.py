"""Grummon et al. (2023) front-of-package nutrition label RCT training-set builder and loader."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from data.base import (
    ALL_PERSONA_NAMES,
    DATA_DIR,
    DEFAULT_CACHE_DIR,
    add_one_hot,
    build_eval_set,
    build_long_records,
    filter_by_qids,
    group_records_by_state,
    make_view,
)
from data.nutrition.grummon2023.stimulus import (
    ARM_NAMES,
    GRUMMON2023_DIR,
    build_product_catalog,
    get_arm_variant_spec,
    load_analytic_cohort,
)
import data.persona as persona

GRUMMON2023_PARQUET_PATH = GRUMMON2023_DIR / "grummon2023_US_long.parquet"
GRUMMON2023_AUDIT_CSV_PATH = GRUMMON2023_DIR / "question_audit.csv"

HISPANIC_MAP = {
    0.0: "No",
    1.0: "Yes",
}

NFPUSE_MAP = {
    1.0: "Never",
    2.0: "Rarely",
    3.0: "Sometimes",
    4.0: "Most of the time",
    5.0: "Always",
}

MEDHELP_MAP = {
    1.0: "Never",
    2.0: "Rarely",
    3.0: "Sometimes",
    4.0: "Most of the time",
    5.0: "Always",
}

# Confirmed persona attributes for Grummon 2023 (expanded step-by-step as attributes are finalized)
GRUMMON2023_PERSONA_COLS = [
    "SEX",
    "AGE",
    "EDULEVEL",
    "US_INC",
    "HISPANIC",
    "RACE",
    "HHSIZE",
    "NFPUSE",
    "MEDHELP",
]

GRUMMON2023_PERSONA_NAMES = {
    "c_alphan": ALL_PERSONA_NAMES["c_alphan"],
    **{col: ALL_PERSONA_NAMES[col] for col in GRUMMON2023_PERSONA_COLS},
}


def strict_code_map(series: pd.Series, mapping: dict[float, str], name: str) -> pd.Series:
    """Map numeric survey codes to string labels with strict fail-fast validation."""
    s_num = pd.to_numeric(series, errors="raise")
    observed = set(s_num.dropna().unique())
    unexpected = observed - set(mapping.keys())
    if unexpected:
        raise ValueError(f"grummon2023.{name}: unexpected values {sorted(unexpected)} not in {sorted(mapping.keys())}")
    return pd.Series(
        [mapping[float(v)] if pd.notna(v) else float("nan") for v in s_num],
        index=series.index,
        name=name,
        dtype=object,
    )


def derive_grummon2023_race(clean_df: pd.DataFrame) -> pd.Series:
    """Construct raw race categories from `Race_1..Race_6` matching Grummon 2025 `racecat`."""
    race_cols = [f"Race_{i}" for i in range(1, 7)]
    missing_cols = [c for c in race_cols if c not in clean_df.columns]
    if missing_cols:
        raise KeyError(f"grummon2023: missing expected race columns {missing_cols}")

    r_mat = clean_df[race_cols].apply(pd.to_numeric, errors="raise")
    observed = set(r_mat.stack().dropna().unique())
    unexpected = observed - {0.0, 1.0}
    if unexpected:
        raise ValueError(f"grummon2023.Race_1..6: unexpected indicator values {sorted(unexpected)}")

    out: list[object] = []
    for _, row in r_mat.iterrows():
        if row.isna().all():
            out.append(float("nan"))
            continue
        if row.isna().any():
            raise ValueError("grummon2023.Race_1..6: partial NaN across race checkboxes")
        checked = [idx for idx in range(1, 7) if float(row[f"Race_{idx}"]) == 1.0]
        if not checked:
            raise ValueError("grummon2023.Race_1..6: respondent checked zero race boxes without all-NaN")
        if checked == [5]:
            out.append("White")
        elif checked == [3]:
            out.append("Black")
        elif checked == [2]:
            out.append("Asian")
        else:
            out.append("Other")
    raw_race = pd.Series(out, index=clean_df.index, name="RACE", dtype=object)
    return persona.map_persona_series(raw_race, "RACE", "grummon2023")


def build_grummon2023_persona_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Build the validated respondent-level persona DataFrame for Grummon 2023 (`N = 7,945`)."""
    expected_arms = set(ARM_NAMES.values())
    observed_arms = set(clean_df["arm"].dropna().unique())
    if observed_arms != expected_arms or clean_df["arm"].isna().any():
        raise ValueError(f"grummon2023: unexpected arm values {observed_arms} vs {expected_arms}")

    resp_df = pd.DataFrame(
        {
            "CASEID": clean_df["CASEID"].astype(str),
            "arm": clean_df["arm"].astype(str),
            "c_alphan": persona.map_persona_series(
                pd.Series(["United States"] * len(clean_df), index=clean_df.index),
                "c_alphan",
                "grummon2023",
            ),
            "SEX": persona.map_persona_series(clean_df["gender"], "SEX", "grummon2023"),
            "AGE": persona.map_persona_series(clean_df["age"], "AGE", "grummon2023"),
            "EDULEVEL": persona.map_persona_series(clean_df["educ"], "EDULEVEL", "grummon2023"),
            "US_INC": persona.map_persona_series(clean_df["income_10cat"], "US_INC", "grummon2023"),
            "HISPANIC": strict_code_map(clean_df["latino"], HISPANIC_MAP, "HISPANIC"),
            "RACE": derive_grummon2023_race(clean_df),
            "HHSIZE": persona.map_persona_series(clean_df["hhsize_num"], "HHSIZE", "grummon2023"),
            "NFPUSE": strict_code_map(clean_df["NFPuse"], NFPUSE_MAP, "NFPUSE"),
            "MEDHELP": strict_code_map(clean_df["healthlit"], MEDHELP_MAP, "MEDHELP"),
        },
        index=clean_df.index,
    )
    expected_cols = ["CASEID", "arm", "c_alphan", *GRUMMON2023_PERSONA_COLS]
    if list(resp_df.columns) != expected_cols:
        raise ValueError(f"Persona column mismatch: {list(resp_df.columns)} != {expected_cols}")
    return resp_df


def build_grummon2023_long_df(
    clean_df: pd.DataFrame | None = None,
    bev_chars: pd.DataFrame | None = None,
    snack_chars: pd.DataFrame | None = None,
    audit_csv_path: Path = GRUMMON2023_AUDIT_CSV_PATH,
) -> pd.DataFrame:
    """Build the standardized long-format DataFrame across all included outcomes in `question_audit.csv`."""
    if clean_df is None or bev_chars is None or snack_chars is None:
        clean_df, bev_chars, snack_chars = load_analytic_cohort()
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")

    audit_df = pd.read_csv(audit_csv_path)
    inc_df = audit_df[audit_df["include"].astype(bool)].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    product_catalog = build_product_catalog(bev_chars, snack_chars)
    resp_df = build_grummon2023_persona_df(clean_df)

    frames: list[pd.DataFrame] = []
    for _, a_row in inc_df.iterrows():
        v_name = str(a_row["qid"])
        q_type = str(a_row["kev_type"]).strip()
        if q_type not in {"choice", "score"}:
            raise ValueError(f"{v_name}: unexpected kev_type {q_type!r}")

        valid_mask = clean_df[v_name].notna()
        sub_resp = resp_df.loc[valid_mask].copy()
        ans_raw = pd.to_numeric(clean_df.loc[valid_mask, v_name], errors="raise")

        for arm_name in ARM_NAMES.values():
            arm_mask = sub_resp["arm"] == arm_name
            if not arm_mask.any():
                raise ValueError(f"{v_name}: zero respondents in arm {arm_name!r}")

            arm_sub = sub_resp.loc[arm_mask].copy()
            arm_ans = ans_raw.loc[arm_mask]
            spec = get_arm_variant_spec(v_name, arm_name, product_catalog)

            int_codes = [int(c) for c in spec["option_codes"]]
            texts = [str(t) for t in spec["option_texts"]]
            descs = spec["option_descriptions"]
            code_to_text = dict(zip(int_codes, texts, strict=True))

            ans_codes: list[int] = []
            for val in arm_ans:
                ival = int(val)
                if ival not in code_to_text or float(ival) != float(val):
                    raise ValueError(f"{v_name} ({arm_name}): unexpected answer value {val!r}")
                ans_codes.append(ival)
            ans_texts = [code_to_text[c] for c in ans_codes]

            full_q_text = (
                f"{spec['context']}\n\n{spec['question_stem']}"
                if spec["context"]
                else spec["question_stem"]
            )
            qid = f"grummon2023:{arm_name}:{v_name}"
            n_rows = len(arm_sub)

            arm_sub["v"] = v_name
            arm_sub["qid"] = qid
            arm_sub["label"] = f"{arm_name}: {v_name}"
            arm_sub["question"] = [full_q_text] * n_rows
            arm_sub["type"] = q_type
            arm_sub["reversed"] = False
            arm_sub["option_codes"] = [list(int_codes)] * n_rows
            arm_sub["option_texts"] = [list(texts)] * n_rows
            arm_sub["option_descriptions"] = (
                [list(descs)] * n_rows if descs is not None else [None] * n_rows
            )
            arm_sub["answer_code"] = ans_codes
            arm_sub["answer_text"] = ans_texts
            frames.append(arm_sub.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def export_grummon2023_parquet(
    parquet_path: Path = GRUMMON2023_PARQUET_PATH,
    audit_csv_path: Path = GRUMMON2023_AUDIT_CSV_PATH,
) -> pd.DataFrame:
    """Build and write `grummon2023_US_long.parquet`, returning the exported DataFrame."""
    long_df = build_grummon2023_long_df(audit_csv_path=audit_csv_path)
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_grummon2023_df(qids=None, parquet_path=GRUMMON2023_PARQUET_PATH) -> pd.DataFrame:
    """Load `grummon2023_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Grummon 2023 parquet not found: {path}")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    if "option_descriptions" in df.columns:
        df["option_descriptions"] = [
            list(x) if x is not None else None for x in df["option_descriptions"]
        ]
    df["year"] = "grummon2023"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_grummon2023_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols=GRUMMON2023_PERSONA_COLS,
    parquet_path=GRUMMON2023_PARQUET_PATH,
):
    """Load respondent-level Grummon 2023 records (preserving `option_descriptions`), optionally state-grouped."""
    df = load_grummon2023_df(qids=qids, parquet_path=parquet_path)
    records = build_long_records(
        df,
        persona_cols=persona_cols,
        include_country=include_country,
        year="grummon2023",
    )
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="grummon2023")
    return records


def load_grummon2023_eval(
    *,
    attr_sets,
    include_country: bool,
    qids=None,
    min_n=None,
    grouped: bool = True,
    persona_cols=GRUMMON2023_PERSONA_COLS,
    name: str = "grummon2023",
    valid: bool = False,
    parquet_path=GRUMMON2023_PARQUET_PATH,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build a Grummon 2023 evaluation bundle (`grouped=True`) or flat records (`grouped=False`)."""
    df = load_grummon2023_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        grouped=grouped,
        persona_cols=persona_cols,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )

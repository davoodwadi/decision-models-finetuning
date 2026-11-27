"""Grummon et al. (2023) front-of-package nutrition label RCT training-set builder and loader."""

from __future__ import annotations

import functools
import json
from pathlib import Path
import pandas as pd

from data.base import (
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
    VENDING_BONUS_INTRO,
    build_product_catalog,
    get_arm_variant_spec,
    load_analytic_cohort,
)
from data.persona_loaders import get_persona_source_columns, load_raw_grummon2023_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

GRUMMON2023_PARQUET_PATH = GRUMMON2023_DIR / "grummon2023_US_long.parquet"
GRUMMON2023_AUDIT_CSV_PATH = GRUMMON2023_DIR / "question_audit.csv"
GRUMMON2023_OVERRIDES_PATH = GRUMMON2023_DIR / "question_overrides.json"
GRUMMON2023_STUDY_DB_TEXTS = {"vending_intro": VENDING_BONUS_INTRO}

# Canonical `persona_spec.yaml` persona attributes for Grummon 2023.
GRUMMON2023_PERSONA_COLS = [
    "gender",
    "age",
    "education",
    "family_income",
    "hispanic",
    "race",
    "household_size",
    "children_in_household",
    "political_party",
    "nutrition_label_use",
    "diet_quality",
    "medical_reading_help",
    "bmi",
]


def _grummon2023_persona_names() -> dict[str, str]:
    specs = load_persona_spec()
    return {"c_alphan": "Country", **{name: spec.prompt for name, spec in specs.items()}}


def build_grummon2023_persona_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical `persona_spec.yaml` respondent persona DataFrame for Grummon 2023 (`N = 7,945`)."""
    expected_arms = set(ARM_NAMES.values())
    observed_arms = set(clean_df["arm"].dropna().unique())
    if observed_arms != expected_arms or clean_df["arm"].isna().any():
        raise ValueError(f"grummon2023: unexpected arm values {observed_arms} vs {expected_arms}")

    std = standardize_dataframe(load_raw_grummon2023_personas(), dataset="grummon2023")
    persona_cols = [c for c in std.columns if c not in ("CASEID", "country")]
    if persona_cols != GRUMMON2023_PERSONA_COLS:
        raise ValueError(f"grummon2023 persona column drift: {persona_cols} != {GRUMMON2023_PERSONA_COLS}")

    case_ids = clean_df["CASEID"].astype(str)
    if set(case_ids) != set(std["CASEID"].astype(str)):
        raise ValueError("grummon2023: CASEID mismatch between clean_df and persona_loaders")
    std_by_case = std.set_index("CASEID")
    aligned = std_by_case.loc[case_ids.values].reset_index()
    aligned.insert(1, "arm", clean_df["arm"].astype(str).values)
    aligned.insert(2, "c_alphan", aligned["country"])
    expected_cols = ["CASEID", "arm", "c_alphan", "country", *GRUMMON2023_PERSONA_COLS]
    resp_df = aligned[expected_cols].copy()
    resp_df.index = clean_df.index
    if len(resp_df) != len(clean_df) or resp_df["CASEID"].nunique() != len(clean_df):
        raise ValueError(f"Expected {len(clean_df)} unique respondents in persona table, got {len(resp_df)}")
    return resp_df


def _resolve_g23_context_text(context_source: str, in_block_ctx: str) -> str:
    cs = context_source.strip()
    if cs == "":
        return ""
    if cs == "in_block_db":
        return in_block_ctx
    if cs.startswith("prefix:"):
        return cs[len("prefix:") :].strip()
    if cs.startswith("qid:"):
        parts: list[str] = []
        for q in cs[len("qid:") :].split("+"):
            q = q.strip()
            if not q:
                continue
            if q not in GRUMMON2023_STUDY_DB_TEXTS:
                raise KeyError(
                    f"Unknown context_source qid {q!r} (available: {sorted(GRUMMON2023_STUDY_DB_TEXTS)})"
                )
            parts.append(GRUMMON2023_STUDY_DB_TEXTS[q])
        return "\n\n".join(parts)
    raise ValueError(f"Unsupported context_source: {context_source!r}")


def build_grummon2023_long_df(
    clean_df: pd.DataFrame | None = None,
    bev_chars: pd.DataFrame | None = None,
    snack_chars: pd.DataFrame | None = None,
    audit_csv_path: Path = GRUMMON2023_AUDIT_CSV_PATH,
    overrides_path: Path = GRUMMON2023_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build the standardized long-format DataFrame across all included outcomes in `question_audit.csv`."""
    if clean_df is None or bev_chars is None or snack_chars is None:
        clean_df, bev_chars, snack_chars = load_analytic_cohort()
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")

    overrides: dict[str, dict] = {}
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))

    audit_df = pd.read_csv(audit_csv_path, dtype=str, keep_default_na=False)
    inc_df = audit_df[audit_df["include"] == "True"].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    persona_src_cols = get_persona_source_columns("grummon2023")
    overlap = sorted(set(inc_df["QuestionID"]) & set(persona_src_cols))
    if overlap:
        raise ValueError(
            "grummon2023: refusing to export questions that overlap with locked persona columns: "
            + ", ".join(f"{c} -> {persona_src_cols[c]}" for c in overlap)
        )

    product_catalog = build_product_catalog(bev_chars, snack_chars)
    resp_df = build_grummon2023_persona_df(clean_df)

    frames: list[pd.DataFrame] = []
    for _, a_row in inc_df.iterrows():
        v_name = str(a_row["QuestionID"])
        row_key = f"{a_row['catalog_idx']}:{v_name}"
        q_type = str(a_row["kev_type"]).strip()
        if q_type not in {"choice", "score"}:
            raise ValueError(f"{v_name}: unexpected kev_type {q_type!r}")
        ctx_source = str(a_row["context_source"]).strip() if "context_source" in a_row else "in_block_db"
        raw_sep = str(a_row["context_sep"]) if "context_sep" in a_row else ""
        ctx_sep = "\n\n" if raw_sep == "" else raw_sep.replace("\\n", "\n")
        drop_raw = str(a_row["drop_options"]) if "drop_options" in a_row else ""
        drop_set = {s.strip() for s in drop_raw.split("|") if s.strip()}

        ov = overrides.get(row_key, {})
        stem_override = ov.get("stem")
        opt_overrides: dict[str, str] = ov.get("options", {})

        valid_mask = clean_df[v_name].notna()
        sub_resp = resp_df.loc[valid_mask].copy()
        ans_raw = pd.to_numeric(clean_df.loc[valid_mask, v_name], errors="raise")

        for arm_name in ARM_NAMES.values():
            arm_mask = sub_resp["arm"] == arm_name
            if not arm_mask.any():
                raise ValueError(f"{v_name}: zero respondents in arm {arm_name!r}")

            spec = get_arm_variant_spec(v_name, arm_name, product_catalog)
            raw_codes = [int(c) for c in spec["option_codes"]]
            raw_texts = [str(t) for t in spec["option_texts"]]
            raw_descs = spec["option_descriptions"]

            kept_codes: list[int] = []
            kept_texts: list[str] = []
            kept_descs: list[str | None] | None = [] if raw_descs is not None else None
            dropped_codes: set[int] = set()
            for idx_o, (c, rt) in enumerate(zip(raw_codes, raw_texts, strict=True)):
                if rt in drop_set:
                    dropped_codes.add(c)
                    continue
                kept_codes.append(c)
                kept_texts.append(opt_overrides.get(rt, rt))
                if kept_descs is not None and raw_descs is not None:
                    kept_descs.append(raw_descs[idx_o])

            if len(kept_codes) < 2:
                raise ValueError(f"{v_name} ({arm_name}): fewer than 2 options remain after drop_options={drop_set}")

            arm_sub = sub_resp.loc[arm_mask].copy()
            arm_ans = ans_raw.loc[arm_mask]
            if dropped_codes:
                keep_ans_mask = ~arm_ans.isin(dropped_codes)
                arm_sub = arm_sub.loc[keep_ans_mask].copy()
                arm_ans = arm_ans.loc[keep_ans_mask]

            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            ans_codes: list[int] = []
            for val in arm_ans:
                ival = int(val)
                if ival not in code_to_text or float(ival) != float(val):
                    raise ValueError(f"{v_name} ({arm_name}): unexpected answer value {val!r}")
                ans_codes.append(ival)
            ans_texts = [code_to_text[c] for c in ans_codes]

            ctx_text = _resolve_g23_context_text(ctx_source, spec["context"])
            stem_text = stem_override if stem_override is not None else spec["question_stem"]
            full_q_text = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text
            qid = f"grummon2023:{arm_name}:{v_name}"
            n_rows = len(arm_sub)

            arm_sub["v"] = v_name
            arm_sub["qid"] = qid
            arm_sub["label"] = f"{arm_name}: {v_name}"
            arm_sub["question"] = [full_q_text] * n_rows
            arm_sub["type"] = q_type
            arm_sub["reversed"] = False
            arm_sub["option_codes"] = [list(kept_codes)] * n_rows
            arm_sub["option_texts"] = [list(kept_texts)] * n_rows
            arm_sub["option_descriptions"] = (
                [list(kept_descs)] * n_rows if kept_descs is not None else [None] * n_rows
            )
            arm_sub["answer_code"] = ans_codes
            arm_sub["answer_text"] = ans_texts
            frames.append(arm_sub.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def export_grummon2023_parquet(
    parquet_path: Path = GRUMMON2023_PARQUET_PATH,
    audit_csv_path: Path = GRUMMON2023_AUDIT_CSV_PATH,
    overrides_path: Path = GRUMMON2023_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `grummon2023_US_long.parquet`, returning the exported DataFrame."""
    long_df = build_grummon2023_long_df(
        audit_csv_path=audit_csv_path,
        overrides_path=overrides_path,
    )
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
    persona_cols: list[str] | None = None,
    parquet_path=GRUMMON2023_PARQUET_PATH,
):
    """Load respondent-level Grummon 2023 records (preserving `option_descriptions`), optionally state-grouped."""
    df = load_grummon2023_df(qids=qids, parquet_path=parquet_path)
    specs = load_persona_spec()
    persona_names = _grummon2023_persona_names()
    if persona_cols is None:
        persona_cols = [c for c in specs if c in df.columns and c != "country"]
    records = build_long_records(
        df,
        persona_cols=persona_cols,
        include_country=include_country,
        year="grummon2023",
        persona_names=persona_names,
    )
    if grouped:
        view_fn = functools.partial(make_view, persona_names=persona_names)
        return group_records_by_state(records, view_fn=view_fn, name="grummon2023")
    return records


def load_grummon2023_eval(
    *,
    attr_sets,
    include_country: bool,
    qids=None,
    min_n=None,
    grouped: bool = True,
    persona_cols: list[str] | None = None,
    name: str = "grummon2023",
    valid: bool = False,
    parquet_path=GRUMMON2023_PARQUET_PATH,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build a Grummon 2023 evaluation bundle (`grouped=True`) or flat records (`grouped=False`)."""
    df = load_grummon2023_df(qids=qids, parquet_path=parquet_path)
    specs = load_persona_spec()
    persona_names = _grummon2023_persona_names()
    if persona_cols is None:
        persona_cols = [c for c in specs if c in df.columns and c != "country"]
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        grouped=grouped,
        persona_cols=persona_cols,
        persona_names=persona_names,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )

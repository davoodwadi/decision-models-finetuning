"""berke2023 (Berke et al. 2023, drone delivery and customer privacy) long-format parquet exporter and loaders.

Exports every row marked `include=True` in `question_audit.csv` to `berke2023_US_long.parquet`, applying the audit UI's
`drop_options`, `context_source`, `context_sep`, and `question_overrides.json` edits.
Persona decisions: `DECISIONS.md` and `persona.py` (dataset key `berke2023`).
"""

from __future__ import annotations

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
import data.persona as persona
from data.berke2023.stimulus import (
    BERKE_DIR,
    CHOICE_OPTIONS,
    FREQ_OPTIONS,
    N_RESPONDENTS,
    PRIVACY_EXPLAINER_TEXT,
    VENDOR_INTRO_TEXTS,
    YES_NO_DK_OPTIONS,
    build_survey_audit_artifacts,
    choice_stem,
    extract_choices_df,
    load_raw_df,
    option_descriptions,
)

BERKE2023_PARQUET_PATH = BERKE_DIR / "berke2023_US_long.parquet"
BERKE2023_AUDIT_CSV_PATH = BERKE_DIR / "question_audit.csv"
BERKE2023_OVERRIDES_PATH = BERKE_DIR / "question_overrides.json"

BERKE2023_PERSONA_COLS = [c for c in persona.DATASET_PERSONA_COLS["berke2023"] if c != "c_alphan"]

STUDY_DB_TEXTS = {
    "privacy_explainer": PRIVACY_EXPLAINER_TEXT,
    **{f"intro_v{v}": text for v, text in enumerate(VENDOR_INTRO_TEXTS)},
}

# Answer options for the single-column questions the exporter supports (codes as stored in the data).
SINGLE_QUESTION_OPTIONS: dict[str, list[tuple[int, str]]] = {
    "Q_online_shopping": FREQ_OPTIONS,
    "Q_takeout_food": FREQ_OPTIONS,
    "Q_online_groceries": FREQ_OPTIONS,
    "Q_v_1": YES_NO_DK_OPTIONS,
    "Q_v_2": YES_NO_DK_OPTIONS,
}


def build_berke2023_persona_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardized respondent persona table (`N = 3,715`) via `persona.apply_persona_mapping`."""
    mapped = persona.apply_persona_mapping(raw_df, "berke2023")
    mapped["CASEID"] = "berke2023_" + mapped["ResponseId"].astype(str)
    out = mapped[["CASEID", "ResponseId", "c_alphan", *BERKE2023_PERSONA_COLS]].reset_index(drop=True)
    if len(out) != N_RESPONDENTS or out["CASEID"].nunique() != N_RESPONDENTS:
        raise ValueError(f"Expected {N_RESPONDENTS} unique respondents in persona table, got {len(out)}")
    return out


def _resolve_context_text(context_source: str, in_block_ctx: str) -> str:
    cs = context_source.strip()
    if cs == "":
        return ""
    if cs == "in_block_db":
        if in_block_ctx == "":
            raise ValueError("context_source is in_block_db but the row has no in-block context")
        return in_block_ctx
    if cs.startswith("prefix:"):
        return cs[len("prefix:"):].strip()
    if cs.startswith("qid:"):
        parts = []
        for q in cs[len("qid:"):].split("+"):
            q = q.strip()
            if not q:
                continue
            if q not in STUDY_DB_TEXTS:
                raise KeyError(f"Unknown context_source qid {q!r} (available: {sorted(STUDY_DB_TEXTS)})")
            parts.append(STUDY_DB_TEXTS[q])
        return "\n\n".join(parts)
    raise ValueError(f"Unsupported context_source: {context_source!r}")


def _apply_option_edits(
    raw_pairs: list[tuple[int, str]], drop_set: set[str], opt_overrides: dict[str, str], v_name: str
) -> tuple[list[int], list[str], set[int]]:
    unknown = drop_set - {t for _, t in raw_pairs}
    if unknown:
        raise ValueError(f"{v_name}: drop_options not among the options: {unknown}")
    kept_codes, kept_texts, dropped = [], [], set()
    for code, raw_txt in raw_pairs:
        if raw_txt in drop_set:
            dropped.add(int(code))
            continue
        kept_codes.append(int(code))
        kept_texts.append(opt_overrides[raw_txt] if raw_txt in opt_overrides else raw_txt)
    if len(kept_codes) < 2:
        raise ValueError(f"{v_name}: fewer than 2 options remain after drop_options={drop_set}")
    return kept_codes, kept_texts, dropped


def build_berke2023_long_df(
    audit_csv_path: Path = BERKE2023_AUDIT_CSV_PATH,
    overrides_path: Path = BERKE2023_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Long-format DataFrame across all included questions in `question_audit.csv`."""
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")
    overrides: dict[str, dict] = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )

    raw_df = load_raw_df()
    resp_persona = build_berke2023_persona_df(raw_df)
    persona_by_resp = resp_persona.set_index("ResponseId")
    choices_df = extract_choices_df(raw_df)

    catalog_df, _, details = build_survey_audit_artifacts()
    catalog = catalog_df.set_index(catalog_df["catalog_idx"] + ":" + catalog_df["QuestionID"])

    audit_df = pd.read_csv(audit_csv_path, dtype=str, keep_default_na=False)
    inc_df = audit_df[audit_df["include"] == "True"].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    frames: list[pd.DataFrame] = []
    for _, a_row in inc_df.iterrows():
        v_name = a_row["QuestionID"]
        row_key = f"{a_row['catalog_idx']}:{v_name}"
        if row_key not in catalog.index:
            raise KeyError(f"{row_key}: not in the current survey catalog (stimulus.py)")
        cat_row = catalog.loc[row_key]
        if a_row["QuestionText"] != cat_row["QuestionText"]:
            raise ValueError(f"{row_key}: QuestionText in the audit CSV differs from stimulus.py; use overrides instead")
        q_type = a_row["kev_type"].strip()
        if q_type not in {"choice", "score"}:
            raise ValueError(f"{v_name}: unexpected kev_type {q_type!r}")

        raw_sep = a_row["context_sep"]
        ctx_sep = "\n\n" if raw_sep == "" else raw_sep.replace("\\n", "\n")
        drop_set = {s.strip() for s in a_row["drop_options"].split("|") if s.strip()}
        ov = overrides[row_key] if row_key in overrides else {}
        stem_text = ov["stem"] if "stem" in ov else cat_row["QuestionText"]
        opt_overrides: dict[str, str] = ov["options"] if "options" in ov else {}
        ctx_text = _resolve_context_text(a_row["context_source"], details["items"][row_key]["in_block_db_text"])
        full_q = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

        if v_name.startswith("delivery_choice_v"):
            if q_type != "choice":
                raise ValueError(f"{v_name} must be exported as kev_type 'choice'")
            v = int(v_name[len("delivery_choice_v"):])
            if stem_text != choice_stem(v) and "stem" not in ov:
                raise ValueError(f"{v_name}: catalog stem differs from the generated choice stem")
            kept_codes, kept_texts, dropped = _apply_option_edits(CHOICE_OPTIONS, drop_set, opt_overrides, v_name)
            if dropped:
                raise ValueError(f"{v_name}: dropping Ground vehicle or Drone is not supported")
            sub_c = choices_df[choices_df["v"] == v].reset_index(drop=True)
            designs = [f"gc{r.gc}_gt{r.gt}_dc{r.dc}_dt{r.dt}_dp{r.dp}" for r in sub_c.itertuples(index=False)]
            sub = persona_by_resp.loc[sub_c["ResponseId"]].reset_index()
            sub = sub[["CASEID", "ResponseId", "c_alphan", *BERKE2023_PERSONA_COLS]].copy()
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            sub["v"] = v_name
            sub["qid"] = [f"berke2023:{v_name}:{d}" for d in designs]
            sub["label"] = [f"{v_name} ({d})" for d in designs]
            sub["question"] = full_q
            sub["type"] = q_type
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = [
                list(option_descriptions(v, r.gc, r.gt, r.dc, r.dt, r.dp)) for r in sub_c.itertuples(index=False)
            ]
            sub["answer_code"] = [int(a) for a in sub_c["answer"]]
            sub["answer_text"] = [code_to_text[int(a)] for a in sub_c["answer"]]
            frames.append(sub.reset_index(drop=True))
            continue

        if v_name not in SINGLE_QUESTION_OPTIONS:
            raise KeyError(f"{v_name}: export supports the delivery choices and {sorted(SINGLE_QUESTION_OPTIONS)} only")
        kept_codes, kept_texts, dropped = _apply_option_edits(
            SINGLE_QUESTION_OPTIONS[v_name], drop_set, opt_overrides, v_name
        )
        code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
        ans = pd.to_numeric(raw_df[v_name], errors="raise")
        valid = ans.notna() & ~ans.isin(dropped)
        sub = resp_persona.loc[valid.values].reset_index(drop=True).copy()
        ans_codes = []
        for val in ans[valid]:
            ival = int(val)
            if float(ival) != float(val) or ival not in code_to_text:
                raise ValueError(f"{v_name}: unexpected answer value {val!r}")
            ans_codes.append(ival)
        sub["v"] = v_name
        sub["qid"] = f"berke2023:{v_name}"
        sub["label"] = v_name
        sub["question"] = full_q
        sub["type"] = q_type
        sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
        sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
        sub["option_descriptions"] = [None for _ in range(len(sub))]
        sub["answer_code"] = ans_codes
        sub["answer_text"] = [code_to_text[c] for c in ans_codes]
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def export_berke2023_parquet(
    parquet_path: Path = BERKE2023_PARQUET_PATH,
    audit_csv_path: Path = BERKE2023_AUDIT_CSV_PATH,
    overrides_path: Path = BERKE2023_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `berke2023_US_long.parquet` (name matches the audit app's `export_<folder>_parquet` call)."""
    long_df = build_berke2023_long_df(audit_csv_path=audit_csv_path, overrides_path=overrides_path)
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_berke2023_df(qids=None, parquet_path=BERKE2023_PARQUET_PATH) -> pd.DataFrame:
    """Load `berke2023_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"berke2023 parquet not found: {path} (export it from the audit app)")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    df["option_descriptions"] = [list(x) if x is not None else None for x in df["option_descriptions"]]
    df["year"] = "berke2023"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_berke2023_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols=BERKE2023_PERSONA_COLS,
    parquet_path=BERKE2023_PARQUET_PATH,
):
    """Respondent-level training records (one per parquet row), optionally state-grouped."""
    df = load_berke2023_df(qids=qids, parquet_path=parquet_path)
    records = build_long_records(df, persona_cols=persona_cols, include_country=include_country, year="berke2023")
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="berke2023")
    return records


def load_berke2023_eval(
    parquet_path=BERKE2023_PARQUET_PATH,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols=BERKE2023_PERSONA_COLS,
    grouped: bool = True,
    name: str = "berke2023",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Evaluation bundle (state-grouped when `grouped=True`).

    No dataset-specific evaluator yet (`eval_fn=None`, so only the generic per-record metrics apply); the paper's
    benchmark would be its mode-choice logit models per product type.
    """
    df = load_berke2023_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        persona_cols=persona_cols,
        grouped=grouped,
        eval_fn=None,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )

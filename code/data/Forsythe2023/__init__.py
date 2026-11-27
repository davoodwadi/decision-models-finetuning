"""Forsythe et al. (2023, *PNAS*) electric vehicle discrete-choice conjoint exporter and loaders.

Exports every row marked `include=True` in `question_audit.csv` (`car_choice`, `suv_choice`) to
`forsythe2023_US_long.parquet`, applying the Audit UI's `drop_options`, `context_source`, `context_sep`,
and `question_overrides.json` edits. Attaches canonical `persona_spec.yaml` persona attributes strictly via
`standardize_dataframe(load_raw_forsythe2023_personas(), dataset="forsythe2023")`.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import pandas as pd

from data.Forsythe2023.stimulus import (
    CAR_CONTEXT_TEXT,
    CHOICE_OPTIONS,
    CHOICE_STEM,
    FORSYTHE_DIR,
    N_TOTAL_RESPONDENTS,
    SUV_CONTEXT_TEXT,
    extract_all_tasks,
    load_raw_pooled_sheets,
)
from data.base import (
    DEFAULT_CACHE_DIR,
    add_one_hot,
    build_eval_set,
    build_long_records,
    filter_by_qids,
    group_records_by_state,
    make_view,
)
from data.persona_loaders import get_persona_source_columns, load_raw_forsythe2023_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

FORSYTHE2023_PARQUET_PATH = FORSYTHE_DIR / "forsythe2023_US_long.parquet"
FORSYTHE2023_AUDIT_CSV_PATH = FORSYTHE_DIR / "question_audit.csv"
FORSYTHE2023_OVERRIDES_PATH = FORSYTHE_DIR / "question_overrides.json"

STUDY_DB_TEXTS: dict[str, str] = {}


def build_forsythe2023_persona_df() -> tuple[pd.DataFrame, list[str]]:
    """Standardized respondent persona table (`N = 1,626`) using canonical `persona_spec.yaml` attributes."""
    specs = load_persona_spec()
    raw_personas = load_raw_forsythe2023_personas()
    persona_cols = [attr for attr in specs if attr in raw_personas.columns]
    std = standardize_dataframe(
        raw_personas[["CASEID", *persona_cols]].copy(), dataset="forsythe2023", specs=specs
    )
    std["c_alphan"] = "US"
    out = std[["CASEID", "c_alphan", *persona_cols]].reset_index(drop=True)
    if len(out) != N_TOTAL_RESPONDENTS or out["CASEID"].nunique() != N_TOTAL_RESPONDENTS:
        raise ValueError(
            f"Forsythe2023: expected {N_TOTAL_RESPONDENTS} unique respondents in persona table, got {len(out)}"
        )
    return out, persona_cols


def _resolve_context_text(context_source: str, in_block_ctx: str) -> str:
    cs = context_source.strip()
    if cs == "":
        return ""
    if cs == "in_block_db":
        if in_block_ctx == "":
            raise ValueError("context_source is in_block_db but the row has no in-block context")
        return in_block_ctx
    if cs.startswith("prefix:"):
        return cs[len("prefix:") :].strip()
    if cs.startswith("qid:"):
        parts = []
        for q in cs[len("qid:") :].split("+"):
            q = q.strip()
            if not q:
                continue
            if q not in STUDY_DB_TEXTS:
                raise KeyError(f"Unknown context_source qid {q!r} (available: {sorted(STUDY_DB_TEXTS)})")
            parts.append(STUDY_DB_TEXTS[q])
        return "\n\n".join(parts)
    raise ValueError(f"Unsupported context_source: {context_source!r}")


def _apply_option_edits(
    raw_pairs: list[tuple[int, str]],
    drop_set: set[str],
    opt_overrides: dict[str, str],
    v_name: str,
) -> tuple[list[int], list[str], set[int]]:
    unknown = drop_set - {t for _, t in raw_pairs}
    if unknown:
        raise ValueError(f"{v_name}: drop_options not among options: {sorted(unknown)}")
    kept_codes: list[int] = []
    kept_texts: list[str] = []
    dropped_codes: set[int] = set()
    for code, raw_txt in raw_pairs:
        if raw_txt in drop_set:
            dropped_codes.add(int(code))
            continue
        kept_codes.append(int(code))
        kept_texts.append(opt_overrides[raw_txt] if raw_txt in opt_overrides else raw_txt)
    if len(kept_codes) < 2:
        raise ValueError(f"{v_name}: fewer than 2 options remain after drop_options={drop_set}")
    return kept_codes, kept_texts, dropped_codes


def build_forsythe2023_long_df(
    audit_csv_path: Path = FORSYTHE2023_AUDIT_CSV_PATH,
    overrides_path: Path = FORSYTHE2023_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build the standardized long-format DataFrame across all included questions in `question_audit.csv`."""
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")
    overrides: dict[str, dict] = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )

    audit_df = pd.read_csv(audit_csv_path, dtype=str, keep_default_na=False)
    inc_df = audit_df[audit_df["include"] == "True"].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    persona_source_cols = get_persona_source_columns("Forsythe2023")
    overlap = [q for q in inc_df["QuestionID"] if q in persona_source_cols]
    if overlap:
        raise ValueError(
            f"Forsythe2023: refusing to export questions that overlap locked persona attributes: {overlap}"
        )

    cbc_car, cbc_suv, _, _ = load_raw_pooled_sheets()
    car_tasks, suv_tasks = extract_all_tasks(cbc_car, cbc_suv)
    resp_persona, persona_cols = build_forsythe2023_persona_df()
    persona_by_id = resp_persona.set_index("CASEID")

    conjoint_map = {
        "car_choice": (car_tasks, CAR_CONTEXT_TEXT),
        "suv_choice": (suv_tasks, SUV_CONTEXT_TEXT),
    }

    frames: list[pd.DataFrame] = []
    for _, a_row in inc_df.iterrows():
        v_name = str(a_row["QuestionID"])
        row_key = f"{a_row['catalog_idx']}:{v_name}"
        q_type = str(a_row["kev_type"]).strip()
        if q_type != "choice":
            raise ValueError(f"{v_name}: must be exported as kev_type 'choice', got {q_type!r}")
        if v_name not in conjoint_map:
            raise KeyError(f"Unsupported Forsythe2023 QuestionID included for export: {v_name!r}")

        tasks_df, in_block_ctx = conjoint_map[v_name]
        ctx_source = str(a_row["context_source"]).strip()
        raw_sep = str(a_row["context_sep"])
        ctx_sep = "\n\n" if raw_sep == "" else raw_sep.replace("\\n", "\n")
        drop_set = {s.strip() for s in str(a_row["drop_options"]).split("|") if s.strip()}

        ov = overrides[row_key] if row_key in overrides else {}
        stem_override = ov["stem"] if "stem" in ov else None
        opt_overrides: dict[str, str] = ov["options"] if "options" in ov else {}

        kept_codes, kept_texts, dropped = _apply_option_edits(
            CHOICE_OPTIONS, drop_set, opt_overrides, v_name
        )
        if dropped:
            raise ValueError(f"{v_name}: dropping vehicle choice options is not supported")

        code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
        ctx_text = _resolve_context_text(ctx_source, in_block_ctx)
        stem_text = stem_override if stem_override is not None else CHOICE_STEM
        full_q = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

        sub = persona_by_id.loc[tasks_df["CASEID"]].reset_index()
        sub = sub[["CASEID", "c_alphan", *persona_cols]].copy()
        designs = [
            f"{r.code_1}_{r.code_2}_{r.code_3}" for r in tasks_df.itertuples(index=False)
        ]
        ans_codes = [int(c) for c in tasks_df["choice_code"]]
        ans_texts = [code_to_text[c] for c in ans_codes]

        sub["v"] = v_name
        sub["qid"] = [f"forsythe2023:{v_name}:{d}" for d in designs]
        sub["label"] = [f"{v_name} ({d})" for d in designs]
        sub["question"] = full_q
        sub["type"] = q_type
        sub["reversed"] = False
        sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
        sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
        sub["option_descriptions"] = [
            [str(r.desc_1), str(r.desc_2), str(r.desc_3)]
            for r in tasks_df.itertuples(index=False)
        ]
        sub["answer_code"] = ans_codes
        sub["answer_text"] = ans_texts
        frames.append(sub.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def export_Forsythe2023_parquet(
    parquet_path: Path = FORSYTHE2023_PARQUET_PATH,
    audit_csv_path: Path = FORSYTHE2023_AUDIT_CSV_PATH,
    overrides_path: Path = FORSYTHE2023_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `forsythe2023_US_long.parquet` atomically (matches audit app `export_<folder>_parquet`)."""
    long_df = build_forsythe2023_long_df(audit_csv_path=audit_csv_path, overrides_path=overrides_path)
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    long_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, parquet_path)
    return long_df


export_forsythe2023_parquet = export_Forsythe2023_parquet


def load_forsythe2023_df(qids=None, parquet_path: Path = FORSYTHE2023_PARQUET_PATH) -> pd.DataFrame:
    """Load `forsythe2023_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Forsythe2023 parquet not found: {path}")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    if "option_descriptions" in df.columns:
        df["option_descriptions"] = [
            list(x) if x is not None else None for x in df["option_descriptions"]
        ]
    df["year"] = "forsythe2023"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_forsythe2023_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols: list[str] | None = None,
    parquet_path: Path = FORSYTHE2023_PARQUET_PATH,
):
    """Load respondent-level Forsythe et al. (2023) records, optionally state-grouped."""
    df = load_forsythe2023_df(qids=qids, parquet_path=parquet_path)
    specs = load_persona_spec()
    persona_names = {"c_alphan": "Country", **{name: spec.prompt for name, spec in specs.items()}}
    if persona_cols is None:
        persona_cols = [c for c in specs if c in df.columns and c != "country"]
    records = build_long_records(
        df,
        persona_cols=persona_cols,
        include_country=include_country,
        year="forsythe2023",
        persona_names=persona_names,
    )
    if grouped:
        view_fn = functools.partial(make_view, persona_names=persona_names)
        return group_records_by_state(records, view_fn=view_fn, name="forsythe2023")
    return records


def load_forsythe2023_eval(
    parquet_path: Path = FORSYTHE2023_PARQUET_PATH,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols: list[str] | None = None,
    grouped: bool = True,
    name: str = "forsythe2023",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Load Forsythe et al. (2023) evaluation bundle (`grouped=True`) or flat `records`."""
    df = load_forsythe2023_df(qids=qids, parquet_path=parquet_path)
    specs = load_persona_spec()
    persona_names = {"c_alphan": "Country", **{name: spec.prompt for name, spec in specs.items()}}
    if persona_cols is None:
        persona_cols = [c for c in specs if c in df.columns and c != "country"]
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        persona_cols=persona_cols,
        persona_names=persona_names,
        grouped=grouped,
        eval_fn=None,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )

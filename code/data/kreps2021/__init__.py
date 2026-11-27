"""Kreps et al. (2021, *npj Vaccines*) COVID-19 vaccine conjoint & misinformation exporter and loaders.

Exports every row marked `include=True` in `question_audit.csv` to `kreps2021_US_long.parquet`, applying the
Audit UI's `drop_options`, `context_source`, `context_sep`, and `question_overrides.json` edits.
Attaches canonical `persona_spec.yaml` persona attributes strictly via
`standardize_dataframe(load_raw_kreps2021_personas(), dataset="kreps2021")`.
"""

from __future__ import annotations

import functools
import json
import os
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
from data.kreps2021.stimulus import (
    KREPS2021_DIR,
    N_RESPONDENTS,
    PROFILE_ID_TO_TUPLE,
    SUBSTANTIVE_ITEM_SPECS,
    VAX_BIN_OPTIONS,
    VAX_BIN_STEM,
    VAX_ORD_OPTIONS,
    VAX_ORD_STEM,
    extract_conjoint_tasks_df,
    format_profile_block_context,
    load_raw_dta,
)
from data.persona_loaders import get_persona_source_columns, load_raw_kreps2021_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

KREPS2021_PARQUET_PATH = KREPS2021_DIR / "kreps2021_US_long.parquet"
KREPS2021_AUDIT_CSV_PATH = KREPS2021_DIR / "question_audit.csv"
KREPS2021_OVERRIDES_PATH = KREPS2021_DIR / "question_overrides.json"

STUDY_DB_TEXTS: dict[str, str] = {}


def build_kreps2021_persona_df() -> tuple[pd.DataFrame, list[str]]:
    """Standardized respondent persona table (`N = 1,096`) using canonical `persona_spec.yaml` attributes."""
    specs = load_persona_spec()
    raw_personas = load_raw_kreps2021_personas()
    persona_cols = [attr for attr in specs if attr in raw_personas.columns]
    std = standardize_dataframe(raw_personas[["CASEID", *persona_cols]].copy(), dataset="kreps2021", specs=specs)
    std["c_alphan"] = "US"
    out = std[["CASEID", "c_alphan", *persona_cols]].reset_index(drop=True)
    if len(out) != N_RESPONDENTS or out["CASEID"].nunique() != N_RESPONDENTS:
        raise ValueError(f"kreps2021: expected {N_RESPONDENTS} unique respondents in persona table, got {len(out)}")
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


def build_kreps2021_long_df(
    audit_csv_path: Path = KREPS2021_AUDIT_CSV_PATH,
    overrides_path: Path = KREPS2021_OVERRIDES_PATH,
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

    persona_source_cols = get_persona_source_columns("kreps2021")
    overlap = [q for q in inc_df["QuestionID"] if q in persona_source_cols]
    if overlap:
        raise ValueError(
            f"kreps2021: refusing to export questions that overlap locked persona attributes: {overlap}"
        )

    raw_df, _, _ = load_raw_dta()
    tasks_df = extract_conjoint_tasks_df(raw_df)
    resp_persona, persona_cols = build_kreps2021_persona_df()
    persona_by_id = resp_persona.set_index("CASEID")

    frames: list[pd.DataFrame] = []
    for _, a_row in inc_df.iterrows():
        v_name = str(a_row["QuestionID"])
        row_key = f"{a_row['catalog_idx']}:{v_name}"
        q_type = str(a_row["kev_type"]).strip()
        if q_type not in {"choice", "score"}:
            raise ValueError(f"{v_name}: unexpected kev_type {q_type!r}")

        ctx_source = str(a_row["context_source"]).strip()
        raw_sep = str(a_row["context_sep"])
        ctx_sep = "\n\n" if raw_sep == "" else raw_sep.replace("\\n", "\n")
        drop_set = {s.strip() for s in str(a_row["drop_options"]).split("|") if s.strip()}

        ov = overrides[row_key] if row_key in overrides else {}
        stem_override = ov["stem"] if "stem" in ov else None
        opt_overrides: dict[str, str] = ov["options"] if "options" in ov else {}

        if v_name in ("vax_bin", "vax_ord"):
            raw_opts = VAX_BIN_OPTIONS if v_name == "vax_bin" else VAX_ORD_OPTIONS
            default_stem = VAX_BIN_STEM if v_name == "vax_bin" else VAX_ORD_STEM
            is_reversed = v_name == "vax_ord"

            kept_codes, kept_texts, dropped_codes = _apply_option_edits(
                raw_opts, drop_set, opt_overrides, v_name
            )
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            valid_tasks = (
                tasks_df.loc[~tasks_df[v_name].isin(dropped_codes)].reset_index(drop=True)
                if dropped_codes
                else tasks_df
            )

            stem_text = stem_override if stem_override is not None else default_stem
            q_by_pid: dict[str, str] = {}
            for pid in PROFILE_ID_TO_TUPLE:
                ctx_text = _resolve_context_text(ctx_source, format_profile_block_context(pid))
                q_by_pid[pid] = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

            sub = persona_by_id.loc[valid_tasks["CASEID"]].reset_index()
            sub = sub[["CASEID", "c_alphan", *persona_cols]].copy()
            pids = valid_tasks["profile_id"].astype(str).tolist()
            ans_codes = [int(c) for c in valid_tasks[v_name]]
            ans_texts = [code_to_text[c] for c in ans_codes]

            sub["v"] = v_name
            sub["qid"] = [f"kreps2021:{v_name}:{pid}" for pid in pids]
            sub["label"] = [f"{v_name} ({pid})" for pid in pids]
            sub["question"] = [q_by_pid[pid] for pid in pids]
            sub["type"] = q_type
            sub["reversed"] = is_reversed
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = [None for _ in range(len(sub))]
            sub["answer_code"] = ans_codes
            sub["answer_text"] = ans_texts
            frames.append(sub.reset_index(drop=True))
            continue

        if v_name not in SUBSTANTIVE_ITEM_SPECS:
            raise KeyError(f"Unsupported kreps2021 QuestionID included for export: {v_name!r}")

        spec = SUBSTANTIVE_ITEM_SPECS[v_name]
        raw_map: dict[int, int] = spec["raw_code_map"]
        kept_codes, kept_texts, dropped_codes = _apply_option_edits(
            spec["options"], drop_set, opt_overrides, v_name
        )
        code_to_text = dict(zip(kept_codes, kept_texts, strict=True))

        col_s = raw_df[v_name]
        valid_idx = col_s.notna()
        raw_vals = pd.to_numeric(col_s.loc[valid_idx], errors="raise").astype(int)
        unexpected = set(raw_vals.unique()) - set(raw_map.keys())
        if unexpected:
            raise ValueError(f"kreps2021.{v_name}: unexpected raw answer values {sorted(unexpected)}")
        mapped_vals = raw_vals.map(raw_map)
        keep_mask = ~mapped_vals.isin(dropped_codes) if dropped_codes else pd.Series(True, index=mapped_vals.index)
        kept_mapped = mapped_vals.loc[keep_mask]
        kept_caseids = raw_df.loc[kept_mapped.index, "ResponseId"].astype(str)

        sub = persona_by_id.loc[kept_caseids].reset_index()
        sub = sub[["CASEID", "c_alphan", *persona_cols]].copy()
        ctx_text = _resolve_context_text(ctx_source, spec["in_block_db_text"])
        stem_text = stem_override if stem_override is not None else spec["question_stem"]
        full_q = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

        ans_codes = [int(c) for c in kept_mapped]
        ans_texts = [code_to_text[c] for c in ans_codes]

        sub["v"] = v_name
        sub["qid"] = f"kreps2021:{v_name}"
        sub["label"] = v_name
        sub["question"] = full_q
        sub["type"] = q_type
        sub["reversed"] = bool(spec["reversed"])
        sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
        sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
        sub["option_descriptions"] = [None for _ in range(len(sub))]
        sub["answer_code"] = ans_codes
        sub["answer_text"] = ans_texts
        frames.append(sub.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def export_kreps2021_parquet(
    parquet_path: Path = KREPS2021_PARQUET_PATH,
    audit_csv_path: Path = KREPS2021_AUDIT_CSV_PATH,
    overrides_path: Path = KREPS2021_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `kreps2021_US_long.parquet` atomically, returning the exported DataFrame."""
    long_df = build_kreps2021_long_df(audit_csv_path=audit_csv_path, overrides_path=overrides_path)
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    long_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, parquet_path)
    return long_df


def load_kreps2021_df(qids=None, parquet_path: Path = KREPS2021_PARQUET_PATH) -> pd.DataFrame:
    """Load `kreps2021_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"kreps2021 parquet not found: {path}")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    if "option_descriptions" in df.columns:
        df["option_descriptions"] = [
            list(x) if x is not None else None for x in df["option_descriptions"]
        ]
    df["year"] = "kreps2021"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_kreps2021_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols: list[str] | None = None,
    parquet_path: Path = KREPS2021_PARQUET_PATH,
):
    """Load respondent-level Kreps et al. (2021) records, optionally state-grouped."""
    df = load_kreps2021_df(qids=qids, parquet_path=parquet_path)
    specs = load_persona_spec()
    persona_names = {"c_alphan": "Country", **{name: spec.prompt for name, spec in specs.items()}}
    if persona_cols is None:
        persona_cols = [c for c in specs if c in df.columns and c != "country"]
    records = build_long_records(
        df,
        persona_cols=persona_cols,
        include_country=include_country,
        year="kreps2021",
        persona_names=persona_names,
    )
    if grouped:
        view_fn = functools.partial(make_view, persona_names=persona_names)
        return group_records_by_state(records, view_fn=view_fn, name="kreps2021")
    return records


def load_kreps2021_eval(
    parquet_path: Path = KREPS2021_PARQUET_PATH,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols: list[str] | None = None,
    grouped: bool = True,
    name: str = "kreps2021",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Load Kreps et al. (2021) evaluation bundle (`grouped=True`) or flat `records`."""
    df = load_kreps2021_df(qids=qids, parquet_path=parquet_path)
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

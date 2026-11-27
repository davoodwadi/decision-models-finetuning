"""kantorowicz2026 (Kantorowicz-Reznichenko et al. 2026, contractual rights conjoint) long-format parquet exporter and loaders.

Exports every row marked `include=True` in `question_audit.csv` to `kantorowicz2026_US_long.parquet`, applying the audit
UI's `drop_options`, `context_source`, `context_sep`, and `question_overrides.json` edits.
Attaches canonical `persona_spec.yaml` persona attributes strictly via
`standardize_dataframe(load_raw_kantorowicz2026_personas(), dataset="kantorowicz2026")`.
"""

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
from data.kantorowicz2026.stimulus import (
    COMPUTER_CHOICE_OPTIONS,
    COMPUTER_CHOICE_STEM,
    KANTOROWICZ_DIR,
    LAUNDRY_CHOICE_OPTIONS,
    LAUNDRY_CHOICE_STEM,
    N_TOTAL_RESPONDENTS,
    build_survey_audit_artifacts,
    extract_all_tasks,
    extract_respondent_df,
    load_raw_dfs,
)
from data.persona_loaders import get_persona_source_columns, load_raw_kantorowicz2026_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

KANTOROWICZ2026_PARQUET_PATH = KANTOROWICZ_DIR / "kantorowicz2026_US_long.parquet"
KANTOROWICZ2026_AUDIT_CSV_PATH = KANTOROWICZ_DIR / "question_audit.csv"
KANTOROWICZ2026_OVERRIDES_PATH = KANTOROWICZ_DIR / "question_overrides.json"

KANTOROWICZ2026_PERSONA_COLS = ["gender", "age", "education", "income"]

STUDY_DB_TEXTS: dict[str, str] = {}


def build_kantorowicz2026_persona_df(comp_df: pd.DataFrame, laundry_df: pd.DataFrame) -> pd.DataFrame:
    """Canonical `persona_spec.yaml` respondent persona table (`N = 2,242`)."""
    resp_df = extract_respondent_df(comp_df, laundry_df)
    std = standardize_dataframe(load_raw_kantorowicz2026_personas(), dataset="kantorowicz2026")
    persona_cols = [c for c in std.columns if c not in ("CASEID", "country")]
    if persona_cols != KANTOROWICZ2026_PERSONA_COLS:
        raise ValueError(f"kantorowicz2026 persona column drift: {persona_cols} != {KANTOROWICZ2026_PERSONA_COLS}")
    case_ids = "kantorowicz2026_" + resp_df["Response.ID"].astype(str)
    if set(case_ids) != set(std["CASEID"].astype(str)):
        raise ValueError("kantorowicz2026: CASEID mismatch between raw_df and persona_loaders")
    std_by_case = std.set_index("CASEID")
    aligned = std_by_case.loc[case_ids.values].reset_index()
    aligned.insert(1, "Response.ID", resp_df["Response.ID"].values)
    aligned.insert(2, "c_alphan", aligned["country"])
    out = aligned[["CASEID", "Response.ID", "c_alphan", "country", *KANTOROWICZ2026_PERSONA_COLS]].reset_index(
        drop=True
    )
    if len(out) != N_TOTAL_RESPONDENTS or out["CASEID"].nunique() != N_TOTAL_RESPONDENTS:
        raise ValueError(f"Expected {N_TOTAL_RESPONDENTS} unique respondents in persona table, got {len(out)}")
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


def build_kantorowicz2026_long_df(
    audit_csv_path: Path = KANTOROWICZ2026_AUDIT_CSV_PATH,
    overrides_path: Path = KANTOROWICZ2026_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Long-format DataFrame across all included questions in `question_audit.csv`."""
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")
    overrides: dict[str, dict] = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )

    comp_df, laundry_df = load_raw_dfs()
    resp_persona = build_kantorowicz2026_persona_df(comp_df, laundry_df)
    persona_by_resp = resp_persona.set_index("Response.ID")
    comp_tasks, laundry_tasks = extract_all_tasks(comp_df, laundry_df)

    catalog_df, _, details = build_survey_audit_artifacts()
    catalog = catalog_df.set_index(catalog_df["catalog_idx"] + ":" + catalog_df["QuestionID"])

    audit_df = pd.read_csv(audit_csv_path, dtype=str, keep_default_na=False)
    inc_df = audit_df[audit_df["include"] == "True"].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    persona_src_cols = get_persona_source_columns("kantorowicz2026")
    overlap = sorted(set(inc_df["QuestionID"]) & set(persona_src_cols))
    if overlap:
        raise ValueError(
            "kantorowicz2026: refusing to export questions that overlap with locked persona columns: "
            + ", ".join(f"{c} -> {persona_src_cols[c]}" for c in overlap)
        )

    conjoint_map = {
        "computer_choice": (comp_tasks, COMPUTER_CHOICE_OPTIONS, COMPUTER_CHOICE_STEM),
        "laundry_choice": (laundry_tasks, LAUNDRY_CHOICE_OPTIONS, LAUNDRY_CHOICE_STEM),
    }

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

        if v_name not in conjoint_map:
            raise KeyError(f"{v_name}: export supports {sorted(conjoint_map)} only")
        if q_type != "choice":
            raise ValueError(f"{v_name} must be exported as kev_type 'choice'")
        tasks_df, raw_options, expected_stem = conjoint_map[v_name]
        if stem_text != expected_stem and "stem" not in ov:
            raise ValueError(f"{v_name}: catalog stem differs from the expected choice stem")
        kept_codes, kept_texts, dropped = _apply_option_edits(raw_options, drop_set, opt_overrides, v_name)
        if dropped:
            raise ValueError(f"{v_name}: dropping Option A or Option B is not supported")

        code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
        sub = persona_by_resp.loc[tasks_df["Response.ID"]].reset_index()
        sub = sub[["CASEID", "Response.ID", "c_alphan", "country", *KANTOROWICZ2026_PERSONA_COLS]].copy()
        designs = [f"{r.code_a}_{r.code_b}" for r in tasks_df.itertuples(index=False)]
        sub["v"] = v_name
        sub["qid"] = [f"kantorowicz2026:{v_name}:{d}" for d in designs]
        sub["label"] = [f"{v_name} ({d})" for d in designs]
        sub["question"] = full_q
        sub["type"] = q_type
        sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
        sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
        sub["option_descriptions"] = [[str(r.desc_a), str(r.desc_b)] for r in tasks_df.itertuples(index=False)]
        sub["answer_code"] = [int(a) for a in tasks_df["choice_code"]]
        sub["answer_text"] = [code_to_text[int(a)] for a in tasks_df["choice_code"]]
        frames.append(sub.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def export_kantorowicz2026_parquet(
    parquet_path: Path = KANTOROWICZ2026_PARQUET_PATH,
    audit_csv_path: Path = KANTOROWICZ2026_AUDIT_CSV_PATH,
    overrides_path: Path = KANTOROWICZ2026_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `kantorowicz2026_US_long.parquet` (matches the audit app's `export_<folder>_parquet` call)."""
    long_df = build_kantorowicz2026_long_df(audit_csv_path=audit_csv_path, overrides_path=overrides_path)
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_kantorowicz2026_df(qids=None, parquet_path=KANTOROWICZ2026_PARQUET_PATH) -> pd.DataFrame:
    """Load `kantorowicz2026_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"kantorowicz2026 parquet not found: {path} (export it from the audit app)")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    df["option_descriptions"] = [list(x) if x is not None else None for x in df["option_descriptions"]]
    df["year"] = "kantorowicz2026"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_kantorowicz2026_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols: list[str] | None = None,
    parquet_path=KANTOROWICZ2026_PARQUET_PATH,
):
    """Respondent-level training records (one per parquet row), optionally state-grouped."""
    df = load_kantorowicz2026_df(qids=qids, parquet_path=parquet_path)
    specs = load_persona_spec()
    persona_names = {"c_alphan": "Country", **{name: spec.prompt for name, spec in specs.items()}}
    if persona_cols is None:
        persona_cols = [c for c in specs if c in df.columns and c != "country"]
    records = build_long_records(
        df,
        persona_cols=persona_cols,
        include_country=include_country,
        year="kantorowicz2026",
        persona_names=persona_names,
    )
    if grouped:
        view_fn = functools.partial(make_view, persona_names=persona_names)
        return group_records_by_state(records, view_fn=view_fn, name="kantorowicz2026")
    return records


def load_kantorowicz2026_eval(
    parquet_path=KANTOROWICZ2026_PARQUET_PATH,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols: list[str] | None = None,
    grouped: bool = True,
    name: str = "kantorowicz2026",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Evaluation bundle (state-grouped when `grouped=True`)."""
    df = load_kantorowicz2026_df(qids=qids, parquet_path=parquet_path)
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

"""magistro2025 (Magistro et al. 2025, AI & offshoring conjoint) long-format parquet exporter and loaders.

Exports every row marked `include=True` in `question_audit.csv` to `magistro2025_long.parquet`, applying the audit
UI's `drop_options`, `context_source`, `context_sep`, and `question_overrides.json` edits.
Persona decisions: `DECISIONS.md` and `persona.py` (dataset key `magistro2025`).
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
from data.magistro2025.stimulus import (
    AGREE_OPTIONS,
    ARM_WORDING,
    CONJOINT_AI_STEM,
    CONJOINT_OFFSHORING_STEM,
    MAGISTRO_DIR,
    N_TOTAL_RESPONDENTS,
    POLICY_ITEMS,
    POLICY_STEM_TEMPLATE,
    SUPPORT_OPTIONS,
    build_survey_audit_artifacts,
    extract_conjoint_tasks,
    load_raw_df,
)
import data.persona as persona

MAGISTRO2025_PARQUET_PATH = MAGISTRO_DIR / "magistro2025_long.parquet"
MAGISTRO2025_AUDIT_CSV_PATH = MAGISTRO_DIR / "question_audit.csv"
MAGISTRO2025_OVERRIDES_PATH = MAGISTRO_DIR / "question_overrides.json"

MAGISTRO2025_PERSONA_COLS = [c for c in persona.DATASET_PERSONA_COLS["magistro2025"] if c != "c_alphan"]


def build_magistro2025_persona_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardized respondent persona table (`N = 6,056`, US + Canada) via `persona.apply_persona_mapping`."""
    mapped = persona.apply_persona_mapping(raw_df, "magistro2025")
    mapped["CASEID"] = "magistro2025_" + mapped["ID"].astype(int).astype(str)
    mapped["respondent"] = mapped["ID"].astype(int)
    out = mapped[["CASEID", "respondent", "c_alphan", *MAGISTRO2025_PERSONA_COLS]].reset_index(drop=True)
    if len(out) != N_TOTAL_RESPONDENTS or out["CASEID"].nunique() != N_TOTAL_RESPONDENTS:
        raise ValueError(f"Expected {N_TOTAL_RESPONDENTS} unique respondents in persona table, got {len(out)}")
    return out


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


def _combine_context(context_source: str, row_ctx: str, stem: str, ctx_sep: str) -> str:
    cs = context_source.strip()
    if cs == "":
        return stem
    if cs == "in_block_db":
        if row_ctx == "":
            raise ValueError("context_source is in_block_db but row has no in-block context")
        return f"{row_ctx}{ctx_sep}{stem}"
    if cs.startswith("prefix:"):
        pref = cs[len("prefix:"):].strip()
        return f"{pref}{ctx_sep}{stem}" if pref else stem
    raise ValueError(f"Unsupported context_source: {context_source!r}")


def build_magistro2025_long_df(
    audit_csv_path: Path = MAGISTRO2025_AUDIT_CSV_PATH,
    overrides_path: Path = MAGISTRO2025_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Long-format DataFrame across all included questions in `question_audit.csv`."""
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")
    overrides: dict[str, dict] = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )

    raw_df = load_raw_df()
    resp_persona = build_magistro2025_persona_df(raw_df)
    persona_by_id = resp_persona.set_index("respondent")
    ai_tasks, off_tasks = extract_conjoint_tasks(raw_df)

    catalog_df, _, _ = build_survey_audit_artifacts(raw_df)
    catalog = catalog_df.set_index(catalog_df["catalog_idx"] + ":" + catalog_df["QuestionID"])

    audit_df = pd.read_csv(audit_csv_path, dtype=str, keep_default_na=False)
    inc_df = audit_df[audit_df["include"] == "True"].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    conjoint_map = {
        "conjoint_ai": (ai_tasks, CONJOINT_AI_STEM),
        "conjoint_offshoring": (off_tasks, CONJOINT_OFFSHORING_STEM),
    }
    policy_specs: dict[str, tuple[str, bool]] = {}
    for prefix in ("policy_ai_offshoregr", "tables_policygr"):
        for idx, text, has_ph in POLICY_ITEMS:
            policy_specs[f"{prefix}_{idx}"] = (text, has_ph)

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
        stem_override = ov["stem"] if "stem" in ov else None
        opt_overrides: dict[str, str] = ov["options"] if "options" in ov else {}

        if v_name in conjoint_map:
            tasks_df, expected_stem = conjoint_map[v_name]
            stem_text = stem_override if stem_override is not None else expected_stem
            kept_codes, kept_texts, dropped = _apply_option_edits(SUPPORT_OPTIONS, drop_set, opt_overrides, v_name)
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            kept_tasks = tasks_df[~tasks_df["answer_code"].isin(dropped)].reset_index(drop=True)
            sub = persona_by_id.loc[kept_tasks["ID"]].reset_index()
            sub = sub[["CASEID", "respondent", "c_alphan", *MAGISTRO2025_PERSONA_COLS]].copy()
            codes = list(kept_tasks["profile_code"])
            sub["v"] = v_name
            sub["qid"] = [f"magistro2025:{v_name}:{c}" for c in codes]
            sub["label"] = [f"{v_name} ({c})" for c in codes]
            sub["question"] = [
                _combine_context(a_row["context_source"], str(ctx), stem_text, ctx_sep)
                for ctx in kept_tasks["context"]
            ]
            sub["type"] = q_type
            sub["reversed"] = False
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = [None for _ in range(len(sub))]
            sub["answer_code"] = [int(a) for a in kept_tasks["answer_code"]]
            sub["answer_text"] = [code_to_text[int(a)] for a in kept_tasks["answer_code"]]
            frames.append(sub.reset_index(drop=True))
        elif v_name in policy_specs:
            item_text, has_ph = policy_specs[v_name]
            kept_codes, kept_texts, dropped = _apply_option_edits(AGREE_OPTIONS, drop_set, opt_overrides, v_name)
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            valid_mask = raw_df[v_name].notna()
            sub_raw = raw_df.loc[valid_mask, ["ID", "treat_offshoring", v_name]].copy()
            sub_raw["answer_code"] = sub_raw[v_name].astype(int)
            sub_raw = sub_raw[~sub_raw["answer_code"].isin(dropped)].reset_index(drop=True)
            sub = persona_by_id.loc[sub_raw["ID"].astype(int)].reset_index()
            sub = sub[["CASEID", "respondent", "c_alphan", *MAGISTRO2025_PERSONA_COLS]].copy()
            arms = list(sub_raw["treat_offshoring"])
            questions = []
            for arm_name in arms:
                if stem_override is not None:
                    q_txt = stem_override
                else:
                    w = ARM_WORDING[str(arm_name)]
                    formatted = item_text.format(policy_regulate=w["policy_regulate"]) if has_ph else item_text
                    q_txt = POLICY_STEM_TEMPLATE.format(
                        policy_challenge=w["policy_challenge"],
                        policy_text=formatted,
                    )
                questions.append(_combine_context(a_row["context_source"], "", q_txt, ctx_sep))
            sub["v"] = v_name
            sub["qid"] = [f"magistro2025:{v_name}:{str(a).lower()}" for a in arms]
            sub["label"] = [f"{v_name} ({a})" for a in arms]
            sub["question"] = questions
            sub["type"] = q_type
            sub["reversed"] = False
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = [None for _ in range(len(sub))]
            sub["answer_code"] = [int(a) for a in sub_raw["answer_code"]]
            sub["answer_text"] = [code_to_text[int(a)] for a in sub_raw["answer_code"]]
            frames.append(sub.reset_index(drop=True))
        else:
            raise KeyError(f"{v_name}: export supports {sorted([*conjoint_map, *policy_specs])} only")

    return pd.concat(frames, ignore_index=True)


def export_magistro2025_parquet(
    parquet_path: Path = MAGISTRO2025_PARQUET_PATH,
    audit_csv_path: Path = MAGISTRO2025_AUDIT_CSV_PATH,
    overrides_path: Path = MAGISTRO2025_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `magistro2025_long.parquet` (matches the audit app's `export_<folder>_parquet` call)."""
    long_df = build_magistro2025_long_df(audit_csv_path=audit_csv_path, overrides_path=overrides_path)
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_magistro2025_df(qids=None, parquet_path=MAGISTRO2025_PARQUET_PATH) -> pd.DataFrame:
    """Load `magistro2025_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"magistro2025 parquet not found: {path} (export it from the audit app)")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    df["option_descriptions"] = [list(x) if x is not None else None for x in df["option_descriptions"]]
    df["year"] = "magistro2025"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_magistro2025_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols=MAGISTRO2025_PERSONA_COLS,
    parquet_path=MAGISTRO2025_PARQUET_PATH,
):
    """Respondent-level training records (one per parquet row), optionally state-grouped."""
    df = load_magistro2025_df(qids=qids, parquet_path=parquet_path)
    records = build_long_records(
        df, persona_cols=persona_cols, include_country=include_country, year="magistro2025"
    )
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="magistro2025")
    return records


def load_magistro2025_eval(
    parquet_path=MAGISTRO2025_PARQUET_PATH,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols=MAGISTRO2025_PERSONA_COLS,
    grouped: bool = True,
    name: str = "magistro2025",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Evaluation bundle (state-grouped when `grouped=True`)."""
    df = load_magistro2025_df(qids=qids, parquet_path=parquet_path)
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

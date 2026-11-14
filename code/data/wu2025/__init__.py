"""wu2025 (Wu et al. 2025, CCA electricity-plan conjoint) long-format parquet exporter and loaders.

Exports every row marked `include=True` in `question_audit.csv` to `wu2025_US_long.parquet`, applying the audit UI's
`drop_options`, `context_source`, `context_sep`, and `question_overrides.json` edits.
Persona decisions: `DECISIONS.md` and `persona.py` (dataset key `wu2025`). IP / location columns are never loaded.
"""

from __future__ import annotations

import itertools
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
from data.wu2025.stimulus import (
    ATTR_LOCAL,
    ATTR_MONEY,
    ATTR_PERCENT,
    CONJOINT_INTRO_TEXT,
    CONJOINT_OPTIONS,
    CONJOINT_STEM,
    N_RESPONDENTS,
    Q,
    WU_DIR,
    build_survey_audit_artifacts,
    extract_choice_tasks_df,
    extract_respondent_df,
    format_plan_description,
    load_raw_df,
)

WU2025_PARQUET_PATH = WU_DIR / "wu2025_US_long.parquet"
WU2025_AUDIT_CSV_PATH = WU_DIR / "question_audit.csv"
WU2025_OVERRIDES_PATH = WU_DIR / "question_overrides.json"

WU2025_PERSONA_COLS = [c for c in persona.DATASET_PERSONA_COLS["wu2025"] if c != "c_alphan"]
STUDY_DB_TEXTS = {"conjoint_intro": CONJOINT_INTRO_TEXT}
PLAN_ATTRIBUTES = [ATTR_MONEY, ATTR_PERCENT, ATTR_LOCAL]


def build_wu2025_persona_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardized respondent persona table (`N = 1,862`) via `persona.apply_persona_mapping`."""
    resp = extract_respondent_df(raw_df)
    mapped = persona.apply_persona_mapping(resp, "wu2025")
    mapped["CASEID"] = "wu2025_" + mapped["CaseID"].astype(int).astype(str)
    out = mapped[["CASEID", "CaseID", "c_alphan", *WU2025_PERSONA_COLS]].reset_index(drop=True)
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


def _plan_stimulus_ids(tasks_df: pd.DataFrame) -> list[str]:
    """Stable id per displayed pair: `<plan 1>_<plan 2>_<attribute order>` (same screen -> same id)."""
    levels: dict[str, set[str]] = {a: set() for a in PLAN_ATTRIBUTES}
    for r in tasks_df.itertuples(index=False):
        for plan in (r.plan1, r.plan2):
            for a, lvl in zip(r.attributes, plan, strict=True):
                levels[a].add(lvl)
    profile_id = {combo: f"p{i:02d}" for i, combo in
                  enumerate(itertools.product(*[sorted(levels[a]) for a in PLAN_ATTRIBUTES]), start=1)}
    order_id = {perm: f"o{i}" for i, perm in enumerate(itertools.permutations(PLAN_ATTRIBUTES), start=1)}
    ids = []
    for r in tasks_df.itertuples(index=False):
        pos = {a: i for i, a in enumerate(r.attributes)}
        p1 = profile_id[tuple(r.plan1[pos[a]] for a in PLAN_ATTRIBUTES)]
        p2 = profile_id[tuple(r.plan2[pos[a]] for a in PLAN_ATTRIBUTES)]
        ids.append(f"{p1}_{p2}_{order_id[tuple(r.attributes)]}")
    return ids


def build_wu2025_long_df(
    audit_csv_path: Path = WU2025_AUDIT_CSV_PATH,
    overrides_path: Path = WU2025_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Long-format DataFrame across all included questions in `question_audit.csv`."""
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")
    overrides: dict[str, dict] = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )

    raw_df = load_raw_df()
    resp_raw = extract_respondent_df(raw_df)
    resp_persona = build_wu2025_persona_df(raw_df)
    persona_by_case = resp_persona.set_index("CaseID")
    if not (resp_raw["CaseID"].values == resp_persona["CaseID"].values).all():
        raise ValueError("Respondent order differs between raw and persona tables")

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

        if v_name == "plan_choice":
            if q_type != "choice":
                raise ValueError("plan_choice must be exported as kev_type 'choice'")
            if stem_text != CONJOINT_STEM and "stem" not in ov:
                raise ValueError("plan_choice: catalog stem differs from CONJOINT_STEM")
            pairs = [(i + 1, t) for i, t in enumerate(CONJOINT_OPTIONS)]
            kept_codes, kept_texts, dropped = _apply_option_edits(pairs, drop_set, opt_overrides, v_name)
            if dropped:
                raise ValueError("plan_choice: dropping Choice 1 or Choice 2 is not supported")
            tasks_df = extract_choice_tasks_df(raw_df)
            stim_ids = _plan_stimulus_ids(tasks_df)
            sub = persona_by_case.loc[tasks_df["CaseID"]].reset_index()
            sub = sub[["CASEID", "CaseID", "c_alphan", *WU2025_PERSONA_COLS]].copy()
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            sub["v"] = v_name
            sub["qid"] = [f"wu2025:plan_choice:{s}" for s in stim_ids]
            sub["label"] = [f"plan_choice ({s})" for s in stim_ids]
            sub["question"] = full_q
            sub["type"] = q_type
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = [
                [format_plan_description(r.attributes, r.plan1), format_plan_description(r.attributes, r.plan2)]
                for r in tasks_df.itertuples(index=False)
            ]
            sub["answer_code"] = [int(c) for c in tasks_df["choice_code"]]
            sub["answer_text"] = [code_to_text[int(c)] for c in tasks_df["choice_code"]]
            frames.append(sub.reset_index(drop=True))
            continue

        if v_name not in Q or "options" not in Q[v_name] or Q[v_name]["family"].startswith("0_"):
            raise KeyError(f"{v_name}: export supports plan_choice and the catalog's answer-option questions only")
        pairs = [(i + 1, t) for i, t in enumerate(Q[v_name]["options"])]
        kept_codes, kept_texts, dropped = _apply_option_edits(pairs, drop_set, opt_overrides, v_name)
        label_to_code = {t: c for c, t in pairs}
        code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
        answers = resp_raw[v_name]
        unexpected = set(answers[answers != ""]) - set(label_to_code)
        if unexpected:
            raise ValueError(f"{v_name}: answers not in the option list: {unexpected}")
        codes = answers.map(lambda t: label_to_code[t] if t != "" else None)
        valid = codes.notna() & ~codes.isin(dropped)
        sub = resp_persona.loc[valid.values].reset_index(drop=True).copy()
        ans_codes = [int(c) for c in codes[valid]]
        sub["v"] = v_name
        sub["qid"] = f"wu2025:{v_name}"
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


def export_wu2025_parquet(
    parquet_path: Path = WU2025_PARQUET_PATH,
    audit_csv_path: Path = WU2025_AUDIT_CSV_PATH,
    overrides_path: Path = WU2025_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `wu2025_US_long.parquet` (name matches the audit app's `export_<folder>_parquet` call)."""
    long_df = build_wu2025_long_df(audit_csv_path=audit_csv_path, overrides_path=overrides_path)
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_wu2025_df(qids=None, parquet_path=WU2025_PARQUET_PATH) -> pd.DataFrame:
    """Load `wu2025_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"wu2025 parquet not found: {path} (export it from the audit app)")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    df["option_descriptions"] = [list(x) if x is not None else None for x in df["option_descriptions"]]
    df["year"] = "wu2025"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_wu2025_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols=WU2025_PERSONA_COLS,
    parquet_path=WU2025_PARQUET_PATH,
):
    """Respondent-level training records (one per parquet row), optionally state-grouped."""
    df = load_wu2025_df(qids=qids, parquet_path=parquet_path)
    records = build_long_records(df, persona_cols=persona_cols, include_country=include_country, year="wu2025")
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="wu2025")
    return records


def load_wu2025_eval(
    parquet_path=WU2025_PARQUET_PATH,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols=WU2025_PERSONA_COLS,
    grouped: bool = True,
    name: str = "wu2025",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Evaluation bundle (state-grouped when `grouped=True`).

    No dataset-specific evaluator yet (`eval_fn=None`); the paper's benchmark would be its AMCEs / marginal means and
    attribute salience scores.
    """
    df = load_wu2025_df(qids=qids, parquet_path=parquet_path)
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

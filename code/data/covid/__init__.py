"""Kreps et al. (2020) COVID-19 vaccination conjoint & Likert preprocessing, loader, and OLS evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

from data.base import (
    DEFAULT_CACHE_DIR,
    add_one_hot,
    build_eval_set,
    build_long_records,
    filter_by_qids,
    group_records_by_state,
    make_view,
)
from data.covid.stimulus import (
    ATTRIBUTE_SPECS,
    CHOICE_OPTIONS,
    CHOICE_STEM,
    CONJOINT_PAIR_INTRO,
    CONSENT_TEXT,
    COVID_DIR,
    DEMOGRAPHICS_INTRO_TEXT,
    EMPLOYMENT_INTRO_TEXT,
    END_OF_SURVEY_TEXT,
    INTRO_CONJOINT_TEXT,
    LIKERT_OPTIONS_DESC,
    LIKERT_STEM,
    POST_CONJOINT_SPECS,
    PROFILE_ID_TO_TUPLE,
    extract_choice_pairs_df,
    extract_respondent_df,
    format_profile_block_context,
    format_profile_description,
    load_raw_conjoint_df,
)
import data.persona as persona

COVID_PARQUET_PATH = COVID_DIR / "covid_US_long.parquet"
COVID_AUDIT_CSV_PATH = COVID_DIR / "question_audit.csv"
COVID_OVERRIDES_PATH = COVID_DIR / "question_overrides.json"

COVID_PERSONA_COLS = [c for c in persona.DATASET_PERSONA_COLS["covid"] if c != "c_alphan"]

STUDY_DB_TEXTS = {
    "consent_info": CONSENT_TEXT,
    "intro_conjoint": INTRO_CONJOINT_TEXT,
    "employment_intro": EMPLOYMENT_INTRO_TEXT,
    "demographics_intro": DEMOGRAPHICS_INTRO_TEXT,
    "end_of_survey": END_OF_SURVEY_TEXT,
}

AMCE_FORMULA = (
    "y ~ C(efficacy_n) + C(duration_n) + C(majorside_n) + C(minorside_n) "
    "+ C(fda_n) + C(origin_n) + C(endorsed_n)"
)

AMCE_TERMS: list[tuple[str, str]] = [
    ("C(efficacy_n)[T.2]", "Efficacy: 70% vs 50%"),
    ("C(efficacy_n)[T.3]", "Efficacy: 90% vs 50%"),
    ("C(duration_n)[T.2]", "Duration: 5y vs 1y"),
    ("C(majorside_n)[T.2]", "Severe side effects: 1/1M vs 1/10k"),
    ("C(minorside_n)[T.2]", "Mild side effects: 1/30 vs 1/10"),
    ("C(fda_n)[T.2]", "FDA: EUA vs Full approval"),
    ("C(origin_n)[T.2]", "Origin: UK vs US"),
    ("C(origin_n)[T.3]", "Origin: China vs US"),
    ("C(endorsed_n)[T.2]", "Endorsed: Biden vs Trump"),
    ("C(endorsed_n)[T.3]", "Endorsed: CDC vs Trump"),
    ("C(endorsed_n)[T.4]", "Endorsed: WHO vs Trump"),
]


def build_covid_persona_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the standardized respondent persona table (`N = 1,971`) via `persona.apply_persona_mapping`."""
    resp_raw = extract_respondent_df(raw_df)
    mapped = persona.apply_persona_mapping(resp_raw.copy(), "covid")
    cols = ["CASEID", "respondent", "c_alphan", *COVID_PERSONA_COLS]
    out = mapped[cols].copy().reset_index(drop=True)
    if len(out) != 1_971 or out["CASEID"].nunique() != 1_971:
        raise ValueError(f"Expected 1,971 unique respondents in persona table, got {len(out)}")
    return out


def _resolve_context_text(context_source: str, in_block_ctx: str) -> str:
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


def build_covid_long_df(
    raw_df: pd.DataFrame | None = None,
    audit_csv_path: Path = COVID_AUDIT_CSV_PATH,
    overrides_path: Path = COVID_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build the standardized long-format DataFrame across all included questions in `question_audit.csv`."""
    if raw_df is None:
        raw_df = load_raw_conjoint_df()
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")

    overrides: dict[str, dict] = {}
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))

    audit_df = pd.read_csv(audit_csv_path, dtype=str, keep_default_na=False)
    inc_df = audit_df[audit_df["include"] == "True"].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    resp_raw = extract_respondent_df(raw_df)
    resp_persona = build_covid_persona_df(raw_df)
    persona_by_resp = resp_persona.set_index("respondent")

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
        drop_raw = str(a_row["drop_options"])
        drop_set = {s.strip() for s in drop_raw.split("|") if s.strip()}

        ov = overrides[row_key] if row_key in overrides else {}
        stem_override = ov["stem"] if "stem" in ov else None
        opt_overrides: dict[str, str] = ov["options"] if "options" in ov else {}

        if v_name == "vax_choice":
            pairs_df = extract_choice_pairs_df(raw_df)
            kept_codes, kept_texts, dropped_codes = _apply_option_edits(
                CHOICE_OPTIONS, drop_set, opt_overrides, v_name
            )
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            valid_pairs = (
                pairs_df.loc[~pairs_df["choice_code"].isin(dropped_codes)].reset_index(drop=True)
                if dropped_codes
                else pairs_df
            )

            ctx_text = _resolve_context_text(ctx_source, CONJOINT_PAIR_INTRO)
            stem_text = stem_override if stem_override is not None else CHOICE_STEM
            full_q = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

            sub = persona_by_resp.loc[valid_pairs["respondent"]].reset_index()
            sub = sub[["CASEID", "respondent", "c_alphan", *COVID_PERSONA_COLS]].copy()

            descs_col: list[list[str | None]] = []
            for pa, pb in zip(valid_pairs["profile_a"], valid_pairs["profile_b"], strict=True):
                raw_descs_by_code: dict[int, str | None] = {
                    1: format_profile_description(str(pa)),
                    2: format_profile_description(str(pb)),
                    3: None,
                }
                descs_col.append([raw_descs_by_code[c] for c in kept_codes])

            ans_codes = [int(c) for c in valid_pairs["choice_code"]]
            ans_texts: list[str] = []
            for c in ans_codes:
                if c not in code_to_text:
                    raise ValueError(f"vax_choice: unexpected answer code {c!r}")
                ans_texts.append(code_to_text[c])

            sub["v"] = "vax_choice"
            sub["qid"] = "covid:vax_choice:" + valid_pairs["pair_key"].astype(str)
            sub["label"] = "vax_choice (" + valid_pairs["pair_key"].astype(str) + ")"
            sub["question"] = full_q
            sub["type"] = q_type
            sub["reversed"] = False
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = descs_col
            sub["answer_code"] = ans_codes
            sub["answer_text"] = ans_texts
            frames.append(sub.reset_index(drop=True))
            continue

        if v_name == "vax_likert":
            kept_codes, kept_texts, dropped_codes = _apply_option_edits(
                LIKERT_OPTIONS_DESC, drop_set, opt_overrides, v_name
            )
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            vaxord_int = raw_df["vaxord"].astype(int)
            valid_mask = ~vaxord_int.isin(dropped_codes) if dropped_codes else pd.Series(True, index=raw_df.index)
            valid_prof = raw_df.loc[valid_mask].reset_index(drop=True)

            stem_text = stem_override if stem_override is not None else LIKERT_STEM
            q_by_pid: dict[str, str] = {}
            for pid in PROFILE_ID_TO_TUPLE:
                ctx_text = _resolve_context_text(ctx_source, format_profile_block_context(pid))
                q_by_pid[pid] = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

            sub = persona_by_resp.loc[valid_prof["respondent"]].reset_index()
            sub = sub[["CASEID", "respondent", "c_alphan", *COVID_PERSONA_COLS]].copy()

            pids = valid_prof["profile_id"].astype(str).tolist()
            ans_codes = [int(v) for v in valid_prof["vaxord"]]
            ans_texts = []
            for c in ans_codes:
                if c not in code_to_text:
                    raise ValueError(f"vax_likert: unexpected vaxord code {c!r}")
                ans_texts.append(code_to_text[c])

            sub["v"] = "vax_likert"
            sub["qid"] = ["covid:vax_likert:" + pid for pid in pids]
            sub["label"] = ["vax_likert (" + pid + ")" for pid in pids]
            sub["question"] = [q_by_pid[pid] for pid in pids]
            sub["type"] = q_type
            sub["reversed"] = True
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = [None for _ in range(len(sub))]
            sub["answer_code"] = ans_codes
            sub["answer_text"] = ans_texts
            frames.append(sub.reset_index(drop=True))
            continue

        if v_name not in POST_CONJOINT_SPECS:
            raise KeyError(f"Unsupported COVID QuestionID included for export: {v_name!r}")

        spec = POST_CONJOINT_SPECS[v_name]
        kept_codes, kept_texts, dropped_codes = _apply_option_edits(
            spec["options"], drop_set, opt_overrides, v_name
        )
        code_to_text = dict(zip(kept_codes, kept_texts, strict=True))

        col_s = resp_raw[v_name]
        valid_mask = col_s.notna()
        if dropped_codes:
            col_num = pd.to_numeric(col_s, errors="raise")
            valid_mask = valid_mask & (~col_num.isin(dropped_codes))

        sub = resp_persona.loc[valid_mask].reset_index(drop=True).copy()
        ans_raw = pd.to_numeric(col_s.loc[valid_mask], errors="raise")
        ans_codes = []
        ans_texts = []
        for val in ans_raw:
            ival = int(val)
            if ival not in code_to_text or float(ival) != float(val):
                raise ValueError(f"{v_name}: unexpected answer value {val!r}")
            ans_codes.append(ival)
            ans_texts.append(code_to_text[ival])

        ctx_text = _resolve_context_text(ctx_source, spec["in_block_db_text"])
        stem_text = stem_override if stem_override is not None else spec["question_stem"]
        full_q = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

        sub["v"] = v_name
        sub["qid"] = f"covid:{v_name}"
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


def export_covid_parquet(
    parquet_path: Path = COVID_PARQUET_PATH,
    audit_csv_path: Path = COVID_AUDIT_CSV_PATH,
    overrides_path: Path = COVID_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `covid_US_long.parquet`, returning the exported DataFrame."""
    long_df = build_covid_long_df(
        audit_csv_path=audit_csv_path,
        overrides_path=overrides_path,
    )
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_covid_df(qids=None, parquet_path=None) -> pd.DataFrame:
    """Load `covid_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path) if parquet_path is not None else COVID_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(f"COVID parquet not found: {path}")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    if "option_descriptions" in df.columns:
        df["option_descriptions"] = [
            list(x) if x is not None else None for x in df["option_descriptions"]
        ]
    df["year"] = "covid"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_covid_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols=COVID_PERSONA_COLS,
    parquet_path=None,
):
    """Load respondent-level COVID conjoint/Likert records, optionally state-grouped."""
    df = load_covid_df(qids=qids, parquet_path=parquet_path)
    records = build_long_records(
        df,
        persona_cols=persona_cols,
        include_country=include_country,
        year="covid",
    )
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="covid")
    return records


def _profile_attr_dict(profile_id: str) -> dict[str, int]:
    codes = PROFILE_ID_TO_TUPLE[profile_id]
    return {col: code for (col, _, _), code in zip(ATTRIBUTE_SPECS, codes, strict=True)}


def _fit_amce_series(df_obs: pd.DataFrame, y_col: str) -> pd.Series:
    work = df_obs.copy()
    work["y"] = work[y_col]
    fit = smf.ols(AMCE_FORMULA, data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["CASEID"]}
    )
    return pd.Series({label: float(fit.params[term]) for term, label in AMCE_TERMS})


def evaluate_covid_ols(eval_data, preds):
    """Replicate Kreps et al. (2020) Table 3 / Figure 1 respondent-clustered AMCEs (`model` vs. `human`)."""
    records = eval_data["records"] if isinstance(eval_data, dict) else eval_data
    if len(records) != len(preds):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(preds)} predictions")

    likert_code_by_idx = [code for code, _ in LIKERT_OPTIONS_DESC]
    accept_indices = [idx for idx, code in enumerate(likert_code_by_idx) if code >= 5]

    choice_rows: list[dict] = []
    likert_rows: list[dict] = []

    for r, p in zip(records, preds, strict=True):
        v_name = str(r["v"])
        probs = [float(x) for x in p]
        h_dist = [float(x) for x in r["human_dist"]]

        if v_name == "vax_choice":
            if len(probs) != 3 or len(h_dist) != 3:
                raise ValueError(f"{r['qid']}: expected 3 choice probabilities, got {len(probs)}")
            pair_key = str(r["qid"]).split(":")[-1]
            pa, pb = pair_key.split("_")
            choice_rows.append(
                {
                    "CASEID": r["CASEID"],
                    **_profile_attr_dict(pa),
                    "human": h_dist[0],
                    "model": probs[0],
                }
            )
            choice_rows.append(
                {
                    "CASEID": r["CASEID"],
                    **_profile_attr_dict(pb),
                    "human": h_dist[1],
                    "model": probs[1],
                }
            )
        elif v_name == "vax_likert":
            if len(probs) != 7 or len(h_dist) != 7:
                raise ValueError(f"{r['qid']}: expected 7 Likert probabilities, got {len(probs)}")
            pid = str(r["qid"]).split(":")[-1]
            likert_rows.append(
                {
                    "CASEID": r["CASEID"],
                    **_profile_attr_dict(pid),
                    "human": sum(h_dist[i] for i in accept_indices),
                    "model": sum(probs[i] for i in accept_indices),
                }
            )

    if not choice_rows and not likert_rows:
        raise ValueError("No `vax_choice` or `vax_likert` records found in eval_data")

    beta_frames: list[pd.DataFrame] = []
    metrics: dict[str, float] = {}

    for task_label, task_rows in (("choice", choice_rows), ("likert", likert_rows)):
        if not task_rows:
            continue
        t_df = pd.DataFrame(task_rows)
        hb = _fit_amce_series(t_df, "human")
        mb = _fit_amce_series(t_df, "model")
        if (hb == 0).any():
            raise ValueError(f"Zero human AMCE encountered in {task_label} MAPE denominator")
        metrics[f"{task_label} OLS MAPE (%)"] = float(((mb - hb).abs() / hb.abs()).mean() * 100.0)
        metrics[f"{task_label} OLS MAE"] = float((mb - hb).abs().mean())
        metrics[f"{task_label} Spearman rho"] = float(spearmanr(mb, hb).statistic)
        sub_b = pd.DataFrame(
            {
                "task": task_label,
                "effect": hb.index,
                "human": hb.values,
                "model": mb.values,
                "abs_diff": (mb - hb).abs().values,
            }
        )
        beta_frames.append(sub_b)

    beta_df = pd.concat(beta_frames, ignore_index=True)
    if len(beta_frames) == 2:
        hb_all = beta_df["human"]
        mb_all = beta_df["model"]
        metrics["OLS MAPE (%)"] = float(((mb_all - hb_all).abs() / hb_all.abs()).mean() * 100.0)
        metrics["OLS MAE"] = float((mb_all - hb_all).abs().mean())
        metrics["Spearman rho"] = float(spearmanr(mb_all, hb_all).statistic)

    return metrics, beta_df


def load_covid_eval(
    parquet_path=None,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols=COVID_PERSONA_COLS,
    grouped: bool = True,
    name: str = "covid",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Load COVID conjoint/Likert benchmark as a state-grouped evaluation bundle (`grouped=True`) or flat `records`."""
    df = load_covid_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        persona_cols=persona_cols,
        grouped=grouped,
        eval_fn=evaluate_covid_ols if attr_sets is None else None,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )


load_covid = load_covid_eval

"""Grummon et al. (2025) front-of-package nutrition label RCT preprocessing, loader, and OLS evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

from data.base import (
    DATA_DIR,
    DEFAULT_CACHE_DIR,
    add_one_hot,
    build_eval_set,
    build_long_records,
    filter_by_qids,
    group_records_by_state,
    make_view,
)
from data.nutrition.grummon2025.stimulus import (
    ARM_NAMES,
    CONSUMER_UNDERSTANDING_INTRO,
    DEBRIEF_TEXT,
    FOPL_STIMULUS,
    GRUMMON2025_DIR,
    PRODUCT_BASE_LABELS,
    SHOPPING_INTRO_TEXT,
    format_product_label,
    get_arm_variant_spec,
    load_analytic_cohort,
)
import data.persona as persona

GRUMMON2025_PARQUET_PATH = GRUMMON2025_DIR / "grummon2025_US_long.parquet"
GRUMMON2025_AUDIT_CSV_PATH = GRUMMON2025_DIR / "question_audit.csv"
GRUMMON2025_OVERRIDES_PATH = GRUMMON2025_DIR / "question_overrides.json"

GRUMMON2025_PERSONA_NAMES = {
    "c_alphan": "Country",
    "SEX": "Sex",
    "AGE": "Age",
    "EDULEVEL": "Education",
    "US_INC": "Income",
    "RACE": "Race",
    "HHSIZE": "How many people, including yourself, live in your household?",
    "HISPANIC": "Hispanic or Latino",
    "CHILDREN": "How many children (ages 0-18) currently live in your household?",
    "SNAP": "SNAP / Food Stamps recipient",
    "WIC": "Do you or anyone else in your household currently receive WIC benefits?",
    "NFPUSE": "Nutrition Facts label usage",
    "ONLINEAMT": "Share of groceries bought online",
    # Derived in Grummon 2025 (scr_litall = 0..4 sum across 4 NVS ice-cream Nutrition Facts math items, not a single verbatim survey question); may be adapted in the future.
    "HEALTHLIT": "Nutrition literacy score",
}

GRUMMON2025_PERSONA_COLS = [c for c in GRUMMON2025_PERSONA_NAMES if c != "c_alphan"]
PRIMARY_CHOICE_STEMS: tuple[str, ...] = tuple(PRODUCT_BASE_LABELS.keys())

TREAT_ARMS = [
    "NutritionInfo",
    "HighIn",
    "Positive+NutritionInfo",
    "Positive+HighIn",
    "Spectrum",
]

LATINO_MAP = {0: "No", 1: "Yes"}
CHILDREN_MAP = {0: "0", 1: "1", 2: "2", **{k: "3 or more" for k in range(3, 16)}}
SNAP_MAP = {0: "No", 1: "Yes", 2: "Don't know"}
WIC_MAP = {0: "No", 1: "Yes", 2: "Don't know"}
NFPUSE_MAP = {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Most of the time", 5: "Always"}
ONLINE_MAP = {1: "Less than 25%", 2: "25-49%", 3: "50-74%", 4: "75% or more"}
HEALTHLIT_MAP = {i: f"{i} out of 4" for i in range(5)}

STUDY_DB_TEXTS = {
    "shopping_intro": SHOPPING_INTRO_TEXT,
    "consumer_understanding_intro": CONSUMER_UNDERSTANDING_INTRO,
    "debrief": DEBRIEF_TEXT,
}


def strict_code_map(series: pd.Series, mapping: dict[int, str], col_name: str) -> pd.Series:
    """Map integer codes strictly; `NaN` maps to `None`, unexpected values raise."""
    out: list[str | None] = []
    for val in series:
        if pd.isna(val):
            out.append(None)
            continue
        ival = int(val)
        if ival not in mapping or float(ival) != float(val):
            raise ValueError(f"{col_name}: unexpected code {val!r} (valid: {sorted(mapping)})")
        out.append(mapping[ival])
    return pd.Series(out, index=series.index, dtype=object)


def build_grummon2025_persona_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Build the standardized 13-attribute + Country respondent persona table (`N = 5,636`)."""
    resp_df = pd.DataFrame(
        {
            "CASEID": "grummon2025_" + clean_df["participantid"].astype(str),
            "participantid": clean_df["participantid"].astype(str),
            "arm": clean_df["foplcond"].astype(str),
            "c_alphan": persona.map_persona_series(
                pd.Series(["United States"] * len(clean_df), index=clean_df.index),
                "c_alphan",
                "grummon2025",
            ),
            "SEX": persona.map_persona_series(clean_df["gender"], "SEX", "grummon2025"),
            "AGE": persona.map_persona_series(clean_df["age"], "AGE", "grummon2025"),
            "EDULEVEL": persona.map_persona_series(clean_df["educ"], "EDULEVEL", "grummon2025"),
            "US_INC": persona.map_persona_series(clean_df["income"], "US_INC", "grummon2025"),
            "RACE": persona.map_persona_series(clean_df["racecat"], "RACE", "grummon2025"),
            "HHSIZE": persona.map_persona_series(clean_df["hhsize_num"], "HHSIZE", "grummon2025"),
            "HISPANIC": strict_code_map(clean_df["latino"], LATINO_MAP, "latino"),
            "CHILDREN": strict_code_map(clean_df["children_num"], CHILDREN_MAP, "children_num"),
            "SNAP": strict_code_map(clean_df["snap"], SNAP_MAP, "snap"),
            "WIC": strict_code_map(clean_df["wic"], WIC_MAP, "wic"),
            "NFPUSE": strict_code_map(clean_df["nfpuse"], NFPUSE_MAP, "nfpuse"),
            "ONLINEAMT": strict_code_map(clean_df["onlineamt"], ONLINE_MAP, "onlineamt"),
            "HEALTHLIT": strict_code_map(clean_df["scr_litall"], HEALTHLIT_MAP, "scr_litall"),
        },
        index=clean_df.index,
    )
    expected_cols = ["CASEID", "participantid", "arm", "c_alphan", *GRUMMON2025_PERSONA_COLS]
    if list(resp_df.columns) != expected_cols:
        raise ValueError(f"Persona column mismatch: {list(resp_df.columns)} != {expected_cols}")
    return resp_df


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


def build_grummon2025_long_df(
    clean_df: pd.DataFrame | None = None,
    audit_csv_path: Path = GRUMMON2025_AUDIT_CSV_PATH,
    overrides_path: Path = GRUMMON2025_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build the standardized long-format DataFrame across all included outcomes in `question_audit.csv`."""
    if clean_df is None:
        clean_df = load_analytic_cohort()
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")

    overrides: dict[str, dict] = {}
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))

    audit_df = pd.read_csv(audit_csv_path, dtype=str, keep_default_na=False)
    inc_df = audit_df[audit_df["include"] == "True"].reset_index(drop=True)
    if len(inc_df) == 0:
        raise ValueError(f"No included questions found in {audit_csv_path}")

    resp_df = build_grummon2025_persona_df(clean_df)
    frames: list[pd.DataFrame] = []

    for _, a_row in inc_df.iterrows():
        v_name = str(a_row["QuestionID"])
        row_key = f"{a_row['catalog_idx']}:{v_name}"
        q_type = str(a_row["kev_type"]).strip()
        if q_type not in {"choice", "score"}:
            raise ValueError(f"{v_name}: unexpected kev_type {q_type!r}")

        ctx_source = str(a_row["context_source"]).strip() if "context_source" in a_row else ""
        raw_sep = str(a_row["context_sep"]) if "context_sep" in a_row else ""
        ctx_sep = "\n\n" if raw_sep == "" else raw_sep.replace("\\n", "\n")
        drop_raw = str(a_row["drop_options"]) if "drop_options" in a_row else ""
        drop_set = {s.strip() for s in drop_raw.split("|") if s.strip()}

        ov = overrides.get(row_key, {})
        stem_override = ov.get("stem")
        opt_overrides: dict[str, str] = ov.get("options", {})

        arm_specs: dict[str, dict] = {}
        dropped_codes: set[int] = set()
        for arm_name in ARM_NAMES:
            raw_spec = get_arm_variant_spec(v_name, arm_name)
            raw_codes = [int(c) for c in raw_spec["option_codes"]]
            raw_texts = [str(t) for t in raw_spec["option_texts"]]
            raw_descs = raw_spec["option_descriptions"]

            kept_codes: list[int] = []
            kept_texts: list[str] = []
            kept_descs: list[str | None] | None = [] if raw_descs is not None else None
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

            ctx_text = _resolve_context_text(ctx_source, raw_spec["context"])
            stem_text = stem_override if stem_override is not None else raw_spec["question_stem"]
            full_q = f"{ctx_text}{ctx_sep}{stem_text}" if ctx_text else stem_text

            arm_specs[arm_name] = {
                "question": full_q,
                "option_codes": kept_codes,
                "option_texts": kept_texts,
                "option_descriptions": kept_descs,
                "code_to_text": dict(zip(kept_codes, kept_texts, strict=True)),
            }

        valid_mask = clean_df[v_name].notna()
        if dropped_codes:
            col_num = pd.to_numeric(clean_df[v_name], errors="raise")
            valid_mask = valid_mask & (~col_num.isin(dropped_codes))

        sub = resp_df.loc[valid_mask].copy()
        ans_raw = pd.to_numeric(clean_df.loc[valid_mask, v_name], errors="raise")
        arms = sub["arm"].tolist()

        q_col: list[str] = []
        codes_col: list[list[int]] = []
        texts_col: list[list[str]] = []
        descs_col: list[list[str | None] | None] = []
        ans_codes: list[int] = []
        ans_texts: list[str] = []

        for arm_name, val in zip(arms, ans_raw, strict=True):
            sp = arm_specs[arm_name]
            ival = int(val)
            if ival not in sp["code_to_text"] or float(ival) != float(val):
                raise ValueError(f"{v_name} ({arm_name}): unexpected answer value {val!r}")
            q_col.append(sp["question"])
            codes_col.append(list(sp["option_codes"]))
            texts_col.append(list(sp["option_texts"]))
            descs_col.append(
                list(sp["option_descriptions"]) if sp["option_descriptions"] is not None else None
            )
            ans_codes.append(ival)
            ans_texts.append(sp["code_to_text"][ival])

        sub["v"] = v_name
        sub["qid"] = "grummon2025:" + sub["arm"] + ":" + v_name
        sub["label"] = sub["arm"] + ": " + v_name
        sub["question"] = q_col
        sub["type"] = q_type
        sub["reversed"] = False
        sub["option_codes"] = codes_col
        sub["option_texts"] = texts_col
        sub["option_descriptions"] = descs_col
        sub["answer_code"] = ans_codes
        sub["answer_text"] = ans_texts
        frames.append(sub.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def export_grummon2025_parquet(
    parquet_path: Path = GRUMMON2025_PARQUET_PATH,
    audit_csv_path: Path = GRUMMON2025_AUDIT_CSV_PATH,
    overrides_path: Path = GRUMMON2025_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `grummon2025_US_long.parquet`, returning the exported DataFrame."""
    long_df = build_grummon2025_long_df(
        audit_csv_path=audit_csv_path,
        overrides_path=overrides_path,
    )
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_grummon2025_df(qids=None, parquet_path=None) -> pd.DataFrame:
    """Load `grummon2025_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path) if parquet_path is not None else GRUMMON2025_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Grummon 2025 parquet not found: {path}")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    if "option_descriptions" in df.columns:
        df["option_descriptions"] = [
            list(x) if x is not None else None for x in df["option_descriptions"]
        ]
    df["year"] = "grummon2025"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_grummon2025_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols=GRUMMON2025_PERSONA_COLS,
    parquet_path=None,
):
    """Load respondent-level Grummon 2025 records (preserving `option_descriptions`), optionally state-grouped."""
    df = load_grummon2025_df(qids=qids, parquet_path=parquet_path)
    records = build_long_records(
        df,
        persona_cols=persona_cols,
        include_country=include_country,
        year="grummon2025",
    )
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="grummon2025")
    return records


def load_grummon2025_eval(
    parquet_path=None,
    *,
    attr_sets,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols=GRUMMON2025_PERSONA_COLS,
    grouped: bool = True,
    name: str = "grummon2025",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Load Grummon 2025 RCT test set as a state-grouped evaluation bundle (`grouped=True`) or flat `records` (`grouped=False`)."""
    df = load_grummon2025_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        persona_cols=persona_cols,
        grouped=grouped,
        eval_fn=evaluate_grummon2025_ols if attr_sets is None else None,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )


load_grummon2025 = load_grummon2025_eval


def _record_stem(rec: dict) -> str:
    if "v" in rec and rec["v"]:
        return str(rec["v"])
    return str(rec["qid"]).rsplit(":", 1)[-1]


def _expected_outcome_value(rec: dict, dist) -> float:
    stem = _record_stem(rec)
    q_type = str(rec["type"])
    probs = [float(x) for x in dist]
    if stem in PRIMARY_CHOICE_STEMS:
        if len(probs) != 3:
            raise ValueError(f"{rec['qid']}: expected 3 options on primary choice item, got {len(probs)}")
        return probs[0]
    if q_type == "score":
        return float(sum((idx + 1) * p for idx, p in enumerate(probs)))
    opts_list = [str(t) for t in rec["option_texts"]]
    if q_type == "choice" and "Yes" in opts_list:
        return probs[opts_list.index("Yes")]
    raise ValueError(
        f"{rec['qid']}: no scalar OLS outcome mapping defined for type={q_type!r}, options={rec['option_texts']!r}"
    )


def evaluate_grummon2025_ols(eval_data, preds):
    """Fit participant-level HC1 OLS regressions vs. Human RCT on the 6 primary `id*` choice items."""
    records = eval_data["records"] if isinstance(eval_data, dict) else eval_data
    if len(records) != len(preds):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(preds)} predictions")

    rows = []
    for r, p in zip(records, preds, strict=True):
        if _record_stem(r) not in PRIMARY_CHOICE_STEMS:
            continue
        rows.append(
            {
                "CASEID": r["CASEID"],
                "arm": r["arm"],
                "human": _expected_outcome_value(r, r["human_dist"]),
                "model": _expected_outcome_value(r, p),
            }
        )
    if not rows:
        raise ValueError(f"No primary choice records ({PRIMARY_CHOICE_STEMS}) found in eval_data")

    ols_df = pd.DataFrame(rows)
    part_df = ols_df.groupby("CASEID", as_index=False).agg(
        arm=("arm", "first"),
        human=("human", "mean"),
        model=("model", "mean"),
    )

    fit_h = smf.ols("human ~ C(arm, Treatment(reference='Positive'))", data=part_df).fit(cov_type="HC1")
    fit_m = smf.ols("model ~ C(arm, Treatment(reference='Positive'))", data=part_df).fit(cov_type="HC1")

    hb = pd.Series({a: float(fit_h.params[f"C(arm, Treatment(reference='Positive'))[T.{a}]"]) for a in TREAT_ARMS})
    mb = pd.Series({a: float(fit_m.params[f"C(arm, Treatment(reference='Positive'))[T.{a}]"]) for a in TREAT_ARMS})
    if (hb == 0).any():
        raise ValueError("Zero human beta encountered in MAPE denominator")

    ols_metrics = {
        "OLS MAPE (%)": float(((mb - hb).abs() / hb.abs()).mean() * 100.0),
        "OLS MSE": float(((mb - hb) ** 2).mean()),
        "OLS MAE": float((mb - hb).abs().mean()),
        "Spearman rho": float(spearmanr(mb, hb).statistic),
    }
    beta_df = pd.DataFrame({
        "human": hb,
        "model": mb,
        "abs_diff": (mb - hb).abs(),
    }).sort_values("human", ascending=False)
    return ols_metrics, beta_df


def evaluate_grummon2025_secondary_ols(eval_data, preds) -> dict[str, pd.DataFrame]:
    """Fit per-outcome HC1 OLS treatment effects (`model` vs. `human`) for every secondary `score`/binary item present."""
    records = eval_data["records"] if isinstance(eval_data, dict) else eval_data
    if len(records) != len(preds):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(preds)} predictions")

    by_stem: dict[str, list[dict]] = {}
    for r, p in zip(records, preds, strict=True):
        stem = _record_stem(r)
        if stem in PRIMARY_CHOICE_STEMS:
            continue
        by_stem.setdefault(stem, []).append(
            {
                "CASEID": r["CASEID"],
                "arm": r["arm"],
                "human": _expected_outcome_value(r, r["human_dist"]),
                "model": _expected_outcome_value(r, p),
            }
        )

    out: dict[str, pd.DataFrame] = {}
    for stem, s_rows in by_stem.items():
        part_df = pd.DataFrame(s_rows).groupby("CASEID", as_index=False).agg(
            arm=("arm", "first"),
            human=("human", "mean"),
            model=("model", "mean"),
        )
        fit_h = smf.ols("human ~ C(arm, Treatment(reference='Positive'))", data=part_df).fit(cov_type="HC1")
        fit_m = smf.ols("model ~ C(arm, Treatment(reference='Positive'))", data=part_df).fit(cov_type="HC1")
        hb = pd.Series({a: float(fit_h.params[f"C(arm, Treatment(reference='Positive'))[T.{a}]"]) for a in TREAT_ARMS})
        mb = pd.Series({a: float(fit_m.params[f"C(arm, Treatment(reference='Positive'))[T.{a}]"]) for a in TREAT_ARMS})
        out[stem] = pd.DataFrame(
            {
                "human": hb,
                "model": mb,
                "abs_diff": (mb - hb).abs(),
            }
        )
    return out


"""Kriner2026 (Goldfarb, Kriner et al. 2026, SAF survey) long-format parquet exporter.

Exports every row marked `include=True` in `question_audit.csv` to `kriner2026_US_long.parquet`, applying the audit
UI's `drop_options`, `context_source`, `context_sep`, and `question_overrides.json` edits.
Persona decisions: `DECISIONS.md` and `persona.py` (dataset key `kriner2026`).
"""

from __future__ import annotations

import itertools
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
import data.persona as persona
from data.Kriner2026.stimulus import (
    CHOICE_OPTIONS,
    CHOICE_STEM,
    CONJOINT_INTRO_TEXT,
    KRINER_DIR,
    N_RESPONDENTS,
    SAF_DEFINITION_TEXT,
    build_persona_input_df,
    build_survey_audit_artifacts,
    extract_choice_tasks_df,
    extract_respondent_df,
    format_flight_description,
    load_raw_df,
)

KRINER2026_PARQUET_PATH = KRINER_DIR / "kriner2026_US_long.parquet"
KRINER2026_AUDIT_CSV_PATH = KRINER_DIR / "question_audit.csv"
KRINER2026_OVERRIDES_PATH = KRINER_DIR / "question_overrides.json"

KRINER2026_PERSONA_COLS = [c for c in persona.DATASET_PERSONA_COLS["kriner2026"] if c != "c_alphan"]

STUDY_DB_TEXTS = {
    "saf_definition": SAF_DEFINITION_TEXT,
    "saf_conjoint_intro": CONJOINT_INTRO_TEXT,
}

# `reversed` for included `score` items: True when the first listed option is the high end of the scale
# (Covid convention, e.g. `vax_likert`). Every included score item must be listed here; export fails otherwise.
SCORE_REVERSED = {  # values confirmed by the user, 2026-11-14
    "saf_familiar_old": True,  # Very informed ... Not at all informed
    "saf_safe_old": True,  # Strongly agree ... Strongly disagree
    "sci_know_self": False,  # Not at All Knowledgeable ... Very Knowledgeable
}

FLIGHT_ATTRIBUTES = ["Airline", "Fuel type", "Included baggage", "Legroom", "Price", "Seat type"]

# On-screen level text -> the authors' Stata codes (`*_n` columns; code 1 = Table 2 baseline).
FLIGHT_LEVEL_CODES: dict[str, tuple[str, dict[str, int]]] = {
    "Airline": ("airline_n", {"Delta": 1, "Frontier": 2, "Southwest": 3, "United": 4}),
    "Fuel type": (
        "fuel_type_n",
        {
            "Conventional aviation fuel produced from petroleum": 1,
            "Sutainable aviation fuel produced from biomass": 2,
        },
    ),
    "Included baggage": (
        "baggage_n",
        {
            "Carry-on bag and personal item": 1,
            "Checked bag, carry-on bag, and personal item": 2,
            "Personal item only": 3,
        },
    ),
    "Legroom": ("legroom_n", {"Standard legroom (31 inches)": 1, "Extra legroom (33 inches)": 2}),
    "Price": ("price_n", {"$250": 1, "$275": 2, "$300": 3, "$325": 4}),
    "Seat type": ("seat_n", {"Aisle": 1, "Middle": 2, "Window": 3}),
}

# Goldfarb, Kriner et al. (2026) Table 2: OLS of choose_flight on attribute indicators, clustered by respondent.
AMCE_FORMULA = "y ~ C(airline_n) + C(fuel_type_n) + C(baggage_n) + C(legroom_n) + C(seat_n) + C(price_n)"
AMCE_TERMS: list[tuple[str, str]] = [
    ("C(airline_n)[T.2]", "Airline: Frontier"),
    ("C(airline_n)[T.3]", "Airline: Southwest"),
    ("C(airline_n)[T.4]", "Airline: United"),
    ("C(fuel_type_n)[T.2]", "Fuel: Sustainable aviation fuel"),
    ("C(baggage_n)[T.2]", "Baggage: Checked bag+carry-on"),
    ("C(baggage_n)[T.3]", "Baggage: Personal item only"),
    ("C(legroom_n)[T.2]", "Space: Extra legroom"),
    ("C(seat_n)[T.2]", "Seat: Middle"),
    ("C(seat_n)[T.3]", "Seat: Window"),
    ("C(price_n)[T.2]", "Price: $275"),
    ("C(price_n)[T.3]", "Price: $300"),
    ("C(price_n)[T.4]", "Price: $325"),
]


def build_kriner2026_persona_df(raw_df: pd.DataFrame, value_labels: dict[str, dict[int, str]]) -> pd.DataFrame:
    """Standardized respondent persona table (`N = 1,000`) via `persona.apply_persona_mapping`."""
    mapped = persona.apply_persona_mapping(build_persona_input_df(raw_df, value_labels), "kriner2026")
    mapped["CASEID"] = "kriner2026_" + mapped["respondent"].astype(str)
    out = mapped[["CASEID", "respondent", "c_alphan", *KRINER2026_PERSONA_COLS]].reset_index(drop=True)
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


def _flight_stimulus_ids(tasks_df: pd.DataFrame) -> list[str]:
    """Stable id per displayed pair: `<profile A>_<profile B>_<attribute order>` (same screen -> same id)."""
    levels: dict[str, set[str]] = {a: set() for a in FLIGHT_ATTRIBUTES}
    for r in tasks_df.itertuples(index=False):
        for flight in (r.flight1, r.flight2):
            for a, lvl in zip(r.attributes, flight, strict=True):
                levels[a].add(lvl)
    level_lists = [sorted(levels[a]) for a in FLIGHT_ATTRIBUTES]
    profile_id = {combo: f"p{i:03d}" for i, combo in enumerate(itertools.product(*level_lists), start=1)}
    order_id = {perm: f"o{i:03d}" for i, perm in enumerate(itertools.permutations(FLIGHT_ATTRIBUTES), start=1)}
    ids = []
    for r in tasks_df.itertuples(index=False):
        pos = {a: i for i, a in enumerate(r.attributes)}
        pa = profile_id[tuple(r.flight1[pos[a]] for a in FLIGHT_ATTRIBUTES)]
        pb = profile_id[tuple(r.flight2[pos[a]] for a in FLIGHT_ATTRIBUTES)]
        ids.append(f"{pa}_{pb}_{order_id[tuple(r.attributes)]}")
    return ids


def build_kriner2026_long_df(
    audit_csv_path: Path = KRINER2026_AUDIT_CSV_PATH,
    overrides_path: Path = KRINER2026_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Long-format DataFrame across all included questions in `question_audit.csv`."""
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")
    overrides: dict[str, dict] = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )

    raw_df, _, value_labels = load_raw_df()
    resp_raw = extract_respondent_df(raw_df)
    resp_persona = build_kriner2026_persona_df(raw_df, value_labels)
    persona_by_resp = resp_persona.set_index("respondent")
    if not (resp_raw["respondent"].values == resp_persona["respondent"].values).all():
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

        if v_name == "flight_choice":
            if q_type != "choice":
                raise ValueError("flight_choice must be exported as kev_type 'choice'")
            tasks_df = extract_choice_tasks_df(raw_df)
            kept_codes, kept_texts, dropped = _apply_option_edits(CHOICE_OPTIONS, drop_set, opt_overrides, v_name)
            if dropped:
                raise ValueError("flight_choice: dropping Flight 1 or Flight 2 is not supported")
            stim_ids = _flight_stimulus_ids(tasks_df)
            sub = persona_by_resp.loc[tasks_df["respondent"]].reset_index()
            sub = sub[["CASEID", "respondent", "c_alphan", *KRINER2026_PERSONA_COLS]].copy()
            code_to_text = dict(zip(kept_codes, kept_texts, strict=True))
            sub["v"] = v_name
            sub["qid"] = [f"kriner2026:flight_choice:{s}" for s in stim_ids]
            sub["label"] = [f"flight_choice ({s})" for s in stim_ids]
            sub["question"] = full_q
            sub["type"] = q_type
            sub["reversed"] = False
            sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
            sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
            sub["option_descriptions"] = [
                [format_flight_description(r.attributes, r.flight1), format_flight_description(r.attributes, r.flight2)]
                for r in tasks_df.itertuples(index=False)
            ]
            sub["answer_code"] = [int(c) for c in tasks_df["choice_code"]]
            sub["answer_text"] = [code_to_text[int(c)] for c in tasks_df["choice_code"]]
            frames.append(sub.reset_index(drop=True))
            continue

        if v_name not in value_labels or v_name not in resp_raw.columns:
            raise KeyError(f"{v_name}: export supports the flight choice and single-variable labelled questions only")
        if q_type == "score" and v_name not in SCORE_REVERSED:
            raise KeyError(f"{v_name}: included as score but has no entry in SCORE_REVERSED")
        labels = value_labels[v_name]
        raw_pairs = [(c, labels[c]) for c in sorted(labels)]
        kept_codes, kept_texts, dropped = _apply_option_edits(raw_pairs, drop_set, opt_overrides, v_name)
        code_to_text = dict(zip(kept_codes, kept_texts, strict=True))

        ans = pd.to_numeric(resp_raw[v_name], errors="raise")
        valid = ans.notna() & ~ans.isin(dropped)
        sub = resp_persona.loc[valid].reset_index(drop=True).copy()
        ans_codes = []
        for val in ans[valid]:
            ival = int(val)
            if float(ival) != float(val) or ival not in code_to_text:
                raise ValueError(f"{v_name}: unexpected answer value {val!r}")
            ans_codes.append(ival)
        sub["v"] = v_name
        sub["qid"] = f"kriner2026:{v_name}"
        sub["label"] = v_name
        sub["question"] = full_q
        sub["type"] = q_type
        sub["reversed"] = SCORE_REVERSED[v_name] if q_type == "score" else False
        sub["option_codes"] = [list(kept_codes) for _ in range(len(sub))]
        sub["option_texts"] = [list(kept_texts) for _ in range(len(sub))]
        sub["option_descriptions"] = [None for _ in range(len(sub))]
        sub["answer_code"] = ans_codes
        sub["answer_text"] = [code_to_text[c] for c in ans_codes]
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def export_Kriner2026_parquet(
    parquet_path: Path = KRINER2026_PARQUET_PATH,
    audit_csv_path: Path = KRINER2026_AUDIT_CSV_PATH,
    overrides_path: Path = KRINER2026_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Build and write `kriner2026_US_long.parquet` (name matches the audit app's `export_<folder>_parquet` call)."""
    long_df = build_kriner2026_long_df(audit_csv_path=audit_csv_path, overrides_path=overrides_path)
    long_df.to_parquet(parquet_path, index=False)
    return long_df


def load_kriner2026_df(qids=None, parquet_path=KRINER2026_PARQUET_PATH) -> pd.DataFrame:
    """Load `kriner2026_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Kriner2026 parquet not found: {path} (export it from the audit app)")
    df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    df["option_descriptions"] = [list(x) if x is not None else None for x in df["option_descriptions"]]
    df["year"] = "kriner2026"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_kriner2026_records(
    *,
    include_country: bool,
    qids=None,
    grouped: bool = False,
    persona_cols=KRINER2026_PERSONA_COLS,
    parquet_path=KRINER2026_PARQUET_PATH,
):
    """Respondent-level training records (one per parquet row), optionally state-grouped."""
    df = load_kriner2026_df(qids=qids, parquet_path=parquet_path)
    records = build_long_records(df, persona_cols=persona_cols, include_country=include_country, year="kriner2026")
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="kriner2026")
    return records


def parse_flight_description(description: str) -> dict[str, int]:
    """`'Airline: Delta; Price: $250; ...'` (display order) -> `{'airline_n': 1, 'price_n': 1, ...}`."""
    parts = description.split("; ")
    if len(parts) != len(FLIGHT_ATTRIBUTES):
        raise ValueError(f"Expected {len(FLIGHT_ATTRIBUTES)} attribute rows, got {len(parts)}: {description!r}")
    out: dict[str, int] = {}
    for part in parts:
        attr, level = part.split(": ", 1)
        col, codes = FLIGHT_LEVEL_CODES[attr]
        if col in out:
            raise ValueError(f"Duplicate attribute {attr!r} in {description!r}")
        out[col] = codes[level]
    return out


def _fit_amce_series(df_obs: pd.DataFrame, y_col: str) -> pd.Series:
    work = df_obs.copy()
    work["y"] = work[y_col]
    fit = smf.ols(AMCE_FORMULA, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["CASEID"]})
    return pd.Series({label: float(fit.params[term]) for term, label in AMCE_TERMS})


def evaluate_kriner2026_ols(eval_data, preds):
    """Replicate Table 2 AMCEs on human choices vs. model probabilities (one row per displayed flight).

    Each `flight_choice` record contributes two rows (Flight 1, Flight 2) with the flight's attribute codes,
    `human` = its one-hot choice and `model` = the model's probability for it. Other included questions are ignored.
    """
    records = eval_data["records"] if isinstance(eval_data, dict) else eval_data
    if len(records) != len(preds):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(preds)} predictions")

    rows: list[dict] = []
    for r, p in zip(records, preds, strict=True):
        if r["v"] != "flight_choice":
            continue
        probs = [float(x) for x in p]
        h_dist = [float(x) for x in r["human_dist"]]
        descs = r["option_descriptions"]
        if len(probs) != 2 or len(h_dist) != 2 or descs is None or len(descs) != 2:
            raise ValueError(f"{r['qid']}: expected 2 flights with descriptions, got {len(probs)} probabilities")
        for k in (0, 1):
            rows.append({"CASEID": r["CASEID"], **parse_flight_description(descs[k]), "human": h_dist[k], "model": probs[k]})
    if not rows:
        raise ValueError("No `flight_choice` records found in eval_data")

    obs = pd.DataFrame(rows)
    hb = _fit_amce_series(obs, "human")
    mb = _fit_amce_series(obs, "model")
    if (hb == 0).any():
        raise ValueError("Zero human AMCE encountered in MAPE denominator")
    metrics = {
        "OLS MAPE (%)": float(((mb - hb).abs() / hb.abs()).mean() * 100.0),
        "OLS MAE": float((mb - hb).abs().mean()),
        "Spearman rho": float(spearmanr(mb, hb).statistic),
    }
    beta_df = pd.DataFrame(
        {"effect": hb.index, "human": hb.values, "model": mb.values, "abs_diff": (mb - hb).abs().values}
    )
    return metrics, beta_df


def load_kriner2026_eval(
    parquet_path=KRINER2026_PARQUET_PATH,
    *,
    attr_sets=None,
    include_country: bool,
    min_n=None,
    qids=None,
    persona_cols=KRINER2026_PERSONA_COLS,
    grouped: bool = True,
    name: str = "kriner2026",
    valid: bool = False,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Evaluation bundle (state-grouped when `grouped=True`); respondent-level runs use the Table 2 AMCE evaluator."""
    df = load_kriner2026_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        persona_cols=persona_cols,
        grouped=grouped,
        eval_fn=evaluate_kriner2026_ols if attr_sets is None else None,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )

"""Grummon et al. (2025) front-of-package nutrition label RCT test-set loader and OLS evaluator."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

from data.persona import PERSONA_NAMES

GRUMMON_PARQUET_PATH = (
    Path(__file__).resolve().parent
    / "nutrition"
    / "Nutrition Info and Other Front-of-Package Labels and Simulated Food and Beverage Purchases"
    / "grummon_US_long.parquet"
)

GRUMMON_PERSONA_NAMES = {
    "c_alphan": "Country",
    "SEX": "Sex",
    "AGE": "Age",
    "EDULEVEL": "Education",
    "US_INC": "Income",
    "RACE": "Race",
    "HISPANIC": "Hispanic or Latino",
    "SNAP": "SNAP / Food Stamps recipient",
    "NFPUSE": "Nutrition Facts label usage",
    "ONLINEAMT": "Share of groceries bought online",
    "HEALTHLIT": "Nutrition literacy score",
}

ALL_PERSONA_NAMES = {**PERSONA_NAMES, **GRUMMON_PERSONA_NAMES}

GRUMMON_PERSONA_COLS = [c for c in GRUMMON_PERSONA_NAMES if c != "c_alphan"]

TREAT_ARMS = [
    "NutritionInfo",
    "HighIn",
    "Positive+NutritionInfo",
    "Positive+HighIn",
    "Spectrum",
]


def add_one_hot(df):
    """Attach one-hot human response vectors `one_hot` to a long-format survey DataFrame."""
    one_hots = []
    for qid, codes, texts, ans in zip(
        df["qid"], df["option_codes"], df["option_texts"], df["answer_code"], strict=True
    ):
        codes_list = [int(c) for c in codes]
        if len(codes_list) != len(texts):
            raise ValueError(f"{qid}: option_codes ({len(codes_list)}) != option_texts ({len(texts)})")
        vec = np.zeros(len(codes_list), dtype=np.float64)
        vec[codes_list.index(int(ans))] = 1.0
        one_hots.append(vec)
    df["one_hot"] = one_hots


def build_long_records(long_df, persona_cols=GRUMMON_PERSONA_COLS, persona_names=GRUMMON_PERSONA_NAMES):
    """Convert each row of a *_long.parquet DataFrame into a record dict compatible with evaluate_records()."""
    if "c_alphan" not in long_df.columns:
        raise KeyError("Expected 'c_alphan' column in long_df")
    if "one_hot" not in long_df.columns:
        raise KeyError("Expected 'one_hot' column in long_df; call add_one_hot(long_df) first")
    has_descs = "option_descriptions" in long_df.columns

    records = []
    for row in long_df.to_dict("records"):
        personas = {c: row[c] for c in persona_cols if c in row and pd.notna(row[c])}
        opt_descs = (
            list(row["option_descriptions"])
            if has_descs and row["option_descriptions"] is not None
            else None
        )
        records.append({
            "CASEID": row["CASEID"],
            "arm": row["arm"],
            "qid": row["qid"],
            "year": "grummon",
            "v": row["v"],
            "type": row["type"],
            "question": row["question"],
            "option_texts": list(row["option_texts"]),
            "option_descriptions": opt_descs,
            "country": row["c_alphan"],
            "personas": personas,
            "attribute": "+".join(personas.keys()) if personas else "none",
            "persona": ", ".join(f"{persona_names[c]}={v}" for c, v in personas.items()) if personas else "All",
            "n": 1,
            "human_dist": row["one_hot"],
        })
    return records


def make_long_view(rec, rng=None, persona_names=ALL_PERSONA_NAMES):
    """Build a single-question Kev view from a record dict (supports optional option_descriptions)."""
    qid = rec["qid"]
    q_type = rec["type"]
    option_texts = rec["option_texts"]
    option_descs = rec["option_descriptions"] if "option_descriptions" in rec else None
    human_dist = rec["human_dist"]

    items = list(rec["personas"].items())
    if rng is not None and len(items) > 1:
        rng.shuffle(items)

    state_lines = [f"{persona_names['c_alphan']}: {rec['country']}"]
    state_lines.extend(f"{persona_names[c]}: {v}" for c, v in items)

    if q_type == "score":
        criteria = option_texts
        target = {str(i): float(p) for i, p in enumerate(human_dist)}
    elif q_type == "choice":
        if option_descs is not None:
            criteria = dict(zip(option_texts, option_descs, strict=True))
        else:
            criteria = {text: None for text in option_texts}
        target = {text: float(p) for text, p in zip(option_texts, human_dist)}
    else:
        raise ValueError(f"{qid}: unexpected question type {q_type!r}")

    return {
        "state": "\n".join(state_lines),
        "questions": {
            qid: {
                "type": q_type,
                "instructions": rec["question"],
                "criteria": criteria,
                "target": target,
            }
        },
    }


def pack_views_by_state(views):
    """Deduplicate (state, qid) pairs and pack all questions sharing a state into one multi-question view."""
    state_to_idx = {}
    packed_views = []
    qid_to_pos = []
    unpack_idx = []

    for v in views:
        state = v["state"]
        if state not in state_to_idx:
            state_to_idx[state] = len(packed_views)
            packed_views.append({"state": state, "questions": {}})
            qid_to_pos.append({})
        s_idx = state_to_idx[state]
        s_questions = packed_views[s_idx]["questions"]
        s_qpos = qid_to_pos[s_idx]

        row_map = []
        for qid, q_spec in v["questions"].items():
            if qid not in s_qpos:
                s_qpos[qid] = len(s_questions)
                s_questions[qid] = q_spec
            else:
                prev = s_questions[qid]
                if (prev["type"], prev["instructions"], prev["criteria"]) != (
                    q_spec["type"], q_spec["instructions"], q_spec["criteria"]
                ):
                    raise ValueError(f"Conflicting question definition for qid={qid!r}")
            row_map.append((s_idx, s_qpos[qid]))
        unpack_idx.append(row_map)

    return packed_views, unpack_idx


def group_records_by_state(records, view_fn=make_long_view, eval_fn=None, name=None):
    """Create the state-grouped evaluation bundle ONCE so it can be reused across validation steps."""
    views = [view_fn(r) for r in records]
    packed_views, unpack_idx = pack_views_by_state(views)
    return {
        "name": name,
        "records": records,
        "packed_views": packed_views,
        "unpack_idx": unpack_idx,
        "encs": None,
        "last_stats": None,
        "eval_fn": eval_fn,
    }


def load_grummon(parquet_path=None, persona_cols=GRUMMON_PERSONA_COLS, grouped=True):
    """Load Grummon RCT test set as a state-grouped evaluation bundle (`grouped=True`) or flat `records` (`grouped=False`)."""
    path = Path(parquet_path) if parquet_path is not None else GRUMMON_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Grummon parquet not found: {path}")
    df = pd.read_parquet(path)
    add_one_hot(df)
    records = build_long_records(df, persona_cols=persona_cols)
    if grouped:
        return group_records_by_state(
            records, view_fn=make_long_view, eval_fn=evaluate_grummon_ols, name="grummon"
        )
    return records


def evaluate_grummon_ols(eval_data, preds):
    """Fit participant-level HC1 OLS regressions vs. Human RCT on flat `preds` from Grummon."""
    records = eval_data["records"] if isinstance(eval_data, dict) else eval_data
    if len(records) != len(preds):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(preds)} predictions")

    ols_df = pd.DataFrame({
        "CASEID": [r["CASEID"] for r in records],
        "arm": [r["arm"] for r in records],
        "human": [float(r["human_dist"][0]) for r in records],
        "model": [float(p[0]) for p in preds],
    })
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
    })
    return ols_metrics, beta_df

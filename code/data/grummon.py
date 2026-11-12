"""Grummon et al. (2025) front-of-package nutrition label RCT test-set loader and OLS evaluator."""

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

from data.base import (
    DATA_DIR,
    add_one_hot,
    build_eval_set,
    filter_by_qids,
)

GRUMMON_PARQUET_PATH = (
    DATA_DIR
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
    "HHSIZE": "How many people, including yourself, live in your household?",
    "HISPANIC": "Hispanic or Latino",
    "SNAP": "SNAP / Food Stamps recipient",
    "NFPUSE": "Nutrition Facts label usage",
    "ONLINEAMT": "Share of groceries bought online",
    # Derived in Grummon 2025 (scr_litall = 0..4 sum across 4 NVS ice-cream Nutrition Facts math items, not a single verbatim survey question); may be adapted in the future.
    "HEALTHLIT": "Nutrition literacy score",
}

GRUMMON_PERSONA_COLS = [c for c in GRUMMON_PERSONA_NAMES if c != "c_alphan"]

TREAT_ARMS = [
    "NutritionInfo",
    "HighIn",
    "Positive+NutritionInfo",
    "Positive+HighIn",
    "Spectrum",
]


def load_grummon_df(qids=None, parquet_path=None):
    """Load `grummon_US_long.parquet`, attach `year` and `one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path) if parquet_path is not None else GRUMMON_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Grummon parquet not found: {path}")
    df = pd.read_parquet(path)
    df["year"] = "grummon"
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_grummon(
    parquet_path=None,
    *,
    attr_sets,
    include_country,
    min_n=None,
    qids=None,
    persona_cols=GRUMMON_PERSONA_COLS,
    grouped=True,
    valid=False,
):
    """Load Grummon RCT test set as a state-grouped evaluation bundle (`grouped=True`) or flat `records` (`grouped=False`)."""
    df = load_grummon_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        persona_cols=persona_cols,
        grouped=grouped,
        eval_fn=evaluate_grummon_ols if attr_sets is None else None,
        name="grummon",
        valid=valid,
    )


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
    }).sort_values("human", ascending=False)
    return ols_metrics, beta_df


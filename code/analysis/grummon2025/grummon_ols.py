"""HC1 OLS treatment-effect estimation, dynamic Kev dataset loader, and plotting utilities for Grummon et al. (2025)."""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from cycler import cycler

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.grummon import (
    GRUMMON_PARQUET_PATH,
    TREAT_ARMS,
    load_grummon_df,
)
from kev_eval import load_kev_jsonl

# Standard six-color academic palettes and neutral structural tones from code/plot.md
PALETTES: dict[str, list[str]] = {
    "nature": ["#1F3B5B", "#B84A39", "#2A7B6F", "#D4912A", "#6A4C7D", "#5C768D"],
    "lancet": ["#1B365D", "#9E2A2B", "#2E8B78", "#D9822B", "#6F5091", "#4A6F8A"],
}

MODEL_PALETTES: dict[str, dict[str, str]] = {
    "nature": {
        "human": "#1F3B5B",
        "off_the_shelf": "#B84A39",
        "fine_tuned": "#2A7B6F",
    },
    "lancet": {
        "human": "#1B365D",
        "off_the_shelf": "#9E2A2B",
        "fine_tuned": "#2E8B78",
    },
}

NEUTRALS: dict[str, str] = {
    "text": "#2B2B2B",
    "muted": "#52514E",
    "ref_line": "#6E6A64",
    "baseline": "#9A9893",
    "grid": "#D4D2CC",
    "legend_edge": "#E0DED8",
}

TREAT_ARM_LABELS: dict[str, str] = {
    "NutritionInfo": "Nutrition Info label (vs. Positive)",
    "HighIn": "High In warning label (vs. Positive)",
    "Positive+NutritionInfo": "Positive + Nutrition Info (vs. Positive)",
    "Positive+HighIn": "Positive + High In (vs. Positive)",
    "Spectrum": "Spectrum 1-to-5 star label (vs. Positive)",
}


def set_academic_palette(name: str = "nature") -> list[str]:
    """Configure matplotlib color cycle using the repository's Nature or Lancet academic palette."""
    if name not in PALETTES:
        raise ValueError(f"Unknown palette {name!r}, expected one of {list(PALETTES.keys())}")
    colors = PALETTES[name]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    return colors


def load_grummon_rct(parquet_path: str | Path = GRUMMON_PARQUET_PATH) -> pd.DataFrame:
    """Load and validate the Grummon et al. (2025) FOP nutrition label RCT choice dataset."""
    df = load_grummon_df(parquet_path=parquet_path)
    if len(df) != 33798 or df["CASEID"].nunique() != 5633:
        raise ValueError(f"Unexpected Grummon shape: {df.shape}, CASEID={df['CASEID'].nunique()}")
    df["y_correct"] = df["one_hot"].map(lambda oh: float(oh[0]))
    return df.sort_values(["CASEID", "v"]).reset_index(drop=True)


def subsample_grummon_participants(
    df: pd.DataFrame,
    n_participants: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample n_participants (seed=42) and order participants round-robin across the 6 arms so any prefix covers all arms."""
    all_caseids = np.sort(df["CASEID"].unique())
    rng = np.random.default_rng(seed)
    sampled_caseids = np.sort(rng.choice(all_caseids, size=n_participants, replace=False))

    sub_df = df[df["CASEID"].isin(sampled_caseids)].copy()
    arm_order = ["Positive", *TREAT_ARMS]
    part_arms = (
        sub_df[["CASEID", "arm"]]
        .drop_duplicates()
        .sort_values(["arm", "CASEID"])
    )
    by_arm = {arm: part_arms[part_arms["arm"] == arm]["CASEID"].tolist() for arm in arm_order}
    ordered_caseids: list[str] = []
    max_len = max(len(v) for v in by_arm.values())
    for i in range(max_len):
        for arm in arm_order:
            if i < len(by_arm[arm]):
                ordered_caseids.append(by_arm[arm][i])

    caseid_rank = {cid: idx for idx, cid in enumerate(ordered_caseids)}
    sub_df["_cid_rank"] = sub_df["CASEID"].map(caseid_rank)
    sub_df = sub_df.sort_values(["_cid_rank", "v"]).drop(columns=["_cid_rank"]).reset_index(drop=True)
    return sub_df


def fit_grummon_ols(
    df_tasks: pd.DataFrame,
    outcome_col: str = "y_correct",
) -> pd.DataFrame:
    """Aggregate task-level outcomes to participant means and fit HC1 OLS treatment effects vs. 'Positive' arm."""
    if outcome_col not in df_tasks.columns:
        raise KeyError(f"Missing outcome column {outcome_col!r} in df_tasks")

    part_df = (
        df_tasks.groupby("CASEID", as_index=False)
        .agg(
            arm=("arm", "first"),
            outcome=(outcome_col, "mean"),
        )
        .sort_values("CASEID")
        .reset_index(drop=True)
    )
    arms_present = set(part_df["arm"].unique())
    expected_arms = {"Positive", *TREAT_ARMS}
    if arms_present != expected_arms:
        raise ValueError(f"Expected all 6 trial arms {expected_arms}, got {arms_present}")

    fit = smf.ols("outcome ~ C(arm, Treatment(reference='Positive'))", data=part_df).fit(cov_type="HC1")
    rows = []
    for arm in TREAT_ARMS:
        term = f"C(arm, Treatment(reference='Positive'))[T.{arm}]"
        se_val = float(fit.bse[term])
        rows.append({
            "arm": arm,
            "label": TREAT_ARM_LABELS[arm],
            "beta": float(fit.params[term]),
            "se": se_val,
            "z": float(fit.tvalues[term]) if se_val > 1e-12 else np.nan,
            "p_value": float(fit.pvalues[term]) if se_val > 1e-12 else np.nan,
        })
    return pd.DataFrame(rows).set_index("label")


def evaluate_grummon_metrics(
    beta_hat: pd.Series | np.ndarray,
    beta_star: pd.Series | np.ndarray,
    eps: float = 0.0,
) -> dict[str, float]:
    """Compute parameter recovery metrics (MAPE and MSE) between estimated beta_hat and target beta_star."""
    b_hat = np.asarray(beta_hat, dtype=np.float64)
    b_star = np.asarray(beta_star, dtype=np.float64)
    if b_hat.shape != b_star.shape:
        raise ValueError(f"Shape mismatch: {b_hat.shape} vs {b_star.shape}")
    mape = float(np.mean(np.abs(b_hat - b_star) / (np.abs(b_star) + eps)) * 100.0)
    mse = float(np.mean((b_hat - b_star) ** 2))
    return {"MAPE (%)": mape, "MSE": mse}


def load_all_model_regressions(
    sub_df: pd.DataFrame,
    *,
    kev_data_dir: str | Path = "data",
) -> dict[str, pd.DataFrame]:
    """Dynamically discover all Kev Grummon JSONL datasets in `data/`, align with sub_df, and fit HC1 OLS regressions."""
    kev_path = Path(kev_data_dir)
    ols_dfs: dict[str, pd.DataFrame] = {}
    if not kev_path.exists():
        return ols_dfs

    sub_keys = [(str(r["CASEID"]), str(r["v"])) for r in sub_df[["CASEID", "v"]].to_dict("records")]
    for jsonl_path in sorted(kev_path.glob("*.jsonl")):
        first_line = next((line.strip() for line in jsonl_path.open(encoding="utf-8") if line.strip()), None)
        if first_line is None:
            continue
        first_rec = json.loads(first_line)
        model_id = str(first_rec["model"])
        persona_mode = str(first_rec["persona_mode"])
        temp = float(first_rec["temperature"])
        persona_label = "Random Persona" if persona_mode == "random" else "Respondent Persona"

        k_map = load_kev_jsonl(
            jsonl_path,
            expected_model=model_id,
            expected_persona_mode=persona_mode,
            expected_temperature=temp,
        )
        avail_idx = [i for i, k in enumerate(sub_keys) if k in k_map]
        if not avail_idx:
            continue
        avail_df = sub_df.iloc[avail_idx].copy().reset_index(drop=True)
        avail_df["prob_correct"] = [float(k_map[sub_keys[i]]["prob_correct"]) for i in avail_idx]

        arms_present = set(avail_df["arm"].unique())
        if arms_present != {"Positive", *TREAT_ARMS}:
            continue
        n_participants = int(avail_df["CASEID"].nunique())
        label = f"{model_id} ({persona_label}, N={n_participants}, n={len(avail_idx)})"
        ols_dfs[label] = fit_grummon_ols(avail_df, outcome_col="prob_correct")

    return ols_dfs


def plot_grummon_error_comparison(
    comparison_metrics_df: pd.DataFrame,
    palette: str = "nature",
    persona_filter: str | None = None,
    sort_by: str = "MSE vs. Full Human",
) -> plt.Figure:
    """Plot a two-panel horizontal bar chart of MAPE (%) and MSE vs. Full Human beta* sorted from best to worst on MSE."""
    pal = set_academic_palette(palette)
    if sort_by not in comparison_metrics_df.columns:
        raise KeyError(f"Column {sort_by!r} not found in comparison_metrics_df: {list(comparison_metrics_df.columns)}")

    df_plot = comparison_metrics_df.copy()
    if persona_filter is not None:
        mask = [
            ("human" in str(idx).lower()) or (persona_filter in str(idx))
            for idx in df_plot.index
        ]
        df_plot = df_plot.loc[mask]
        if df_plot.empty:
            raise ValueError(f"No rows in comparison_metrics_df matched persona_filter={persona_filter!r}")

    df_plot = df_plot.sort_values(sort_by, ascending=True)

    raw_models = [str(idx) for idx in df_plot.index.tolist()]
    if persona_filter is not None:
        models = [m.replace(f"{persona_filter}, ", "").replace(f"({persona_filter})", "").strip() for m in raw_models]
    else:
        models = raw_models

    mape_vals = df_plot["MAPE vs. Full Human (%)"].to_numpy(dtype=np.float64)
    mse_vals = df_plot["MSE vs. Full Human"].to_numpy(dtype=np.float64)

    bar_colors: list[str] = []
    for name in raw_models:
        lower = name.lower()
        if "human" in lower:
            bar_colors.append(MODEL_PALETTES[palette]["human"])
        elif any(tag in lower for tag in ("ft", "fine-tuned", "finetuned", "issp", "twin2k", "/best", "/last")):
            bar_colors.append(MODEL_PALETTES[palette]["fine_tuned"])
        else:
            bar_colors.append(MODEL_PALETTES[palette]["off_the_shelf"])

    fig_height = max(3.4, 0.72 * len(models) + 1.3)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, fig_height), sharey=True, dpi=150)
    y_pos = np.arange(len(models))

    panel_specs = [
        (axes[0], mape_vals, "MAPE vs. Full Human β* (%)", "Mean Absolute Percentage Error (MAPE)", "{:.1f}%"),
        (axes[1], mse_vals, "MSE vs. Full Human β*", "Mean Squared Error (MSE)", "{:.4f}"),
    ]
    for ax, vals, xlabel, title, fmt in panel_specs:
        bars = ax.barh(y_pos, vals, color=bar_colors, height=0.55, edgecolor=NEUTRALS["text"], linewidth=0.6, zorder=3)
        ax.set_yticks(y_pos, labels=models, color=NEUTRALS["text"], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, color=NEUTRALS["muted"], fontsize=9.5)
        ax.set_title(title, color=NEUTRALS["text"], fontsize=10.5, fontweight="bold", pad=8)
        ax.tick_params(axis="x", colors=NEUTRALS["muted"], labelsize=9)
        ax.grid(axis="x", color=NEUTRALS["grid"], linestyle="--", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(NEUTRALS["muted"])
        ax.spines["bottom"].set_color(NEUTRALS["muted"])
        max_val = float(np.max(vals)) if len(vals) else 1.0
        ax.set_xlim(0, max_val * 1.22)
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                va="center",
                ha="left",
                fontsize=9,
                color=NEUTRALS["muted"],
                fontweight="bold",
            )

    heading = (
        f"FOP Nutrition Label RCT (Grummon et al., 2025): {persona_filter} — Parameter Recovery Error"
        if persona_filter is not None
        else "FOP Nutrition Label RCT (Grummon et al., 2025): Parameter Recovery Error"
    )
    fig.suptitle(
        heading,
        fontsize=11.5,
        fontweight="bold",
        color=NEUTRALS["text"],
        y=1.02,
    )
    plt.tight_layout()
    return fig

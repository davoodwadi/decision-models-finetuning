"""Multinomial Logit (MNL) MLE estimation and evaluation utilities for the Wang et al. (2026) COVID-19 vaccine benchmark."""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.optimize import minimize
from scipy.stats import norm

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


def set_academic_palette(name: str = "nature") -> list[str]:
    """Configure matplotlib color cycle using the repository's Nature or Lancet academic palette."""
    if name not in PALETTES:
        raise ValueError(f"Unknown palette {name!r}, expected one of {list(PALETTES.keys())}")
    colors = PALETTES[name]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    return colors

# 11 binary vaccine attribute features relative to Kreps et al. (2020) / Wang (2026) baseline levels
# Each tuple is (short_name, stata_numeric_column, numeric_code_for_level, human_readable_label);
# code 1 in each Stata *_n column is the omitted baseline level (50%, 1 year, 1 in 10k, 1 in 10, Full FDA, US, Trump)
FEATURE_SPECS: list[tuple[str, str, int, str]] = [
    ("efficacy_70", "efficacy_n", 2, "Efficacy: 70% (vs. 50%)"),
    ("efficacy_90", "efficacy_n", 3, "Efficacy: 90% (vs. 50%)"),
    ("duration_5yr", "duration_n", 2, "Duration: 5 years (vs. 1 year)"),
    ("majorside_1m", "majorside_n", 2, "Major side effects: 1 in 1,000,000 (vs. 1 in 10,000)"),
    ("minorside_30", "minorside_n", 2, "Minor side effects: 1 in 30 (vs. 1 in 10)"),
    ("fda_eua", "fda_n", 2, "FDA: Emergency Use Authorization (vs. Full approval)"),
    ("origin_uk", "origin_n", 2, "Origin: United Kingdom (vs. United States)"),
    ("origin_china", "origin_n", 3, "Origin: China (vs. United States)"),
    ("endorsed_biden", "endorsed_n", 2, "Endorsed: Joe Biden (vs. Donald Trump)"),
    ("endorsed_cdc", "endorsed_n", 3, "Endorsed: US CDC (vs. Donald Trump)"),
    ("endorsed_who", "endorsed_n", 4, "Endorsed: WHO (vs. Donald Trump)"),
]

FEATURE_NAMES: list[str] = [spec[0] for spec in FEATURE_SPECS]
FEATURE_LABELS: list[str] = [spec[3] for spec in FEATURE_SPECS]


def load_covid_conjoint(data_path: str | Path) -> pd.DataFrame:
    """Load and validate the raw Kreps et al. (2020) COVID-19 conjoint Stata dataset."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {path.resolve()}")

    # Read Stata file with numeric codes intact rather than converting to shifted pandas Categoricals
    raw_df = pd.read_stata(path, convert_categoricals=False)
    if len(raw_df) != 19710 or raw_df["respondent"].nunique() != 1971:
        raise ValueError(f"Unexpected dataset shape: {raw_df.shape}, respondents={raw_df['respondent'].nunique()}")

    raw_df["respondent"] = raw_df["respondent"].astype(int)
    raw_df["choice_set"] = raw_df["choice_set"].astype(int)
    # choice_set is a 2-digit code {11,12,21,22,31,32,41,42,51,52}: tens digit = task 1..5, ones digit = Vaccine A (1) or B (2)
    raw_df["pair_id"] = raw_df["choice_set"] // 10
    raw_df["option_ab"] = raw_df["choice_set"] % 10
    if set(raw_df["pair_id"].unique()) != {1, 2, 3, 4, 5} or set(raw_df["option_ab"].unique()) != {1, 2}:
        raise ValueError("Unexpected pair_id or option_ab values")
    if raw_df["vaxbin"].isna().any():
        raise ValueError("Found unexpected NaN values in vaxbin choice indicator")
    return raw_df


def pair_choice_sets(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split profile-level rows into aligned Option A and Option B DataFrames and 3-way human choice arrays (y_A, y_B, y_C)."""
    df_a = raw_df[raw_df["option_ab"] == 1].sort_values(["respondent", "pair_id"]).reset_index(drop=True)
    df_b = raw_df[raw_df["option_ab"] == 2].sort_values(["respondent", "pair_id"]).reset_index(drop=True)
    if not (df_a["respondent"].equals(df_b["respondent"]) and df_a["pair_id"].equals(df_b["pair_id"])):
        raise ValueError("Option A and Option B rows are misaligned across respondents or pair_ids")

    y_a = df_a["vaxbin"].to_numpy(dtype=np.float64)
    y_b = df_b["vaxbin"].to_numpy(dtype=np.float64)
    # Outside option C (neither A nor B) is chosen whenever both A and B are 0
    y_c = 1.0 - y_a - y_b
    if not np.isin(y_c, [0.0, 1.0]).all():
        raise ValueError("Found choice sets where both Vaccine A and Vaccine B were simultaneously chosen")
    return df_a, df_b, y_a, y_b, y_c


def encode_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert a profile DataFrame (N rows) into an (N, 11) float64 binary design matrix."""
    cols = []
    for _, col_name, level_code, _ in FEATURE_SPECS:
        vals = df[col_name].to_numpy()
        if pd.isna(vals).any():
            raise ValueError(f"NaN encountered in feature column {col_name}")
        cols.append((vals == level_code).astype(np.float64))
    return np.column_stack(cols)


def fit_mnl_mle(
    X_a: np.ndarray,
    X_b: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_c: np.ndarray,
) -> pd.DataFrame:
    """Fit the 3-way MNL model (Wang 2026 Eq. 1-2) where Option A has utility X_a @ beta,
    Option B has utility X_b @ beta, and outside Option C (neither) has utility 0.
    Works for both hard one-hot choices and soft probability distributions (y_a + y_b + y_c = 1).
    """
    n_pairs, d = X_a.shape
    if X_b.shape != (n_pairs, d):
        raise ValueError(f"Shape mismatch: X_a={X_a.shape}, X_b={X_b.shape}")
    if y_a.shape != (n_pairs,) or y_b.shape != (n_pairs,) or y_c.shape != (n_pairs,):
        raise ValueError("Target arrays must have shape (n_pairs,)")
    if not np.allclose(y_a + y_b + y_c, 1.0, atol=1e-5):
        raise ValueError("Choice probabilities (y_a + y_b + y_c) must sum to 1 for every choice set")

    def nll_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        # Linear utilities for Vaccine A and Vaccine B; outside Option C utility is fixed to 0
        u_a = X_a @ beta
        u_b = X_b @ beta

        # Numerically stable log-sum-exp: subtract row-wise max utility m = max(0, u_a, u_b) before exponentiating
        m = np.maximum(0.0, np.maximum(u_a, u_b))
        e_a = np.exp(u_a - m)
        e_b = np.exp(u_b - m)
        e_c = np.exp(-m)
        denom = e_a + e_b + e_c
        log_denom = m + np.log(denom)  # equals log(1 + exp(u_a) + exp(u_b))

        # Average negative log-likelihood across choice sets
        nll = -float(np.mean(y_a * (u_a - log_denom) + y_b * (u_b - log_denom) + y_c * (-log_denom)))

        # MNL predicted probabilities P(A) and P(B), and analytical gradient d(NLL)/d(beta)
        p_a = e_a / denom
        p_b = e_b / denom
        grad = (X_a.T @ (p_a - y_a) + X_b.T @ (p_b - y_b)) / n_pairs
        return nll, grad

    def hess(beta: np.ndarray) -> np.ndarray:
        # Analytical Hessian d^2(NLL)/d(beta)^2 using the multinomial covariance weights
        u_a = X_a @ beta
        u_b = X_b @ beta
        m = np.maximum(0.0, np.maximum(u_a, u_b))
        e_a = np.exp(u_a - m)
        e_b = np.exp(u_b - m)
        e_c = np.exp(-m)
        denom = e_a + e_b + e_c
        p_a = e_a / denom
        p_b = e_b / denom

        # Variance and cross-covariance of indicator choices (A, B) under MNL
        w_aa = p_a * (1.0 - p_a)
        w_bb = p_b * (1.0 - p_b)
        w_ab = -p_a * p_b
        H = (
            (X_a.T * w_aa) @ X_a
            + (X_b.T * w_bb) @ X_b
            + (X_a.T * w_ab) @ X_b
            + (X_b.T * w_ab) @ X_a
        ) / n_pairs
        return H

    # Minimize NLL using exact second-order Newton-CG optimization starting from beta = 0
    res = minimize(
        nll_and_grad,
        np.zeros(d, dtype=np.float64),
        jac=True,
        hess=hess,
        method="Newton-CG",
        options={"xtol": 1e-10, "maxiter": 200},
    )
    if not res.success:
        raise RuntimeError(f"MNL MLE failed to converge: {res.message}")

    # Compute asymptotic standard errors, z-statistics, and two-sided p-values from inverse Fisher information
    beta_hat = res.x
    H_total = hess(beta_hat) * n_pairs
    cov = np.linalg.inv(H_total)
    se = np.sqrt(np.diag(cov))
    z_stat = beta_hat / se
    p_val = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))

    return pd.DataFrame({
        "feature": FEATURE_NAMES,
        "label": FEATURE_LABELS,
        "beta": beta_hat,
        "se": se,
        "z": z_stat,
        "p_value": p_val,
    }).set_index("label")


def evaluate_wang_metrics(
    beta_hat: pd.Series | np.ndarray,
    beta_star: pd.Series | np.ndarray,
    eps: float = 0.0,
) -> dict[str, float]:
    """Compute Wang (2026) parameter recovery metrics (MAPE and MSE) between estimated beta_hat and target beta_star."""
    b_hat = np.asarray(beta_hat, dtype=np.float64)
    b_star = np.asarray(beta_star, dtype=np.float64)
    if b_hat.shape != b_star.shape:
        raise ValueError(f"Shape mismatch: {b_hat.shape} vs {b_star.shape}")
    # Wang (2026) Section 5.3 Eq: average relative percentage error across the d=11 coefficients
    mape = float(np.mean(np.abs(b_hat - b_star) / (np.abs(b_star) + eps)) * 100.0)
    # Wang (2026) Appendix EC.3.1 Eq: mean squared error across the d=11 coefficients
    mse = float(np.mean((b_hat - b_star) ** 2))
    return {"MAPE (%)": mape, "MSE": mse}


def _persona_label_from_stem(stem: str, mode_field: str | None = None) -> str:
    """Map filename stem or persona_mode field to a human-readable persona label."""
    if mode_field == "random" or stem.endswith("_random"):
        return "Random Persona"
    if mode_field == "persona" or stem.endswith("_persona"):
        return "Respondent Persona"
    raise ValueError(f"Cannot determine persona mode from filename stem {stem!r} or mode {mode_field!r}")


def load_all_model_regressions(
    X_a_sub: np.ndarray,
    X_b_sub: np.ndarray,
    subsample_keys: list[tuple[int, int]],
    *,
    gemini_data_dir: str | Path = "../../proprietary-llms/gemini/data",
    kev_data_dir: str | Path = "data",
) -> dict[str, pd.DataFrame]:
    """Dynamically discover all Gemini and Kev JSONL datasets, validate metadata, and fit MNL MLE regressions."""
    gemini_dir = Path(gemini_data_dir).resolve().parent
    if str(gemini_dir) not in sys.path:
        sys.path.insert(0, str(gemini_dir))
    from gemini_api import load_collected_jsonl, parse_letter_choices
    from kev_eval import extract_kev_prob_targets, load_kev_jsonl

    mnl_dfs: dict[str, pd.DataFrame] = {}

    # 1. Dynamically discover all Gemini JSONL files
    gem_path = Path(gemini_data_dir)
    if gem_path.exists():
        for jsonl_path in sorted(gem_path.glob("*.jsonl")):
            first_line = next((line.strip() for line in jsonl_path.open(encoding="utf-8") if line.strip()), None)
            if first_line is None:
                continue
            first_rec = json.loads(first_line)
            model_id = str(first_rec["model"])
            thinking_level = first_rec["thinking_level"]
            persona_label = _persona_label_from_stem(jsonl_path.stem)

            rec_map = load_collected_jsonl(
                jsonl_path,
                expected_model=model_id,
                expected_thinking_level=thinking_level,
            )
            avail_idx = [i for i, k in enumerate(subsample_keys) if k in rec_map]
            if not avail_idx:
                continue
            valid_idx: list[int] = []
            valid_raw_resps: list[str] = []
            refusal_keys: list[tuple[int, int]] = []
            for i in avail_idx:
                raw_r = str(rec_map[subsample_keys[i]]["raw_response"])
                if raw_r.strip().upper() in {"A", "B", "C"}:
                    valid_idx.append(i)
                    valid_raw_resps.append(raw_r)
                else:
                    refusal_keys.append(subsample_keys[i])
            if not valid_idx:
                raise ValueError(f"No valid A/B/C responses found in {jsonl_path}")
            if refusal_keys:
                print(
                    f"[WARNING] {jsonl_path.name}: dropped {len(refusal_keys)}/{len(avail_idx)} non-A/B/C refusal responses "
                    f"(fitting on valid n={len(valid_idx)}; first 3 refusal keys: {refusal_keys[:3]})"
                )
            y_a, y_b, y_c = parse_letter_choices(valid_raw_resps)
            label = f"{model_id} ({thinking_level}, {persona_label}, n={len(valid_idx)})"
            mnl_dfs[label] = fit_mnl_mle(X_a_sub[valid_idx], X_b_sub[valid_idx], y_a, y_b, y_c)

    # 2. Dynamically discover all Kev JSONL files
    kev_path = Path(kev_data_dir)
    if kev_path.exists():
        for jsonl_path in sorted(kev_path.glob("*.jsonl")):
            first_line = next((line.strip() for line in jsonl_path.open(encoding="utf-8") if line.strip()), None)
            if first_line is None:
                continue
            first_rec = json.loads(first_line)
            model_id = str(first_rec["model"])
            persona_mode = str(first_rec["persona_mode"])
            temp = float(first_rec["temperature"])
            persona_label = _persona_label_from_stem(jsonl_path.stem, mode_field=persona_mode)

            k_map = load_kev_jsonl(
                jsonl_path,
                expected_model=model_id,
                expected_persona_mode=persona_mode,
                expected_temperature=temp,
            )
            avail_idx = [i for i, k in enumerate(subsample_keys) if k in k_map]
            if not avail_idx:
                continue
            k_recs = [k_map[subsample_keys[i]] for i in avail_idx]
            p_a, p_b, p_c = extract_kev_prob_targets(k_recs)
            label = f"{model_id} ({persona_label}, n={len(avail_idx)})"
            mnl_dfs[label] = fit_mnl_mle(X_a_sub[avail_idx], X_b_sub[avail_idx], p_a, p_b, p_c)

    return mnl_dfs


def plot_wang_error_comparison(
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

    # Sort from best (lowest error) to worst (highest error) on sort_by (default: MSE vs. Full Human)
    df_plot = df_plot.sort_values(sort_by, ascending=True)

    raw_models = [str(idx) for idx in df_plot.index.tolist()]
    if persona_filter is not None:
        models = [m.replace(f"{persona_filter}, ", "").replace(f"({persona_filter})", "").strip() for m in raw_models]
    else:
        models = raw_models

    mape_vals = df_plot["MAPE vs. Full Human (%)"].to_numpy(dtype=np.float64)
    mse_vals = df_plot["MSE vs. Full Human"].to_numpy(dtype=np.float64)

    # Assign colors deterministically from the academic palette so every model keeps the same color across plots:
    # Human -> navy, Fine-tuned Kev -> viridian, Base Kev -> terracotta, other models -> deterministic palette mapping
    other_palette = [pal[3], pal[4], pal[5], NEUTRALS["baseline"]]
    other_keys = sorted({
        m.replace("Random Persona", "").replace("Respondent Persona", "").split(", n=")[0].strip()
        for m in comparison_metrics_df.index.astype(str)
        if "human" not in m.lower() and "kev-9b" not in m.lower()
    })
    other_color_map = {k: other_palette[i % len(other_palette)] for i, k in enumerate(other_keys)}

    bar_colors: list[str] = []
    for name in raw_models:
        lower = name.lower()
        if "human" in lower:
            bar_colors.append(MODEL_PALETTES[palette]["human"])
        elif any(tag in lower for tag in ("ft", "fine-tuned", "finetuned", "issp", "twin2k", "/best")):
            bar_colors.append(MODEL_PALETTES[palette]["fine_tuned"])
        elif "kev-9b" in lower:
            bar_colors.append(MODEL_PALETTES[palette]["off_the_shelf"])
        else:
            key = name.replace("Random Persona", "").replace("Respondent Persona", "").split(", n=")[0].strip()
            bar_colors.append(other_color_map[key])

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
        f"COVID-19 Vaccine Conjoint (Wang et al., 2026): {persona_filter} — Parameter Recovery Error"
        if persona_filter is not None
        else "COVID-19 Vaccine Conjoint (Wang et al., 2026): Parameter Recovery Error"
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


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


import re


def _persona_label_from_stem(stem: str, mode_field: str | None = None) -> str:
    """Map filename stem or persona_mode field to a human-readable persona label."""
    if mode_field == "random" or stem.endswith("_random"):
        return "Random Persona"
    if mode_field == "persona" or stem.endswith("_persona"):
        return "Respondent Persona"
    raise ValueError(f"Cannot determine persona mode from filename stem {stem!r} or mode {mode_field!r}")


def format_raw_model_name(raw_model_id: str, thinking_level: str | None = None) -> str:
    """Format a raw model identifier or checkpoint path into a clean publication display name."""
    s = raw_model_id.strip()
    lower = s.lower()

    if "human subsample" in lower:
        return "Human subsample"
    if "full human" in lower:
        return "Full human"

    # Match Kev models (base or fine-tuned)
    kev_match = re.search(r"kev-(\d+(?:\.\d+)?b)", lower)
    if kev_match:
        size_str = kev_match.group(1).upper()
        if s.startswith("jaredpalmer/kev-"):
            return f"Kev {size_str} (base)"
        suffix = ""
        if "-ep2" in lower:
            suffix = " (ep2)"
        elif lower.endswith("/last"):
            suffix = " (last)"
        return f"Kev {size_str} FT{suffix}"

    # Match Gemini or other generative LLMs
    if lower.startswith("gemini-"):
        body = s[len("gemini-"):]
        parts = body.split("-")
        formatted_parts = [p.capitalize() if not p.replace(".", "").isdigit() else p for p in parts]
        name = "Gemini " + " ".join(formatted_parts)
        name = name.replace("Flash Lite", "Flash-Lite")
    else:
        name = s

    if thinking_level is not None and str(thinking_level).strip():
        return f"{name} ({str(thinking_level).strip().lower()})"
    return name


def format_entry_display_label(entry_key: str, default_n: int = 1000) -> str:
    """Convert an internal regression dictionary key into a concise publication plot/table label."""
    s = entry_key.strip()
    if s.lower().startswith("human subsample"):
        return f"Human subsample (n={default_n:,})"

    # Expected format from load_all_model_regressions:
    # "<model_id> (<thinking_level>, <Persona Label>, n=<N>)" or "<model_id> (<Persona Label>, n=<N>)"
    m = re.match(r"^(.*?)\s*\((.*),\s*(Random Persona|Respondent Persona),\s*n=(\d+)\)$", s)
    if m:
        raw_model, thinking, _, n_str = m.groups()
        base_name = format_raw_model_name(raw_model, thinking_level=thinking)
        n_val = int(n_str)
        return f"{base_name} (n={n_val:,})" if n_val != default_n else base_name

    m2 = re.match(r"^(.*?)\s*\((Random Persona|Respondent Persona),\s*n=(\d+)\)$", s)
    if m2:
        raw_model, _, n_str = m2.groups()
        base_name = format_raw_model_name(raw_model, thinking_level=None)
        n_val = int(n_str)
        return f"{base_name} (n={n_val:,})" if n_val != default_n else base_name

    return s


def parse_entry_metadata(entry_key: str) -> dict[str, str | int]:
    """Parse model display name, persona regime, and valid sample size from a regression dictionary key."""
    s = entry_key.strip()
    m = re.match(r"^(.*?)\s*\((.*),\s*(Random Persona|Respondent Persona),\s*n=(\d+)\)$", s)
    if m:
        raw_model, thinking, persona, n_str = m.groups()
        return {
            "model": format_raw_model_name(raw_model, thinking_level=thinking),
            "persona": persona,
            "n": int(n_str),
        }
    m2 = re.match(r"^(.*?)\s*\((Random Persona|Respondent Persona),\s*n=(\d+)\)$", s)
    if m2:
        raw_model, persona, n_str = m2.groups()
        return {
            "model": format_raw_model_name(raw_model, thinking_level=None),
            "persona": persona,
            "n": int(n_str),
        }
    raise ValueError(f"Unrecognized regression entry key format: {entry_key!r}")


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
    default_n: int = 1000,
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
    models = [format_entry_display_label(m, default_n=default_n) for m in raw_models]

    mape_vals = df_plot["MAPE vs. Full Human (%)"].to_numpy(dtype=np.float64)
    mse_vals = df_plot["MSE vs. Full Human"].to_numpy(dtype=np.float64)

    # Assign colors deterministically from the academic palette so every model keeps the same color across plots:
    # Human -> navy, Fine-tuned Kev -> viridian, Base Kev -> terracotta, other models -> deterministic palette mapping
    other_palette = [pal[3], pal[4], pal[5], NEUTRALS["baseline"]]
    other_keys = sorted({
        m.replace("Random Persona", "").replace("Respondent Persona", "").split(", n=")[0].split(", N=")[0].strip()
        for m in comparison_metrics_df.index.astype(str)
        if "human" not in m.lower() and "kev-" not in m.lower()
    })
    other_color_map = {k: other_palette[i % len(other_palette)] for i, k in enumerate(other_keys)}

    bar_colors: list[str] = []
    for name in raw_models:
        lower = name.lower()
        if "human" in lower:
            bar_colors.append(MODEL_PALETTES[palette]["human"])
        elif any(tag in lower for tag in ("ft", "fine-tuned", "finetuned", "issp", "twin2k", "/best", "/last")):
            bar_colors.append(MODEL_PALETTES[palette]["fine_tuned"])
        elif "kev-" in lower:
            bar_colors.append(MODEL_PALETTES[palette]["off_the_shelf"])
        else:
            key = name.replace("Random Persona", "").replace("Respondent Persona", "").split(", n=")[0].split(", N=")[0].strip()
            bar_colors.append(other_color_map[key])

    fig_height = max(3.3, 0.56 * len(models) + 0.9)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, fig_height), sharey=True, dpi=150)
    y_pos = np.arange(len(models))

    panel_specs = [
        (axes[0], mape_vals, "MAPE vs. full human β* (%)", "A. Mean absolute percentage error (MAPE)", "{:.1f}%"),
        (axes[1], mse_vals, "MSE vs. full human β*", "B. Mean squared error (MSE)", "{:.4f}"),
    ]
    for ax, vals, xlabel, title, fmt in panel_specs:
        bars = ax.barh(y_pos, vals, color=bar_colors, height=0.55, edgecolor=NEUTRALS["text"], linewidth=0.6, zorder=3)
        ax.set_yticks(y_pos, labels=models, color=NEUTRALS["text"], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, color=NEUTRALS["muted"], fontsize=9.5)
        ax.set_title(title, color=NEUTRALS["text"], fontsize=10.5, fontweight="bold", loc="left", pad=8)
        ax.tick_params(axis="x", colors=NEUTRALS["muted"], labelsize=9)
        ax.grid(axis="x", color=NEUTRALS["grid"], linestyle="--", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(NEUTRALS["muted"])
        ax.spines["bottom"].set_color(NEUTRALS["muted"])
        max_val = float(np.max(vals)) if len(vals) else 1.0
        ax.set_xlim(0, max_val * 1.24)
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

    plt.tight_layout()
    return fig


def build_wang_summary_table(
    comparison_metrics_df: pd.DataFrame,
    human_sub_key: str = "Human Subsample Baseline (n=1000)",
) -> pd.DataFrame:
    """Build a publication-ready side-by-side table comparing Random Persona and Respondent Persona MNL metrics."""
    if human_sub_key not in comparison_metrics_df.index:
        raise KeyError(f"Missing {human_sub_key!r} in comparison_metrics_df")

    by_model: dict[str, dict[str, dict[str, float | int]]] = {}
    for entry_key, row in comparison_metrics_df.iterrows():
        key_str = str(entry_key)
        if key_str == human_sub_key:
            continue
        meta = parse_entry_metadata(key_str)
        model_name = str(meta["model"])
        persona = str(meta["persona"])
        if model_name not in by_model:
            by_model[model_name] = {}
        by_model[model_name][persona] = {
            "n": int(meta["n"]),
            "mape_full": float(row["MAPE vs. Full Human (%)"]),
            "mse_full": float(row["MSE vs. Full Human"]),
            "mape_sub": float(row["MAPE vs. Human Subsample (%)"]),
            "mse_sub": float(row["MSE vs. Human Subsample"]),
        }

    # Sort models by Respondent Persona MSE vs. Full Human (or Random Persona if Respondent is absent)
    def sort_key(m_name: str) -> float:
        entries = by_model[m_name]
        if "Respondent Persona" in entries:
            return float(entries["Respondent Persona"]["mse_full"])
        return float(entries["Random Persona"]["mse_full"])

    sorted_models = sorted(by_model.keys(), key=sort_key)

    h_row = comparison_metrics_df.loc[human_sub_key]
    table_rows: list[dict[str, str]] = [
        {
            "Estimator": "Human subsample (n = 1,000)",
            "Random persona MAPE (%)": "—",
            "Random persona MSE": "—",
            "Respondent persona MAPE (%)": f"{float(h_row['MAPE vs. Full Human (%)']):.1f}%",
            "Respondent persona MSE": f"{float(h_row['MSE vs. Full Human']):.4f}",
        }
    ]

    for m_name in sorted_models:
        entries = by_model[m_name]
        if "Random Persona" not in entries or "Respondent Persona" not in entries:
            raise ValueError(f"Expected both Random and Respondent Persona for {m_name}, got {list(entries.keys())}")
        rnd = entries["Random Persona"]
        prs = entries["Respondent Persona"]
        n_note = f" (random n = {rnd['n']})" if int(rnd["n"]) < 1000 else ""
        table_rows.append({
            "Estimator": f"{m_name}{n_note}",
            "Random persona MAPE (%)": f"{rnd['mape_full']:.1f}%",
            "Random persona MSE": f"{rnd['mse_full']:.4f}",
            "Respondent persona MAPE (%)": f"{prs['mape_full']:.1f}%",
            "Respondent persona MSE": f"{prs['mse_full']:.4f}",
        })

    return pd.DataFrame(table_rows)


def build_wang_coef_table(
    human_mnl_df: pd.DataFrame,
    human_sub_mnl_df: pd.DataFrame,
    model_mnl_dfs: dict[str, pd.DataFrame],
    persona_mode: str = "Respondent Persona",
) -> pd.DataFrame:
    """Build a Web Appendix table of the d=11 MNL part-worth estimates (and standard errors) across estimators."""
    selected_models: list[tuple[str, pd.DataFrame]] = [
        ("Full human (N = 9,855)", human_mnl_df),
        ("Human subsample (n = 1,000)", human_sub_mnl_df),
    ]

    # Collect models matching persona_mode sorted cleanly (Kev FT, Kev base, then LLMs)
    matched: list[tuple[str, pd.DataFrame]] = []
    for key, df_m in model_mnl_dfs.items():
        meta = parse_entry_metadata(key)
        if meta["persona"] == persona_mode:
            matched.append((str(meta["model"]), df_m))

    def model_order(item: tuple[str, pd.DataFrame]) -> tuple[int, str]:
        name = item[0].lower()
        if "kev" in name and "ft" in name:
            return (0, name)
        if "kev" in name:
            return (1, name)
        return (2, name)

    selected_models.extend(sorted(matched, key=model_order))

    rows: list[dict[str, str]] = []
    for feat_label in human_mnl_df.index:
        row_dict: dict[str, str] = {"Vaccine attribute level": str(feat_label)}
        for col_name, df_est in selected_models:
            b = float(df_est.loc[feat_label, "beta"])
            se = float(df_est.loc[feat_label, "se"])
            row_dict[col_name] = f"{b:+.3f} ({se:.3f})"
        rows.append(row_dict)

    return pd.DataFrame(rows)


def plot_wang_coefficients(
    human_mnl_df: pd.DataFrame,
    human_sub_mnl_df: pd.DataFrame,
    model_mnl_dfs: dict[str, pd.DataFrame],
    palette: str = "nature",
    persona_mode: str = "Respondent Persona",
) -> plt.Figure:
    """Plot a two-panel forest comparison of the d=11 MNL part-worth coefficients:
    Panel A: Human ground truth vs. off-the-shelf and fine-tuned Kev decision models (with 95% CIs).
    Panel B: Human ground truth and Kev 9B FT vs. generative LLMs (showing logit-scale inflation).
    """
    pal = set_academic_palette(palette)
    by_name: dict[str, pd.DataFrame] = {}
    for key, df_m in model_mnl_dfs.items():
        meta = parse_entry_metadata(key)
        if meta["persona"] == persona_mode:
            by_name[str(meta["model"])] = df_m

    kev_base_names = sorted([k for k in by_name if k.startswith("Kev ") and "(base)" in k])
    kev_ft_names = sorted([k for k in by_name if k.startswith("Kev ") and "FT" in k])
    llm_names = sorted([k for k in by_name if not k.startswith("Kev ")])
    if not kev_ft_names or not kev_base_names:
        raise ValueError(f"Expected both base and FT Kev models for {persona_mode}, found {list(by_name.keys())}")

    labels = list(human_mnl_df.index)
    short_labels = [
        lbl.replace("Major side effects: 1 in 1,000,000 (vs. 1 in 10,000)", "Major side effects: 1 in 1M (vs. 1 in 10k)")
           .replace("FDA: Emergency Use Authorization (vs. Full approval)", "FDA: EUA (vs. Full approval)")
           .replace("Origin: United Kingdom (vs. United States)", "Origin: UK (vs. US)")
           .replace("Origin: China (vs. United States)", "Origin: China (vs. US)")
        for lbl in labels
    ]
    y_base = np.arange(len(labels), dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.4), sharey=True, dpi=150)
    fig.subplots_adjust(wspace=0.10)

    # Panel A series: Full human, Human subsample, Kev base, Kev FT
    panel_a_specs: list[tuple[str, pd.DataFrame, str, str]] = [
        ("Full human (N = 9,855)", human_mnl_df, MODEL_PALETTES[palette]["human"], "o"),
        ("Human subsample (n = 1,000)", human_sub_mnl_df, NEUTRALS["baseline"], "s"),
    ]
    for kb in kev_base_names:
        panel_a_specs.append((kb, by_name[kb], MODEL_PALETTES[palette]["off_the_shelf"], "^"))
    for kft in kev_ft_names:
        panel_a_specs.append((kft, by_name[kft], MODEL_PALETTES[palette]["fine_tuned"], "D"))

    offsets_a = np.linspace(-0.24, 0.24, len(panel_a_specs))
    ax1.axvline(0.0, color=NEUTRALS["ref_line"], linestyle=":", linewidth=1.0, zorder=1)
    for offset, (name, df_m, color, marker) in zip(offsets_a, panel_a_specs, strict=True):
        betas = df_m.loc[labels, "beta"].to_numpy(dtype=np.float64)
        ses = df_m.loc[labels, "se"].to_numpy(dtype=np.float64)
        ax1.errorbar(
            betas,
            y_base + offset,
            xerr=1.96 * ses,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.2,
            markersize=5.2,
            label=name,
            zorder=3,
        )

    ax1.set_yticks(y_base, labels=short_labels, color=NEUTRALS["text"], fontsize=9.0)
    ax1.invert_yaxis()
    ax1.set_xlabel("Estimated MNL part-worth coefficient β̂ (log-odds)", color=NEUTRALS["muted"], fontsize=9.5)
    ax1.set_title("A. Decision models vs. human ground truth", color=NEUTRALS["text"], fontsize=10.5, fontweight="bold", loc="left", pad=8)
    ax1.tick_params(axis="x", colors=NEUTRALS["muted"], labelsize=9.0)
    ax1.grid(axis="x", color=NEUTRALS["grid"], linestyle="--", linewidth=0.8, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color(NEUTRALS["muted"])
    ax1.spines["bottom"].set_color(NEUTRALS["muted"])
    ax1.legend(
        frameon=True,
        facecolor="white",
        edgecolor=NEUTRALS["legend_edge"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8.5,
    )

    # Panel B series: Full human, best Kev FT, and generative LLMs
    best_kft = kev_ft_names[-1]
    panel_b_specs: list[tuple[str, pd.DataFrame, str, str]] = [
        ("Full human (N = 9,855)", human_mnl_df, MODEL_PALETTES[palette]["human"], "o"),
        (best_kft, by_name[best_kft], MODEL_PALETTES[palette]["fine_tuned"], "D"),
    ]
    llm_colors = [pal[3], pal[4], pal[5], NEUTRALS["baseline"]]
    llm_markers = ["v", "<", ">", "P"]
    for i, llm_name in enumerate(llm_names):
        panel_b_specs.append((
            llm_name,
            by_name[llm_name],
            llm_colors[i % len(llm_colors)],
            llm_markers[i % len(llm_markers)],
        ))

    offsets_b = np.linspace(-0.26, 0.26, len(panel_b_specs))
    ax2.axvline(0.0, color=NEUTRALS["ref_line"], linestyle=":", linewidth=1.0, zorder=1)
    for offset, (name, df_m, color, marker) in zip(offsets_b, panel_b_specs, strict=True):
        betas = df_m.loc[labels, "beta"].to_numpy(dtype=np.float64)
        ses = df_m.loc[labels, "se"].to_numpy(dtype=np.float64)
        ax2.errorbar(
            betas,
            y_base + offset,
            xerr=1.96 * ses,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.0,
            markersize=5.0,
            label=name,
            zorder=3,
        )

    ax2.set_xlabel("Estimated MNL part-worth coefficient β̂ (log-odds)", color=NEUTRALS["muted"], fontsize=9.5)
    ax2.set_title("B. Generative LLM scale inflation vs. Kev 9B FT", color=NEUTRALS["text"], fontsize=10.5, fontweight="bold", loc="left", pad=8)
    ax2.tick_params(axis="x", colors=NEUTRALS["muted"], labelsize=9.0)
    ax2.grid(axis="x", color=NEUTRALS["grid"], linestyle="--", linewidth=0.8, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(NEUTRALS["muted"])
    ax2.spines["bottom"].set_color(NEUTRALS["muted"])
    ax2.legend(
        frameon=True,
        facecolor="white",
        edgecolor=NEUTRALS["legend_edge"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8.5,
    )

    plt.tight_layout()
    return fig


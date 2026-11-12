"""HC1 OLS treatment-effect estimation, dynamic Kev dataset loader, and plotting utilities for Grummon et al. (2025)."""

from pathlib import Path
import json
import math
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from cycler import cycler

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.grummon import (
    GRUMMON_PARQUET_PATH,
    GRUMMON_PERSONA_NAMES,
    TREAT_ARMS,
    load_grummon_df,
)
from kev_eval import (
    SEEN_PERSONA_COLS,
    UNSEEN_PERSONA_COLS,
    load_kev_jsonl,
    load_single_attr_jsonl,
)

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
    """Compute parameter recovery metrics (MAPE, MSE, and Spearman rho) between estimated beta_hat and target beta_star."""
    b_hat = np.asarray(beta_hat, dtype=np.float64)
    b_star = np.asarray(beta_star, dtype=np.float64)
    if b_hat.shape != b_star.shape:
        raise ValueError(f"Shape mismatch: {b_hat.shape} vs {b_star.shape}")
    mape = float(np.mean(np.abs(b_hat - b_star) / (np.abs(b_star) + eps)) * 100.0)
    mse = float(np.mean((b_hat - b_star) ** 2))
    rho = float(spearmanr(b_hat, b_star).statistic)
    return {"MAPE (%)": mape, "MSE": mse, "Spearman rho": rho}


def format_raw_model_name(raw_model_id: str) -> str:
    """Format a raw model identifier or checkpoint path into a clean publication display name."""
    s = raw_model_id.strip()
    lower = s.lower()

    if "human subsample" in lower:
        return "Human subsample"
    if "full human" in lower:
        return "Full human"

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

    return s


def parse_entry_metadata(entry_key: str) -> dict[str, str | int]:
    """Parse model display name, persona regime, participant count N, and task count n from a regression key."""
    s = entry_key.strip()
    m = re.match(r"^(.*?)\s*\((Random Persona|Respondent Persona),\s*N=(\d+),\s*n=(\d+)\)$", s)
    if not m:
        raise ValueError(f"Unrecognized regression entry key format: {entry_key!r}")
    raw_model, persona, n_part_str, n_task_str = m.groups()
    return {
        "raw_model": raw_model,
        "model": format_raw_model_name(raw_model),
        "persona": persona,
        "N": int(n_part_str),
        "n": int(n_task_str),
    }


def format_entry_display_label(entry_key: str, default_n: int = 1200) -> str:
    """Convert an internal regression dictionary key into a concise publication plot/table label."""
    s = entry_key.strip()
    if s.lower().startswith("human subsample"):
        return f"Human subsample (n={default_n:,})"
    meta = parse_entry_metadata(s)
    base_name = str(meta["model"])
    n_val = int(meta["n"])
    return f"{base_name} (n={n_val:,})" if n_val != default_n else base_name


def compute_choice_ce_stats(
    sub_df: pd.DataFrame,
    k_map: dict[tuple[str, str], dict[str, object]],
    avail_idx: list[int],
) -> dict[str, float]:
    """Compute respondent-level 3-option choice cross-entropy (in nats and bits) on aligned tasks."""
    sub_records = sub_df.iloc[avail_idx].to_dict("records")
    ce_nats_list: list[float] = []
    for row in sub_records:
        key = (str(row["CASEID"]), str(row["v"]))
        rec = k_map[key]
        oh = list(row["one_hot"])
        if 1.0 not in oh:
            raise ValueError(f"Invalid one_hot target {oh} for {key}")
        ans_idx = oh.index(1.0)
        probs = [float(rec["prob_correct"]), float(rec["prob_wrong"]), float(rec["prob_same"])]
        p_true = max(probs[ans_idx], 1e-12)
        ce_nats_list.append(-math.log(p_true))
    mean_nats = float(np.mean(ce_nats_list))
    return {
        "ce_nats": mean_nats,
        "ce_bits": mean_nats / math.log(2.0),
    }


def load_all_model_regressions(
    sub_df: pd.DataFrame,
    *,
    kev_data_dir: str | Path = "data",
    only_newvalid_ft: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Discover Kev Grummon JSONL datasets in `data/`, compute choice CE, and fit HC1 OLS regressions."""
    kev_path = Path(kev_data_dir)
    ols_dfs: dict[str, pd.DataFrame] = {}
    ce_stats: dict[str, dict[str, float]] = {}
    if not kev_path.exists():
        return ols_dfs, ce_stats

    sub_keys = [(str(r["CASEID"]), str(r["v"])) for r in sub_df[["CASEID", "v"]].to_dict("records")]
    for jsonl_path in sorted(kev_path.glob("*.jsonl")):
        first_line = next((line.strip() for line in jsonl_path.open(encoding="utf-8") if line.strip()), None)
        if first_line is None:
            continue
        first_rec = json.loads(first_line)
        model_id = str(first_rec["model"])
        if only_newvalid_ft and not model_id.startswith("jaredpalmer/kev-") and "newvalid" not in model_id:
            continue
        persona_mode = str(first_rec["persona_mode"])
        if persona_mode not in {"random", "persona"}:
            raise ValueError(f"Unexpected persona_mode={persona_mode!r} in {jsonl_path}")
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
        ce_info = compute_choice_ce_stats(sub_df, k_map, avail_idx)
        ce_info["temperature"] = temp
        ce_info["prob_correct_mean"] = float(avail_df["prob_correct"].mean())
        ce_info["prob_same_mean"] = float(np.mean([float(k_map[sub_keys[i]]["prob_same"]) for i in avail_idx]))
        ce_stats[label] = ce_info

    return ols_dfs, ce_stats


def build_grummon_summary_table(
    comparison_metrics_df: pd.DataFrame,
    ce_stats: dict[str, dict[str, float]],
    human_sub_key: str = "Human Subsample Baseline (N=200, n=1200)",
) -> pd.DataFrame:
    """Build a publication-ready table of choice cross-entropy and Random vs. Respondent Persona OLS recovery."""
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
        ce_row = ce_stats[key_str]
        by_model[model_name][persona] = {
            "n": int(meta["n"]),
            "ce_nats": float(ce_row["ce_nats"]),
            "ce_bits": float(ce_row["ce_bits"]),
            "mape_full": float(row["MAPE vs. Full Human (%)"]),
            "mse_full": float(row["MSE vs. Full Human"]),
            "rho_full": float(row["Spearman rho vs. Full Human"]),
        }

    scale_order = {"0.8B": 0, "4B": 1, "9B": 2}

    def sort_key(m_name: str) -> tuple[int, int]:
        m = re.search(r"(\d+(?:\.\d+)?B)", m_name)
        s_idx = scale_order[m.group(1)] if m and m.group(1) in scale_order else 99
        ft_idx = 1 if "FT" in m_name else 0
        return (s_idx, ft_idx)

    sorted_models = sorted(by_model.keys(), key=sort_key)

    h_row = comparison_metrics_df.loc[human_sub_key]
    table_rows: list[dict[str, str]] = [
        {
            "Estimator": "Human subsample (N = 200, n = 1,200)",
            "Random CE (nats)": "—",
            "Random MAPE (%)": "—",
            "Random MSE": "—",
            "Respondent CE (nats)": "—",
            "Respondent MAPE (%)": f"{float(h_row['MAPE vs. Full Human (%)']):.1f}%",
            "Respondent MSE": f"{float(h_row['MSE vs. Full Human']):.4f}",
            "Spearman rho": f"{float(h_row['Spearman rho vs. Full Human']):+.2f}",
        }
    ]

    for m_name in sorted_models:
        entries = by_model[m_name]
        if "Random Persona" not in entries or "Respondent Persona" not in entries:
            raise ValueError(f"Expected both Random and Respondent Persona for {m_name}, got {list(entries.keys())}")
        rnd = entries["Random Persona"]
        prs = entries["Respondent Persona"]
        table_rows.append({
            "Estimator": m_name,
            "Random CE (nats)": f"{rnd['ce_nats']:.4f}",
            "Random MAPE (%)": f"{rnd['mape_full']:.1f}%",
            "Random MSE": f"{rnd['mse_full']:.4f}",
            "Respondent CE (nats)": f"{prs['ce_nats']:.4f}",
            "Respondent MAPE (%)": f"{prs['mape_full']:.1f}%",
            "Respondent MSE": f"{prs['mse_full']:.4f}",
            "Spearman rho": f"{prs['rho_full']:+.2f}",
        })

    return pd.DataFrame(table_rows)


def build_grummon_coef_table(
    human_ols_df: pd.DataFrame,
    human_sub_ols_df: pd.DataFrame,
    model_ols_dfs: dict[str, pd.DataFrame],
    persona_mode: str = "Respondent Persona",
) -> pd.DataFrame:
    """Build a Web Appendix table of the d=5 HC1 OLS treatment-effect estimates (and SEs) across estimators."""
    selected_models: list[tuple[str, pd.DataFrame]] = [
        ("Full human (N = 5,633)", human_ols_df),
        ("Human subsample (N = 200)", human_sub_ols_df),
    ]

    scale_order = {"0.8B": 0, "4B": 1, "9B": 2}
    matched: list[tuple[str, pd.DataFrame]] = []
    for key, df_m in model_ols_dfs.items():
        meta = parse_entry_metadata(key)
        if meta["persona"] == persona_mode:
            matched.append((str(meta["model"]), df_m))

    def model_order(item: tuple[str, pd.DataFrame]) -> tuple[int, int]:
        m_name = item[0]
        m = re.search(r"(\d+(?:\.\d+)?B)", m_name)
        s_idx = scale_order[m.group(1)] if m and m.group(1) in scale_order else 99
        ft_idx = 1 if "FT" in m_name else 0
        return (s_idx, ft_idx)

    selected_models.extend(sorted(matched, key=model_order))

    # Order rows from largest to smallest Full Human treatment effect
    ordered_labels = list(human_ols_df["beta"].sort_values(ascending=False).index)
    rows: list[dict[str, str]] = []
    for arm_label in ordered_labels:
        row_dict: dict[str, str] = {"Trial arm (vs. Positive)": str(arm_label)}
        for col_name, df_est in selected_models:
            b = float(df_est.loc[arm_label, "beta"])
            se = float(df_est.loc[arm_label, "se"])
            row_dict[col_name] = f"{b:+.3f} ({se:.3f})"
        rows.append(row_dict)

    return pd.DataFrame(rows)


def load_single_attr_ce_table(
    sub_df: pd.DataFrame,
    *,
    ablation_dir: str | Path = "data/persona_ablation",
) -> pd.DataFrame:
    """Compute per-attribute choice cross-entropy (in nats) across the 5 seen and 5 unseen shopper attributes."""
    abl_path = Path(ablation_dir)
    ans_map: dict[tuple[str, str], int] = {}
    for row in sub_df.to_dict("records"):
        oh = list(row["one_hot"])
        if 1.0 not in oh:
            raise ValueError(f"Invalid one_hot target {oh} for CASEID={row['CASEID']!r}")
        ans_map[(str(row["CASEID"]), str(row["v"]))] = oh.index(1.0)

    model_specs = [
        ("Kev 0.8B (base)", "kev-0.8b-base"),
        ("Kev 0.8B FT", "kev-0.8b-resp-issp+twin2k-grouped-newvalid_best"),
        ("Kev 4B (base)", "kev-4b-base"),
        ("Kev 4B FT", "kev-4b-resp-issp+twin2k-grouped-newvalid_best"),
        ("Kev 9B (base)", "kev-9b-base"),
        ("Kev 9B FT", "kev-9b-resp-issp+twin2k-grouped-newvalid_best"),
    ]

    all_cols = list(SEEN_PERSONA_COLS) + list(UNSEEN_PERSONA_COLS)
    per_attr: dict[str, dict[str, float]] = {c: {} for c in all_cols}
    for label, slug in model_specs:
        smap = load_single_attr_jsonl(abl_path / f"{slug}_single_attrs.jsonl")
        if not smap:
            raise FileNotFoundError(f"Missing single-attribute records in {abl_path / f'{slug}_single_attrs.jsonl'}")
        by_col: dict[str, list[float]] = {c: [] for c in all_cols}
        for (col, cid, v), rec in smap.items():
            ans_idx = ans_map[(cid, v)]
            probs = [float(rec["prob_correct"]), float(rec["prob_wrong"]), float(rec["prob_same"])]
            by_col[col].append(-math.log(max(probs[ans_idx], 1e-12)))
        for col in all_cols:
            if not by_col[col]:
                raise ValueError(f"No single-attribute items found for {col!r} in {slug}")
            per_attr[col][label] = float(np.mean(by_col[col]))

    rows: list[dict[str, object]] = []
    for col in all_cols:
        grp = "Seen in training" if col in SEEN_PERSONA_COLS else "Unseen in training"
        rows.append({
            "Attribute group": grp,
            "Code": col,
            "Shopper attribute": GRUMMON_PERSONA_NAMES[col],
            **{lbl: per_attr[col][lbl] for lbl, _ in model_specs},
        })
    return pd.DataFrame(rows)


def plot_grummon_error_comparison(
    comparison_metrics_df: pd.DataFrame,
    palette: str = "nature",
    persona_filter: str | None = None,
    sort_by: str = "MSE vs. Full Human",
    default_n: int = 1200,
) -> plt.Figure:
    """Plot a two-panel horizontal bar chart of MAPE (%) and MSE vs. Full Human beta* sorted from best to worst on MSE."""
    set_academic_palette(palette)
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
    models = [format_entry_display_label(m, default_n=default_n) for m in raw_models]

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


def plot_grummon_coefs_and_attributes(
    human_ols_df: pd.DataFrame,
    human_sub_ols_df: pd.DataFrame,
    model_ols_dfs: dict[str, pd.DataFrame],
    single_attr_ce_df: pd.DataFrame,
    palette: str = "nature",
    persona_mode: str = "Respondent Persona",
) -> plt.Figure:
    """Plot a two-panel main-text figure for the Grummon et al. (2025) nutrition labeling RCT:
    Panel A: HC1 OLS treatment-effect coefficients and 95% CIs across the 5 FOP arms (Human vs. Kev 9B).
    Panel B: Single-attribute choice cross-entropy (nats) across the 5 seen vs. 5 unseen shopper attributes.
    """
    set_academic_palette(palette)
    by_name: dict[str, pd.DataFrame] = {}
    for key, df_m in model_ols_dfs.items():
        meta = parse_entry_metadata(key)
        if meta["persona"] == persona_mode:
            by_name[str(meta["model"])] = df_m

    if "Kev 9B (base)" not in by_name or "Kev 9B FT" not in by_name:
        raise KeyError(f"Expected Kev 9B (base) and Kev 9B FT in {list(by_name.keys())}")

    # Order Panel A arms from largest to smallest Full Human treatment effect
    ordered_labels = list(human_ols_df["beta"].sort_values(ascending=False).index)
    short_arm_labels = [
        lbl.replace(" (vs. Positive)", "")
           .replace("Nutrition Info label", "Nutrition Info")
           .replace("High In warning label", "High In warning")
           .replace("Spectrum 1-to-5 star label", "Spectrum (1–5 stars)")
        for lbl in ordered_labels
    ]
    y_arms = np.arange(len(ordered_labels), dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.9), dpi=150)
    fig.subplots_adjust(wspace=0.38)

    # Panel A: Forest plot of d=5 HC1 OLS treatment effects
    panel_a_specs: list[tuple[str, pd.DataFrame, str, str]] = [
        ("Full human (N = 5,633)", human_ols_df, MODEL_PALETTES[palette]["human"], "o"),
        ("Human subsample (N = 200)", human_sub_ols_df, NEUTRALS["baseline"], "s"),
        ("Kev 9B (base)", by_name["Kev 9B (base)"], MODEL_PALETTES[palette]["off_the_shelf"], "^"),
        ("Kev 9B FT", by_name["Kev 9B FT"], MODEL_PALETTES[palette]["fine_tuned"], "D"),
    ]
    offsets_a = np.linspace(-0.24, 0.24, len(panel_a_specs))
    ax1.axvline(0.0, color=NEUTRALS["ref_line"], linestyle=":", linewidth=1.0, zorder=1)
    for offset, (name, df_m, color, marker) in zip(offsets_a, panel_a_specs, strict=True):
        betas = df_m.loc[ordered_labels, "beta"].to_numpy(dtype=np.float64)
        ses = df_m.loc[ordered_labels, "se"].to_numpy(dtype=np.float64)
        ax1.errorbar(
            betas,
            y_arms + offset,
            xerr=1.96 * ses,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.3,
            markersize=5.4,
            label=name,
            zorder=3,
        )

    ax1.set_yticks(y_arms, labels=short_arm_labels, color=NEUTRALS["text"], fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlabel("HC1 OLS treatment effect vs. Positive arm β̂", color=NEUTRALS["muted"], fontsize=9.5)
    ax1.set_title("A. Causal treatment-effect estimates (5 arms)", color=NEUTRALS["text"], fontsize=10.5, fontweight="bold", loc="left", pad=8)
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
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8.5,
    )

    # Panel B: Seen vs. Unseen shopper attributes single-attribute choice cross-entropy (nats)
    short_attr_map = {
        "Sex": "Sex",
        "Age": "Age",
        "Education": "Education",
        "Income": "Household income",
        "Race": "Race",
        "Hispanic or Latino": "Hispanic ethnicity*",
        "SNAP / Food Stamps recipient": "SNAP participation*",
        "Nutrition Facts label usage": "Nutrition Facts usage*",
        "Share of groceries bought online": "Online grocery share*",
        "Nutrition literacy score": "Nutrition literacy*",
    }
    attr_labels = [short_attr_map[str(a)] for a in single_attr_ce_df["Shopper attribute"]]
    y_attrs = np.arange(len(attr_labels), dtype=np.float64)

    ce_9b_base = single_attr_ce_df["Kev 9B (base)"].to_numpy(dtype=np.float64)
    ce_9b_ft = single_attr_ce_df["Kev 9B FT"].to_numpy(dtype=np.float64)

    uniform_ce_nats = math.log(3.0)
    ax2.axvline(uniform_ce_nats, color=NEUTRALS["ref_line"], linestyle=":", linewidth=1.1, zorder=1, label="Uniform 3-option (1.099)")
    ax2.axhline(4.5, color=NEUTRALS["muted"], linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)

    for idx in range(len(y_attrs)):
        ax2.plot(
            [ce_9b_base[idx], ce_9b_ft[idx]],
            [y_attrs[idx], y_attrs[idx]],
            color=NEUTRALS["baseline"],
            linewidth=1.5,
            zorder=2,
        )
    ax2.scatter(
        ce_9b_base,
        y_attrs,
        marker="^",
        s=42,
        color=MODEL_PALETTES[palette]["off_the_shelf"],
        label="Kev 9B (base)",
        zorder=3,
    )
    ax2.scatter(
        ce_9b_ft,
        y_attrs,
        marker="D",
        s=36,
        color=MODEL_PALETTES[palette]["fine_tuned"],
        label="Kev 9B FT",
        zorder=4,
    )

    ax2.set_yticks(y_attrs, labels=attr_labels, color=NEUTRALS["text"], fontsize=8.9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Choice cross-entropy (nats; *unseen in training)", color=NEUTRALS["muted"], fontsize=9.5)
    ax2.set_title("B. Seen vs. unseen shopper attributes (10 attrs)", color=NEUTRALS["text"], fontsize=10.5, fontweight="bold", loc="left", pad=8)
    ax2.tick_params(axis="x", colors=NEUTRALS["muted"], labelsize=9.0)
    ax2.grid(axis="x", color=NEUTRALS["grid"], linestyle="--", linewidth=0.8, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(NEUTRALS["muted"])
    ax2.spines["bottom"].set_color(NEUTRALS["muted"])

    # Annotate Seen vs. Unseen regions cleanly on the right margin inside Panel B
    x_max = max(float(np.max(ce_9b_base)), uniform_ce_nats) + 0.035
    x_min = float(np.min(ce_9b_ft)) - 0.025
    ax2.set_xlim(x_min, x_max)
    ax2.text(
        x_max - 0.005,
        0.35,
        "Seen in survey training (5)",
        ha="right",
        va="center",
        fontsize=8.0,
        fontstyle="italic",
        color=NEUTRALS["muted"],
    )
    ax2.text(
        x_max - 0.005,
        5.35,
        "*Unseen in training (5)",
        ha="right",
        va="center",
        fontsize=8.0,
        fontstyle="italic",
        color=NEUTRALS["muted"],
    )
    ax2.legend(
        frameon=True,
        facecolor="white",
        edgecolor=NEUTRALS["legend_edge"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8.5,
    )

    plt.tight_layout()
    return fig

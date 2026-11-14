"""HC1 OLS treatment-effect estimation, dynamic Kev dataset loader, and plotting utilities for Grummon et al. (2025)."""

from collections.abc import Sequence
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

from data.grummon2025 import (
    GRUMMON2025_PARQUET_PATH,
    GRUMMON2025_PERSONA_NAMES,
    PRIMARY_CHOICE_STEMS,
    TREAT_ARMS,
    load_grummon2025_df,
)
from kev_eval import (
    load_kev_jsonl,
    load_single_attr_jsonl,
)

# Label prefix for Kev runs collected on the audited grummon2025_US_long.parquet prompts (data-audit/)
AUDIT_PREFIX = "audit-"

# Persona mode saved in records -> label used in regression keys and tables
PERSONA_LABELS: dict[str, str] = {"random": "Random Persona", "persona": "Respondent Persona"}

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


def _row_scalar_outcome(row: pd.Series) -> float:
    oh = [float(x) for x in row["one_hot"]]
    v_name = str(row["v"])
    q_type = str(row["type"])
    if v_name in PRIMARY_CHOICE_STEMS:
        return oh[0]
    if q_type == "score":
        return float(sum((idx + 1) * p for idx, p in enumerate(oh)))
    opts_list = [str(t) for t in row["option_texts"]]
    if q_type == "choice" and "Yes" in opts_list:
        return oh[opts_list.index("Yes")]
    raise ValueError(f"{row['qid']}: unsupported outcome type={q_type!r}, options={row['option_texts']!r}")


def load_grummon2025_rct(
    parquet_path: str | Path = GRUMMON2025_PARQUET_PATH,
    *,
    all_included: bool = False,
) -> pd.DataFrame:
    """Load and validate the Grummon et al. (2025) FOP nutrition label RCT dataset.

    By default (`all_included=False`), filters to the 6 primary `id*` consumer-understanding choice tasks
    (`33,798` rows across `5,633` participants) even when `grummon2025_US_long.parquet` also includes
    audited secondary `score` outcomes. Pass `all_included=True` to return all included outcomes with
    `y_outcome` (`P(healthier)` for `id*`, expected 1..K scale value for `score` items).
    """
    df = load_grummon2025_df(parquet_path=parquet_path)
    if all_included:
        df = df.copy()
        df["y_outcome"] = [_row_scalar_outcome(r) for _, r in df.iterrows()]
        return df.sort_values(["CASEID", "v"]).reset_index(drop=True)

    df = df[df["v"].isin(PRIMARY_CHOICE_STEMS)].copy()
    if len(df) != 33798 or df["CASEID"].nunique() != 5633:
        raise ValueError(
            f"Unexpected Grummon 2025 primary choice shape: {df.shape}, CASEID={df['CASEID'].nunique()}"
        )
    df["y_correct"] = df["one_hot"].map(lambda oh: float(oh[0]))
    return df.sort_values(["CASEID", "v"]).reset_index(drop=True)


def subsample_grummon2025_participants(
    df: pd.DataFrame,
    *,
    n_participants: int,
    seed: int,
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


def fit_grummon2025_ols(
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


def evaluate_grummon2025_metrics(
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
    # Audited-prompt Kev runs keep their prefix in front of the formatted name (e.g. "audit-Kev 9B (FT)")
    if s.startswith(AUDIT_PREFIX):
        return AUDIT_PREFIX + format_raw_model_name(s[len(AUDIT_PREFIX):])
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
            suffix = ", ep2"
        elif lower.endswith("/last"):
            suffix = ", last"
        return f"Kev {size_str} (FT{suffix})"

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


def format_entry_display_label(entry_key: str, *, default_n: int) -> str:
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
    kev_data_dir: str | Path,
    model_ids: Sequence[str],
    persona_modes: Sequence[str],
    kev_label_prefix: str = "",
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Load the `persona_modes` JSONLs ("random" and/or "persona") of each model in `model_ids` from `kev_data_dir`,
    compute choice CE, and fit HC1 OLS regressions (labels follow the order of `model_ids`).

    Files of models not in `model_ids` are ignored; a missing (model, persona mode) file raises.
    `kev_label_prefix` is prepended to every label (e.g. AUDIT_PREFIX for data-audit/).
    """
    kev_path = Path(kev_data_dir)
    if not kev_path.is_dir():
        raise FileNotFoundError(f"Kev data folder not found: {kev_path.resolve()}")

    # (model_id, persona_mode) -> JSONL path, for the requested models only
    files: dict[tuple[str, str], Path] = {}
    for jsonl_path in sorted(kev_path.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            first_rec = json.loads(f.readline())
        model_id = str(first_rec["model"])
        if model_id not in model_ids:
            continue
        persona_mode = str(first_rec["persona_mode"])
        if persona_mode not in PERSONA_LABELS:
            raise ValueError(f"Unexpected persona_mode={persona_mode!r} in {jsonl_path}")
        if (model_id, persona_mode) in files:
            raise ValueError(f"Two files for {(model_id, persona_mode)}: {files[(model_id, persona_mode)]}, {jsonl_path}")
        files[(model_id, persona_mode)] = jsonl_path

    sub_keys = [(str(r["CASEID"]), str(r["v"])) for r in sub_df[["CASEID", "v"]].to_dict("records")]
    ols_dfs: dict[str, pd.DataFrame] = {}
    ce_stats: dict[str, dict[str, float]] = {}
    for model_id in model_ids:
        for persona_mode in persona_modes:
            persona_label = PERSONA_LABELS[persona_mode]
            if (model_id, persona_mode) not in files:
                raise FileNotFoundError(f"No {persona_mode!r} JSONL for {model_id!r} in {kev_path}")
            jsonl_path = files[(model_id, persona_mode)]
            with jsonl_path.open(encoding="utf-8") as f:
                temp = float(json.loads(f.readline())["temperature"])
            k_map = load_kev_jsonl(
                jsonl_path,
                expected_model=model_id,
                expected_persona_mode=persona_mode,
                expected_temperature=temp,
            )
            avail_idx = [i for i, k in enumerate(sub_keys) if k in k_map]
            if not avail_idx:
                raise ValueError(f"{jsonl_path} has no records for the evaluation subsample")
            avail_df = sub_df.iloc[avail_idx].copy().reset_index(drop=True)
            avail_df["prob_correct"] = [float(k_map[sub_keys[i]]["prob_correct"]) for i in avail_idx]
            if set(avail_df["arm"].unique()) != {"Positive", *TREAT_ARMS}:
                raise ValueError(f"{jsonl_path} does not cover every trial arm: {sorted(avail_df['arm'].unique())}")

            n_participants = int(avail_df["CASEID"].nunique())
            label = f"{kev_label_prefix}{model_id} ({persona_label}, N={n_participants}, n={len(avail_idx)})"
            ols_dfs[label] = fit_grummon2025_ols(avail_df, outcome_col="prob_correct")
            ce_info = compute_choice_ce_stats(sub_df, k_map, avail_idx)
            ce_info["temperature"] = temp
            ce_info["prob_correct_mean"] = float(avail_df["prob_correct"].mean())
            ce_info["prob_same_mean"] = float(np.mean([float(k_map[sub_keys[i]]["prob_same"]) for i in avail_idx]))
            ce_stats[label] = ce_info

    return ols_dfs, ce_stats


def _declared_model_order(entry_keys) -> dict[str, int]:
    """Map each display name (without AUDIT_PREFIX) to the position of its first appearance in `entry_keys`."""
    order: dict[str, int] = {}
    for key in entry_keys:
        base = str(parse_entry_metadata(str(key))["model"]).removeprefix(AUDIT_PREFIX)
        order.setdefault(base, len(order))
    return order


def _order_key(m_name: str, order: dict[str, int]) -> tuple[int, int]:
    """Sort by declared model order, each original row followed by its audited row."""
    return (order[m_name.removeprefix(AUDIT_PREFIX)], int(m_name.startswith(AUDIT_PREFIX)))


def build_grummon2025_summary_table(
    comparison_metrics_df: pd.DataFrame,
    ce_stats: dict[str, dict[str, float]],
    *,
    human_sub_key: str,
    persona_modes: Sequence[str],
    n_sub_participants: int,
    n_sub_tasks: int,
) -> pd.DataFrame:
    """Build a publication-ready table of choice cross-entropy and OLS recovery for each persona mode in `persona_modes`.

    Columns are CE (bits), MAPE (%) and MSE per mode (prefixed "Random" / "Respondent" only when there are several
    modes), plus Spearman rho for the respondent persona when it is included. The human subsample row fills the
    respondent-persona columns (it has no CE).
    """
    if human_sub_key not in comparison_metrics_df.index:
        raise KeyError(f"Missing {human_sub_key!r} in comparison_metrics_df")
    unknown = [m for m in persona_modes if m not in PERSONA_LABELS]
    if unknown or not persona_modes:
        raise ValueError(f"persona_modes must be a non-empty subset of {list(PERSONA_LABELS)}, got {list(persona_modes)}")
    regimes = [PERSONA_LABELS[m] for m in persona_modes]

    by_model: dict[str, dict[str, dict[str, float]]] = {}
    for entry_key, row in comparison_metrics_df.iterrows():
        key_str = str(entry_key)
        if key_str == human_sub_key:
            continue
        meta = parse_entry_metadata(key_str)
        persona = str(meta["persona"])
        if persona not in regimes:
            continue
        ce_row = ce_stats[key_str]
        by_model.setdefault(str(meta["model"]), {})[persona] = {
            "ce_bits": float(ce_row["ce_bits"]),
            "mape_full": float(row["MAPE vs. Full Human (%)"]),
            "mse_full": float(row["MSE vs. Full Human"]),
            "rho_full": float(row["Spearman rho vs. Full Human"]),
        }

    # Row order follows the order models were loaded in (ce_stats insertion order)
    order = _declared_model_order(ce_stats.keys())
    sorted_models = sorted(by_model.keys(), key=lambda m_name: _order_key(m_name, order))

    def col(regime: str, metric: str) -> str:
        return f"{regime.split()[0]} {metric}" if len(regimes) > 1 else metric

    h_row = comparison_metrics_df.loc[human_sub_key]
    human_row: dict[str, str] = {"Estimator": f"Human subsample (N = {n_sub_participants:,}, n = {n_sub_tasks:,})"}
    for regime in regimes:
        is_resp = regime == "Respondent Persona"
        human_row[col(regime, "CE (bits)")] = "—"
        human_row[col(regime, "MAPE (%)")] = f"{float(h_row['MAPE vs. Full Human (%)']):.1f}%" if is_resp else "—"
        human_row[col(regime, "MSE")] = f"{float(h_row['MSE vs. Full Human']):.4f}" if is_resp else "—"
    if "Respondent Persona" in regimes:
        human_row["Spearman rho"] = f"{float(h_row['Spearman rho vs. Full Human']):+.2f}"
    table_rows = [human_row]

    for m_name in sorted_models:
        entries = by_model[m_name]
        missing = [r for r in regimes if r not in entries]
        if missing:
            raise ValueError(f"{m_name} has no rows for {missing}")
        model_row: dict[str, str] = {"Estimator": m_name}
        for regime in regimes:
            e = entries[regime]
            model_row[col(regime, "CE (bits)")] = f"{e['ce_bits']:.4f}"
            model_row[col(regime, "MAPE (%)")] = f"{e['mape_full']:.1f}%"
            model_row[col(regime, "MSE")] = f"{e['mse_full']:.4f}"
        if "Respondent Persona" in regimes:
            model_row["Spearman rho"] = f"{entries['Respondent Persona']['rho_full']:+.2f}"
        table_rows.append(model_row)

    return pd.DataFrame(table_rows)


def build_grummon2025_coef_table(
    human_ols_df: pd.DataFrame,
    human_sub_ols_df: pd.DataFrame,
    model_ols_dfs: dict[str, pd.DataFrame],
    persona_mode: str = "Respondent Persona",
    *,
    n_full_participants: int,
    n_sub_participants: int,
) -> pd.DataFrame:
    """Build a Web Appendix table of the d=5 HC1 OLS treatment-effect estimates (and SEs) across estimators."""
    selected_models: list[tuple[str, pd.DataFrame]] = [
        (f"Full human (N = {n_full_participants:,})", human_ols_df),
        (f"Human subsample (N = {n_sub_participants:,})", human_sub_ols_df),
    ]

    matched: list[tuple[str, pd.DataFrame]] = []
    for key, df_m in model_ols_dfs.items():
        meta = parse_entry_metadata(key)
        if meta["persona"] == persona_mode:
            matched.append((str(meta["model"]), df_m))

    # Column order follows the order models were loaded in (model_ols_dfs insertion order)
    order = _declared_model_order(model_ols_dfs.keys())
    selected_models.extend(sorted(matched, key=lambda item: _order_key(item[0], order)))

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
    ablation_dir: str | Path,
    model_ids: Sequence[str],
    seen_cols: Sequence[str],
    unseen_cols: Sequence[str],
    label_prefix: str = "",
) -> pd.DataFrame:
    """Compute per-attribute choice cross-entropy (in bits) across seen and unseen shopper attributes.

    Reads `<ablation_dir>/*_single_attrs.jsonl` for each model in `model_ids` (columns in that order, named by
    format_raw_model_name). Each file must cover exactly `seen_cols` + `unseen_cols`.
    `label_prefix` is prepended to every model column (e.g. AUDIT_PREFIX for data-audit/persona_ablation).
    """
    abl_path = Path(ablation_dir)
    ans_map: dict[tuple[str, str], int] = {}
    for row in sub_df.to_dict("records"):
        oh = list(row["one_hot"])
        if 1.0 not in oh:
            raise ValueError(f"Invalid one_hot target {oh} for CASEID={row['CASEID']!r}")
        ans_map[(str(row["CASEID"]), str(row["v"]))] = oh.index(1.0)

    # model_id -> single-attribute JSONL path
    files: dict[str, Path] = {}
    for jsonl_path in sorted(abl_path.glob("*_single_attrs.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            model_id = str(json.loads(f.readline())["model"])
        if model_id in files:
            raise ValueError(f"Two single-attribute files for {model_id!r}: {files[model_id]}, {jsonl_path}")
        files[model_id] = jsonl_path
    missing = [m for m in model_ids if m not in files]
    if missing:
        raise FileNotFoundError(f"No single-attribute JSONL in {abl_path} for {missing}")
    model_specs = [(f"{label_prefix}{format_raw_model_name(m)}", files[m]) for m in model_ids]

    all_cols = list(seen_cols) + list(unseen_cols)
    per_attr: dict[str, dict[str, float]] = {c: {} for c in all_cols}
    for label, jsonl_path in model_specs:
        smap = load_single_attr_jsonl(jsonl_path)
        file_cols = {key[0] for key in smap}
        if file_cols != set(all_cols):
            raise ValueError(f"{jsonl_path} covers {sorted(file_cols)}, expected {sorted(all_cols)}")
        by_col: dict[str, list[float]] = {c: [] for c in all_cols}
        for (col, cid, v), rec in smap.items():
            ans_idx = ans_map[(cid, v)]
            probs = [float(rec["prob_correct"]), float(rec["prob_wrong"]), float(rec["prob_same"])]
            by_col[col].append(-math.log2(max(probs[ans_idx], 1e-12)))
        for col in all_cols:
            if not by_col[col]:
                raise ValueError(f"No single-attribute items found for {col!r} in {jsonl_path}")
            per_attr[col][label] = float(np.mean(by_col[col]))

    rows: list[dict[str, object]] = []
    for col in all_cols:
        grp = "Seen in training" if col in seen_cols else "Unseen in training"
        rows.append({
            "Attribute group": grp,
            "Code": col,
            "Shopper attribute": GRUMMON2025_PERSONA_NAMES[col],
            **{lbl: per_attr[col][lbl] for lbl, _ in model_specs},
        })
    return pd.DataFrame(rows)

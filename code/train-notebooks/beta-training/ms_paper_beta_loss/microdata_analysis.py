"""Shared data loading, validation, formatting, and plotting helpers for the microdata choice-training analysis notebooks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

PALETTE_NAME = "lancet"
PALETTES = {
    "nature": ["#1B365D", "#9E2A2B", "#2E8B78", "#D9822B", "#6F5091", "#4A6F8A"],
    "lancet": ["#1B365D", "#9E2A2B", "#2E8B78", "#D9822B", "#6F5091", "#4A6F8A"],
}
NEUTRALS = {
    "text": "#2B2B2B",
    "muted": "#52514E",
    "ref_line": "#6E6A64",
    "baseline": "#9A9893",
    "grid": "#D4D2CC",
    "legend_edge": "#E0DED8",
}
PAL = PALETTES[PALETTE_NAME]

PAPER_DIR = Path(__file__).resolve().parent
BETA_TRAINING_DIR = PAPER_DIR.parent
CODE_DIR = BETA_TRAINING_DIR.parent.parent
if not all((CODE_DIR / d).is_dir() for d in ("kev", "data", "training")):
    raise RuntimeError(f"Expected code/ root containing kev/, data/, training/ at {CODE_DIR}")

PAPER_RUNS_ROOT = PAPER_DIR / "runs-rorqual"
REGIME_RUN_ROOTS = (PAPER_DIR / "runs-rorqual", PAPER_DIR / "runs-nibi")
TEST_BEST_SUBDIR = "test_best"
FIGURES_DIR = CODE_DIR.parent / "manuscript" / "figures"
if not FIGURES_DIR.is_dir():
    raise FileNotFoundError(f"Expected manuscript figures directory at {FIGURES_DIR}")

MODEL_SLUGS = ("kev-0.8b", "kev-4b", "kev-9b")
MODEL_DISPLAY = {
    "kev-0.8b": "KEV-0.8B",
    "kev-4b": "KEV-4B",
    "kev-9b": "KEV-9B",
}
VALIDATORS = ("kriner2026", "grummon2025")

BASE_SLUG = "row1_base_untrained"
MSE_SCALE = 1000.0
REGIME_LABELS = {
    "row1_base_untrained": "Off-the-shelf",
    "row2b_wu_ce_oracle": "Electricity",
    "row3_berke_ce_only": "Drones",
    "row5_full_berke_wu_ce_oracle": "Drones + Electricity",
}
ALL_SLUGS = tuple(REGIME_LABELS.keys())
TRAINED_SLUGS = tuple(k for k in ALL_SLUGS if k != BASE_SLUG)

STORY_COLUMNS = (
    ("kriner", "Flights"),
    ("kantorowicz", "Consumer rights"),
    ("magistro", "AI & offshoring"),
    ("grummon", "Food labels"),
    ("covid", "Vaccines"),
)
STORY_ROW_GROUPS = (
    ("Validation", ("kriner",)),
    ("Test\n(price studies)", ("kantorowicz", "magistro")),
    ("Test\n(non-price studies)", ("grummon", "covid")),
)
STORY_LABELS = dict(REGIME_LABELS)
STORY_COLORS = {
    "row1_base_untrained": NEUTRALS["baseline"],
    "row2b_wu_ce_oracle": PAL[2],
    "row3_berke_ce_only": PAL[4],
    "row5_full_berke_wu_ce_oracle": PAL[0],
}
ROLES_NOTE = (
    "Food labels (Grummon et al., 2025), Vaccines (Kreps et al., 2020), "
    "Consumer rights (Kantorowicz-Reznichenko et al., 2026), and AI & offshoring (Magistro et al., 2026) "
    "are held-out test datasets; Flights (Goldfarb et al., 2026) is the validation dataset and selects each run's checkpoint (best.pt)."
)
TABLE_LETTERS = "abcdefgh"

FIGURE2_SLUGS = (BASE_SLUG, "row5_full_berke_wu_ce_oracle")
HUMAN_TICK_HALF_HEIGHT = 0.34


def setup_analysis_env() -> None:
    """Configure pandas/matplotlib styling, sys.path, and working directory for analysis notebooks."""
    plt.rcParams["axes.prop_cycle"] = cycler(color=PAL)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", None)
    os.chdir(CODE_DIR)
    for path_str in (str(CODE_DIR), str(CODE_DIR / "kev"), str(BETA_TRAINING_DIR), str(PAPER_DIR)):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def print_caption(label: str, title: str) -> None:
    """APA caption printed above a table or figure."""
    print(label)
    print(title)


def print_note(note: str) -> None:
    """APA general note printed below a table or figure."""
    print(f"Note. {note}")


def save_manuscript_fig(fig: plt.Figure, stem: str) -> None:
    """Save a manuscript figure as 300-DPI PNG and vector PDF into FIGURES_DIR."""
    for ext in ("png", "pdf"):
        out_path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / stem}.{{png,pdf}}")


def find_run_dir(model_slug: str, slug: str) -> Path:
    """Return the unique regime run directory across REGIME_RUN_ROOTS, or the default path if missing."""
    found = []
    for root in REGIME_RUN_ROOTS:
        candidate = root / model_slug / slug
        if (candidate / "run.json").exists():
            found.append(candidate)
    if len(found) > 1:
        raise ValueError(f"{model_slug}/{slug} exists in more than one run root: {found}")
    if len(found) == 1:
        return found[0]
    return REGIME_RUN_ROOTS[0] / model_slug / slug


def run_is_finished(run_dir: Path) -> bool:
    """True when the run folder has run.json and status == finished."""
    setup_analysis_env()
    from beta_train import runs as run_store

    if not (run_dir / "run.json").exists():
        return False
    return run_store.read_run_json(run_dir)["status"] == run_store.STATUS_FINISHED


def load_test_best_entry(run_or_row1_dir: Path, expected_step: int) -> dict | None:
    """Load test_best/ metrics and betas and verify checkpoint step and dataset alignment."""
    test_dir = run_or_row1_dir / TEST_BEST_SUBDIR
    test_json = test_dir / "test.json"
    if not test_json.exists():
        return None
    metrics_csv = test_dir / "metrics.csv"
    betas_csv = test_dir / "betas.csv"
    if not metrics_csv.exists() or not betas_csv.exists():
        raise FileNotFoundError(f"Incomplete test_best directory in {test_dir}")

    test_info = json.loads(test_json.read_text(encoding="utf-8"))
    actual_step = int(test_info["checkpoint_step"])
    if actual_step != expected_step:
        raise ValueError(f"Stale test_best in {test_dir}: checkpoint_step={actual_step}, expected {expected_step}")

    metrics_df = pd.read_csv(metrics_csv).set_index("dataset")
    for ds in ("grummon2025", "covid"):
        if ds not in metrics_df.index:
            raise ValueError(f"Dataset {ds!r} missing from {metrics_csv}")

    betas_df = pd.read_csv(betas_csv)
    required_cols = {"validator", "term", "human_pop_beta", "model_pop_beta"}
    if not required_cols.issubset(betas_df.columns):
        raise ValueError(f"Missing columns in {betas_csv}: expected {required_cols}, got {set(betas_df.columns)}")

    k26_in_metrics = "kantorowicz2026" in metrics_df.index
    k26_in_betas = bool((betas_df["validator"] == "kantorowicz2026").any())
    if k26_in_metrics != k26_in_betas:
        raise ValueError(
            f"Inconsistent kantorowicz2026 state in {test_dir}: "
            f"in metrics.csv={k26_in_metrics}, in betas.csv={k26_in_betas}"
        )
    kantorowicz_betas = (
        betas_df[betas_df["validator"] == "kantorowicz2026"].set_index("term") if k26_in_metrics else None
    )

    m25_in_metrics = "magistro2025" in metrics_df.index
    m25_in_betas = bool((betas_df["validator"] == "magistro2025").any())
    if m25_in_metrics != m25_in_betas:
        raise ValueError(
            f"Inconsistent magistro2025 state in {test_dir}: "
            f"in metrics.csv={m25_in_metrics}, in betas.csv={m25_in_betas}"
        )
    magistro_betas = (
        betas_df[betas_df["validator"] == "magistro2025"].set_index("term") if m25_in_metrics else None
    )

    return {
        "metrics_df": metrics_df,
        "covid_betas": betas_df[betas_df["validator"] == "covid"].set_index("term"),
        "grummon_betas": betas_df[betas_df["validator"] == "grummon2025"].set_index("term"),
        "kantorowicz_betas": kantorowicz_betas,
        "has_kantorowicz": k26_in_metrics,
        "magistro_betas": magistro_betas,
        "has_magistro": m25_in_metrics,
    }


@dataclass(frozen=True)
class MicrodataContext:
    regimes_by_model: dict[str, pd.DataFrame]
    grummon_betas_by_model: dict[str, dict[str, pd.DataFrame]]
    covid_betas_by_model: dict[str, dict[str, pd.DataFrame]]
    kantorowicz_betas_by_model: dict[str, dict[str, pd.DataFrame]]
    magistro_betas_by_model: dict[str, dict[str, pd.DataFrame]]
    kriner_betas_by_model: dict[str, dict[str, pd.DataFrame]]
    kriner_terms: list[str]
    kriner_transferred_count: int
    status_grid: pd.DataFrame


def load_microdata_context() -> MicrodataContext:
    """Load all microdata choice-training runs, validation slopes, and held-out test_best evaluations."""
    setup_analysis_env()
    from beta_train import runs as run_store
    from paper_beta import extract_step0_from_run_dir, load_paper_run

    regimes_by_model: dict[str, pd.DataFrame] = {}
    grummon_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    covid_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    kantorowicz_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    magistro_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    kriner_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}

    for model_slug in MODEL_SLUGS:
        model_dir = PAPER_RUNS_ROOT / model_slug
        existing_dirs = [find_run_dir(model_slug, s) for s in TRAINED_SLUGS if (find_run_dir(model_slug, s) / "run.json").exists()]
        if not existing_dirs:
            raise FileNotFoundError(f"No canonical microdata runs for {model_slug} in {REGIME_RUN_ROOTS}")

        step0 = extract_step0_from_run_dir(existing_dirs[0], VALIDATORS)
        step0_kriner_betas = step0["betas_by_validator"]["kriner2026"][["human_pop_beta", "model_pop_beta"]].copy()
        human_kriner = step0_kriner_betas["human_pop_beta"]

        grummon_by_slug: dict[str, pd.DataFrame] = {}
        covid_by_slug: dict[str, pd.DataFrame] = {}
        kantorowicz_by_slug: dict[str, pd.DataFrame] = {}
        magistro_by_slug: dict[str, pd.DataFrame] = {}
        kriner_by_slug: dict[str, pd.DataFrame] = {
            BASE_SLUG: step0_kriner_betas.loc[human_kriner.index, ["human_pop_beta", "model_pop_beta"]]
        }

        row1_tb = load_test_best_entry(model_dir / BASE_SLUG, expected_step=0)
        if row1_tb is not None:
            g0_tb = float(row1_tb["metrics_df"].loc["grummon2025", "beta_mse"])
            g0_val = step0["metrics"]["grummon2025/beta_mse"]
            if abs(g0_tb - g0_val) > 2e-4:
                raise ValueError(f"{model_slug} Off-the-shelf Grummon mismatch: val={g0_val:.6f} vs test_best={g0_tb:.6f}")
            row1_grummon_mse = g0_tb
            row1_grummon_kl = float(row1_tb["metrics_df"].loc["grummon2025", "kl_loss"])
            row1_covid_mse = float(row1_tb["metrics_df"].loc["covid", "beta_mse"])
            row1_covid_kl = float(row1_tb["metrics_df"].loc["covid", "kl_loss"])
            grummon_by_slug[BASE_SLUG] = row1_tb["grummon_betas"]
            covid_by_slug[BASE_SLUG] = row1_tb["covid_betas"]
            if row1_tb["has_kantorowicz"]:
                row1_k26_mse = float(row1_tb["metrics_df"].loc["kantorowicz2026", "beta_mse"])
                row1_k26_kl = float(row1_tb["metrics_df"].loc["kantorowicz2026", "kl_loss"])
                kantorowicz_by_slug[BASE_SLUG] = row1_tb["kantorowicz_betas"]
            else:
                row1_k26_mse, row1_k26_kl = np.nan, np.nan
            if row1_tb["has_magistro"]:
                row1_m25_mse = float(row1_tb["metrics_df"].loc["magistro2025", "beta_mse"])
                row1_m25_kl = float(row1_tb["metrics_df"].loc["magistro2025", "kl_loss"])
                magistro_by_slug[BASE_SLUG] = row1_tb["magistro_betas"]
            else:
                row1_m25_mse, row1_m25_kl = np.nan, np.nan
            row1_pending = [
                tag for ok, tag in ((row1_tb["has_kantorowicz"], "k26"), (row1_tb["has_magistro"], "m25")) if not ok
            ]
            row1_progress = "step 0 [test_best ✓]" if not row1_pending else f"step 0 [test_best: {', '.join(row1_pending)} pending]"
        else:
            row1_grummon_mse = step0["metrics"]["grummon2025/beta_mse"]
            row1_grummon_kl = step0["metrics"]["grummon2025/kl_loss"]
            row1_covid_mse = row1_covid_kl = row1_k26_mse = row1_k26_kl = row1_m25_mse = row1_m25_kl = np.nan
            row1_progress = "step 0 [test_best pending]"

        rows = [
            {
                "slug": BASE_SLUG,
                "regime": REGIME_LABELS[BASE_SLUG],
                "progress": row1_progress,
                "kriner_mse": step0["metrics"]["kriner2026/beta_mse"],
                "kriner_kl": step0["metrics"]["kriner2026/kl_loss"],
                "grummon_mse": row1_grummon_mse,
                "grummon_kl": row1_grummon_kl,
                "covid_mse": row1_covid_mse,
                "covid_kl": row1_covid_kl,
                "kantorowicz_mse": row1_k26_mse,
                "kantorowicz_kl": row1_k26_kl,
                "magistro_mse": row1_m25_mse,
                "magistro_kl": row1_m25_kl,
            }
        ]

        for slug in TRAINED_SLUGS:
            run_dir = find_run_dir(model_slug, slug)
            if not (run_dir / "run.json").exists():
                kriner_by_slug[slug] = pd.DataFrame({"human_pop_beta": human_kriner, "model_pop_beta": np.nan})
                rows.append(
                    {
                        "slug": slug,
                        "regime": REGIME_LABELS[slug],
                        "progress": "—",
                        "kriner_mse": np.nan,
                        "kriner_kl": np.nan,
                        "grummon_mse": np.nan,
                        "grummon_kl": np.nan,
                        "covid_mse": np.nan,
                        "covid_kl": np.nan,
                        "kantorowicz_mse": np.nan,
                        "kantorowicz_kl": np.nan,
                        "magistro_mse": np.nan,
                        "magistro_kl": np.nan,
                    }
                )
                continue

            run_info = run_store.read_run_json(run_dir)
            raw_status = run_info["status"]
            result = load_paper_run(run_dir)
            steps_done = len(result.step_df)
            best_step = int(result.best["step"])

            if raw_status != run_store.STATUS_FINISHED:
                kriner_by_slug[slug] = pd.DataFrame({"human_pop_beta": human_kriner, "model_pop_beta": np.nan})
                rows.append(
                    {
                        "slug": slug,
                        "regime": REGIME_LABELS[slug],
                        "progress": f"{raw_status} (@{best_step}/{steps_done})",
                        "kriner_mse": np.nan,
                        "kriner_kl": np.nan,
                        "grummon_mse": np.nan,
                        "grummon_kl": np.nan,
                        "covid_mse": np.nan,
                        "covid_kl": np.nan,
                        "kantorowicz_mse": np.nan,
                        "kantorowicz_kl": np.nan,
                        "magistro_mse": np.nan,
                        "magistro_kl": np.nan,
                    }
                )
                continue

            k_metrics, k_betas = result.best_validation["kriner2026"]
            if list(k_betas.index) != list(human_kriner.index):
                raise ValueError(f"{model_slug}/{slug}: Flights terms {list(k_betas.index)} != {list(human_kriner.index)}")
            kriner_by_slug[slug] = k_betas[["human_pop_beta", "model_pop_beta"]]

            tb_entry = load_test_best_entry(run_dir, expected_step=best_step)
            if tb_entry is not None:
                g_mse = float(tb_entry["metrics_df"].loc["grummon2025", "beta_mse"])
                g_kl = float(tb_entry["metrics_df"].loc["grummon2025", "kl_loss"])
                if "grummon2025" in result.best_validation:
                    g_val = float(result.best_validation["grummon2025"][0]["beta_mse"])
                    if abs(g_mse - g_val) > 2e-4:
                        raise ValueError(f"{model_slug}/{slug} Grummon mismatch: val={g_val:.6f} vs test_best={g_mse:.6f}")
                c_mse = float(tb_entry["metrics_df"].loc["covid", "beta_mse"])
                c_kl = float(tb_entry["metrics_df"].loc["covid", "kl_loss"])
                grummon_by_slug[slug] = tb_entry["grummon_betas"]
                covid_by_slug[slug] = tb_entry["covid_betas"]
                if tb_entry["has_kantorowicz"]:
                    k26_mse = float(tb_entry["metrics_df"].loc["kantorowicz2026", "beta_mse"])
                    k26_kl = float(tb_entry["metrics_df"].loc["kantorowicz2026", "kl_loss"])
                    kantorowicz_by_slug[slug] = tb_entry["kantorowicz_betas"]
                else:
                    k26_mse, k26_kl = np.nan, np.nan
                if tb_entry["has_magistro"]:
                    m25_mse = float(tb_entry["metrics_df"].loc["magistro2025", "beta_mse"])
                    m25_kl = float(tb_entry["metrics_df"].loc["magistro2025", "kl_loss"])
                    magistro_by_slug[slug] = tb_entry["magistro_betas"]
                else:
                    m25_mse, m25_kl = np.nan, np.nan
                pending = [
                    tag for ok, tag in ((tb_entry["has_kantorowicz"], "k26"), (tb_entry["has_magistro"], "m25")) if not ok
                ]
                tb_tag = "[test_best ✓]" if not pending else f"[test_best: {', '.join(pending)} pending]"
            else:
                c_mse = c_kl = k26_mse = k26_kl = m25_mse = m25_kl = np.nan
                tb_tag = "[test_best pending]"
                if "grummon2025" in result.best_validation:
                    g_mse = float(result.best_validation["grummon2025"][0]["beta_mse"])
                    g_kl = float(result.best_validation["grummon2025"][0]["kl_loss"])
                else:
                    g_valid = result.val_history_df["grummon2025/beta_mse"].dropna()
                    g_step = int(g_valid.index[np.abs(g_valid.index.to_numpy() - best_step).argmin()])
                    g_mse = float(g_valid.loc[g_step])
                    g_kl = float(result.val_history_df.loc[g_step, "grummon2025/kl_loss"])

            rows.append(
                {
                    "slug": slug,
                    "regime": REGIME_LABELS[slug],
                    "progress": f"{raw_status} (@{best_step}/{steps_done}) {tb_tag}",
                    "kriner_mse": float(k_metrics["beta_mse"]),
                    "kriner_kl": float(k_metrics["kl_loss"]),
                    "grummon_mse": g_mse,
                    "grummon_kl": g_kl,
                    "covid_mse": c_mse,
                    "covid_kl": c_kl,
                    "kantorowicz_mse": k26_mse,
                    "kantorowicz_kl": k26_kl,
                    "magistro_mse": m25_mse,
                    "magistro_kl": m25_kl,
                }
            )

        regimes_by_model[model_slug] = pd.DataFrame(rows).set_index("slug")
        grummon_betas_by_model[model_slug] = grummon_by_slug
        covid_betas_by_model[model_slug] = covid_by_slug
        kantorowicz_betas_by_model[model_slug] = kantorowicz_by_slug
        magistro_betas_by_model[model_slug] = magistro_by_slug
        kriner_betas_by_model[model_slug] = kriner_by_slug

    all_kriner_terms = list(kriner_betas_by_model[MODEL_SLUGS[0]][BASE_SLUG].index)
    price_terms = [t for t in all_kriner_terms if t.startswith("Price:")]
    fuel_terms = [t for t in all_kriner_terms if t.startswith("Fuel:")]
    if not price_terms or not fuel_terms:
        raise ValueError(f"Expected Price and Fuel terms in Flights: {all_kriner_terms}")
    transferred = price_terms + fuel_terms
    kriner_terms = transferred + [t for t in all_kriner_terms if t not in transferred]

    status_grid = pd.DataFrame(
        {MODEL_DISPLAY[m]: regimes_by_model[m].loc[list(ALL_SLUGS), "progress"].values for m in MODEL_SLUGS},
        index=[REGIME_LABELS[s] for s in ALL_SLUGS],
    )
    status_grid.index.name = "Regime"

    return MicrodataContext(
        regimes_by_model=regimes_by_model,
        grummon_betas_by_model=grummon_betas_by_model,
        covid_betas_by_model=covid_betas_by_model,
        kantorowicz_betas_by_model=kantorowicz_betas_by_model,
        magistro_betas_by_model=magistro_betas_by_model,
        kriner_betas_by_model=kriner_betas_by_model,
        kriner_terms=kriner_terms,
        kriner_transferred_count=len(transferred),
        status_grid=status_grid,
    )


def build_metric_table(
    regimes_by_model: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    slugs: tuple[str, ...],
    metric_suffix: str,
    columns: tuple = STORY_COLUMNS,
) -> pd.DataFrame:
    """Numeric table (model × regime slug, by dataset column) for one metric."""
    frames = []
    for model_slug in models:
        sub = regimes_by_model[model_slug].reindex(list(slugs))
        frame = pd.DataFrame({ds_label: sub[f"{ds_key}_{metric_suffix}"].astype(float) for ds_key, ds_label in columns})
        frame.index = pd.MultiIndex.from_tuples(
            [(MODEL_DISPLAY[model_slug], s) for s in sub.index],
            names=["Model", "Trained on"],
        )
        frames.append(frame)
    return pd.concat(frames)


def format_mse_cell(value: float, base_value: float, slug: str) -> str:
    if np.isnan(value):
        return "NaN"
    scaled = value * MSE_SCALE
    if slug == BASE_SLUG:
        return f"{scaled:.2f}"
    change_pct = 100.0 * (value - base_value) / base_value
    return f"{scaled:.2f} ({change_pct:+.0f}%)"


def best_cell_css(values: pd.DataFrame) -> pd.DataFrame:
    """Bold the lowest trained value per model and column when it improves on Off-the-shelf."""
    css = pd.DataFrame("", index=values.index, columns=values.columns)
    for model_name in values.index.get_level_values("Model").unique():
        model_block = values.xs(model_name, level="Model", drop_level=False)
        trained_block = model_block.drop(index=BASE_SLUG, level="Trained on")
        for col in values.columns:
            if trained_block[col].notna().sum() < 2:
                continue
            best_index = trained_block[col].idxmin()
            best_value = float(trained_block.loc[best_index, col])
            base_value = float(values.loc[(model_name, BASE_SLUG), col])
            if not np.isnan(base_value) and best_value < base_value:
                css.loc[best_index, col] = "font-weight: bold"
    return css


def relabel_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.MultiIndex.from_tuples(
        [(model_name, STORY_LABELS[slug]) for model_name, slug in out.index],
        names=out.index.names,
    )
    return out


def build_mse_styler(
    regimes_by_model: dict[str, pd.DataFrame],
    models: tuple[str, ...] = MODEL_SLUGS,
    slugs: tuple[str, ...] = ALL_SLUGS,
    columns: tuple = STORY_COLUMNS,
):
    """β MSE ×10³ with % change vs. Off-the-shelf; bold = best trained value per model and column."""
    values = build_metric_table(regimes_by_model, models, slugs, "mse", columns)
    text = pd.DataFrame("", index=values.index, columns=values.columns)
    for model_name, slug in values.index:
        for col in values.columns:
            value = float(values.loc[(model_name, slug), col])
            base_value = float(values.loc[(model_name, BASE_SLUG), col])
            text.loc[(model_name, slug), col] = format_mse_cell(value, base_value, slug)
    css = relabel_rows(best_cell_css(values))
    return relabel_rows(text).style.apply(lambda _frame: css, axis=None)


def plot_dot_panels(
    regimes_by_model: dict[str, pd.DataFrame],
    models: tuple[str, ...] = MODEL_SLUGS,
    slugs: tuple[str, ...] = ALL_SLUGS,
    datasets: tuple = STORY_COLUMNS,
    y_offsets: dict | None = None,
    x_ticks: tuple | None = (0.5, 1, 2, 5, 10, 20, 50, 100),
    reference_slug: str = "row5_full_berke_wu_ce_oracle",
    scale: str = "log",
    row_groups: tuple[tuple[str, tuple[str, ...]], ...] | None = STORY_ROW_GROUPS,
    save_stem: str | None = None,
) -> None:
    """One panel per model: datasets on y (optionally grouped), β MSE ×10³ on a shared x axis (`scale` = 'log' or 'linear')."""
    if y_offsets is None:
        y_offsets = {
            "row1_base_untrained": 0.0,
            "row2b_wu_ce_oracle": 0.14,
            "row3_berke_ce_only": -0.14,
            "row5_full_berke_wu_ce_oracle": 0.0,
        }
    if scale not in ("log", "linear"):
        raise ValueError(f"scale must be 'log' or 'linear', got {scale!r}")
    if reference_slug not in slugs:
        raise ValueError(f"reference_slug {reference_slug!r} not in slugs {slugs}")
    if y_offsets[reference_slug] != y_offsets[BASE_SLUG]:
        raise ValueError(f"reference {reference_slug!r} offset != Off-the-shelf offset; the bar would not meet the dot")

    ds_keys = [k for k, _ in datasets]
    group_spans: list[tuple[str, float, float]] = []
    divider_ys: list[float] = []
    if row_groups is not None:
        grouped_keys = [k for _, keys in row_groups for k in keys]
        if grouped_keys != ds_keys:
            raise ValueError(f"row_groups keys {grouped_keys} must match datasets order {ds_keys}")
        y_list: list[float] = []
        cursor = 0.0
        group_gap = 0.45
        for g_idx, (g_label, g_keys) in enumerate(row_groups):
            if g_idx > 0:
                divider_ys.append(cursor - 0.5 + group_gap / 2.0)
                cursor += group_gap
            g_start = cursor
            for _ in g_keys:
                y_list.append(cursor)
                cursor += 1.0
            group_spans.append((g_label, g_start, cursor - 1.0))
        y_positions = np.array(y_list, dtype=float)
    else:
        y_positions = np.arange(len(datasets), dtype=float)
        if len(datasets) >= 2 and datasets[0][0] == "kriner":
            divider_ys.append(0.5)
        elif len(datasets) >= 2 and datasets[-1][0] == "kriner":
            divider_ys.append(len(datasets) - 1.5)

    y_min = float(y_positions[0]) - 0.55
    y_max = float(y_positions[-1]) + 0.55
    left_pad = 2.6 if row_groups is not None else 1.6
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(3.3 * len(models) + left_pad, 0.72 * (y_max - y_min) + 1.65),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, model_slug in zip(axes, models):
        sub = regimes_by_model[model_slug].reindex(list(slugs))
        if row_groups is not None:
            for g_idx, (_, g_start, g_end) in enumerate(group_spans):
                if g_idx % 2 == 1:
                    ax.axhspan(g_start - 0.45, g_end + 0.45, color=NEUTRALS["grid"], alpha=0.18, zorder=0, linewidth=0)
        for div_y in divider_ys:
            ax.axhline(div_y, color=NEUTRALS["ref_line"], linestyle="--", linewidth=0.85, alpha=0.65, zorder=0)

        for y, (ds_key, _) in zip(y_positions, datasets):
            base_value = float(sub.loc[BASE_SLUG, f"{ds_key}_mse"]) * MSE_SCALE
            end_value = float(sub.loc[reference_slug, f"{ds_key}_mse"]) * MSE_SCALE
            bar_y = y + y_offsets[BASE_SLUG]
            if not np.isnan(base_value) and not np.isnan(end_value):
                ax.plot([end_value, base_value], [bar_y, bar_y], color=NEUTRALS["grid"], linewidth=2.0, zorder=1)

        for slug in slugs:
            values = [float(sub.loc[slug, f"{ds_key}_mse"]) * MSE_SCALE for ds_key, _ in datasets]
            y_shifted = y_positions + y_offsets[slug]
            if slug == BASE_SLUG:
                ax.scatter(values, y_shifted, s=60, facecolors="white", edgecolors=NEUTRALS["text"], linewidths=1.4, zorder=3, label=STORY_LABELS[slug])
            else:
                ax.scatter(values, y_shifted, s=60, color=STORY_COLORS[slug], edgecolors=NEUTRALS["text"], linewidths=0.5, zorder=3, label=STORY_LABELS[slug])

        ax.set_xscale(scale)
        if scale == "linear":
            ax.set_xlim(left=0.0)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{t:g}" for t in x_ticks])
        if scale == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(MODEL_DISPLAY[model_slug], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\mathrm{MSE}_\beta\ (\times 10^3)$")
        ax.grid(axis="x", color=NEUTRALS["grid"], alpha=0.6, which="both" if scale == "log" else "major")

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([label for _, label in datasets])
    axes[0].set_ylim(y_min, y_max)
    axes[0].invert_yaxis()

    if row_groups is not None:
        trans = blended_transform_factory(axes[0].transAxes, axes[0].transData)
        bx = -0.45
        tick_w = 0.03
        for g_label, g_start, g_end in group_spans:
            y0, y1 = g_start - 0.32, g_end + 0.32
            y_mid = 0.5 * (g_start + g_end)
            axes[0].plot(
                [bx + tick_w, bx, bx, bx + tick_w],
                [y0, y0, y1, y1],
                transform=trans,
                color=NEUTRALS["ref_line"],
                linewidth=0.9,
                clip_on=False,
            )
            axes[0].text(
                bx - 0.03,
                y_mid,
                g_label,
                transform=trans,
                fontsize=8.5,
                fontstyle="italic",
                color=NEUTRALS["muted"],
                ha="right",
                va="center",
                clip_on=False,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.02), ncol=len(slugs), fontsize=9.0, frameon=False)
    fig.tight_layout(rect=(0.06 if row_groups is not None else 0.0, 0.0, 1.0, 0.92))
    if save_stem is not None:
        save_manuscript_fig(fig, save_stem)
    plt.show()


def plot_slope_bars(
    betas_by_model: dict[str, dict[str, pd.DataFrame]],
    terms: list[str],
    slugs: tuple[str, ...] = FIGURE2_SLUGS,
    display_labels: list[str] | None = None,
    divider_ys: tuple[float, ...] = (),
    left_pad: float = 2.0,
    row_height: float = 0.44,
    legend_x: float = 0.55,
    x_label: str = r"OLS treatment effect ($\beta$)",
    save_stem: str | None = None,
) -> None:
    """One panel per model: attribute slopes on y, horizontal regime bars vs. black vertical Human (OLS) tick."""
    if not terms:
        raise ValueError("terms must be non-empty")
    if not slugs:
        raise ValueError("slugs must be non-empty")
    labels_y = list(terms) if display_labels is None else list(display_labels)
    if len(labels_y) != len(terms):
        raise ValueError(f"display_labels length {len(labels_y)} != terms length {len(terms)}")

    if len(slugs) == 1:
        offsets = [0.0]
        bar_lw = 5.5
    elif len(slugs) == 2:
        offsets = [-0.14, 0.14]
        bar_lw = 5.0
    else:
        offsets = np.linspace(-0.24, 0.24, len(slugs)).tolist()
        bar_lw = 3.4

    y_positions = np.arange(len(terms), dtype=float)
    fig_height = max(3.6, row_height * len(terms) + 1.8)
    fig, axes = plt.subplots(
        1,
        len(MODEL_SLUGS),
        figsize=(3.4 * len(MODEL_SLUGS) + left_pad, fig_height),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, model_slug in zip(axes, MODEL_SLUGS):
        ax.axvline(0.0, color=NEUTRALS["ref_line"], linewidth=1.0, linestyle=":", zorder=1)
        for div_y in divider_ys:
            ax.axhline(div_y, color=NEUTRALS["ref_line"], linestyle="--", linewidth=0.9, zorder=2)
        for y in y_positions:
            if int(y) % 2 == 1:
                ax.axhspan(y - 0.5, y + 0.5, color=NEUTRALS["grid"], alpha=0.25, zorder=0, linewidth=0)

        for slug, y_off in zip(slugs, offsets):
            if slug not in betas_by_model[model_slug]:
                raise KeyError(f"{model_slug}/{slug}: missing betas table")
            betas = betas_by_model[model_slug][slug].loc[terms]
            model_values = betas["model_pop_beta"].to_numpy(dtype=float)
            ax.hlines(
                y_positions + y_off,
                0.0,
                model_values,
                color=STORY_COLORS[slug],
                linewidth=bar_lw,
                zorder=3,
                label=STORY_LABELS[slug],
            )

        human_values = betas_by_model[model_slug][BASE_SLUG].loc[terms, "human_pop_beta"].to_numpy(dtype=float)
        ax.vlines(
            human_values,
            y_positions - HUMAN_TICK_HALF_HEIGHT,
            y_positions + HUMAN_TICK_HALF_HEIGHT,
            color=NEUTRALS["text"],
            linewidth=2.3,
            zorder=4,
        )

        ax.set_title(MODEL_DISPLAY[model_slug], fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.grid(axis="x", color=NEUTRALS["grid"], alpha=0.6)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([lbl.replace("$", r"\$") for lbl in labels_y], fontsize=9)
    axes[0].set_ylim(-0.5, len(terms) - 0.5)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D([], [], color=NEUTRALS["text"], marker="|", linestyle="None", markersize=11, markeredgewidth=2.3)
    )
    labels.append("Human (OLS)")
    top_rect = 0.90 if len(terms) <= 6 else (0.94 if len(terms) <= 14 else 0.96)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(legend_x, 1.01), ncol=len(slugs) + 1, fontsize=9.0, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
    if save_stem is not None:
        save_manuscript_fig(fig, save_stem)
    plt.show()

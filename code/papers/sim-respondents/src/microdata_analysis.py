"""Data loading for the manuscript-asset notebooks: saved microdata choice-training runs and their test_best evaluations.

Only loads numbers. Every label, colour, ordering, and output path lives in the notebooks in manuscript-training-assets/.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PAPER_DIR = SRC_DIR.parent
CODE_DIR = PAPER_DIR.parent.parent
if not all((CODE_DIR / d).is_dir() for d in ("kev", "data", "models")):
    raise RuntimeError(f"Expected code/ root containing kev/, data/, models/ at {CODE_DIR}")

TEST_BEST_SUBDIR = "test_best"


def setup_analysis_env() -> None:
    """Put code/ and code/kev on sys.path, then this paper's src/ ahead of them."""
    for path_str in (str(CODE_DIR), str(CODE_DIR / "kev")):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    src_str = str(SRC_DIR)
    while src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)


def find_run_dir(run_roots: tuple[Path, ...], model_slug: str, slug: str) -> Path:
    """Return the unique regime run directory across run_roots, or the path under run_roots[0] if missing."""
    found = []
    for root in run_roots:
        candidate = root / model_slug / slug
        if (candidate / "run.json").exists():
            found.append(candidate)
    if len(found) > 1:
        raise ValueError(f"{model_slug}/{slug} exists in more than one run root: {found}")
    if len(found) == 1:
        return found[0]
    return run_roots[0] / model_slug / slug


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
    status_grid: pd.DataFrame


def load_microdata_context(
    run_roots: tuple[Path, ...],
    models: tuple[str, ...],
    base_slug: str,
    trained_slugs: tuple[str, ...],
    validators: tuple[str, ...],
) -> MicrodataContext:
    """Load all microdata choice-training runs, validation slopes, and held-out test_best evaluations.

    run_roots[0] must hold the off-the-shelf row (base_slug) test_best folders. Rows of regimes_by_model and
    status_grid are indexed by regime slug (base_slug first, then trained_slugs), columns of status_grid by model slug.
    """
    setup_analysis_env()
    from beta_train import runs as run_store
    from paper_beta import extract_step0_from_run_dir, load_paper_run

    regimes_by_model: dict[str, pd.DataFrame] = {}
    grummon_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    covid_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    kantorowicz_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    magistro_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    kriner_betas_by_model: dict[str, dict[str, pd.DataFrame]] = {}

    for model_slug in models:
        model_dir = run_roots[0] / model_slug
        existing_dirs = []
        for trained_slug in trained_slugs:
            candidate = find_run_dir(run_roots, model_slug, trained_slug)
            if (candidate / "run.json").exists():
                existing_dirs.append(candidate)
        if not existing_dirs:
            raise FileNotFoundError(f"No canonical microdata runs for {model_slug} in {run_roots}")

        step0 = extract_step0_from_run_dir(existing_dirs[0], validators)
        step0_kriner_betas = step0["betas_by_validator"]["kriner2026"][["human_pop_beta", "model_pop_beta"]].copy()
        human_kriner = step0_kriner_betas["human_pop_beta"]

        grummon_by_slug: dict[str, pd.DataFrame] = {}
        covid_by_slug: dict[str, pd.DataFrame] = {}
        kantorowicz_by_slug: dict[str, pd.DataFrame] = {}
        magistro_by_slug: dict[str, pd.DataFrame] = {}
        kriner_by_slug: dict[str, pd.DataFrame] = {
            base_slug: step0_kriner_betas.loc[human_kriner.index, ["human_pop_beta", "model_pop_beta"]]
        }

        row1_tb = load_test_best_entry(model_dir / base_slug, expected_step=0)
        if row1_tb is not None:
            g0_tb = float(row1_tb["metrics_df"].loc["grummon2025", "beta_mse"])
            g0_val = step0["metrics"]["grummon2025/beta_mse"]
            if abs(g0_tb - g0_val) > 2e-4:
                raise ValueError(f"{model_slug} Off-the-shelf Grummon mismatch: val={g0_val:.6f} vs test_best={g0_tb:.6f}")
            row1_grummon_mse = g0_tb
            row1_grummon_kl = float(row1_tb["metrics_df"].loc["grummon2025", "kl_loss"])
            row1_covid_mse = float(row1_tb["metrics_df"].loc["covid", "beta_mse"])
            row1_covid_kl = float(row1_tb["metrics_df"].loc["covid", "kl_loss"])
            grummon_by_slug[base_slug] = row1_tb["grummon_betas"]
            covid_by_slug[base_slug] = row1_tb["covid_betas"]
            if row1_tb["has_kantorowicz"]:
                row1_k26_mse = float(row1_tb["metrics_df"].loc["kantorowicz2026", "beta_mse"])
                row1_k26_kl = float(row1_tb["metrics_df"].loc["kantorowicz2026", "kl_loss"])
                kantorowicz_by_slug[base_slug] = row1_tb["kantorowicz_betas"]
            else:
                row1_k26_mse, row1_k26_kl = np.nan, np.nan
            if row1_tb["has_magistro"]:
                row1_m25_mse = float(row1_tb["metrics_df"].loc["magistro2025", "beta_mse"])
                row1_m25_kl = float(row1_tb["metrics_df"].loc["magistro2025", "kl_loss"])
                magistro_by_slug[base_slug] = row1_tb["magistro_betas"]
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
                "slug": base_slug,
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

        for slug in trained_slugs:
            run_dir = find_run_dir(run_roots, model_slug, slug)
            if not (run_dir / "run.json").exists():
                kriner_by_slug[slug] = pd.DataFrame({"human_pop_beta": human_kriner, "model_pop_beta": np.nan})
                rows.append(
                    {
                        "slug": slug,
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

    all_slugs = [base_slug, *trained_slugs]
    status_columns = {}
    for model_slug in models:
        status_columns[model_slug] = regimes_by_model[model_slug].loc[all_slugs, "progress"].to_numpy()
    status_grid = pd.DataFrame(status_columns, index=all_slugs)
    status_grid.index.name = "regime"

    return MicrodataContext(
        regimes_by_model=regimes_by_model,
        grummon_betas_by_model=grummon_betas_by_model,
        covid_betas_by_model=covid_betas_by_model,
        kantorowicz_betas_by_model=kantorowicz_betas_by_model,
        magistro_betas_by_model=magistro_betas_by_model,
        kriner_betas_by_model=kriner_betas_by_model,
        status_grid=status_grid,
    )



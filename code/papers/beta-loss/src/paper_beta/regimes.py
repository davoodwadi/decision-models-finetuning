"""Step-0 and cross-run summary helpers for the beta-loss paper.

Each regime is declared in its own training-notebook cell; its design and paper row code are saved in its run's run.json,
which is where these helpers read them from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from beta_train import runs as run_store
from beta_train.validation import METRIC_NAMES
from .trainer import load_paper_run


def _sign_accuracy(beta_df: pd.DataFrame) -> float:
    model_sign = np.sign(beta_df["model_pop_beta"].to_numpy(dtype=float))
    human_sign = np.sign(beta_df["human_pop_beta"].to_numpy(dtype=float))
    return float(np.mean(model_sign == human_sign))


def extract_step0_from_run_dir(run_dir: Path, validators: tuple[str, ...]) -> dict:
    """Row 1 (untrained base model, step 0): metrics and term-level betas from any saved run directory."""
    run_dir = Path(run_dir)
    val_history_df = pd.read_csv(run_dir / "val_history.csv").set_index("step")
    if 0 not in val_history_df.index:
        raise ValueError(f"Step 0 missing from {run_dir / 'val_history.csv'}")
    step0_row = val_history_df.loc[0]
    step0_betas = run_store.read_betas(run_dir, 0)
    metrics_out = {}
    for validator_name in validators:
        for metric in (*METRIC_NAMES, "std_beta_mse"):
            col = f"{validator_name}/{metric}"
            metrics_out[col] = float(step0_row[col])
        metrics_out[f"{validator_name}/sign_acc"] = _sign_accuracy(step0_betas[validator_name])
    betas_by_validator = {}
    for validator_name in validators:
        betas_by_validator[validator_name] = step0_betas[validator_name]
    return {
        "row_code": "Row 1",
        "paper_label": "Untrained Base LLM (Zero-Shot)",
        "data_regime": "None (Zero-Shot)",
        "metrics": metrics_out,
        "betas_by_validator": betas_by_validator,
    }


def saved_run_dirs(runs_dir: Path) -> list[Path]:
    """Run folders of `runs_dir` that hold a run.json, ordered by paper row code (run.json), then folder name."""
    runs_dir = Path(runs_dir)
    keyed_dirs = []
    for run_dir in runs_dir.iterdir():
        if (run_dir / "run.json").exists():
            row_code = run_store.read_run_json(run_dir)["identity"]["run_metadata"]["paper_row"]
            keyed_dirs.append((row_code, run_dir.name, run_dir))
    keyed_dirs.sort()
    return [run_dir for _, _, run_dir in keyed_dirs]


def summarize_paper_runs_dir(runs_dir: Path, validators: tuple[str, ...], metrics: tuple[str, ...]) -> pd.DataFrame:
    """One row per paper row: Row 1 (step 0) plus every saved run in `runs_dir`, metrics at best.pt.

    Raises if `runs_dir` holds no run (Row 1 needs one). A run that is not finished gets its status
    ("running", "interrupted") and NaN metrics; a finished run must have every metric.
    """
    runs_dir = Path(runs_dir)
    run_dirs = saved_run_dirs(runs_dir)
    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No saved paper runs in {runs_dir}")

    rows = []
    base_info = extract_step0_from_run_dir(run_dirs[0], validators)
    base_row = {
        "row": base_info["row_code"],
        "slug": "row1_base_untrained",
        "status": "step0",
        "best_step": 0,
        "steps_done": 0,
    }
    for v in validators:
        for m in metrics:
            base_row[f"{v}/{m}"] = base_info["metrics"][f"{v}/{m}"]
    rows.append(base_row)

    for run_dir in run_dirs:
        run_info = run_store.read_run_json(run_dir)
        row = {
            "row": run_info["identity"]["run_metadata"]["paper_row"],
            "slug": run_dir.name,
            "status": run_info["status"],
        }
        if row["status"] != run_store.STATUS_FINISHED:
            row["best_step"] = float("nan")
            row["steps_done"] = float("nan")
            for v in validators:
                for m in metrics:
                    row[f"{v}/{m}"] = float("nan")
            rows.append(row)
            continue

        result = load_paper_run(run_dir)
        row["best_step"] = int(result.best["step"])
        row["steps_done"] = len(result.step_df)
        for v in validators:
            for m in metrics:
                col = f"{v}/{m}"
                if m == "sign_acc":
                    row[col] = _sign_accuracy(result.best_validation[v][1])
                else:
                    row[col] = float(result.best_row[col])
        rows.append(row)

    return pd.DataFrame(rows).set_index("slug")

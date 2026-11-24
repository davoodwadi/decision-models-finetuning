"""Tables comparing finished runs: final validation metrics, per-dataset train losses, betas, MSE decomposition.

`runs` is {run name: RunResult}; `step0_validation` is {validator: (metrics, beta_df)} of the untrained model.
"""

from __future__ import annotations

import pandas as pd

from beta_train.validation import METRIC_NAMES


def comparison_table(runs: dict, step0_validation: dict) -> pd.DataFrame:
    # One row per run (plus the untrained model): settings, final validation metrics per validator, timings
    rows = []
    step0_row = {"run": "step_0 (untrained)", "train_datasets": "", "beta_lambda": float("nan"), "num_epochs": 0}
    for name, (metrics, _) in step0_validation.items():
        for metric in METRIC_NAMES:
            step0_row[f"{name}/{metric}"] = metrics[metric]
    rows.append(step0_row)

    for run_name, run in runs.items():
        row = {
            "run": run_name,
            "train_datasets": "+".join(run.config["train_datasets"]),
            "beta_lambda": run.config["beta_lambda"],
            "num_epochs": run.config["num_epochs"],
        }
        for name, (metrics, _) in run.final_validation.items():
            for metric in METRIC_NAMES:
                row[f"{name}/{metric}"] = metrics[metric]
        row["steps_per_epoch"] = run.summary["steps_per_epoch"]
        row["mean_step_ms"] = run.summary["mean_step_ms"]
        row["train_only_sec"] = run.summary["train_only_sec"]
        row["eval_only_sec"] = run.summary["eval_only_sec"]
        row["total_wall_sec"] = run.summary["total_wall_sec"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


def per_dataset_table(runs: dict) -> pd.DataFrame:
    # Per-dataset train KL and beta loss (first vs last 15% of that dataset's steps) for every run
    frames = []
    for run_name, run in runs.items():
        frame = run.per_dataset_df.reset_index()
        frame.insert(0, "run", run_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).set_index(["run", "dataset"])


def eval_beta_table(runs: dict, step0_validation: dict, validator: str) -> pd.DataFrame:
    # Human betas, untrained-model betas, and every run's final betas for one validator
    _, step0_beta_df = step0_validation[validator]
    columns = {"human_pop_beta": step0_beta_df["human_pop_beta"], "step_0": step0_beta_df["model_pop_beta"]}
    for run_name, run in runs.items():
        _, beta_df = run.final_validation[validator]
        if not beta_df.index.equals(step0_beta_df.index):
            raise ValueError(f"Beta index mismatch for run {run_name!r}")
        columns[run_name] = beta_df["model_pop_beta"]
    return pd.DataFrame(columns)


def mse_decomposition(runs: dict, step0_validation: dict, validator: str) -> pd.DataFrame:
    # Squared error per beta for every run; the MEAN row equals that validator's beta MSE
    betas = eval_beta_table(runs, step0_validation, validator)
    human = betas["human_pop_beta"]
    columns = {"human_beta": human}
    for run_name in runs:
        columns[f"pred | {run_name}"] = betas[run_name]
    for run_name in runs:
        columns[f"sq_err | {run_name}"] = (betas[run_name] - human) ** 2
    table = pd.DataFrame(columns)
    mean_row = pd.DataFrame([table.mean()], index=pd.Index(["MEAN (= beta_mse on sq_err columns)"]))
    return pd.concat([table, mean_row])


def best_table(runs: dict) -> pd.DataFrame:
    # One row per run: the best-metric rule, the best step, and every validation column of best.pt (validators that
    # did not run at that step were validated on best.pt at the end of the run)
    rows = []
    for run_name, run in runs.items():
        row = {
            "run": run_name,
            "beta_lambda": run.config["beta_lambda"],
            "best_metric": run.best["metric"],
            "best_mode": run.best["mode"],
            "best_step": run.best["step"],
            "best_value": run.best["value"],
        }
        for column, value in run.best_row.items():
            if column != "step":
                row[column] = value
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


def best_beta_table(runs: dict, step0_validation: dict, validator: str) -> pd.DataFrame:
    # Human betas, untrained-model betas, and every run's betas at its best step for one validator
    _, step0_beta_df = step0_validation[validator]
    columns = {"human_pop_beta": step0_beta_df["human_pop_beta"], "step_0": step0_beta_df["model_pop_beta"]}
    for run_name, run in runs.items():
        _, beta_df = run.best_validation[validator]
        if not beta_df.index.equals(step0_beta_df.index):
            raise ValueError(f"Beta index mismatch for run {run_name!r}")
        columns[f"{run_name} (best step {run.best['step']})"] = beta_df["model_pop_beta"]
    return pd.DataFrame(columns)

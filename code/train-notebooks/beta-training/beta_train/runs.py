"""Run folders: the validation loss, saved stats, the best checkpoint, and the state to resume a run.

One folder per run (the notebook declares its path):
    run.json                  identity (config, notebook metadata, data fingerprints, best-metric rule), git state,
                              status (running / interrupted / finished), resume history, final summary
    steps.csv                 one row per optimization step
    val_history.csv           one row per validation step (step 0 included); a validator that did not run there: NaN
    betas/step_000124.csv     the beta table of every validator that ran at that step
    betas/best_step_000120.csv  every validator's betas for best.pt (ones that did not run at that step: end of run)
    best_validation.csv       one row: every validator's columns for best.pt
    checkpoints/best.pt       trainable weights at the best validation so far (by the run's best metric)
    checkpoints/last.pt       everything needed to resume, overwritten at every validation

Validation loss = the training loss's form on each validator: KL + beta_lambda * (mean over regressions of the mean
squared sigma-standardized beta error), reported per validator as `<validator>/val_loss`. The best checkpoint is
picked on ONE validator: `best_metric` must be one of that validator's columns (e.g. "kriner2026/val_loss").
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from beta_train.model_state import snapshot_trainable

BEST_MODES = ("min", "max")
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_FINISHED = "finished"

# Config settings that only change memory use, not the gradient; a resume may change them (e.g. a smaller micro-chunk
# after an OOM). Dropout masks after the resume then differ from an uninterrupted run, the gradients' expectation does not.
MEMORY_ONLY_SETTINGS = ("micro_chunk", "gradient_checkpointing")


# ---------------------------------------------------------------- validation loss

def standardized_beta_mse(validator, beta_df: pd.DataFrame) -> float:
    # Per regression: mean of ((model beta - human beta) / sigma)^2; then mean over regressions (each counts equally).
    # Same units and weighting as the training beta loss (beta_ols.unified_loss).
    specs = validator.eval_ops["specs"]
    groups_in_table = beta_df["model"].unique().tolist()
    if set(groups_in_table) != set(specs):
        raise ValueError(f"{validator.name}: beta table regressions {groups_in_table} != specs {list(specs)}")
    group_mses = []
    for group in groups_in_table:
        group_diffs = beta_df.loc[beta_df["model"] == group, "diff"]
        standardized_diffs = group_diffs / specs[group].sigma
        group_mses.append(float((standardized_diffs**2).mean()))
    return sum(group_mses) / len(group_mses)


def validation_losses(validators: dict, validation: dict, beta_lambda: float) -> dict[str, float]:
    # {"<validator>/std_beta_mse", "<validator>/val_loss"} for each validator in `validation` (the ones that ran;
    # never pooled across validators)
    for name in validation:
        if name not in validators:
            raise ValueError(f"Validation result for unknown validator {name!r}; validators are {sorted(validators)}")
    losses = {}
    for name in validation:
        validator = validators[name]
        metrics, beta_df = validation[name]
        std_mse = standardized_beta_mse(validator, beta_df)
        validator_loss = metrics["kl_loss"] + beta_lambda * std_mse
        losses[f"{name}/std_beta_mse"] = std_mse
        losses[f"{name}/val_loss"] = validator_loss
    return losses


def check_best_rule(best_metric: str, best_mode: str, val_row: dict, validator_names: list[str]) -> None:
    # The best metric must be one validator's column ("<validator>/<metric>"), and the mode "min" or "max"
    if best_mode not in BEST_MODES:
        raise ValueError(f"best_mode must be one of {BEST_MODES}, got {best_mode!r}")
    single_validator_columns = []
    for column in val_row:
        validator_name = column.split("/")[0]
        if validator_name in validator_names and not column.endswith("/eval_wall_sec"):
            single_validator_columns.append(column)
    if best_metric not in single_validator_columns:
        raise KeyError(
            f"best_metric {best_metric!r} must be one validator's column; choose one of {single_validator_columns}"
        )


def is_better(value: float, best_value: float, best_mode: str) -> bool:
    # Strictly better under the run's best mode (NaN is never better)
    if best_mode == "min":
        return value < best_value
    if best_mode == "max":
        return value > best_value
    raise ValueError(f"best_mode must be one of {BEST_MODES}, got {best_mode!r}")


# ---------------------------------------------------------------- run identity (what a resume must match)

def items_fingerprint(items_by_dataset: dict[str, list[dict]], train_datasets: tuple[str, ...]) -> dict[str, dict]:
    # Per training dataset: item count and a hash of the (caseid, qids) sequence. A resume must see the same items in
    # the same order, since the saved epoch plan holds item positions.
    fingerprints = {}
    for dataset_name in train_datasets:
        item_keys = []
        for item in items_by_dataset[dataset_name]:
            item_keys.append([item["caseid"], item["qids"]])
        key_text = json.dumps(item_keys)
        fingerprints[dataset_name] = {"items": len(item_keys), "keys_sha256": hashlib.sha256(key_text.encode()).hexdigest()}
    return fingerprints


def run_identity(
    config_dict: dict,
    run_metadata: dict,
    items_by_dataset: dict[str, list[dict]],
    validators: dict,
    init_snapshot: dict[str, torch.Tensor],
    best_metric: str,
    best_mode: str,
) -> dict:
    # Everything that defines the run; stored in run.json and compared on resume. JSON round trip, so it compares
    # equal to what is read back (and non-JSON metadata fails here, not later).
    validator_info = {}
    for name, validator in validators.items():
        validator_info[name] = {"kind": validator.KIND, "choice_rows": validator.choice_fingerprint}
    trainable_shapes = {}
    for param_name, tensor in init_snapshot.items():
        trainable_shapes[param_name] = list(tensor.shape)
    identity = {
        "config": config_dict,
        "run_metadata": run_metadata,
        "train_items": items_fingerprint(items_by_dataset, tuple(config_dict["train_datasets"])),
        "validators": validator_info,
        "trainable_shapes": trainable_shapes,
        "best_metric": best_metric,
        "best_mode": best_mode,
    }
    return json.loads(json.dumps(identity))


def identity_differences(stored: dict, current: dict) -> list[str]:
    # Names of the identity parts that differ; config settings are listed one by one, memory-only settings excluded
    differences = []
    for key in current:
        if key == "config":
            continue
        if stored[key] != current[key]:
            differences.append(key)
    stored_config = stored["config"]
    current_config = current["config"]
    if set(stored_config) != set(current_config):
        differences.append("config keys")
    for setting in current_config:
        if setting in MEMORY_ONLY_SETTINGS:
            continue
        if stored_config[setting] != current_config[setting]:
            differences.append(f"config.{setting}: {stored_config[setting]!r} -> {current_config[setting]!r}")
    return differences


# ---------------------------------------------------------------- run.json

def git_state(folder: Path) -> dict:
    # Commit of the repository holding `folder`, and whether its working tree has uncommitted changes
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=folder, capture_output=True, text=True, check=True)
    return {"commit": commit.stdout.strip(), "uncommitted_changes": len(status.stdout.strip()) > 0}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_text_atomic(path: Path, text: str) -> None:
    # Write to a temporary file, then rename: an interrupted write never leaves a broken file
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(text)
    temporary_path.replace(path)


def read_run_json(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def write_run_json(run_dir: Path, run_info: dict) -> None:
    write_text_atomic(Path(run_dir) / "run.json", json.dumps(run_info, indent=2))


def create_run_folder(run_dir: Path, identity: dict) -> dict:
    # New run: the folder must not exist yet (nothing is ever overwritten)
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists but holds no run.json, so it is not a run folder; choose another run folder")
    (run_dir / "betas").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    run_info = {
        "identity": identity,
        "created": now_text(),
        "git": git_state(run_dir),
        "status": STATUS_RUNNING,
        "resumes": [],
        "summary": None,
    }
    write_run_json(run_dir, run_info)
    return run_info


def open_run_for_resume(run_dir: Path, identity: dict) -> dict:
    # Existing run: its identity must match (memory-only settings may change). Returns run.json as stored.
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").exists():
        raise FileNotFoundError(f"No run.json in {run_dir}")
    run_info = read_run_json(run_dir)
    differences = identity_differences(run_info["identity"], identity)
    if differences:
        raise ValueError(
            f"Cannot resume {run_dir}: the run differs in {differences}. "
            "Use another run folder, or from_scratch=True to delete this run and train it again"
        )
    return run_info


def delete_run_folder(run_dir: Path) -> None:
    # Remove an existing run (training from scratch). Only a folder with a run.json is deleted.
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").exists():
        raise FileNotFoundError(f"{run_dir} has no run.json; refusing to delete a folder that is not a run folder")
    shutil.rmtree(run_dir)


def record_resume(run_dir: Path, run_info: dict, identity: dict, resumed_from_step: int) -> dict:
    # Log the resume (time, step, git state, changed memory-only settings) and set the status back to running
    changed_settings = {}
    for setting in MEMORY_ONLY_SETTINGS:
        stored_value = run_info["identity"]["config"][setting]
        current_value = identity["config"][setting]
        if stored_value != current_value:
            changed_settings[setting] = [stored_value, current_value]
    run_info["resumes"].append(
        {"at": now_text(), "from_step": resumed_from_step, "git": git_state(run_dir), "changed_settings": changed_settings}
    )
    run_info["identity"]["config"] = identity["config"]
    run_info["status"] = STATUS_RUNNING
    write_run_json(run_dir, run_info)
    return run_info


def set_status(run_dir: Path, status: str, summary: dict | None) -> None:
    run_info = read_run_json(run_dir)
    run_info["status"] = status
    if summary is not None:
        run_info["summary"] = summary
    write_run_json(run_dir, run_info)


# ---------------------------------------------------------------- tables and betas

def write_tables(run_dir: Path, step_rows: list[dict], val_rows: list[dict]) -> None:
    # Rewrite steps.csv and val_history.csv from the in-memory rows (after a resume they hold the restored history)
    run_dir = Path(run_dir)
    write_text_atomic(run_dir / "steps.csv", pd.DataFrame(step_rows).to_csv(index=False))
    write_text_atomic(run_dir / "val_history.csv", pd.DataFrame(val_rows).to_csv(index=False))


def betas_file(run_dir: Path, step: int) -> Path:
    return Path(run_dir) / "betas" / f"step_{step:06d}.csv"


def write_betas(run_dir: Path, step: int, validation: dict) -> None:
    # One file per validation row: the beta table of every validator that ran, stacked, with a `validator` column
    write_betas_file(betas_file(run_dir, step), validation)


def write_betas_file(path: Path, validation: dict) -> None:
    frames = []
    for name, (_, beta_df) in validation.items():
        frame = beta_df.reset_index()
        frame.insert(0, "validator", name)
        frames.append(frame)
    write_text_atomic(path, pd.concat(frames, ignore_index=True).to_csv(index=False))


def read_betas(run_dir: Path, step: int) -> dict[str, pd.DataFrame]:
    # {validator: beta_df indexed by term}, as written by write_betas
    return read_betas_file(betas_file(run_dir, step))


def best_betas_file(run_dir: Path, step: int) -> Path:
    # Every validator's betas for the best checkpoint (validators that did not run at that step: validated at the end)
    return Path(run_dir) / "betas" / f"best_step_{step:06d}.csv"


def write_best_betas(run_dir: Path, step: int, validation: dict) -> None:
    write_betas_file(best_betas_file(run_dir, step), validation)


def read_best_betas(run_dir: Path, step: int) -> dict[str, pd.DataFrame]:
    return read_betas_file(best_betas_file(run_dir, step))


def write_best_row(run_dir: Path, best_row: dict) -> None:
    # best_validation.csv: one row, every validator's columns at the best step (complete after the end-of-run check)
    write_text_atomic(Path(run_dir) / "best_validation.csv", pd.DataFrame([best_row]).to_csv(index=False))


def read_best_row(run_dir: Path) -> dict:
    return pd.read_csv(Path(run_dir) / "best_validation.csv").iloc[0].to_dict()


def read_betas_file(path: Path) -> dict[str, pd.DataFrame]:
    stacked = pd.read_csv(path)
    beta_dfs = {}
    for name, frame in stacked.groupby("validator", sort=False):
        beta_dfs[name] = frame.drop(columns="validator").set_index("term")
    return beta_dfs


# ---------------------------------------------------------------- checkpoints

def save_torch_atomic(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def save_best(run_dir: Path, model, best: dict) -> None:
    # Trainable weights (CPU copy) with the step, metric, mode, and value they were chosen by
    payload = {"trainable": snapshot_trainable(model), "best": dict(best)}
    save_torch_atomic(Path(run_dir) / "checkpoints" / "best.pt", payload)


def load_best(run_dir: Path) -> dict:
    # {"trainable": {name: tensor}, "best": {metric, mode, step, value}}; load into a model with reset_trainable
    return torch.load(Path(run_dir) / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)


def save_last(run_dir: Path, state: dict) -> None:
    save_torch_atomic(Path(run_dir) / "checkpoints" / "last.pt", state)


def load_last(run_dir: Path) -> dict:
    # Our own file: holds Python objects (RNG state, rows, beta tables), so it is not a weights-only load
    return torch.load(Path(run_dir) / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- resume pieces

def rng_states(rng: random.Random) -> dict:
    # Every RNG a run draws from: sample order (Python), CPU torch, and CUDA (dropout); CUDA is None without a GPU
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    return {"python": rng.getstate(), "torch": torch.get_rng_state(), "cuda": cuda_states}


def restore_rng_states(rng: random.Random, states: dict) -> None:
    rng.setstate(states["python"])
    torch.set_rng_state(states["torch"])
    if states["cuda"] is None:
        if torch.cuda.is_available():
            raise ValueError("The saved run state has no CUDA RNG state, but CUDA is available")
        return
    torch.cuda.set_rng_state_all(states["cuda"])


def epoch_plan_positions(epoch_samples: list[tuple[str, list[dict]]], items_by_dataset: dict[str, list[dict]]) -> list:
    # The epoch's samples as (dataset, [item positions in items_by_dataset[dataset]])
    position_by_dataset = {}
    for dataset_name, _ in epoch_samples:
        if dataset_name not in position_by_dataset:
            positions = {}
            for position, item in enumerate(items_by_dataset[dataset_name]):
                positions[id(item)] = position
            position_by_dataset[dataset_name] = positions
    plan = []
    for dataset_name, sample_items in epoch_samples:
        sample_positions = []
        for item in sample_items:
            sample_positions.append(position_by_dataset[dataset_name][id(item)])
        plan.append((dataset_name, sample_positions))
    return plan


def epoch_plan_from_positions(plan: list, items_by_dataset: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    # Inverse of epoch_plan_positions
    epoch_samples = []
    for dataset_name, sample_positions in plan:
        dataset_items = items_by_dataset[dataset_name]
        sample_items = []
        for position in sample_positions:
            sample_items.append(dataset_items[position])
        epoch_samples.append((dataset_name, sample_items))
    return epoch_samples

"""Unified trainer for the Marketing Science beta-loss paper (`run_paper_training` and `load_paper_run`).

Supports every row of the paper's experimental design under one code path:
  - `ce_datasets`: one-pass micro-chunked respondent cross-entropy (exact gradient, no p0 pass needed).
  - `beta_datasets`: two-pass micro-chunked split population-beta cross loss + optional forward KL(p0 || p_theta)
    anchor against frozen base probabilities p0 (respondent choices y never enter the loss or gradient).
  - Hybrid runs interleave `ce_datasets` samples and `beta_datasets` samples within each shuffled epoch.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training import free_gpu

from beta_train import runs
from beta_train.batching import build_epoch_samples, split_into_chunks, steps_per_epoch
from beta_train.live_display import LiveDisplay, make_val_row
from beta_train.model_state import reset_trainable, set_checkpointing, snapshot_trainable
from beta_train.progress import PROGRESS_INTERVAL_SEC, HtmlProgress
from beta_train.trainer import RunResult, json_ready, make_scheduler, resume_state, validators_due
from beta_train.validation import METRIC_NAMES, validate_all

from .config import PaperTrainConfig
from .loss import (
    FrozenBaseEstimator,
    beta_sample_loss,
    forward_chunk_logits,
    sample_num_options,
    sync_device,
    task_log_probs,
)


def paper_validation_losses(validators: dict, validation: dict, config: PaperTrainConfig) -> dict[str, float]:
    """Per-validator `<name>/std_beta_mse` and `<name>/val_loss`.

    For table-only runs (`ce_datasets == ()`), `val_loss = std_beta_mse` so respondent choices y do not enter
    `val_loss` either. When `ce_datasets` is non-empty, `val_loss = kl_loss + beta_lambda * std_beta_mse`.
    Note: paper checkpoint selection uses `best_metric = "kriner2026/beta_mse"` uniformly across all rows.
    """
    for name in validation:
        if name not in validators:
            raise ValueError(f"Validation result for unknown validator {name!r}; validators are {sorted(validators)}")
    losses = {}
    pure_table_mode = len(config.ce_datasets) == 0
    for name in validation:
        validator = validators[name]
        metrics, beta_df = validation[name]
        std_mse = runs.standardized_beta_mse(validator, beta_df)
        losses[f"{name}/std_beta_mse"] = std_mse
        if pure_table_mode:
            losses[f"{name}/val_loss"] = std_mse
        else:
            losses[f"{name}/val_loss"] = metrics["kl_loss"] + config.beta_lambda * std_mse
    return losses


def single_task_beta_datasets(items_by_dataset: dict[str, list[dict]], config: PaperTrainConfig) -> list[str]:
    """Table datasets whose epoch would end in a 1-task remainder (N mod S == 1), which cannot be split."""
    datasets = []
    for dataset_name in config.beta_datasets:
        num_items = len(items_by_dataset[dataset_name])
        if num_items % config.sample_size_by_dataset[dataset_name] == 1:
            datasets.append(dataset_name)
    return datasets


def paper_steps_per_epoch(items_by_dataset: dict[str, list[dict]], config: PaperTrainConfig) -> int:
    return steps_per_epoch(items_by_dataset, config) - len(single_task_beta_datasets(items_by_dataset, config))


def build_paper_epoch_samples(
    rng: random.Random, items_by_dataset: dict[str, list[dict]], config: PaperTrainConfig
) -> list[tuple[str, list[dict]]]:
    """Build one shuffled epoch of single-dataset samples; merge any 1-task remainder on `beta_datasets`."""
    epoch_samples = build_epoch_samples(rng, items_by_dataset, config)
    beta_set = set(config.beta_datasets)
    single_positions = [
        pos for pos, (ds_name, sample_items) in enumerate(epoch_samples) if ds_name in beta_set and len(sample_items) == 1
    ]
    expected_singles = single_task_beta_datasets(items_by_dataset, config)
    found_singles = [epoch_samples[pos][0] for pos in single_positions]
    if sorted(found_singles) != sorted(expected_singles):
        raise AssertionError(f"1-task beta samples found for {found_singles}, expected {expected_singles}")

    if not single_positions:
        return epoch_samples

    merged_samples = list(epoch_samples)
    for single_pos in single_positions:
        single_ds, single_items = epoch_samples[single_pos]
        target_pos = None
        for pos, (ds_name, _) in enumerate(merged_samples):
            if ds_name == single_ds and pos != single_pos:
                target_pos = pos
                break
        if target_pos is None:
            raise ValueError(f"{single_ds}: no other sample available to absorb its 1-task remainder")
        target_ds, target_items = merged_samples[target_pos]
        merged_samples[target_pos] = (target_ds, list(target_items) + list(single_items))

    return [sample for pos, sample in enumerate(merged_samples) if pos not in single_positions]


def _get_device_rng_state(device: str) -> torch.Tensor:
    if device == "cuda":
        return torch.cuda.get_rng_state()
    if device == "cpu":
        return torch.get_rng_state()
    raise ValueError(f"Unsupported device {device!r}; expected 'cuda' or 'cpu'")


def _set_device_rng_state(device: str, state: torch.Tensor) -> None:
    if device == "cuda":
        torch.cuda.set_rng_state(state)
    elif device == "cpu":
        torch.set_rng_state(state)
    else:
        raise ValueError(f"Unsupported device {device!r}; expected 'cuda' or 'cpu'")


def ce_one_pass_step(
    model, sample_items: list[dict], micro_chunk: int, device: str, chunk_bar
) -> tuple[dict, dict]:
    """Exact one-pass micro-chunked cross-entropy gradient for a microdata study (`ce_datasets`).

    Because per-task cross-entropy is additive across tasks, accumulating -(y_chunk * log_p_chunk).sum() / total_tasks
    chunk by chunk produces the exact full-sample CE gradient in a single forward+backward pass per chunk.
    """
    total_tasks = len(sample_num_options(sample_items))
    if total_tasks == 0:
        raise ValueError("sample_items has zero tasks")
    chunks = split_into_chunks(sample_items, micro_chunk)
    chunk_bar.reset(total=len(chunks))
    chunk_bar.set_description("ce (forward + backward)")
    sync_device(device)
    t_start = time.perf_counter()

    ce_sum = 0.0
    fwd_sec = 0.0
    bwd_sec = 0.0
    for chunk in chunks:
        t_f0 = time.perf_counter()
        chunk_logits = forward_chunk_logits(model, chunk, device)
        chunk_log_probs = task_log_probs(chunk_logits, sample_num_options(chunk))
        y_chunk = torch.cat([item["y_flat"] for item in chunk]).to(
            device=chunk_log_probs.device, dtype=chunk_log_probs.dtype
        )
        if chunk_log_probs.shape[0] != y_chunk.shape[0]:
            raise ValueError(
                f"Chunk log_probs length {chunk_log_probs.shape[0]} != y_chunk length {y_chunk.shape[0]}"
            )
        chunk_nll_sum = -(y_chunk * chunk_log_probs).sum()
        chunk_loss = chunk_nll_sum / total_tasks
        sync_device(device)
        t_f1 = time.perf_counter()
        fwd_sec += t_f1 - t_f0

        chunk_loss.backward()
        ce_sum += float(chunk_nll_sum.detach())
        del chunk_logits, chunk_log_probs, chunk_nll_sum, chunk_loss
        sync_device(device)
        bwd_sec += time.perf_counter() - t_f1
        chunk_bar.update(1)

    ce_val = ce_sum / total_tasks
    nan = float("nan")
    loss_out = {
        "mode": "ce",
        "ce_loss": ce_val,
        "cross_loss": nan,
        "weighted_cross_loss": nan,
        "kl_anchor_loss": nan,
        "weighted_kl_anchor_loss": nan,
        "total_loss": ce_val,
        "sample_beta_mse": nan,
        "monitor_human_kl": ce_val,
        "half_sizes": (0, 0),
        "group_cross_loss": {},
    }
    timings_ms = {
        "fwd_ms": fwd_sec * 1000.0,
        "loss_ms": 0.0,
        "bwd_ms": bwd_sec * 1000.0,
        "total_ms": (time.perf_counter() - t_start) * 1000.0,
    }
    return loss_out, timings_ms


def beta_two_pass_step(
    model,
    sample_items: list[dict],
    estimator: FrozenBaseEstimator,
    beta_lambda: float,
    kl_anchor_alpha: float,
    micro_chunk: int,
    device: str,
    chunk_bar,
) -> tuple[dict, dict]:
    """Exact two-pass micro-chunked gradient for `beta_lambda * split_cross_loss + kl_anchor_alpha * KL(p0 || p)`."""
    chunks = split_into_chunks(sample_items, micro_chunk)
    chunk_bar.reset(total=2 * len(chunks))
    chunk_bar.set_description("pop_beta pass 1 (forward)")
    sync_device(device)
    t_start = time.perf_counter()

    chunk_rng_states = []
    chunk_logits_list = []
    with torch.no_grad():
        for chunk in chunks:
            chunk_rng_states.append(_get_device_rng_state(device))
            chunk_logits_list.append(forward_chunk_logits(model, chunk, device))
            sync_device(device)
            chunk_bar.update(1)
    flat_logits = torch.cat(chunk_logits_list, dim=0)
    sync_device(device)
    t_after_forward = time.perf_counter()

    flat_logits.requires_grad_(True)
    loss, loss_out = beta_sample_loss(flat_logits, sample_items, estimator, beta_lambda, kl_anchor_alpha)
    loss.backward()
    if flat_logits.grad is None:
        raise RuntimeError("flat_logits.grad is None after beta_sample_loss.backward()")
    options_per_chunk = [int(chunk_logits.shape[0]) for chunk_logits in chunk_logits_list]
    grad_per_chunk = flat_logits.grad.split(options_per_chunk, dim=0)
    sync_device(device)
    t_after_loss = time.perf_counter()

    chunk_bar.set_description("pop_beta pass 2 (fwd + bwd)")
    for chunk, rng_state, chunk_grad in zip(chunks, chunk_rng_states, grad_per_chunk, strict=True):
        _set_device_rng_state(device, rng_state)
        chunk_logits = forward_chunk_logits(model, chunk, device)
        chunk_logits.backward(chunk_grad)
        del chunk_logits
        sync_device(device)
        chunk_bar.update(1)
    t_after_backward = time.perf_counter()

    timings_ms = {
        "fwd_ms": (t_after_forward - t_start) * 1000.0,
        "loss_ms": (t_after_loss - t_after_forward) * 1000.0,
        "bwd_ms": (t_after_backward - t_after_loss) * 1000.0,
    }
    return loss_out, timings_ms


def per_dataset_summary(step_df: pd.DataFrame, config: PaperTrainConfig) -> pd.DataFrame:
    """First-15% vs last-15% training summary per dataset (`ce_datasets` and `beta_datasets`)."""
    ce_set = set(config.ce_datasets)
    summary_rows = []
    for dataset_name in config.train_datasets:
        dataset_steps = step_df[step_df["dataset"] == dataset_name]
        if len(dataset_steps) == 0:
            raise AssertionError(f"No steps recorded for training dataset {dataset_name!r}")
        edge_window = max(1, round(0.15 * len(dataset_steps)))
        first_slice = dataset_steps.iloc[:edge_window]
        last_slice = dataset_steps.iloc[-edge_window:]
        is_ce = dataset_name in ce_set
        nan = float("nan")
        summary_rows.append(
            {
                "dataset": dataset_name,
                "mode": "ce" if is_ce else "pop_beta",
                "sample_size": config.sample_size_by_dataset[dataset_name],
                "steps": len(dataset_steps),
                "ce_loss_first15pct": float(first_slice["ce_loss"].mean()) if is_ce else nan,
                "ce_loss_last15pct": float(last_slice["ce_loss"].mean()) if is_ce else nan,
                "cross_loss_first15pct": nan if is_ce else float(first_slice["cross_loss"].mean()),
                "cross_loss_last15pct": nan if is_ce else float(last_slice["cross_loss"].mean()),
                "kl_anchor_first15pct": nan if is_ce else float(first_slice["kl_anchor_loss"].mean()),
                "kl_anchor_last15pct": nan if is_ce else float(last_slice["kl_anchor_loss"].mean()),
                "sample_beta_mse_first15pct": nan if is_ce else float(first_slice["sample_beta_mse"].mean()),
                "sample_beta_mse_last15pct": nan if is_ce else float(last_slice["sample_beta_mse"].mean()),
                "monitor_human_kl_first15pct": float(first_slice["monitor_human_kl"].mean()),
                "monitor_human_kl_last15pct": float(last_slice["monitor_human_kl"].mean()),
            }
        )
    return pd.DataFrame(summary_rows).set_index("dataset")


def target_identity(
    target_std_betas_by_dataset: dict[str, dict[str, torch.Tensor]], beta_datasets: tuple[str, ...]
) -> dict:
    """Serialize table target betas for `beta_datasets` into JSON-safe lists for `run.json`."""
    identity = {}
    for dataset_name in beta_datasets:
        targets = target_std_betas_by_dataset[dataset_name]
        identity[dataset_name] = {
            group: [float(value) for value in target.tolist()] for group, target in targets.items()
        }
    return identity


def load_paper_run(run_dir: Path) -> RunResult:
    """Read back a saved `paper_beta` run (finished or incomplete) from disk."""
    run_dir = Path(run_dir)
    run_info = runs.read_run_json(run_dir)
    config = PaperTrainConfig.from_dict(run_info["identity"]["config"])
    step_df = pd.read_csv(run_dir / "steps.csv")
    val_history_df = pd.read_csv(run_dir / "val_history.csv").set_index("step")
    validator_names = list(run_info["identity"]["validators"])

    def validation_from(val_row, beta_dfs: dict) -> dict:
        validation = {}
        for name in validator_names:
            if name in beta_dfs:
                metrics = {metric: float(val_row[f"{name}/{metric}"]) for metric in (*METRIC_NAMES, "eval_wall_sec")}
                validation[name] = (metrics, beta_dfs[name])
        return validation

    best_info = runs.load_best(run_dir)["best"]
    best_step = int(best_info["step"])
    last_step = int(val_history_df.index[-1])
    final_validation = validation_from(val_history_df.loc[last_step], runs.read_betas(run_dir, last_step))
    if runs.best_betas_file(run_dir, best_step).exists():
        best_row = runs.read_best_row(run_dir)
        best_validation = validation_from(best_row, runs.read_best_betas(run_dir, best_step))
    else:
        best_row = val_history_df.loc[best_step].to_dict()
        best_row["step"] = best_step
        best_validation = validation_from(val_history_df.loc[best_step], runs.read_betas(run_dir, best_step))
    summary = run_info["summary"]
    if summary is None:
        summary = {"label": config.label(), "status": run_info["status"]}
    return RunResult(
        config=config.as_dict(),
        summary=pd.Series(summary, name=config.label()),
        step_df=step_df,
        val_history_df=val_history_df,
        per_dataset_df=per_dataset_summary(step_df, config) if len(step_df) > 0 else pd.DataFrame(),
        final_validation=final_validation,
        best=best_info,
        best_validation=best_validation,
        best_row=best_row,
        run_dir=run_dir,
    )


def run_paper_training(
    *,
    model,
    items_by_dataset: dict[str, list[dict]],
    specs_by_dataset: dict[str, dict],
    target_std_betas_by_dataset: dict[str, dict[str, torch.Tensor]],
    validators: dict,
    step0_validation: dict,
    init_snapshot: dict[str, torch.Tensor],
    config: PaperTrainConfig,
    device: str,
    run_dir: Path,
    best_metric: str,
    best_mode: str,
    run_metadata: dict,
    from_scratch: bool,
) -> RunResult:
    """Train (or resume / load if finished) one paper regime run and evaluate best.pt on all held-out validators."""
    if not isinstance(config, PaperTrainConfig):
        raise TypeError(f"config must be a PaperTrainConfig, got {type(config).__name__}")
    if device not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported device {device!r}; expected 'cuda' or 'cpu'")
    for dataset_name in config.train_datasets:
        if dataset_name not in items_by_dataset:
            raise ValueError(f"No items for training dataset {dataset_name!r}; have {sorted(items_by_dataset)}")
        if len(items_by_dataset[dataset_name]) == 0:
            raise ValueError(f"Training dataset {dataset_name!r} has 0 items")
    for dataset_name in config.beta_datasets:
        if len(items_by_dataset[dataset_name]) < 2:
            raise ValueError(
                f"beta_dataset {dataset_name!r} needs >= 2 items for split cross loss, "
                f"got {len(items_by_dataset[dataset_name])}"
            )
        if dataset_name not in specs_by_dataset:
            raise ValueError(f"No regression specs for beta_dataset {dataset_name!r}; have {sorted(specs_by_dataset)}")
        if dataset_name not in target_std_betas_by_dataset:
            raise ValueError(
                f"No target betas for beta_dataset {dataset_name!r}; have {sorted(target_std_betas_by_dataset)}"
            )
    if set(step0_validation) != set(validators):
        raise ValueError(
            f"step0_validation keys {sorted(step0_validation)} != validators {sorted(validators)}"
        )
    if not isinstance(from_scratch, bool):
        raise TypeError(f"from_scratch must be a bool, got {from_scratch!r}")
    if set(config.eval_every_by_validator) != set(validators):
        raise ValueError(
            f"eval_every_by_validator has {sorted(config.eval_every_by_validator)}, validators are {sorted(validators)}"
        )

    validator_names = list(validators)
    run_dir = Path(run_dir)
    step0_losses = paper_validation_losses(validators, step0_validation, config)
    step0_row = make_val_row(0, 0, 0, float("nan"), validator_names, step0_validation, step0_losses, float("nan"), 0.0)
    runs.check_best_rule(best_metric, best_mode, step0_row, validator_names)
    best_validator = best_metric.split("/")[0]

    identity = runs.run_identity(
        config.as_dict(), run_metadata, items_by_dataset, validators, init_snapshot, best_metric, best_mode
    )
    identity["population_targets"] = target_identity(target_std_betas_by_dataset, config.beta_datasets)

    run_exists = (run_dir / "run.json").exists()
    if run_exists and from_scratch:
        print(f"from_scratch=True: deleting existing run in {run_dir}")
        runs.delete_run_folder(run_dir)
    resume = run_exists and not from_scratch
    if resume:
        run_info = runs.open_run_for_resume(run_dir, identity)
        if run_info["status"] == runs.STATUS_FINISHED:
            print(f"{run_dir} already finished; loading from disk instead of training")
            return load_paper_run(run_dir)
    else:
        runs.create_run_folder(run_dir, identity)

    t_wall0 = time.perf_counter()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rng = random.Random(config.seed)

    merged_datasets = single_task_beta_datasets(items_by_dataset, config)
    for dataset_name in merged_datasets:
        print(
            f"{dataset_name}: {len(items_by_dataset[dataset_name]):,} tasks mod sample size "
            f"{config.sample_size_by_dataset[dataset_name]} = 1, so each epoch has one sample of "
            f"{config.sample_size_by_dataset[dataset_name] + 1} tasks"
        )
    epoch_steps = paper_steps_per_epoch(items_by_dataset, config)
    total_steps = epoch_steps * config.num_epochs
    postfix_every = max(1, epoch_steps // 30)
    rolling_steps = max(1, epoch_steps // 20)

    # Build FrozenBaseEstimator ONLY for beta_datasets (ce_datasets never compute p0 or OLS operators)
    estimators: dict[str, FrozenBaseEstimator] = {}
    all_beta_group_columns: list[str] = []
    for dataset_name in config.beta_datasets:
        estimator = FrozenBaseEstimator(
            dataset_name,
            items_by_dataset[dataset_name],
            specs_by_dataset[dataset_name],
            target_std_betas_by_dataset[dataset_name],
            device,
        )
        estimators[dataset_name] = estimator
        for op in estimator.operators:
            all_beta_group_columns.append(f"cross_loss | {dataset_name}/{op['group']}")
    ce_dataset_set = set(config.ce_datasets)

    trainable_params = None
    optimizer = None
    step_bar = None
    chunk_bar = None
    validation_bar = None
    scheduler = None
    finished = False
    try:
        reset_trainable(model, init_snapshot)
        set_checkpointing(model, config.gradient_checkpointing)
        trainable_params = list(model.trainable_parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=config.lr, weight_decay=config.weight_decay, fused=(device == "cuda")
        )
        scheduler = make_scheduler(optimizer, config, total_steps)
        initial_chunk_desc = "p0" if estimators else "micro-chunks (waiting)"
        chunk_bar = HtmlProgress(
            total=0, initial=0, description=initial_chunk_desc, unit="pass", min_interval_sec=PROGRESS_INTERVAL_SEC
        )

        if resume:
            state = runs.load_last(run_dir)
            reset_trainable(model, state["trainable"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            runs.restore_rng_states(rng, state["rng"])
            if set(state["p0_by_dataset"]) != set(estimators):
                raise ValueError(
                    f"Saved p0 for {sorted(state['p0_by_dataset'])}, expected beta_datasets {sorted(estimators)}"
                )
            for dataset_name, estimator in estimators.items():
                estimator.load_p0(state["p0_by_dataset"][dataset_name])
            start_step = state["step"] + 1
            if state["epoch_plan"] is None:
                epoch_samples = None
            else:
                epoch_samples = runs.epoch_plan_from_positions(state["epoch_plan"], items_by_dataset)
            step_rows = state["step_rows"]
            val_rows = state["val_rows"]
            rows_seen = state["rows_seen"]
            cum_train_sec = state["cum_train_sec"]
            cum_eval_sec = state["cum_eval_sec"]
            wall_sec_before = state["wall_sec"]
            best = state["best"]
            best_validation = state["best_validation"]
            final_validation = state["final_validation"]
            runs.record_resume(run_dir, run_info, identity, state["step"])
            state = None
        else:
            torch.manual_seed(config.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(config.seed)
            t_fill0 = time.perf_counter()
            for dataset_name, estimator in estimators.items():
                estimator.compute_p0(model, items_by_dataset[dataset_name], config.p0_chunk, chunk_bar)
            start_step = 1
            epoch_samples = None
            step_rows = []
            val_rows = [step0_row]
            rows_seen = 0
            cum_train_sec = time.perf_counter() - t_fill0
            cum_eval_sec = 0.0
            wall_sec_before = 0.0
            best = {"metric": best_metric, "mode": best_mode, "step": 0, "value": float(val_rows[0][best_metric])}
            best_validation = step0_validation
            final_validation = step0_validation
            runs.save_best(run_dir, model, best)
            runs.write_betas(run_dir, 0, step0_validation)
            runs.write_tables(run_dir, step_rows, val_rows)

        if start_step > 1 and (start_step - 1) % epoch_steps != 0 and epoch_samples is None:
            raise AssertionError(f"Resuming mid-epoch at step {start_step} requires saved epoch_plan")

        def save_last_state(step: int) -> None:
            epoch_plan = None if epoch_samples is None else runs.epoch_plan_positions(epoch_samples, items_by_dataset)
            p0_states = {dataset_name: estimator.p0_state() for dataset_name, estimator in estimators.items()}
            progress = {
                "step_rows": step_rows,
                "val_rows": val_rows,
                "rows_seen": rows_seen,
                "cum_train_sec": cum_train_sec,
                "cum_eval_sec": cum_eval_sec,
                "wall_sec": wall_sec_before + (time.perf_counter() - t_wall0),
                "best": best,
                "best_validation": best_validation,
                "final_validation": final_validation,
                "p0_by_dataset": p0_states,
            }
            runs.save_last(run_dir, resume_state(step, model, optimizer, scheduler, rng, epoch_plan, progress))

        if not resume:
            save_last_state(0)

        live = LiveDisplay(list(validators), epoch_steps, rolling_steps, False)
        live.update_table(val_rows, best)
        live.redraw(step_rows, val_rows, best)

        step_bar = HtmlProgress(
            total=total_steps,
            initial=start_step - 1,
            description=config.label(),
            unit="step",
            min_interval_sec=PROGRESS_INTERVAL_SEC,
        )
        validation_bar = HtmlProgress(
            total=0, initial=0, description="validation (waiting)", unit="batch", min_interval_sec=PROGRESS_INTERVAL_SEC
        )

        for step_idx in range(start_step, total_steps + 1):
            epoch_idx = (step_idx - 1) // epoch_steps + 1
            step_in_epoch = (step_idx - 1) % epoch_steps + 1
            if step_in_epoch == 1:
                epoch_samples = build_paper_epoch_samples(rng, items_by_dataset, config)
                if len(epoch_samples) != epoch_steps:
                    raise AssertionError(f"Expected {epoch_steps} samples per epoch, got {len(epoch_samples)}")
            sample_dataset, sample_items = epoch_samples[step_in_epoch - 1]
            rows_seen += len(sample_items)

            t_step0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if sample_dataset in ce_dataset_set:
                loss_out, timings_ms = ce_one_pass_step(
                    model, sample_items, config.micro_chunk, device, chunk_bar
                )
            else:
                loss_out, timings_ms = beta_two_pass_step(
                    model,
                    sample_items,
                    estimators[sample_dataset],
                    config.beta_lambda,
                    config.kl_anchor_alpha,
                    config.micro_chunk,
                    device,
                    chunk_bar,
                )

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"Non-finite gradient norm {float(grad_norm)} at step {step_idx}")
            step_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if step_idx < total_steps:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            sync_device(device)
            step_ms = (time.perf_counter() - t_step0) * 1000.0
            cum_train_sec += step_ms / 1000.0

            step_row = {
                "step": step_idx,
                "epoch": epoch_idx,
                "dataset": sample_dataset,
                "mode": loss_out["mode"],
                "num_rows": len(sample_items),
                "rows_seen": rows_seen,
                "half_1_rows": loss_out["half_sizes"][0],
                "half_2_rows": loss_out["half_sizes"][1],
                "ce_loss": loss_out["ce_loss"],
                "cross_loss": loss_out["cross_loss"],
                "weighted_cross_loss": loss_out["weighted_cross_loss"],
                "kl_anchor_loss": loss_out["kl_anchor_loss"],
                "weighted_kl_anchor_loss": loss_out["weighted_kl_anchor_loss"],
                "total_loss": loss_out["total_loss"],
                "sample_beta_mse": loss_out["sample_beta_mse"],
                "monitor_human_kl": loss_out["monitor_human_kl"],
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": step_lr,
                "fwd_ms": timings_ms["fwd_ms"],
                "loss_ms": timings_ms["loss_ms"],
                "bwd_ms": timings_ms["bwd_ms"],
                "step_ms": step_ms,
                "ms_per_row": step_ms / len(sample_items),
            }
            for col in all_beta_group_columns:
                step_row[col] = float("nan")
            for group, group_cross in loss_out["group_cross_loss"].items():
                step_row[f"cross_loss | {sample_dataset}/{group}"] = group_cross
            step_rows.append(step_row)

            if step_idx % postfix_every == 0 or step_idx == total_steps:
                recent = step_rows[-rolling_steps:]
                postfix = {}
                ce_vals = [r["ce_loss"] for r in recent if r["mode"] == "ce"]
                beta_vals = [r["cross_loss"] for r in recent if r["mode"] == "pop_beta"]
                kl_p0_vals = [r["kl_anchor_loss"] for r in recent if r["mode"] == "pop_beta"]
                if ce_vals:
                    postfix["ce"] = f"{float(np.mean(ce_vals)):.4f}"
                if beta_vals:
                    postfix["cross"] = f"{float(np.mean(beta_vals)):.5f}"
                    postfix["kl_p0"] = f"{float(np.mean(kl_p0_vals)):.4f}"
                postfix["step_ms"] = f"{float(np.mean([r['step_ms'] for r in recent])):.1f}"
                step_bar.set_postfix(**postfix)

            due_validators = validators_due(step_idx, total_steps, validator_names, config)
            if due_validators:
                step_validators = {name: validators[name] for name in due_validators}
                step_validation = validate_all(model, step_validators, validation_bar)
                for metrics, _ in step_validation.values():
                    cum_eval_sec += metrics["eval_wall_sec"]
                losses = paper_validation_losses(validators, step_validation, config)
                previous_eval_step = val_rows[-1]["step"]
                interval_losses = [r["total_loss"] for r in step_rows if r["step"] > previous_eval_step]
                val_row = make_val_row(
                    step_idx,
                    epoch_idx,
                    rows_seen,
                    float(np.mean(interval_losses)),
                    validator_names,
                    step_validation,
                    losses,
                    step_lr,
                    cum_train_sec,
                )
                val_rows.append(val_row)
                if step_idx == total_steps:
                    final_validation = step_validation
                if best_validator in step_validation:
                    value = float(val_row[best_metric])
                    if runs.is_better(value, best["value"], best_mode):
                        best = {"metric": best_metric, "mode": best_mode, "step": step_idx, "value": value}
                        best_validation = step_validation
                        runs.save_best(run_dir, model, best)
                runs.write_betas(run_dir, step_idx, step_validation)
                runs.write_tables(run_dir, step_rows, val_rows)
                save_last_state(step_idx)
                live.update_table(val_rows, best)

            if step_idx % postfix_every == 0 or due_validators:
                live.redraw(step_rows, val_rows, best)
            step_bar.update(1)

        # Held-out evaluation of best.pt on any validator (e.g. grummon2025) that did not run at best['step']
        missing_validators = [name for name in validator_names if name not in best_validation]
        if missing_validators:
            final_weights = snapshot_trainable(model)
            reset_trainable(model, runs.load_best(run_dir)["trainable"])
            held_out_validators = {name: validators[name] for name in missing_validators}
            held_out_validation = validate_all(model, held_out_validators, validation_bar)
            for metrics, _ in held_out_validation.values():
                cum_eval_sec += metrics["eval_wall_sec"]
            reset_trainable(model, final_weights)
            final_weights = None
            best_validation = {
                name: (best_validation[name] if name in best_validation else held_out_validation[name])
                for name in validator_names
            }
            print(f"best.pt (step {best['step']}) validated at the end on: {', '.join(missing_validators)}")

        best_row_source = val_rows[0]
        for row in val_rows:
            if row["step"] == best["step"]:
                best_row_source = row
        best_losses = paper_validation_losses(validators, best_validation, config)
        best_row = make_val_row(
            best["step"],
            best_row_source["epoch"],
            best_row_source["rows_seen"],
            best_row_source["train_loss"],
            validator_names,
            best_validation,
            best_losses,
            best_row_source["lr"],
            best_row_source["train_elapsed_sec"],
        )
        runs.write_best_betas(run_dir, best["step"], best_validation)
        runs.write_best_row(run_dir, best_row)
        finished = True
    finally:
        for param in model.trainable_parameters():
            param.grad = None
        trainable_params = None
        optimizer = None
        scheduler = None
        estimators = None
        if chunk_bar is not None:
            chunk_bar.close()
        if step_bar is not None:
            step_bar.close()
        if validation_bar is not None:
            validation_bar.close()
        free_gpu()
        if not finished:
            runs.set_status(run_dir, runs.STATUS_INTERRUPTED, None)

    step_df = pd.DataFrame(step_rows)
    val_history_df = pd.DataFrame(val_rows).set_index("step")
    eval_wall_columns = [f"{name}/eval_wall_sec" for name in validators]
    summary = pd.Series(
        {
            "label": config.label(),
            "steps_per_epoch": epoch_steps,
            "total_steps": total_steps,
            "rows_seen": rows_seen,
            "best_metric": best["metric"],
            "best_mode": best["mode"],
            "best_step": best["step"],
            "best_value": best["value"],
            "mean_grad_norm": float(step_df["grad_norm"].mean()),
            "mean_step_ms": float(step_df["step_ms"].mean()),
            "mean_ms_per_row": (cum_train_sec * 1000.0) / rows_seen,
            "mean_eval_wall_sec": float(val_history_df[eval_wall_columns].iloc[1:].sum(axis=1).mean()),
            "train_only_sec": cum_train_sec,
            "eval_only_sec": cum_eval_sec,
            "total_wall_sec": wall_sec_before + (time.perf_counter() - t_wall0),
        },
        name=config.label(),
    )
    runs.set_status(run_dir, runs.STATUS_FINISHED, json_ready(summary.to_dict()))
    live.update_table(val_rows, best)
    return RunResult(
        config=config.as_dict(),
        summary=summary,
        step_df=step_df,
        val_history_df=val_history_df,
        per_dataset_df=per_dataset_summary(step_df, config),
        final_validation=final_validation,
        best=best,
        best_validation=best_validation,
        best_row=best_row,
        run_dir=run_dir,
    )

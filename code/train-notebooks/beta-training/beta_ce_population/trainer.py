"""The CE + lambda x population beta training run: a copy of `beta_population.trainer` with the loss of
`beta_ce_population.loss.combined_loss`.

Differences from `beta_population.trainer.run_population_training`:
- the gradient comes from CE + beta_lambda * cross loss (`combined_loss`); CE uses the respondents' y;
- `<validator>/val_loss` = that validator's KL + beta_lambda * std beta MSE (`beta_train.runs.validation_losses`,
  the same form as v8-v12);
- per step, `ce_loss`, `cross_loss`, `weighted_cross_loss` and `total_loss` are logged (no `monitor_kl`: CE is that KL).
Unchanged from it: frozen p0 computed at the start (eval mode) and saved in last.pt; one-task samples merged when
N mod S == 1 (the cross loss needs 2 tasks); sampling, validation intervals, run folder, best.pt, held-out check of
best.pt and resume reuse `beta_train`.
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
from beta_train.step import backprop_chunks, predict_chunk
from beta_train.trainer import RunResult, json_ready, make_scheduler, resume_state, validators_due
from beta_train.validation import METRIC_NAMES, validate_all

from beta_ce_population.config import CEPopulationTrainConfig
from beta_ce_population.loss import FrozenBaseEstimator, combined_loss


def single_task_datasets(items_by_dataset: dict[str, list[dict]], config: CEPopulationTrainConfig) -> list[str]:
    # Training datasets whose epoch would end in a one-task sample (N mod S == 1), which the cross loss cannot split
    datasets = []
    for dataset_name in config.train_datasets:
        num_items = len(items_by_dataset[dataset_name])
        if num_items % config.sample_size_by_dataset[dataset_name] == 1:
            datasets.append(dataset_name)
    return datasets


def population_steps_per_epoch(items_by_dataset: dict[str, list[dict]], config: CEPopulationTrainConfig) -> int:
    # The shared steps per epoch, minus one per dataset whose one-task sample is merged into another sample
    return steps_per_epoch(items_by_dataset, config) - len(single_task_datasets(items_by_dataset, config))


def build_population_epoch_samples(
    rng: random.Random, items_by_dataset: dict[str, list[dict]], config: CEPopulationTrainConfig
) -> list[tuple[str, list[dict]]]:
    # The shared epoch plan (same shuffles, same RNG use), then every one-task sample is added to the first other
    # sample of its dataset in epoch order (one sample of S + 1 tasks); every task is still seen once per epoch
    epoch_samples = build_epoch_samples(rng, items_by_dataset, config)
    single_positions = []
    for position, (_, sample_items) in enumerate(epoch_samples):
        if len(sample_items) == 1:
            single_positions.append(position)
    expected_singles = single_task_datasets(items_by_dataset, config)
    single_datasets = [epoch_samples[position][0] for position in single_positions]
    if sorted(single_datasets) != sorted(expected_singles):
        raise AssertionError(f"One-task samples for {single_datasets}, expected {expected_singles}")

    merged_samples = list(epoch_samples)
    for single_position in single_positions:
        single_dataset, single_items = epoch_samples[single_position]
        target_position = None
        for position, (dataset_name, _) in enumerate(merged_samples):
            if dataset_name == single_dataset and position != single_position:
                target_position = position
                break
        if target_position is None:
            raise ValueError(f"{single_dataset}: no other sample to merge its one-task sample into")
        target_dataset, target_items = merged_samples[target_position]
        merged_samples[target_position] = (target_dataset, list(target_items) + list(single_items))

    kept_samples = []
    for position, sample in enumerate(merged_samples):
        if position not in single_positions:
            kept_samples.append(sample)
    return kept_samples


def ce_population_step(
    model, sample_items: list[dict], estimator: FrozenBaseEstimator, beta_lambda: float, micro_chunk: int, chunk_bar
) -> tuple[dict, dict]:
    # Fill every trainable parameter's .grad with the gradient of CE + beta_lambda * cross loss, by two-pass
    # micro-chunking (`beta_train.step` explains why the two passes give the exact gradient). Returns (loss_out, timings_ms).
    chunks = split_into_chunks(sample_items, micro_chunk)
    chunk_bar.reset(total=2 * len(chunks))
    chunk_bar.set_description("pass 1 (forward)")
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # 1. Pass 1: forward every chunk without a graph; save the RNG state right before each chunk
    chunk_rng_states = []
    chunk_logits_list = []
    with torch.no_grad():
        for chunk in chunks:
            chunk_rng_states.append(torch.cuda.get_rng_state())
            chunk_logits_list.append(predict_chunk(model, chunk))
            torch.cuda.synchronize()
            chunk_bar.update(1)
    flat_logits = torch.cat(chunk_logits_list, dim=0)
    torch.cuda.synchronize()
    t_after_forward = time.perf_counter()

    # 2. CE + beta_lambda * cross loss of the two half-sample beta estimates; its backward fills flat_logits.grad
    flat_logits.requires_grad_(True)
    loss, loss_out = combined_loss(flat_logits, sample_items, estimator, beta_lambda)
    loss.backward()
    options_per_chunk = [int(chunk_logits.shape[0]) for chunk_logits in chunk_logits_list]
    grad_per_chunk = flat_logits.grad.split(options_per_chunk, dim=0)
    torch.cuda.synchronize()
    t_after_loss = time.perf_counter()

    # 3. Pass 2: replay each chunk with its saved RNG state and push its gradient slice into the weights
    chunk_bar.set_description("pass 2 (forward + backward)")
    backprop_chunks(model, chunks, chunk_rng_states, grad_per_chunk, chunk_bar)
    torch.cuda.synchronize()
    t_after_backward = time.perf_counter()

    timings_ms = {
        "fwd_ms": (t_after_forward - t_start) * 1000.0,
        "loss_ms": (t_after_loss - t_after_forward) * 1000.0,
        "bwd_ms": (t_after_backward - t_after_loss) * 1000.0,
    }
    return loss_out, timings_ms


def per_dataset_summary(step_df: pd.DataFrame, config: CEPopulationTrainConfig) -> pd.DataFrame:
    # CE, cross loss and full-sample beta MSE over the first and last 15% of each training dataset's steps
    summary_rows = []
    for dataset_name in config.train_datasets:
        dataset_steps = step_df[step_df["dataset"] == dataset_name]
        edge_window = max(1, round(0.15 * len(dataset_steps)))
        summary_rows.append(
            {
                "dataset": dataset_name,
                "sample_size": config.sample_size_by_dataset[dataset_name],
                "steps": len(dataset_steps),
                "ce_loss_first15pct": float(dataset_steps["ce_loss"].iloc[:edge_window].mean()),
                "ce_loss_last15pct": float(dataset_steps["ce_loss"].iloc[-edge_window:].mean()),
                "cross_loss_first15pct": float(dataset_steps["cross_loss"].iloc[:edge_window].mean()),
                "cross_loss_last15pct": float(dataset_steps["cross_loss"].iloc[-edge_window:].mean()),
                "sample_beta_mse_first15pct": float(dataset_steps["sample_beta_mse"].iloc[:edge_window].mean()),
                "sample_beta_mse_last15pct": float(dataset_steps["sample_beta_mse"].iloc[-edge_window:].mean()),
            }
        )
    return pd.DataFrame(summary_rows).set_index("dataset")


def target_identity(target_std_betas_by_dataset: dict[str, dict[str, torch.Tensor]]) -> dict:
    # The targets as plain lists, so they are part of the run identity (a resume must use the same targets)
    identity = {}
    for dataset_name, targets in target_std_betas_by_dataset.items():
        identity[dataset_name] = {}
        for group, target in targets.items():
            identity[dataset_name][group] = [float(value) for value in target.tolist()]
    return identity


def load_ce_population_run(run_dir: Path) -> RunResult:
    # A saved CE + population beta run (finished or not) read back from its folder: config, stats, final and best validation
    run_dir = Path(run_dir)
    run_info = runs.read_run_json(run_dir)
    config_dict = run_info["identity"]["config"]
    config_args = dict(config_dict)
    config_args["train_datasets"] = tuple(config_dict["train_datasets"])
    config = CEPopulationTrainConfig(**config_args)
    step_df = pd.read_csv(run_dir / "steps.csv")
    val_history_df = pd.read_csv(run_dir / "val_history.csv").set_index("step")
    validator_names = list(run_info["identity"]["validators"])

    def validation_from(val_row, beta_dfs: dict) -> dict:
        # {validator: (metrics, beta_df)} for the validators that have betas (the ones that ran)
        validation = {}
        for name in validator_names:
            if name in beta_dfs:
                metrics = {}
                for metric in (*METRIC_NAMES, "eval_wall_sec"):
                    metrics[metric] = float(val_row[f"{name}/{metric}"])
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
        # Unfinished run: only the validators that ran at the best step
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
        per_dataset_df=per_dataset_summary(step_df, config),
        final_validation=final_validation,
        best=best_info,
        best_validation=best_validation,
        best_row=best_row,
        run_dir=run_dir,
    )


def run_ce_population_training(
    *,
    model,
    items_by_dataset: dict[str, list[dict]],
    specs_by_dataset: dict[str, dict],             # {dataset: {group: RegressionSpec}}
    target_std_betas_by_dataset: dict[str, dict[str, torch.Tensor]],   # {dataset: {group: standardized target betas}}
    validators: dict,                              # {name: PopulationValidator or TrainingPathValidator}
    step0_validation: dict,
    init_snapshot: dict[str, torch.Tensor],
    config: CEPopulationTrainConfig,
    device: str,
    run_dir: Path,                                 # this run's folder: resumed if it holds a run, created if missing
    best_metric: str,                              # ONE validator's column that picks checkpoints/best.pt, e.g. "kriner2026/beta_mse"
    best_mode: str,                                # "min" or "max"
    run_metadata: dict,                            # notebook-declared settings to record (model key, notebook, ...); JSON
    from_scratch: bool,                            # True: delete the run already in run_dir (if any) and train again
) -> RunResult:
    # 0. Check inputs, then resume the run in run_dir, or create it (deleting an existing one first if from_scratch)
    if not isinstance(config, CEPopulationTrainConfig):
        raise TypeError(f"config must be a CEPopulationTrainConfig, got {type(config).__name__}")
    for dataset_name in config.train_datasets:
        if dataset_name not in items_by_dataset:
            raise ValueError(f"No items for training dataset {dataset_name!r}; have {sorted(items_by_dataset)}")
        if dataset_name not in specs_by_dataset:
            raise ValueError(f"No regression specs for training dataset {dataset_name!r}; have {sorted(specs_by_dataset)}")
        if dataset_name not in target_std_betas_by_dataset:
            raise ValueError(f"No target betas for training dataset {dataset_name!r}; have {sorted(target_std_betas_by_dataset)}")
    if set(target_std_betas_by_dataset) != set(config.train_datasets):
        raise ValueError(f"Target betas for {sorted(target_std_betas_by_dataset)}, training datasets are {sorted(config.train_datasets)}")
    if set(step0_validation) != set(validators):
        raise ValueError("step0_validation must have one entry per validator")
    if not isinstance(from_scratch, bool):
        raise TypeError(f"from_scratch must be a bool, got {from_scratch!r}")
    if set(config.eval_every_by_validator) != set(validators):
        raise ValueError(
            f"eval_every_by_validator has {sorted(config.eval_every_by_validator)}, validators are {sorted(validators)}"
        )
    validator_names = list(validators)
    run_dir = Path(run_dir)
    step0_losses = runs.validation_losses(validators, step0_validation, config.beta_lambda)
    step0_row = make_val_row(0, 0, 0, float("nan"), validator_names, step0_validation, step0_losses, float("nan"), 0.0)
    runs.check_best_rule(best_metric, best_mode, step0_row, validator_names)
    best_validator = best_metric.split("/")[0]
    identity = runs.run_identity(
        config.as_dict(), run_metadata, items_by_dataset, validators, init_snapshot, best_metric, best_mode
    )
    identity["population_targets"] = target_identity(target_std_betas_by_dataset)
    run_exists = (run_dir / "run.json").exists()
    if run_exists and from_scratch:
        print(f"from_scratch=True: deleting the existing run in {run_dir}")
        runs.delete_run_folder(run_dir)
    resume = run_exists and not from_scratch
    if resume:
        run_info = runs.open_run_for_resume(run_dir, identity)
        if run_info["status"] == runs.STATUS_FINISHED:
            print(f"{run_dir} already finished; loading it instead of training")
            return load_ce_population_run(run_dir)
    else:
        runs.create_run_folder(run_dir, identity)

    t_wall0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = random.Random(config.seed)

    for dataset_name in config.train_datasets:
        if len(items_by_dataset[dataset_name]) < 2:
            raise ValueError(f"{dataset_name}: the cross loss needs at least 2 tasks, the dataset has {len(items_by_dataset[dataset_name])}")
    merged_datasets = single_task_datasets(items_by_dataset, config)
    for dataset_name in merged_datasets:
        print(
            f"{dataset_name}: {len(items_by_dataset[dataset_name]):,} tasks mod sample size "
            f"{config.sample_size_by_dataset[dataset_name]} = 1, so each epoch has one sample of "
            f"{config.sample_size_by_dataset[dataset_name] + 1} tasks"
        )
    epoch_steps = population_steps_per_epoch(items_by_dataset, config)
    total_steps = epoch_steps * config.num_epochs
    postfix_every = max(1, epoch_steps // 30)
    rolling_steps = max(1, epoch_steps // 20)

    # One estimator per training dataset (operators and targets now; p0 computed or loaded below)
    estimators = {}
    for dataset_name in config.train_datasets:
        estimators[dataset_name] = FrozenBaseEstimator(
            dataset_name,
            items_by_dataset[dataset_name],
            specs_by_dataset[dataset_name],
            target_std_betas_by_dataset[dataset_name],
            device,
        )

    trainable_params = None
    optimizer = None
    step_bar = None
    chunk_bar = None
    validation_bar = None
    scheduler = None
    finished = False
    try:
        # 1. Fresh start from the snapshot weights with seeded RNGs, or the saved state of the last validation
        reset_trainable(model, init_snapshot)
        set_checkpointing(model, config.gradient_checkpointing)
        trainable_params = list(model.trainable_parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay, fused=True)
        scheduler = make_scheduler(optimizer, config, total_steps)
        # The micro-chunk bar is created before the step loop because computing p0 also uses it
        chunk_bar = HtmlProgress(
            total=0, initial=0, description="p0", unit="pass", min_interval_sec=PROGRESS_INTERVAL_SEC
        )
        if resume:
            state = runs.load_last(run_dir)
            reset_trainable(model, state["trainable"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            runs.restore_rng_states(rng, state["rng"])
            if set(state["p0_by_dataset"]) != set(estimators):
                raise ValueError(f"Saved p0 for {sorted(state['p0_by_dataset'])}, training datasets are {sorted(estimators)}")
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
            torch.cuda.manual_seed_all(config.seed)
            # Frozen base probabilities p0 of the snapshot weights (eval mode), timed as training
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
            raise AssertionError(f"Resuming mid-epoch at step {start_step} needs the saved epoch plan")

        def save_last_state(step: int) -> None:
            if epoch_samples is None:
                epoch_plan = None
            else:
                epoch_plan = runs.epoch_plan_positions(epoch_samples, items_by_dataset)
            p0_states = {}
            for dataset_name, estimator in estimators.items():
                p0_states[dataset_name] = estimator.p0_state()
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
            total=total_steps, initial=start_step - 1, description=config.label(), unit="step", min_interval_sec=PROGRESS_INTERVAL_SEC
        )
        validation_bar = HtmlProgress(
            total=0, initial=0, description="validation (waiting)", unit="batch", min_interval_sec=PROGRESS_INTERVAL_SEC
        )
        for step_idx in range(start_step, total_steps + 1):
            # 2. This step's sample (a new shuffled epoch plan at the start of every epoch)
            epoch_idx = (step_idx - 1) // epoch_steps + 1
            step_in_epoch = (step_idx - 1) % epoch_steps + 1
            if step_in_epoch == 1:
                epoch_samples = build_population_epoch_samples(rng, items_by_dataset, config)
                if len(epoch_samples) != epoch_steps:
                    raise AssertionError(f"Expected {epoch_steps} samples per epoch, got {len(epoch_samples)}")
            sample_dataset, sample_items = epoch_samples[step_in_epoch - 1]
            rows_seen += len(sample_items)

            # 3. CE + lambda x cross-loss gradient by two-pass micro-chunking
            t_step0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_out, timings_ms = ce_population_step(
                model, sample_items, estimators[sample_dataset], config.beta_lambda, config.micro_chunk, chunk_bar
            )

            # 4. Clip, update, advance the LR schedule
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"Non-finite gradient norm {float(grad_norm)} at step {step_idx}")
            step_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if step_idx < total_steps:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t_step0) * 1000.0
            cum_train_sec += step_ms / 1000.0

            # 5. Log the step (total_loss = CE + lambda x cross loss, the training objective the live display plots)
            step_row = {
                "step": step_idx,
                "epoch": epoch_idx,
                "dataset": sample_dataset,
                "num_rows": len(sample_items),
                "rows_seen": rows_seen,
                "half_1_rows": loss_out["half_sizes"][0],
                "half_2_rows": loss_out["half_sizes"][1],
                "ce_loss": loss_out["ce_loss"],
                "cross_loss": loss_out["cross_loss"],
                "weighted_cross_loss": loss_out["weighted_cross_loss"],
                "total_loss": loss_out["total_loss"],
                "sample_beta_mse": loss_out["sample_beta_mse"],
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": step_lr,
                "fwd_ms": timings_ms["fwd_ms"],
                "loss_ms": timings_ms["loss_ms"],
                "bwd_ms": timings_ms["bwd_ms"],
                "step_ms": step_ms,
                "ms_per_row": step_ms / len(sample_items),
            }
            for group, group_cross in loss_out["group_cross_loss"].items():
                step_row[f"cross_loss | {group}"] = group_cross
            step_rows.append(step_row)

            if step_idx % postfix_every == 0 or step_idx == total_steps:
                recent = step_rows[-rolling_steps:]
                step_bar.set_postfix(
                    ce=f"{float(np.mean([r['ce_loss'] for r in recent])):.4f}",
                    cross=f"{float(np.mean([r['cross_loss'] for r in recent])):.5f}",
                    sample_beta=f"{float(np.mean([r['sample_beta_mse'] for r in recent])):.5f}",
                    step_ms=f"{float(np.mean([r['step_ms'] for r in recent])):.1f}",
                )

            # 6. Run the validators due at this step (each on its own interval; all at the last step), write the run folder
            due_validators = validators_due(step_idx, total_steps, validator_names, config)
            if due_validators:
                step_validators = {}
                for name in due_validators:
                    step_validators[name] = validators[name]
                step_validation = validate_all(model, step_validators, validation_bar)
                for metrics, _ in step_validation.values():
                    cum_eval_sec += metrics["eval_wall_sec"]
                losses = runs.validation_losses(validators, step_validation, config.beta_lambda)
                previous_eval_step = val_rows[-1]["step"]
                interval_losses = [r["total_loss"] for r in step_rows if r["step"] > previous_eval_step]
                val_row = make_val_row(
                    step_idx, epoch_idx, rows_seen, float(np.mean(interval_losses)), validator_names, step_validation,
                    losses, step_lr, cum_train_sec,
                )
                val_rows.append(val_row)
                if step_idx == total_steps:
                    final_validation = step_validation
                # The best checkpoint can only change when its validator ran
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

        # 7. Held-out check of best.pt: validate it on every validator that did not run at its step, then put the
        #    final weights back
        missing_validators = []
        for name in validator_names:
            if name not in best_validation:
                missing_validators.append(name)
        if missing_validators:
            final_weights = snapshot_trainable(model)
            reset_trainable(model, runs.load_best(run_dir)["trainable"])
            held_out_validators = {}
            for name in missing_validators:
                held_out_validators[name] = validators[name]
            held_out_validation = validate_all(model, held_out_validators, validation_bar)
            for metrics, _ in held_out_validation.values():
                cum_eval_sec += metrics["eval_wall_sec"]
            reset_trainable(model, final_weights)
            final_weights = None
            completed_best_validation = {}
            for name in validator_names:
                if name in best_validation:
                    completed_best_validation[name] = best_validation[name]
                else:
                    completed_best_validation[name] = held_out_validation[name]
            best_validation = completed_best_validation
            print(f"best.pt (step {best['step']}) validated at the end on: {', '.join(missing_validators)}")
        best_row_source = val_rows[0]
        for row in val_rows:
            if row["step"] == best["step"]:
                best_row_source = row
        best_losses = runs.validation_losses(validators, best_validation, config.beta_lambda)
        best_row = make_val_row(
            best["step"], best_row_source["epoch"], best_row_source["rows_seen"], best_row_source["train_loss"],
            validator_names, best_validation, best_losses, best_row_source["lr"], best_row_source["train_elapsed_sec"],
        )
        runs.write_best_betas(run_dir, best["step"], best_validation)
        runs.write_best_row(run_dir, best_row)
        finished = True
    finally:
        # Free everything this run allocated on the GPU, also after an error or interrupt; mark an unfinished run
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

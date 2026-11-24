"""The training run: optimizer, LR schedule, epoch loop over one-dataset samples, periodic validation, run folder.

Everything a run depends on is an argument; nothing is read from notebook globals. Each validator runs on its own
interval (`config.eval_every_by_validator`; all of them at step 0 and the last step). Every validation step writes the
run folder (`beta_train.runs`): stats, that step's betas, the best checkpoint (by `best_metric`, one validator's column),
and the state to resume from. At the end, best.pt is validated on every validator that did not run at its step. By default an existing run in `run_dir` is resumed from its last validation (a finished one is loaded
instead) and a missing one is created; `from_scratch=True` deletes an existing run and trains it again.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training import free_gpu

from beta_ols.unified_loss import build_sample_beta_operators

from beta_train import runs
from beta_train.batching import build_epoch_samples, steps_per_epoch
from beta_train.config import TrainConfig
from beta_train.live_display import LiveDisplay, make_val_row
from beta_train.model_state import reset_trainable, set_checkpointing, snapshot_trainable
from beta_train.progress import PROGRESS_INTERVAL_SEC, HtmlProgress
from beta_train.step import two_pass_gradients
from beta_train.validation import METRIC_NAMES, validate_all

# Gradient-split columns (config.gradient_split) averaged over each validation interval in the validation table
GRADIENT_SPLIT_TABLE_COLUMNS = ["grad_ce_norm", "grad_beta_norm", "grad_lambda_beta_norm", "grad_total_norm", "grad_cos_ce_lambda_beta"]

# OneCycle shape (fixed across runs; only the peak LR `config.lr` varies)
ONECYCLE_PCT_START = 0.1
ONECYCLE_DIV_FACTOR = 25.0
ONECYCLE_FINAL_DIV_FACTOR = 1e4


@dataclass
class RunResult:
    """Everything one run produced, with the settings that produced it."""

    config: dict                                  # TrainConfig.as_dict()
    summary: pd.Series                            # run-level settings, timings, mean grad norm
    step_df: pd.DataFrame                         # one row per optimization step
    val_history_df: pd.DataFrame                  # one row per validation (step 0 included)
    per_dataset_df: pd.DataFrame                  # train KL / beta loss, first vs last 15% of each dataset's steps
    final_validation: dict                        # {validator: (metrics, beta_df)} at the last step
    best: dict                                    # {metric, mode, step, value} of the best validation
    best_validation: dict                         # {validator: (metrics, beta_df)} of best.pt, every validator
    best_row: dict                                # every validation column of best.pt (one row, no NaN for finished runs)
    run_dir: Path                                 # the run folder (stats, betas, checkpoints/best.pt, checkpoints/last.pt)


def constant_lr_factor(step: int) -> float:
    # LambdaLR multiplier for the constant schedule
    return 1.0


def make_scheduler(optimizer, config: TrainConfig, total_steps: int):
    if config.lr_schedule == "onecycle" and total_steps > 1:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=ONECYCLE_PCT_START,
            div_factor=ONECYCLE_DIV_FACTOR,
            final_div_factor=ONECYCLE_FINAL_DIV_FACTOR,
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=constant_lr_factor)


def per_dataset_summary(step_df: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    # Train KL and beta loss over the first and last 15% of each training dataset's steps
    summary_rows = []
    for dataset_name in config.train_datasets:
        dataset_steps = step_df[step_df["dataset"] == dataset_name]
        beta_steps = dataset_steps[dataset_steps["has_beta"]]
        edge_window = max(1, round(0.15 * len(dataset_steps)))
        if len(beta_steps) > 0:
            beta_edge_window = max(1, round(0.15 * len(beta_steps)))
            raw_beta_first15 = float(beta_steps["raw_beta_loss"].iloc[:beta_edge_window].mean())
            raw_beta_last15 = float(beta_steps["raw_beta_loss"].iloc[-beta_edge_window:].mean())
        else:
            raw_beta_first15 = float("nan")
            raw_beta_last15 = float("nan")
        summary_rows.append(
            {
                "dataset": dataset_name,
                "sample_size": config.sample_size_by_dataset[dataset_name],
                "steps": len(dataset_steps),
                "steps_without_beta": int((~dataset_steps["has_beta"]).sum()),
                "skipped_groups_total": int(dataset_steps["skipped_groups"].sum()),
                "train_kl_first15pct": float(dataset_steps["kl_loss"].iloc[:edge_window].mean()),
                "train_kl_last15pct": float(dataset_steps["kl_loss"].iloc[-edge_window:].mean()),
                "train_raw_beta_first15pct": raw_beta_first15,
                "train_raw_beta_last15pct": raw_beta_last15,
            }
        )
    return pd.DataFrame(summary_rows).set_index("dataset")


def validators_due(step: int, total_steps: int, validator_names: list[str], config: TrainConfig) -> list[str]:
    # Validators that run at this step: each when the step is a multiple of its interval, all at the last step
    due = []
    for name in validator_names:
        if step % config.eval_every_by_validator[name] == 0 or step == total_steps:
            due.append(name)
    return due


def resume_state(
    step: int,
    model,
    optimizer,
    scheduler,
    rng: random.Random,
    epoch_plan: list | None,
    progress: dict,
) -> dict:
    # Everything `run_training` needs to continue after `step` exactly as if it had not stopped
    state = {
        "step": step,
        "trainable": snapshot_trainable(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": runs.rng_states(rng),
        "epoch_plan": epoch_plan,                  # current epoch's samples as item positions (None before step 1)
    }
    state.update(progress)
    return state


def load_run(run_dir: Path) -> RunResult:
    # A saved run (finished or not) read back from its folder: config, stats, final and best validation
    run_dir = Path(run_dir)
    run_info = runs.read_run_json(run_dir)
    config_dict = run_info["identity"]["config"]
    config_args = dict(config_dict)
    config_args["train_datasets"] = tuple(config_dict["train_datasets"])
    config = TrainConfig(**config_args)
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


def run_training(
    *,
    model,
    items_by_dataset: dict[str, list[dict]],
    specs_by_dataset: dict[str, dict],             # {(pseudo) dataset: {group: RegressionSpec}}; a step uses its dataset's specs
    validators: dict,                              # {name: PopulationValidator or TrainingPathValidator}
    step0_validation: dict,
    init_snapshot: dict[str, torch.Tensor],
    config: TrainConfig,
    device: str,
    run_dir: Path,                                 # this run's folder: resumed if it holds a run, created if missing
    best_metric: str,                              # ONE validator's column that picks checkpoints/best.pt, e.g. "kriner2026/val_loss"
    best_mode: str,                                # "min" or "max" (e.g. "max" for a spearman_rho column)
    run_metadata: dict,                            # notebook-declared settings to record (model key, notebook, ...); JSON
    from_scratch: bool = False,                    # True: delete the run already in run_dir (if any) and train again
) -> RunResult:
    # 0. Check inputs, then resume the run in run_dir, or create it (deleting an existing one first if from_scratch)
    for dataset_name in config.train_datasets:
        if dataset_name not in items_by_dataset:
            raise ValueError(f"No items for training dataset {dataset_name!r}; have {sorted(items_by_dataset)}")
        if dataset_name not in specs_by_dataset:
            raise ValueError(f"No regression specs for training dataset {dataset_name!r}; have {sorted(specs_by_dataset)}")
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
    run_exists = (run_dir / "run.json").exists()
    if run_exists and from_scratch:
        print(f"from_scratch=True: deleting the existing run in {run_dir}")
        runs.delete_run_folder(run_dir)
    resume = run_exists and not from_scratch
    if resume:
        run_info = runs.open_run_for_resume(run_dir, identity)
        if run_info["status"] == runs.STATUS_FINISHED:
            print(f"{run_dir} already finished; loading it instead of training")
            return load_run(run_dir)
    else:
        runs.create_run_folder(run_dir, identity)

    t_wall0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = random.Random(config.seed)

    epoch_steps = steps_per_epoch(items_by_dataset, config)
    total_steps = epoch_steps * config.num_epochs
    postfix_every = max(1, epoch_steps // 30)
    rolling_steps = max(1, epoch_steps // 20)

    trainable_params = None
    optimizer = None
    step_bar = None
    chunk_bar = None
    validation_bar = None
    scheduler = None
    sample_ops = None
    finished = False
    try:
        # 1. Fresh start from the snapshot weights with seeded RNGs, or the saved state of the last validation
        reset_trainable(model, init_snapshot)
        set_checkpointing(model, config.gradient_checkpointing)
        trainable_params = list(model.trainable_parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay, fused=True)
        scheduler = make_scheduler(optimizer, config, total_steps)
        if resume:
            state = runs.load_last(run_dir)
            reset_trainable(model, state["trainable"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            runs.restore_rng_states(rng, state["rng"])
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
            start_step = 1
            epoch_samples = None
            step_rows = []
            val_rows = [step0_row]
            rows_seen = 0
            cum_train_sec = 0.0
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
            }
            runs.save_last(run_dir, resume_state(step, model, optimizer, scheduler, rng, epoch_plan, progress))

        if not resume:
            save_last_state(0)

        live = LiveDisplay(list(validators), epoch_steps, rolling_steps, config.gradient_split)
        live.update_table(val_rows, best)
        live.redraw(step_rows, val_rows, best)

        # HTML bars redrawn in place, like the table and figure above (see HtmlProgress): one for the steps, one for the
        # micro-chunk passes, reset every step (a new bar per step would add one cell output per step)
        step_bar = HtmlProgress(
            total=total_steps, initial=start_step - 1, description=config.label(), unit="step", min_interval_sec=PROGRESS_INTERVAL_SEC
        )
        chunk_bar = HtmlProgress(
            total=0, initial=0, description="pass 1 (forward)", unit="pass", min_interval_sec=PROGRESS_INTERVAL_SEC
        )
        # One bar for the validations' GPU batches, reset by every validator (also for the end-of-run check of best.pt)
        validation_bar = HtmlProgress(
            total=0, initial=0, description="validation (waiting)", unit="batch", min_interval_sec=PROGRESS_INTERVAL_SEC
        )
        for step_idx in range(start_step, total_steps + 1):
            # 2. This step's sample (a new shuffled epoch plan at the start of every epoch)
            epoch_idx = (step_idx - 1) // epoch_steps + 1
            step_in_epoch = (step_idx - 1) % epoch_steps + 1
            if step_in_epoch == 1:
                epoch_samples = build_epoch_samples(rng, items_by_dataset, config)
                if len(epoch_samples) != epoch_steps:
                    raise AssertionError(f"Expected {epoch_steps} samples per epoch, got {len(epoch_samples)}")
            sample_dataset, sample_items = epoch_samples[step_in_epoch - 1]
            rows_seen += len(sample_items)

            # 3. Exact full-sample gradient by two-pass micro-chunking
            t_step0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            # Specs of this sample's own (pseudo) dataset: pieces of one dataset can share a group name with their own sigma
            sample_ops = build_sample_beta_operators(sample_items, specs_by_dataset[sample_dataset], device)
            loss_out, timings_ms, split_stats = two_pass_gradients(
                model, sample_items, sample_ops, config.beta_lambda, config.micro_chunk, chunk_bar, config.gradient_split
            )
            sample_ops = None

            # 4. Clip, update, advance the LR schedule
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"Non-finite gradient norm {float(grad_norm)} at step {step_idx}")
            if split_stats is not None:
                # The split's total must be the gradient that was clipped (float32 vs float64 sums)
                clipped_norm = float(grad_norm.detach().cpu())
                if abs(split_stats["grad_total_norm"] - clipped_norm) > 1e-3 * max(clipped_norm, 1e-12):
                    raise AssertionError(
                        f"Step {step_idx}: split total gradient norm {split_stats['grad_total_norm']:.6g} "
                        f"!= clipped gradient norm {clipped_norm:.6g}"
                    )
            step_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if step_idx < total_steps:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t_step0) * 1000.0
            cum_train_sec += step_ms / 1000.0

            # 5. Log the step
            has_beta = len(loss_out["group_beta_mse"]) > 0
            step_row = {
                "step": step_idx,
                "epoch": epoch_idx,
                "dataset": sample_dataset,
                "num_rows": len(sample_items),
                "rows_seen": rows_seen,
                "has_beta": has_beta,
                "skipped_groups": len(loss_out["skipped_groups"]),
                "kl_loss": loss_out["kl_loss"],
                "raw_beta_loss": loss_out["beta_loss"],
                "weighted_beta_loss": config.beta_lambda * loss_out["beta_loss"] if has_beta else 0.0,
                "total_loss": loss_out["total_loss"],
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": step_lr,
                "fwd_ms": timings_ms["fwd_ms"],
                "loss_ms": timings_ms["loss_ms"],
                "bwd_ms": timings_ms["bwd_ms"],
                "step_ms": step_ms,
                "ms_per_row": step_ms / len(sample_items),
            }
            if split_stats is not None:
                step_row.update(split_stats)
            for group, group_mse in loss_out["group_beta_mse"].items():
                step_row[f"beta_mse | {group}"] = group_mse
            step_rows.append(step_row)

            if step_idx % postfix_every == 0 or step_idx == total_steps:
                recent = step_rows[-rolling_steps:]
                recent_betas = [r["raw_beta_loss"] for r in recent if r["has_beta"]]
                step_bar.set_postfix(
                    kl=f"{float(np.mean([r['kl_loss'] for r in recent])):.3f}",
                    beta=f"{float(np.mean(recent_betas)):.4f}" if recent_betas else "n/a",
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
                if config.gradient_split:
                    # Mean gradient-split norms over the steps since the previous validation (per-step values: steps.csv)
                    # (the cosine is NaN at lambda = 0, where the beta part is zero; then the mean is NaN too)
                    for column in GRADIENT_SPLIT_TABLE_COLUMNS:
                        interval_values = []
                        for r in step_rows:
                            if r["step"] > previous_eval_step and not np.isnan(r[column]):
                                interval_values.append(r[column])
                        if interval_values:
                            val_row[f"mean {column}"] = float(np.mean(interval_values))
                        else:
                            val_row[f"mean {column}"] = float("nan")
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

            # With gradient_split (debugging), the figure is redrawn every step so the per-step split is visible as it
            # happens (costs a figure redraw per step); otherwise at the progress-bar interval and at validations
            if config.gradient_split or step_idx % postfix_every == 0 or due_validators:
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
        sample_ops = None
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


def json_ready(values: dict) -> dict:
    # numpy scalars -> plain Python numbers, so the summary can go into run.json
    plain = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            plain[key] = value.item()
        else:
            plain[key] = value
    return plain

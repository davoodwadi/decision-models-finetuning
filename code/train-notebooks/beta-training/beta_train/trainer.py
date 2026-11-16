"""The training run: optimizer, LR schedule, epoch loop over one-dataset samples, periodic validation, cleanup.

Everything a run depends on is an argument; nothing is read from notebook globals. The full config is recorded in
the returned `RunResult`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from training import free_gpu

from beta_ols.unified_loss import build_sample_beta_operators

from beta_train.batching import build_epoch_samples, steps_per_epoch
from beta_train.config import TrainConfig
from beta_train.live_display import LiveDisplay, make_val_row
from beta_train.model_state import reset_trainable, set_checkpointing
from beta_train.step import two_pass_gradients
from beta_train.validation import validate_all

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
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


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


def run_training(
    *,
    model,
    items_by_dataset: dict[str, list[dict]],
    specs: dict,
    validators: dict,                              # {name: PopulationValidator or TrainingPathValidator}
    step0_validation: dict,
    init_snapshot: dict[str, torch.Tensor],
    config: TrainConfig,
    device: str,
    show_chunk_progress: bool,                     # per-step bar over the micro-chunk passes (useful for big models)
) -> RunResult:
    # 0. Check every training dataset has items, then start from the snapshot weights with seeded RNGs
    for dataset_name in config.train_datasets:
        if dataset_name not in items_by_dataset:
            raise ValueError(f"No items for training dataset {dataset_name!r}; have {sorted(items_by_dataset)}")
    if set(step0_validation) != set(validators):
        raise ValueError("step0_validation must have one entry per validator")
    t_wall0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    reset_trainable(model, init_snapshot)
    set_checkpointing(model, config.gradient_checkpointing)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    epoch_steps = steps_per_epoch(items_by_dataset, config)
    total_steps = epoch_steps * config.num_epochs
    postfix_every = max(1, epoch_steps // 30)
    rolling_steps = max(1, epoch_steps // 20)

    trainable_params = None
    optimizer = None
    scheduler = None
    sample_ops = None
    try:
        trainable_params = list(model.trainable_parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay, fused=True)
        scheduler = make_scheduler(optimizer, config, total_steps)

        step_rows: list[dict] = []
        val_rows: list[dict] = [make_val_row(0, 0, 0, float("nan"), step0_validation, float("nan"), 0.0)]
        live = LiveDisplay(list(validators), epoch_steps, rolling_steps)
        live.update_table(val_rows)
        live.redraw(step_rows, val_rows)
        final_validation = step0_validation
        cum_train_sec = 0.0
        cum_eval_sec = 0.0
        respondents_seen = 0

        pbar = tqdm(range(1, total_steps + 1), total=total_steps, desc=config.label())
        for step_idx in pbar:
            # 1. This step's sample (a new shuffled epoch plan at the start of every epoch)
            epoch_idx = (step_idx - 1) // epoch_steps + 1
            step_in_epoch = (step_idx - 1) % epoch_steps + 1
            if step_in_epoch == 1:
                epoch_samples = build_epoch_samples(rng, items_by_dataset, config)
                if len(epoch_samples) != epoch_steps:
                    raise AssertionError(f"Expected {epoch_steps} samples per epoch, got {len(epoch_samples)}")
            sample_dataset, sample_items = epoch_samples[step_in_epoch - 1]
            respondents_seen += len(sample_items)

            # 2. Exact full-sample gradient by two-pass micro-chunking
            t_step0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            sample_ops = build_sample_beta_operators(sample_items, specs, device)
            loss_out, timings_ms = two_pass_gradients(
                model, sample_items, sample_ops, config.beta_lambda, config.micro_chunk, show_chunk_progress
            )
            sample_ops = None

            # 3. Clip, update, advance the LR schedule
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

            # 4. Log the step
            has_beta = len(loss_out["group_beta_mse"]) > 0
            step_row = {
                "step": step_idx,
                "epoch": epoch_idx,
                "dataset": sample_dataset,
                "num_respondents": len(sample_items),
                "respondents_seen": respondents_seen,
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
                "ms_per_respondent": step_ms / len(sample_items),
            }
            for group, group_mse in loss_out["group_beta_mse"].items():
                step_row[f"beta_mse | {group}"] = group_mse
            step_rows.append(step_row)

            if step_idx % postfix_every == 0 or step_idx == total_steps:
                recent = step_rows[-rolling_steps:]
                recent_betas = [r["raw_beta_loss"] for r in recent if r["has_beta"]]
                pbar.set_postfix(
                    kl=f"{float(np.mean([r['kl_loss'] for r in recent])):.3f}",
                    beta=f"{float(np.mean(recent_betas)):.4f}" if recent_betas else "n/a",
                    step_ms=f"{float(np.mean([r['step_ms'] for r in recent])):.1f}",
                )

            # 5. Validate every `eval_every` steps and at the last step
            if step_idx % config.eval_every == 0 or step_idx == total_steps:
                final_validation = validate_all(model, validators)
                for metrics, _ in final_validation.values():
                    cum_eval_sec += metrics["eval_wall_sec"]
                previous_eval_step = val_rows[-1]["step"]
                interval_losses = [r["total_loss"] for r in step_rows if r["step"] > previous_eval_step]
                val_rows.append(
                    make_val_row(
                        step_idx, epoch_idx, respondents_seen, float(np.mean(interval_losses)), final_validation, step_lr, cum_train_sec
                    )
                )
                live.update_table(val_rows)

            if step_idx % postfix_every == 0 or step_idx % config.eval_every == 0 or step_idx == total_steps:
                live.redraw(step_rows, val_rows)
    finally:
        # Free everything this run allocated on the GPU, also after an error or interrupt
        for param in model.trainable_parameters():
            param.grad = None
        trainable_params = None
        optimizer = None
        scheduler = None
        sample_ops = None
        free_gpu()

    step_df = pd.DataFrame(step_rows)
    val_history_df = pd.DataFrame(val_rows).set_index("step")
    eval_wall_columns = [f"{name}/eval_wall_sec" for name in validators]
    summary = pd.Series(
        {
            "label": config.label(),
            "steps_per_epoch": epoch_steps,
            "total_steps": total_steps,
            "respondents_seen": respondents_seen,
            "mean_grad_norm": float(step_df["grad_norm"].mean()),
            "mean_step_ms": float(step_df["step_ms"].mean()),
            "mean_ms_per_respondent": (cum_train_sec * 1000.0) / respondents_seen,
            "mean_eval_wall_sec": float(val_history_df[eval_wall_columns].iloc[1:].sum(axis=1).mean()),
            "train_only_sec": cum_train_sec,
            "eval_only_sec": cum_eval_sec,
            "total_wall_sec": time.perf_counter() - t_wall0,
        },
        name=config.label(),
    )
    live.update_table(val_rows)
    return RunResult(
        config=config.as_dict(),
        summary=summary,
        step_df=step_df,
        val_history_df=val_history_df,
        per_dataset_df=per_dataset_summary(step_df, config),
        final_validation=final_validation,
    )

"""Unified training loss for respondents from any dataset module in `beta_ols`.

Every regression is a fixed linear map of the sample's option probabilities p (all tasks, flattened):
    betas = B @ p,   B = (1 / sigma) * pinv(X)[target coefficients] @ W
- X: the regression's design rows in this sample (one per outcome row)
- W: how each outcome row is built from option probabilities (OutcomeRow.option_idx / weights)
- sigma: the regression's fixed outcome SD from the full human dataset (RegressionSpec.sigma)
Human betas use the same B with one-hot choices y, so the beta error is B @ (p - y).

Loss = KL (mean over tasks) + beta_lambda * beta loss, where beta loss = mean over regressions of each
regression's mean squared standardized beta error (every regression counts equally).
A sample must come from exactly one dataset (one dataset per optimization step).
"""

from __future__ import annotations

import torch

from beta_ols.core import RegressionSpec, raw_beta_operator


def build_sample_beta_operators(sample_items: list[dict], specs: dict[str, RegressionSpec], device: str) -> dict:
    # Fixed per-sample pieces of the loss: flat human y, option counts per task, and one B per usable regression.
    # A regression that is rank-deficient in this sample is skipped and reported.

    # 1. Give every option in the sample a position in one flat vector, in sample order
    datasets_in_sample = {item["dataset"] for item in sample_items}
    if len(datasets_in_sample) != 1:
        raise ValueError(f"A sample must come from exactly one dataset, got {sorted(datasets_in_sample)}")
    y_parts = []
    num_options = []
    rows_by_group: dict[str, list[tuple[list[float], list[int], list[float]]]] = {}
    offset = 0
    for item in sample_items:
        y_parts.append(item["y_flat"])
        num_options.extend(item["num_options"])
        for row in item["outcome_rows"]:
            if row.group not in rows_by_group:
                rows_by_group[row.group] = []
            shifted_option_idx = [offset + i for i in row.option_idx]
            rows_by_group[row.group].append((row.x, shifted_option_idx, row.weights))
        offset += item["y_flat"].shape[0]
    total_options = offset

    # 2. Per regression: standardized B = pinv(X)[target rows] @ W / sigma (skipped when X is rank-deficient)
    operators = []
    skipped_groups = []
    for group in sorted(rows_by_group):
        spec = specs[group]
        B = raw_beta_operator(rows_by_group[group], spec, total_options)
        if B is None:
            skipped_groups.append(group)
            continue
        operators.append({"group": group, "B": (B / spec.sigma).to(device)})

    return {
        "y_flat": torch.cat(y_parts).to(device=device, dtype=torch.float64),
        "num_options": num_options,
        "operators": operators,
        "skipped_groups": skipped_groups,
    }


def compute_unified_loss(flat_logits: torch.Tensor, sample_ops: dict, beta_lambda: float) -> dict:
    # Loss for one sample. `flat_logits` = every option logit of the sample, task by task, in sample order.
    y_flat = sample_ops["y_flat"]
    num_options = sample_ops["num_options"]
    num_tasks = len(num_options)
    if flat_logits.ndim != 1 or flat_logits.shape[0] != y_flat.shape[0]:
        raise ValueError(f"Expected flat logits of length {y_flat.shape[0]}, got shape {tuple(flat_logits.shape)}")
    if beta_lambda < 0.0:
        raise ValueError(f"beta_lambda must be non-negative, got {beta_lambda}")

    # 1. Softmax within each task; tasks with the same option count are softmaxed together as one batch
    task_logits = torch.split(flat_logits, num_options)
    tasks_by_count: dict[int, list[int]] = {}
    for t_idx, n_opt in enumerate(num_options):
        if n_opt not in tasks_by_count:
            tasks_by_count[n_opt] = []
        tasks_by_count[n_opt].append(t_idx)
    log_probs_by_task: dict[int, torch.Tensor] = {}
    for t_indices in tasks_by_count.values():
        stacked_log_probs = torch.log_softmax(torch.stack([task_logits[t] for t in t_indices], dim=0), dim=-1)
        for row_idx, t_idx in enumerate(t_indices):
            log_probs_by_task[t_idx] = stacked_log_probs[row_idx]
    log_probs_flat = torch.cat([log_probs_by_task[t_idx] for t_idx in range(num_tasks)])

    # 2. KL(y || p) per task; for one-hot y this is the cross-entropy -log p(chosen). Mean over tasks.
    kl_loss = -(y_flat * log_probs_flat).sum() / num_tasks

    # 3. Beta loss: per regression, mean squared standardized beta error; then mean over regressions
    def beta_errors(probs_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        group_mse = {}
        for op in sample_ops["operators"]:
            beta_error = op["B"] @ (probs_flat - y_flat)
            group_mse[op["group"]] = beta_error.pow(2).mean()
        return group_mse

    has_beta = len(sample_ops["operators"]) > 0
    if beta_lambda == 0.0 or not has_beta:
        # Beta loss is logged but does not enter the gradient
        with torch.no_grad():
            group_mse = beta_errors(log_probs_flat.detach().exp())
        total_loss = kl_loss
    else:
        group_mse = beta_errors(log_probs_flat.exp())
        total_loss = kl_loss + beta_lambda * torch.stack(list(group_mse.values())).mean()

    if has_beta:
        beta_loss = float(torch.stack(list(group_mse.values())).mean().detach())
    else:
        beta_loss = float("nan")
    return {
        "total_loss": total_loss,
        "kl_loss": float(kl_loss.detach()),
        "beta_loss": beta_loss,
        "group_beta_mse": {group: float(mse.detach()) for group, mse in group_mse.items()},
        "skipped_groups": list(sample_ops["skipped_groups"]),
    }

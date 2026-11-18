"""CE + lambda x population beta loss (`combined_loss`, at the end).

The population beta part below is copied unchanged from `beta_population.loss`; `ce_loss` and `combined_loss` are new.

Population beta loss: frozen base probabilities p0 and a split-sample cross loss (no y, no stale state).

The model's population betas of one dataset are a fixed linear map of every task's option probabilities:
    betas = B_full @ p,   B_full = pinv(X_full)[target coefficients] @ W / sigma
over ALL N tasks of the dataset (the operator `beta_ols.unified_loss` builds per sample, here built once from every
item). Targets are standardized betas (beta / sigma), e.g. the `std_beta` column of `beta_ols.core.fit_human_ols`, the
stand-in for a published table. The respondents' choices y are not used.

A step sees S tasks, but the population betas need all N. With p0 = the untrained model's probabilities of every task
(computed once, eval mode, frozen) and beta0 = B_full @ p0, the betas at the current weights are
    B_full @ p(theta) = beta0 + sum over tasks i of B_full[:, i] (p_i(theta) - p0_i).
A uniform sample S_m of tasks estimates that sum without bias:
    beta_hat_m = beta0 + (N / |S_m|) * sum over i in S_m of B_full[:, i] (p_i(theta) - p0_i).
Every term uses the current weights or the frozen p0, so nothing is stale; p0 only lowers the variance (it cancels in
expectation, so its exact values do not bias anything).

Squaring one estimate would be biased: E||beta_hat - target||^2 = ||B p - target||^2 + trace Cov(beta_hat), and the
covariance grows with how much p varies across tasks (personas). So each sample is split into two disjoint halves
S_1, S_2, and the loss is their cross product:
    cross loss = mean over coefficients of (beta_hat_1 - target) * (beta_hat_2 - target).
The halves are independent apart from sampling without replacement (an O(1/N) covariance), so its expectation is the
population loss mean((B p - target)^2), and its gradient is an unbiased estimate of the population gradient. A single
step's cross loss can be negative.
"""

from __future__ import annotations

import torch

from beta_ols.core import RegressionSpec
from beta_ols.unified_loss import build_sample_beta_operators

from beta_train.batching import split_into_chunks
from beta_train.step import predict_chunk


def task_log_probs(flat_logits: torch.Tensor, num_options: list[int]) -> torch.Tensor:
    # Log-softmax within each task; tasks with the same option count are softmaxed together as one batch.
    # Returns the flat log-probabilities in the same task order.
    if flat_logits.ndim != 1 or flat_logits.shape[0] != sum(num_options):
        raise ValueError(f"Expected flat logits of length {sum(num_options)}, got shape {tuple(flat_logits.shape)}")
    task_logits = torch.split(flat_logits, num_options)
    tasks_by_count: dict[int, list[int]] = {}
    for task_idx, option_count in enumerate(num_options):
        if option_count not in tasks_by_count:
            tasks_by_count[option_count] = []
        tasks_by_count[option_count].append(task_idx)
    log_probs_by_task: dict[int, torch.Tensor] = {}
    for task_indices in tasks_by_count.values():
        stacked = torch.stack([task_logits[task_idx] for task_idx in task_indices], dim=0)
        stacked_log_probs = torch.log_softmax(stacked, dim=-1)
        for row_idx, task_idx in enumerate(task_indices):
            log_probs_by_task[task_idx] = stacked_log_probs[row_idx]
    ordered = []
    for task_idx in range(len(num_options)):
        ordered.append(log_probs_by_task[task_idx])
    return torch.cat(ordered)


def sample_num_options(sample_items: list[dict]) -> list[int]:
    # Option count of every task of the sample, item by item
    num_options = []
    for item in sample_items:
        num_options.extend(item["num_options"])
    return num_options


class FrozenBaseEstimator:
    """One dataset's population operators, target betas, frozen base probabilities p0, and the cross loss."""

    def __init__(
        self,
        dataset_name: str,
        items: list[dict],                              # every item of the dataset, in the order of the flat vector
        specs: dict[str, RegressionSpec],
        target_std_betas: dict[str, torch.Tensor],      # {group: standardized target betas, spec target order}
        device: str,
    ):
        self.dataset_name = dataset_name
        self.num_items = len(items)                     # N: one item = one task (data row)
        if self.num_items == 0:
            raise ValueError(f"{dataset_name}: no items")

        # 1. Where each item's options sit in the flat option vector (items in the given order)
        self.option_slice_by_item_id: dict[int, tuple[int, int]] = {}
        offset = 0
        for item in items:
            if item["dataset"] != dataset_name:
                raise ValueError(f"Item of {item['dataset']!r} in the items of {dataset_name!r}")
            if id(item) in self.option_slice_by_item_id:
                raise ValueError(f"{dataset_name}: the same item object appears twice")
            item_options = sum(item["num_options"])
            self.option_slice_by_item_id[id(item)] = (offset, item_options)
            offset += item_options
        self.total_options = offset

        # 2. Standardized population operators over every item (the full design must be full rank). The operator
        #    builder also stacks the items' y; it is not kept, so it cannot enter the loss.
        full_ops = build_sample_beta_operators(items, specs, device)
        if full_ops["skipped_groups"]:
            raise ValueError(f"{dataset_name}: full-data design is rank-deficient for {full_ops['skipped_groups']}")
        if full_ops["y_flat"].shape[0] != self.total_options:
            raise AssertionError(f"{dataset_name}: operator width {full_ops['y_flat'].shape[0]} != {self.total_options} options")
        self.operators = full_ops["operators"]

        # 3. Targets, one vector per regression, in the spec's target order
        operator_groups = [op["group"] for op in self.operators]
        if set(target_std_betas) != set(operator_groups):
            raise ValueError(f"{dataset_name}: targets for {sorted(target_std_betas)}, regressions are {sorted(operator_groups)}")
        self.targets: dict[str, torch.Tensor] = {}
        for op in self.operators:
            target = target_std_betas[op["group"]].to(device=device, dtype=torch.float64)
            if target.ndim != 1 or target.shape[0] != op["B"].shape[0]:
                raise ValueError(
                    f"{dataset_name}/{op['group']}: target has shape {tuple(target.shape)}, expected ({op['B'].shape[0]},)"
                )
            self.targets[op["group"]] = target

        self.device = device
        self.p0: torch.Tensor | None = None            # frozen base probabilities of every option, set by compute_p0 / load_p0
        self.beta0: dict[str, torch.Tensor] = {}       # {group: B_full @ p0}

    def compute_p0(self, model, items: list[dict], chunk_size: int, chunk_bar) -> None:
        # No-grad forward of every item in eval mode (no dropout), through the training forward (`predict_chunk`);
        # the model's mode is restored afterwards
        if len(items) != self.num_items:
            raise ValueError(f"{self.dataset_name}: compute_p0 got {len(items)} items, the estimator has {self.num_items}")
        positions = self.positions(items)
        expected_positions = torch.arange(self.total_options, device=self.device)
        if not torch.equal(positions, expected_positions):
            raise AssertionError(f"{self.dataset_name}: p0 items are not in the estimator's order")
        chunks = split_into_chunks(items, chunk_size)
        chunk_bar.reset(total=len(chunks))
        chunk_bar.set_description(f"p0 (eval mode): {self.dataset_name}")
        was_training = model.training
        model.eval()
        prob_parts = []
        try:
            with torch.no_grad():
                for chunk in chunks:
                    chunk_logits = predict_chunk(model, chunk)
                    prob_parts.append(task_log_probs(chunk_logits, sample_num_options(chunk)).exp())
                    torch.cuda.synchronize()
                    chunk_bar.update(1)
        finally:
            if was_training:
                model.train()
        if model.training != was_training:
            raise AssertionError(f"Expected model.training={was_training} after computing p0, got {model.training}")
        self.load_p0(torch.cat(prob_parts))

    def load_p0(self, p0: torch.Tensor) -> None:
        # Set the frozen base probabilities (computed here or from a saved run state) and beta0 = B_full @ p0
        if p0.ndim != 1 or p0.shape[0] != self.total_options:
            raise ValueError(f"{self.dataset_name}: p0 has shape {tuple(p0.shape)}, expected ({self.total_options},)")
        self.p0 = p0.detach().to(device=self.device, dtype=torch.float64)
        self.beta0 = {}
        for op in self.operators:
            self.beta0[op["group"]] = op["B"] @ self.p0

    def p0_state(self) -> torch.Tensor:
        # CPU copy of p0, for the resume state
        if self.p0 is None:
            raise RuntimeError(f"{self.dataset_name}: p0 is not set")
        return self.p0.cpu().clone()

    def positions(self, sample_items: list[dict]) -> torch.Tensor:
        # Flat-vector positions of the sample's options, in sample order (the order of the sample's logits)
        positions = []
        for item in sample_items:
            start, count = self.option_slice_by_item_id[id(item)]
            positions.extend(range(start, start + count))
        return torch.tensor(positions, dtype=torch.long, device=self.device)

    def estimate_betas(self, probs: torch.Tensor, items: list[dict]) -> dict[str, torch.Tensor]:
        # {group: beta0 + (N / |items|) * B_full[:, items] @ (p - p0)}, unbiased for B_full @ p(theta) over a uniform
        # sample of tasks; gradients reach `probs`
        if self.p0 is None:
            raise RuntimeError(f"{self.dataset_name}: p0 is not set")
        positions = self.positions(items)
        if probs.ndim != 1 or probs.shape[0] != positions.shape[0]:
            raise ValueError(f"Expected {positions.shape[0]} probabilities, got shape {tuple(probs.shape)}")
        if torch.unique(positions).shape[0] != positions.shape[0]:
            raise ValueError(f"{self.dataset_name}: the same task appears twice")
        delta = probs - self.p0[positions]
        scale = self.num_items / len(items)
        estimates = {}
        for op in self.operators:
            estimates[op["group"]] = self.beta0[op["group"]] + scale * (op["B"][:, positions] @ delta)
        return estimates

    def cross_loss(self, sample_probs: torch.Tensor, sample_items: list[dict]) -> tuple[torch.Tensor, dict]:
        # Split the sample into two disjoint halves (first / second half of the already shuffled sample), estimate the
        # betas on each, and return (cross loss, logs). Logs: the cross loss per regression, and the squared error of
        # the full-sample estimate (readable, but biased upward by the estimate's variance; never in the gradient).
        if len(sample_items) < 2:
            raise ValueError(f"{self.dataset_name}: the cross loss needs at least 2 tasks, got {len(sample_items)}")
        half_size = len(sample_items) // 2
        items_1 = sample_items[:half_size]
        items_2 = sample_items[half_size:]
        options_1 = sum(sample_num_options(items_1))
        if sample_probs.ndim != 1 or sample_probs.shape[0] != sum(sample_num_options(sample_items)):
            raise ValueError(f"Expected {sum(sample_num_options(sample_items))} sample probabilities, got shape {tuple(sample_probs.shape)}")
        probs_1 = sample_probs[:options_1]
        probs_2 = sample_probs[options_1:]
        estimates_1 = self.estimate_betas(probs_1, items_1)
        estimates_2 = self.estimate_betas(probs_2, items_2)
        group_losses = []
        group_cross = {}
        for op in self.operators:
            group = op["group"]
            error_1 = estimates_1[group] - self.targets[group]
            error_2 = estimates_2[group] - self.targets[group]
            group_loss = (error_1 * error_2).mean()
            group_losses.append(group_loss)
            group_cross[group] = float(group_loss.detach())
        loss = torch.stack(group_losses).mean()

        with torch.no_grad():
            full_estimates = self.estimate_betas(sample_probs.detach(), sample_items)
            full_mses = []
            for op in self.operators:
                group = op["group"]
                full_mses.append((full_estimates[group] - self.targets[group]).pow(2).mean())
            sample_beta_mse = float(torch.stack(full_mses).mean())
        logs = {
            "group_cross_loss": group_cross,
            "sample_beta_mse": sample_beta_mse,
            "half_sizes": (len(items_1), len(items_2)),
        }
        return loss, logs


def ce_loss(log_probs: torch.Tensor, sample_items: list[dict]) -> torch.Tensor:
    # Mean over tasks of the cross-entropy against the respondents' one-hot choices: -sum(y * log p) / tasks.
    # The same KL term as `beta_ols.unified_loss.compute_unified_loss`.
    y_flat = torch.cat([item["y_flat"] for item in sample_items]).to(device=log_probs.device, dtype=log_probs.dtype)
    if log_probs.ndim != 1 or log_probs.shape[0] != y_flat.shape[0]:
        raise ValueError(f"Expected {y_flat.shape[0]} log-probabilities, got shape {tuple(log_probs.shape)}")
    num_tasks = len(sample_num_options(sample_items))
    return -(y_flat * log_probs).sum() / num_tasks


def combined_loss(
    flat_logits: torch.Tensor, sample_items: list[dict], estimator: FrozenBaseEstimator, beta_lambda: float
) -> tuple[torch.Tensor, dict]:
    # total = CE + beta_lambda * cross loss, on one sample's flat logits (task by task, in sample order).
    # Returns (total loss with its graph, logs). At beta_lambda = 0 the cross loss is logged and its gradient is zero.
    if beta_lambda < 0.0:
        raise ValueError(f"beta_lambda must be non-negative, got {beta_lambda}")
    log_probs = task_log_probs(flat_logits, sample_num_options(sample_items))
    ce = ce_loss(log_probs, sample_items)
    cross, cross_logs = estimator.cross_loss(log_probs.exp(), sample_items)
    total = ce + beta_lambda * cross
    logs = {
        "ce_loss": float(ce.detach()),
        "cross_loss": float(cross.detach()),
        "weighted_cross_loss": beta_lambda * float(cross.detach()),
        "total_loss": float(total.detach()),
        "group_cross_loss": cross_logs["group_cross_loss"],
        "sample_beta_mse": cross_logs["sample_beta_mse"],
        "half_sizes": cross_logs["half_sizes"],
    }
    return total, logs

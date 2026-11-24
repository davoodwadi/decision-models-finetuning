"""Losses for microdata studies (CE) and table studies (split population-beta cross loss + KL-to-p0 anchor).

For a table dataset (`beta_datasets`), respondent choices y are never used in the differentiable objective. Its targets
are standardized OLS slopes fitted on the study's choices, standing in for a published table:
  1. `FrozenBaseEstimator` precomputes the untrained model's eval-mode option probabilities p0 across all N tasks once
     and sets beta0 = B_full @ p0, where B_full = pinv(X_full)[target_indices] / sigma.
  2. Each step draws S tasks, splits them into two disjoint halves S_1 and S_2, forms the control-variate estimates
         beta_hat_m = beta0 + (N / |S_m|) * B_full[:, S_m] @ (p_{S_m}(theta) - p0_{S_m}),
     and computes their unbiased cross product:
         cross_loss = mean_g mean_k (beta_hat_{1,g,k} - target_{g,k}) * (beta_hat_{2,g,k} - target_{g,k}).
  3. Because B_full constrains only K aggregate linear projections out of N tasks, an optional forward KL anchor to the
     frozen base model
         kl_from_p0 = (1 / T) * sum_{t, m} p0_{t, m} * (log p0_{t, m} - log p_{t, m}(theta))
     regularizes the (N - K)-dimensional nullspace ker(B_full) against logit-sharpness drift with logit gradient
     (p(theta) - p0) / T and zero extra forward passes (reusing the cached p0).
"""

from __future__ import annotations

import torch

from beta_ols.core import RegressionSpec
from beta_ols.unified_loss import build_sample_beta_operators

from beta_train.batching import split_into_chunks
from beta_train.step import predict_chunk


def forward_chunk_logits(model, chunk_items: list[dict], device: str) -> torch.Tensor:
    """Shared-prefix forward for one micro-chunk -> flat float64 option logits in item/question order."""
    if device == "cuda":
        return predict_chunk(model, chunk_items)
    if device == "cpu":
        logits_per_item = model.forward([item["enc"] for item in chunk_items], shared_prefix=True)
        flat_parts = []
        for item, item_logits in zip(chunk_items, logits_per_item, strict=True):
            option_counts = [int(question_logits.shape[0]) for question_logits in item_logits]
            if option_counts != item["num_options"]:
                raise ValueError(f"{item['caseid']}: logits option counts {option_counts} != expected {item['num_options']}")
            for question_logits in item_logits:
                flat_parts.append(question_logits.to(dtype=torch.float64))
        return torch.cat(flat_parts, dim=0)
    raise ValueError(f"Unsupported device {device!r}; expected 'cuda' or 'cpu'")


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device != "cpu":
        raise ValueError(f"Unsupported device {device!r}; expected 'cuda' or 'cpu'")


def task_log_probs(flat_logits: torch.Tensor, num_options: list[int]) -> torch.Tensor:
    """Log-softmax within each task; tasks sharing an option count are batched together."""
    if len(num_options) == 0:
        raise ValueError("num_options must be non-empty")
    expected_len = sum(num_options)
    if flat_logits.ndim != 1 or flat_logits.shape[0] != expected_len:
        raise ValueError(f"Expected flat logits of length {expected_len}, got shape {tuple(flat_logits.shape)}")
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
    return torch.cat([log_probs_by_task[task_idx] for task_idx in range(len(num_options))])


def sample_num_options(sample_items: list[dict]) -> list[int]:
    num_options = []
    for item in sample_items:
        num_options.extend(item["num_options"])
    return num_options


class FrozenBaseEstimator:
    """Full-dataset OLS operators, standardized target betas, frozen base p0, split cross loss, and KL-to-p0 anchor."""

    def __init__(
        self,
        dataset_name: str,
        items: list[dict],
        specs: dict[str, RegressionSpec],
        target_std_betas: dict[str, torch.Tensor],
        device: str,
    ):
        if device not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported device {device!r}; expected 'cuda' or 'cpu'")
        self.dataset_name = dataset_name
        self.num_items = len(items)
        if self.num_items < 2:
            raise ValueError(f"{dataset_name}: need at least 2 items for split cross loss, got {self.num_items}")

        # 1. Option slice of each item in the flat dataset option vector
        self.option_slice_by_item_id: dict[int, tuple[int, int]] = {}
        offset = 0
        for item in items:
            if item["dataset"] != dataset_name:
                raise ValueError(f"Item of {item['dataset']!r} passed to estimator for {dataset_name!r}")
            if id(item) in self.option_slice_by_item_id:
                raise ValueError(f"{dataset_name}: duplicate item object in items list")
            item_options = sum(item["num_options"])
            self.option_slice_by_item_id[id(item)] = (offset, item_options)
            offset += item_options
        self.total_options = offset

        # 2. Full-dataset linear OLS operators; y_flat returned by builder is discarded immediately and never stored
        full_ops = build_sample_beta_operators(items, specs, device)
        if full_ops["skipped_groups"]:
            raise ValueError(f"{dataset_name}: full-dataset design is rank-deficient for {full_ops['skipped_groups']}")
        if full_ops["y_flat"].shape[0] != self.total_options:
            raise AssertionError(
                f"{dataset_name}: operator width {full_ops['y_flat'].shape[0]} != {self.total_options} options"
            )
        self.operators = full_ops["operators"]

        # 3. Table target standardized betas (fitted OLS beta / sigma), one vector per regression group
        operator_groups = [op["group"] for op in self.operators]
        if set(target_std_betas) != set(operator_groups):
            raise ValueError(
                f"{dataset_name}: target_std_betas has groups {sorted(target_std_betas)}, "
                f"expected {sorted(operator_groups)}"
            )
        self.targets: dict[str, torch.Tensor] = {}
        for op in self.operators:
            target = target_std_betas[op["group"]].to(device=device, dtype=torch.float64)
            if target.ndim != 1 or target.shape[0] != op["B"].shape[0]:
                raise ValueError(
                    f"{dataset_name}/{op['group']}: target has shape {tuple(target.shape)}, expected ({op['B'].shape[0]},)"
                )
            self.targets[op["group"]] = target

        self.device = device
        self.p0: torch.Tensor | None = None
        self.beta0: dict[str, torch.Tensor] = {}

    def compute_p0(self, model, items: list[dict], chunk_size: int, chunk_bar) -> None:
        if len(items) != self.num_items:
            raise ValueError(f"{self.dataset_name}: compute_p0 got {len(items)} items, expected {self.num_items}")
        positions = self.positions(items)
        expected_positions = torch.arange(self.total_options, device=self.device)
        if not torch.equal(positions, expected_positions):
            raise AssertionError(f"{self.dataset_name}: items passed to compute_p0 are out of order")
        chunks = split_into_chunks(items, chunk_size)
        chunk_bar.reset(total=len(chunks))
        chunk_bar.set_description(f"p0 (eval mode): {self.dataset_name}")
        was_training = model.training
        model.eval()
        prob_parts = []
        try:
            with torch.no_grad():
                for chunk in chunks:
                    chunk_logits = forward_chunk_logits(model, chunk, self.device)
                    prob_parts.append(task_log_probs(chunk_logits, sample_num_options(chunk)).exp())
                    sync_device(self.device)
                    chunk_bar.update(1)
        finally:
            if was_training:
                model.train()
        if model.training != was_training:
            raise AssertionError(f"Expected model.training={was_training} after compute_p0, got {model.training}")
        self.load_p0(torch.cat(prob_parts))

    def load_p0(self, p0: torch.Tensor) -> None:
        if p0.ndim != 1 or p0.shape[0] != self.total_options:
            raise ValueError(f"{self.dataset_name}: p0 has shape {tuple(p0.shape)}, expected ({self.total_options},)")
        if not bool(torch.isfinite(p0).all()):
            raise FloatingPointError(f"{self.dataset_name}: p0 contains non-finite values")
        if bool((p0 < 0.0).any()):
            raise ValueError(f"{self.dataset_name}: p0 contains negative probabilities")
        self.p0 = p0.detach().to(device=self.device, dtype=torch.float64)
        self.beta0 = {}
        for op in self.operators:
            self.beta0[op["group"]] = op["B"] @ self.p0

    def p0_state(self) -> torch.Tensor:
        if self.p0 is None:
            raise RuntimeError(f"{self.dataset_name}: p0 has not been initialized")
        return self.p0.cpu().clone()

    def positions(self, sample_items: list[dict]) -> torch.Tensor:
        positions = []
        for item in sample_items:
            start, count = self.option_slice_by_item_id[id(item)]
            positions.extend(range(start, start + count))
        return torch.tensor(positions, dtype=torch.long, device=self.device)

    def estimate_betas(self, probs: torch.Tensor, items: list[dict]) -> dict[str, torch.Tensor]:
        if self.p0 is None:
            raise RuntimeError(f"{self.dataset_name}: p0 has not been initialized")
        positions = self.positions(items)
        if probs.ndim != 1 or probs.shape[0] != positions.shape[0]:
            raise ValueError(f"Expected {positions.shape[0]} probabilities, got shape {tuple(probs.shape)}")
        if torch.unique(positions).shape[0] != positions.shape[0]:
            raise ValueError(f"{self.dataset_name}: duplicate item in sample")
        delta = probs - self.p0[positions]
        scale = self.num_items / len(items)
        estimates = {}
        for op in self.operators:
            estimates[op["group"]] = self.beta0[op["group"]] + scale * (op["B"][:, positions] @ delta)
        return estimates

    def cross_loss(self, sample_probs: torch.Tensor, sample_items: list[dict]) -> tuple[torch.Tensor, dict]:
        if len(sample_items) < 2:
            raise ValueError(f"{self.dataset_name}: split cross loss needs >= 2 tasks, got {len(sample_items)}")
        half_size = len(sample_items) // 2
        items_1 = sample_items[:half_size]
        items_2 = sample_items[half_size:]
        options_1 = sum(sample_num_options(items_1))
        total_sample_options = sum(sample_num_options(sample_items))
        if sample_probs.ndim != 1 or sample_probs.shape[0] != total_sample_options:
            raise ValueError(f"Expected {total_sample_options} sample probabilities, got shape {tuple(sample_probs.shape)}")
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
            full_mses = [(full_estimates[op["group"]] - self.targets[op["group"]]).pow(2).mean() for op in self.operators]
            sample_beta_mse = float(torch.stack(full_mses).mean())
        logs = {
            "group_cross_loss": group_cross,
            "sample_beta_mse": sample_beta_mse,
            "half_sizes": (len(items_1), len(items_2)),
        }
        return loss, logs

    def kl_from_p0(self, sample_log_probs: torch.Tensor, sample_items: list[dict]) -> torch.Tensor:
        """Mean over tasks of forward KL(p0 || p_theta) = sum_m p0_m * (log p0_m - log p_m).

        Strictly convex in logits with d/dz = (p_theta - p0) / num_tasks; uses zero respondent choices y.
        """
        if self.p0 is None:
            raise RuntimeError(f"{self.dataset_name}: p0 has not been initialized")
        num_options = sample_num_options(sample_items)
        positions = self.positions(sample_items)
        if sample_log_probs.ndim != 1 or sample_log_probs.shape[0] != positions.shape[0]:
            raise ValueError(f"Expected {positions.shape[0]} log-probabilities, got shape {tuple(sample_log_probs.shape)}")
        p0_sample = self.p0[positions]
        num_tasks = len(num_options)
        return (torch.special.xlogy(p0_sample, p0_sample) - p0_sample * sample_log_probs).sum() / num_tasks


def ce_loss(log_probs: torch.Tensor, sample_items: list[dict]) -> torch.Tensor:
    """Mean per-task cross-entropy against observed one-hot respondent choices y."""
    y_flat = torch.cat([item["y_flat"] for item in sample_items]).to(device=log_probs.device, dtype=log_probs.dtype)
    if log_probs.ndim != 1 or log_probs.shape[0] != y_flat.shape[0]:
        raise ValueError(f"Expected {y_flat.shape[0]} log-probabilities, got shape {tuple(log_probs.shape)}")
    num_tasks = len(sample_num_options(sample_items))
    return -(y_flat * log_probs).sum() / num_tasks


def ce_sample_loss(flat_logits: torch.Tensor, sample_items: list[dict]) -> tuple[torch.Tensor, dict]:
    """Loss for a microdata study (`ce_datasets`): pure respondent cross-entropy."""
    log_probs = task_log_probs(flat_logits, sample_num_options(sample_items))
    ce = ce_loss(log_probs, sample_items)
    ce_val = float(ce.detach())
    nan = float("nan")
    logs = {
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
    return ce, logs


def beta_sample_loss(
    flat_logits: torch.Tensor,
    sample_items: list[dict],
    estimator: FrozenBaseEstimator,
    beta_lambda: float,
    kl_anchor_alpha: float,
) -> tuple[torch.Tensor, dict]:
    """Loss for a table study (`beta_datasets`): beta_lambda * split_cross_loss + kl_anchor_alpha * KL(p0 || p).

    Respondent choices y never enter `total` or its computation graph; `monitor_human_kl` is computed detached for
    diagnostic logging only.
    """
    if beta_lambda <= 0.0:
        raise ValueError(f"beta_lambda must be > 0 for beta_sample_loss, got {beta_lambda}")
    if kl_anchor_alpha < 0.0:
        raise ValueError(f"kl_anchor_alpha must be >= 0, got {kl_anchor_alpha}")
    log_probs = task_log_probs(flat_logits, sample_num_options(sample_items))
    cross, cross_logs = estimator.cross_loss(log_probs.exp(), sample_items)
    kl_anchor = estimator.kl_from_p0(log_probs, sample_items)
    total = beta_lambda * cross + kl_anchor_alpha * kl_anchor
    with torch.no_grad():
        monitor_human_kl = float(ce_loss(log_probs.detach(), sample_items))
    cross_val = float(cross.detach())
    kl_anchor_val = float(kl_anchor.detach())
    logs = {
        "mode": "pop_beta",
        "ce_loss": float("nan"),
        "cross_loss": cross_val,
        "weighted_cross_loss": beta_lambda * cross_val,
        "kl_anchor_loss": kl_anchor_val,
        "weighted_kl_anchor_loss": kl_anchor_alpha * kl_anchor_val,
        "total_loss": float(total.detach()),
        "sample_beta_mse": cross_logs["sample_beta_mse"],
        "monitor_human_kl": monitor_human_kl,
        "half_sizes": cross_logs["half_sizes"],
        "group_cross_loss": cross_logs["group_cross_loss"],
    }
    return total, logs

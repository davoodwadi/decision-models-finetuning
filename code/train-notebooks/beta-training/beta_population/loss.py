"""Population beta loss with a probability cache.

The model's population betas of one dataset are a fixed linear map of every task's option probabilities:
    betas = B_full @ p,   B_full = pinv(X_full)[target coefficients] @ W / sigma
over ALL tasks of the dataset (the same operator `beta_ols.unified_loss` builds per sample, here built once from every
item). The loss is, per regression, the mean squared error between these betas and the target betas, then the mean
over regressions. Targets are standardized betas (beta / sigma), e.g. the `std_beta` column of
`beta_ols.core.fit_human_ols`, the stand-in for a published table. The respondents' choices y are not used.

Why a cache: a step sees S tasks, but the loss needs all N. Comparing the S-task sample betas with the population
target instead would be biased: its expected value adds the variance of the S-task beta estimate, which grows with
how much p varies across personas, so it would push the model toward ignoring the persona. Instead:
- p_cache holds one probability per option of every task, filled at step 0 (no-grad forward of every item);
- a step recomputes p for its S tasks, puts them into a copy of p_cache, and computes the loss from that full vector;
  only the S fresh entries carry gradients;
- the S fresh probabilities are then written into p_cache.
d loss / d p_i is exact for the sampled tasks (at the cached values of the others). Scaling it by N / S makes the
weight gradient an unbiased estimate of the full population gradient (uniform sampling of tasks), apart from the
staleness of the cached entries.
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


class ProbabilityCache:
    """One dataset's population operators, target betas, and cached option probabilities of every task."""

    def __init__(
        self,
        dataset_name: str,
        items: list[dict],                              # every item of the dataset, in the order of the flat vector
        specs: dict[str, RegressionSpec],
        target_std_betas: dict[str, torch.Tensor],      # {group: standardized target betas, spec target order}
        device: str,
    ):
        self.dataset_name = dataset_name
        self.num_items = len(items)
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
        self.probs: torch.Tensor | None = None   # flat float64 probabilities of every option, set by fill / load

    def fill(self, model, items: list[dict], chunk_size: int, chunk_bar) -> None:
        # No-grad forward of every item in the model's current mode (training: dropout on, like the fresh entries of
        # every step), in the order the cache was built with
        if len(items) != self.num_items:
            raise ValueError(f"{self.dataset_name}: fill got {len(items)} items, the cache has {self.num_items}")
        chunks = split_into_chunks(items, chunk_size)
        chunk_bar.reset(total=len(chunks))
        chunk_bar.set_description(f"cache fill: {self.dataset_name}")
        prob_parts = []
        with torch.no_grad():
            for chunk in chunks:
                chunk_logits = predict_chunk(model, chunk)
                prob_parts.append(task_log_probs(chunk_logits, sample_num_options(chunk)).exp())
                torch.cuda.synchronize()
                chunk_bar.update(1)
        probs = torch.cat(prob_parts).to(device=self.device, dtype=torch.float64)
        if probs.shape[0] != self.total_options:
            raise AssertionError(f"{self.dataset_name}: filled {probs.shape[0]} options, expected {self.total_options}")
        positions = self.positions(items)
        expected_positions = torch.arange(self.total_options, device=self.device)
        if not torch.equal(positions, expected_positions):
            raise AssertionError(f"{self.dataset_name}: fill items are not in the cache's order")
        self.probs = probs

    def load(self, probs: torch.Tensor) -> None:
        # Cached probabilities from a saved run state (resume)
        if probs.ndim != 1 or probs.shape[0] != self.total_options:
            raise ValueError(f"{self.dataset_name}: saved cache has shape {tuple(probs.shape)}, expected ({self.total_options},)")
        self.probs = probs.to(device=self.device, dtype=torch.float64)

    def state(self) -> torch.Tensor:
        # CPU copy of the cached probabilities, for the resume state
        if self.probs is None:
            raise RuntimeError(f"{self.dataset_name}: cache is not filled")
        return self.probs.detach().cpu().clone()

    def positions(self, sample_items: list[dict]) -> torch.Tensor:
        # Flat-vector positions of the sample's options, in sample order (the order of the sample's logits)
        positions = []
        for item in sample_items:
            start, count = self.option_slice_by_item_id[id(item)]
            positions.extend(range(start, start + count))
        return torch.tensor(positions, dtype=torch.long, device=self.device)

    def population_loss(self, sample_probs: torch.Tensor, sample_items: list[dict]) -> tuple[torch.Tensor, dict[str, float]]:
        # Loss of the full probability vector = cache with the sample's fresh probabilities in place; gradients reach
        # `sample_probs` only. Returns (loss, {group: that regression's mean squared standardized beta error}).
        if self.probs is None:
            raise RuntimeError(f"{self.dataset_name}: cache is not filled")
        positions = self.positions(sample_items)
        if sample_probs.ndim != 1 or sample_probs.shape[0] != positions.shape[0]:
            raise ValueError(f"Expected {positions.shape[0]} sample probabilities, got shape {tuple(sample_probs.shape)}")
        if torch.unique(positions).shape[0] != positions.shape[0]:
            raise ValueError(f"{self.dataset_name}: a sample holds the same task twice")
        full_probs = self.probs.detach().index_put((positions,), sample_probs)
        group_losses = []
        group_mse = {}
        for op in self.operators:
            beta_error = op["B"] @ full_probs - self.targets[op["group"]]
            mse = beta_error.pow(2).mean()
            group_losses.append(mse)
            group_mse[op["group"]] = float(mse.detach())
        loss = torch.stack(group_losses).mean()
        return loss, group_mse

    def update(self, sample_items: list[dict], sample_probs: torch.Tensor) -> None:
        # Write the sample's fresh probabilities into the cache
        if self.probs is None:
            raise RuntimeError(f"{self.dataset_name}: cache is not filled")
        positions = self.positions(sample_items)
        self.probs[positions] = sample_probs.detach().to(dtype=torch.float64)

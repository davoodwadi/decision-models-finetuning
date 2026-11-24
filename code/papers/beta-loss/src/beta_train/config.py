"""Run settings for `beta_train.trainer.run_training`, declared in the notebook and recorded in every result."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

LR_SCHEDULES = ("onecycle", "constant")


@dataclass(frozen=True)
class TrainConfig:
    """Every setting that can vary between runs. Invalid values raise on creation."""

    train_datasets: tuple[str, ...]          # dataset names (keys of the items dict); each step uses one of them
    sample_size_by_dataset: dict[str, int]   # items (data rows = answered tasks) per sample, per training dataset
    beta_lambda: float                       # weight of the beta loss (0 = KL only; beta loss still logged)
    lr: float                                # peak learning rate (OneCycle max, or the constant LR)
    num_epochs: int                          # full passes over every item (data row) of every training dataset
    eval_every_by_validator: dict[str, int]  # {validator: steps between its validations}; all run at step 0 and the last step
    lr_schedule: str                         # "onecycle" or "constant"
    weight_decay: float                      # AdamW weight decay
    micro_chunk: int                         # items (data rows) per forward/backward chunk (VRAM), does not change gradients
    gradient_checkpointing: bool             # recompute activations in backward (less VRAM, slower)
    grad_clip: float                         # max global gradient norm
    seed: int                                # sample order and dropout RNG seed
    # Debugging only: split every step's gradient into its CE and beta parts and log their norms (beta_train.step).
    # Costs a second pass 2 per step. Off unless a notebook turns it on; runs made before it existed had it off.
    gradient_split: bool = False

    def __post_init__(self):
        if not isinstance(self.train_datasets, tuple) or len(self.train_datasets) == 0:
            raise ValueError(f"train_datasets must be a non-empty tuple, got {self.train_datasets!r}")
        if len(set(self.train_datasets)) != len(self.train_datasets):
            raise ValueError(f"train_datasets has duplicates: {self.train_datasets!r}")
        for dataset_name in self.train_datasets:
            sample_size = self.sample_size_by_dataset[dataset_name]
            if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
                raise ValueError(f"sample size for {dataset_name} must be a positive integer, got {sample_size!r}")
        for name in ("num_epochs", "micro_chunk", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.num_epochs < 1 or self.micro_chunk < 1:
            raise ValueError("num_epochs and micro_chunk must be positive")
        if not isinstance(self.eval_every_by_validator, dict) or len(self.eval_every_by_validator) == 0:
            raise ValueError(f"eval_every_by_validator must be a non-empty dict, got {self.eval_every_by_validator!r}")
        for validator_name, eval_every in self.eval_every_by_validator.items():
            if not isinstance(eval_every, int) or isinstance(eval_every, bool) or eval_every < 1:
                raise ValueError(f"eval_every for {validator_name} must be a positive integer, got {eval_every!r}")
        if self.beta_lambda < 0.0:
            raise ValueError(f"beta_lambda must be non-negative, got {self.beta_lambda}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError(f"gradient_checkpointing must be a bool, got {self.gradient_checkpointing!r}")
        if self.grad_clip <= 0.0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")
        if not isinstance(self.gradient_split, bool):
            raise ValueError(f"gradient_split must be a bool, got {self.gradient_split!r}")

    def label(self) -> str:
        # Short run name used in progress bars and tables
        return f"{'+'.join(self.train_datasets)} | lambda={self.beta_lambda:g}"

    def as_dict(self) -> dict:
        # Plain dict of every setting (sample sizes restricted to the training datasets), for recording in results
        settings = asdict(self)
        settings["train_datasets"] = list(self.train_datasets)
        settings["sample_size_by_dataset"] = {name: self.sample_size_by_dataset[name] for name in self.train_datasets}
        settings["eval_every_by_validator"] = dict(self.eval_every_by_validator)
        # Recorded only when on, so runs without the split keep the identity they were saved with (and still resume)
        if not self.gradient_split:
            del settings["gradient_split"]
        return settings


def scale_lr_for_sample_size(base_lr: float, sample_size: int) -> float:
    # Square-root LR scaling (Adam heuristic): the gradient noise per step falls as 1/sqrt(S), so the LR rises as
    # sqrt(S). It keeps the noise-driven part of one epoch's parameter travel comparable across sample sizes; the
    # gradient-driven part (about lr per Adam step x steps per epoch) still shrinks as 1/sqrt(S).
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
        raise ValueError(f"sample_size must be a positive integer, got {sample_size!r}")
    return base_lr * math.sqrt(sample_size)

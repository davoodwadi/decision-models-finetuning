"""Configuration dataclass for `paper_beta.trainer.run_paper_training`."""

from __future__ import annotations

import math
from dataclasses import dataclass

from beta_train.config import LR_SCHEDULES


@dataclass(frozen=True)
class PaperTrainConfig:
    """All training hyperparameters for a paper regime run. Invalid combinations raise immediately.

    `ce_datasets` and `beta_datasets` partition the training corpus into disjoint subsets:
      - `ce_datasets`: respondent microdata choices y are observed; supervised with cross-entropy.
      - `beta_datasets`: only standardized population betas are observed (fitted OLS on the study's choices, standing in for a
        published table; y itself is withheld); supervised with
        `beta_lambda * split_cross_loss + kl_anchor_alpha * KL(p0 || p_theta)`.
    `train_datasets` exposes `ce_datasets + beta_datasets` so `beta_train.batching` and `beta_train.runs` work unchanged.
    """

    ce_datasets: tuple[str, ...]             # microdata datasets supervised via CE(y, p_theta); may be ()
    beta_datasets: tuple[str, ...]           # table datasets supervised via split cross-loss (no y); may be ()
    sample_size_by_dataset: dict[str, int]   # answered tasks (items) per step, for every dataset in train_datasets
    beta_lambda: float                       # weight on split population-beta cross loss (>0 iff beta_datasets non-empty)
    kl_anchor_alpha: float                   # weight on KL(p0 || p_theta) on beta_datasets (0 when beta_datasets empty)
    lr: float                                # peak learning rate (OneCycle max, or constant LR)
    num_epochs: int                          # full passes over every item of every training dataset
    eval_every_by_validator: dict[str, int]  # {validator_name: steps between validations}; all run at step 0 and end
    lr_schedule: str                         # "onecycle" or "constant"
    weight_decay: float                      # AdamW weight decay
    micro_chunk: int                         # items per forward/backward micro-chunk (VRAM only)
    p0_chunk: int                            # items per eval-mode no-grad forward when computing p0 on beta_datasets
    gradient_checkpointing: bool             # activation checkpointing in backbone (VRAM only)
    grad_clip: float                         # max global gradient norm
    seed: int                                # batch-order and dropout RNG seed

    def __post_init__(self) -> None:
        if not isinstance(self.ce_datasets, tuple):
            raise TypeError(f"ce_datasets must be a tuple of dataset names, got {self.ce_datasets!r}")
        if not isinstance(self.beta_datasets, tuple):
            raise TypeError(f"beta_datasets must be a tuple of dataset names, got {self.beta_datasets!r}")
        for dataset_name in (*self.ce_datasets, *self.beta_datasets):
            if not isinstance(dataset_name, str) or len(dataset_name) == 0:
                raise ValueError(f"Dataset names must be non-empty strings, got {dataset_name!r}")
        if len(set(self.ce_datasets)) != len(self.ce_datasets):
            raise ValueError(f"ce_datasets contains duplicates: {self.ce_datasets!r}")
        if len(set(self.beta_datasets)) != len(self.beta_datasets):
            raise ValueError(f"beta_datasets contains duplicates: {self.beta_datasets!r}")
        overlap = set(self.ce_datasets) & set(self.beta_datasets)
        if overlap:
            raise ValueError(
                f"ce_datasets and beta_datasets must be disjoint (a study either has microdata y or table-only betas); "
                f"overlap: {sorted(overlap)}"
            )
        if len(self.train_datasets) == 0:
            raise ValueError("At least one dataset must be provided in ce_datasets or beta_datasets")

        if not isinstance(self.sample_size_by_dataset, dict):
            raise TypeError(f"sample_size_by_dataset must be a dict, got {self.sample_size_by_dataset!r}")
        for dataset_name in self.ce_datasets:
            if dataset_name not in self.sample_size_by_dataset:
                raise KeyError(f"Missing sample size for ce_dataset {dataset_name!r}")
            sample_size = self.sample_size_by_dataset[dataset_name]
            if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
                raise ValueError(f"sample size for ce_dataset {dataset_name} must be an int >= 1, got {sample_size!r}")
        for dataset_name in self.beta_datasets:
            if dataset_name not in self.sample_size_by_dataset:
                raise KeyError(f"Missing sample size for beta_dataset {dataset_name!r}")
            sample_size = self.sample_size_by_dataset[dataset_name]
            if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 2:
                raise ValueError(
                    f"sample size for beta_dataset {dataset_name} must be an int >= 2 (needed for split halves), "
                    f"got {sample_size!r}"
                )

        for field_name, value in (
            ("num_epochs", self.num_epochs),
            ("micro_chunk", self.micro_chunk),
            ("p0_chunk", self.p0_chunk),
            ("seed", self.seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be an int, got {value!r}")
        if self.num_epochs < 1 or self.micro_chunk < 1 or self.p0_chunk < 1:
            raise ValueError("num_epochs, micro_chunk, and p0_chunk must all be >= 1")

        if not isinstance(self.eval_every_by_validator, dict) or len(self.eval_every_by_validator) == 0:
            raise ValueError(f"eval_every_by_validator must be a non-empty dict, got {self.eval_every_by_validator!r}")
        for validator_name, eval_every in self.eval_every_by_validator.items():
            if not isinstance(eval_every, int) or isinstance(eval_every, bool) or eval_every < 1:
                raise ValueError(f"eval_every for {validator_name} must be a positive int, got {eval_every!r}")

        for float_name, float_val in (
            ("beta_lambda", self.beta_lambda),
            ("kl_anchor_alpha", self.kl_anchor_alpha),
            ("lr", self.lr),
            ("weight_decay", self.weight_decay),
            ("grad_clip", self.grad_clip),
        ):
            if isinstance(float_val, bool) or not isinstance(float_val, (int, float)) or not math.isfinite(float(float_val)):
                raise ValueError(f"{float_name} must be a finite number, got {float_val!r}")

        if len(self.beta_datasets) > 0:
            if self.beta_lambda <= 0.0:
                raise ValueError(f"beta_lambda must be > 0 when beta_datasets is non-empty, got {self.beta_lambda}")
            if self.kl_anchor_alpha < 0.0:
                raise ValueError(f"kl_anchor_alpha must be >= 0, got {self.kl_anchor_alpha}")
        else:
            if self.beta_lambda != 0.0:
                raise ValueError(f"beta_lambda must be 0.0 when beta_datasets is empty, got {self.beta_lambda}")
            if self.kl_anchor_alpha != 0.0:
                raise ValueError(f"kl_anchor_alpha must be 0.0 when beta_datasets is empty, got {self.kl_anchor_alpha}")

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

    @property
    def train_datasets(self) -> tuple[str, ...]:
        return self.ce_datasets + self.beta_datasets

    def label(self) -> str:
        parts = []
        if self.ce_datasets:
            parts.append(f"CE({'+'.join(self.ce_datasets)})")
        if self.beta_datasets:
            beta_part = f"lambda={self.beta_lambda:g}*pop_beta({'+'.join(self.beta_datasets)})"
            if self.kl_anchor_alpha > 0.0:
                beta_part += f"+alpha={self.kl_anchor_alpha:g}*KL_p0"
            parts.append(beta_part)
        return " + ".join(parts)

    def as_dict(self) -> dict:
        train_ds = self.train_datasets
        return {
            "ce_datasets": list(self.ce_datasets),
            "beta_datasets": list(self.beta_datasets),
            "train_datasets": list(train_ds),
            "sample_size_by_dataset": {name: self.sample_size_by_dataset[name] for name in train_ds},
            "beta_lambda": float(self.beta_lambda),
            "kl_anchor_alpha": float(self.kl_anchor_alpha),
            "lr": float(self.lr),
            "num_epochs": self.num_epochs,
            "eval_every_by_validator": dict(self.eval_every_by_validator),
            "lr_schedule": self.lr_schedule,
            "weight_decay": float(self.weight_decay),
            "micro_chunk": self.micro_chunk,
            "p0_chunk": self.p0_chunk,
            "gradient_checkpointing": self.gradient_checkpointing,
            "grad_clip": float(self.grad_clip),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> PaperTrainConfig:
        ce_tuple = tuple(config_dict["ce_datasets"])
        beta_tuple = tuple(config_dict["beta_datasets"])
        expected_train = list(ce_tuple + beta_tuple)
        if config_dict["train_datasets"] != expected_train:
            raise ValueError(
                f"Stored train_datasets {config_dict['train_datasets']} != ce_datasets + beta_datasets {expected_train}"
            )
        return cls(
            ce_datasets=ce_tuple,
            beta_datasets=beta_tuple,
            sample_size_by_dataset=dict(config_dict["sample_size_by_dataset"]),
            beta_lambda=float(config_dict["beta_lambda"]),
            kl_anchor_alpha=float(config_dict["kl_anchor_alpha"]),
            lr=float(config_dict["lr"]),
            num_epochs=int(config_dict["num_epochs"]),
            eval_every_by_validator=dict(config_dict["eval_every_by_validator"]),
            lr_schedule=str(config_dict["lr_schedule"]),
            weight_decay=float(config_dict["weight_decay"]),
            micro_chunk=int(config_dict["micro_chunk"]),
            p0_chunk=int(config_dict["p0_chunk"]),
            gradient_checkpointing=bool(config_dict["gradient_checkpointing"]),
            grad_clip=float(config_dict["grad_clip"]),
            seed=int(config_dict["seed"]),
        )

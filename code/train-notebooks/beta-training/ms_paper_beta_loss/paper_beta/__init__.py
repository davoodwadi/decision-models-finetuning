"""Unified semi-supervised choice-model training for the Marketing Science beta-loss paper.

Covers every row of the paper's main experimental table under one objective:
    L(theta) = E_{d in ce_datasets}[ CE(y_d, p_theta) ]
             + beta_lambda * E_{d in beta_datasets}[ split_cross_loss(p_theta, beta*_d) ]
             + kl_anchor_alpha * E_{d in beta_datasets}[ KL(p_0 || p_theta) ]

Each row's design (which datasets get CE, which get the table loss, beta_lambda, kl_anchor_alpha, epochs, LR) is
declared in the training notebooks; `regimes.py` holds only the rows' labels. See README.md for the rows.

Datasets in `ce_datasets` use one-pass micro-chunked cross-entropy on observed choices y and never compute p0.
Datasets in `beta_datasets` never use y in the loss or gradient: they use `FrozenBaseEstimator` (frozen step-0 eval-mode
probabilities p0, disjoint half-sample unbiased cross product against standardized fitted OLS betas, which stand in
for a published table, plus optional forward KL(p_0 || p_theta) anchor against p0 using the already-cached p0 vector).
"""

from .config import PaperTrainConfig
from .loss import FrozenBaseEstimator, beta_sample_loss, ce_loss, ce_sample_loss, task_log_probs
from .regimes import (
    ALL_REGIME_ORDER,
    MAIN_TABLE_ORDER,
    PAPER_REGIMES,
    PaperRegimeSpec,
    extract_step0_from_run_dir,
    summarize_paper_runs_dir,
)
from .trainer import load_paper_run, run_paper_training

__all__ = [
    "ALL_REGIME_ORDER",
    "FrozenBaseEstimator",
    "MAIN_TABLE_ORDER",
    "PAPER_REGIMES",
    "PaperRegimeSpec",
    "PaperTrainConfig",
    "beta_sample_loss",
    "ce_loss",
    "ce_sample_loss",
    "extract_step0_from_run_dir",
    "load_paper_run",
    "run_paper_training",
    "summarize_paper_runs_dir",
    "task_log_probs",
]

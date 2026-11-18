"""Beta-only training against population betas (a published-table stand-in): frozen base p0 and a split-sample cross loss.

The respondents' choices y never enter the gradient. Each training dataset has one target: its population betas
(standardized, one vector per regression), declared in the notebook. The model's population betas are
B_full @ p over every task of the dataset; a step estimates them from its sample with the frozen untrained
probabilities p0 as a control variate, on two disjoint halves, and minimizes the cross product of the two errors
(unbiased for the population loss; see `loss.py`).

Modules:
    config    PopulationTrainConfig (every run setting)
    loss      task_log_probs, FrozenBaseEstimator (p0, half-sample estimates, cross loss)
    trainer   run_population_training -> RunResult, load_population_run
Uses `beta_ols` and `beta_train` through imports only; nothing there imports this package.
"""

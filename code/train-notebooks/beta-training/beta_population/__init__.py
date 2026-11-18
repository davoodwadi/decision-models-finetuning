"""Beta-only training against population betas (a published-table stand-in), with a probability cache.

The respondents' choices y never enter the gradient. Each training dataset has one target: its population betas
(standardized, one vector per regression), declared in the notebook. The model's population betas are
B_full @ p over every task of the dataset, so the loss of one 40-task step needs every task's probability: they come
from a cache filled at step 0 and refreshed for the sampled tasks at every step (see `loss.py`).

Modules:
    config    PopulationTrainConfig (every run setting)
    loss      task_log_probs, ProbabilityCache (cache + population loss)
    trainer   run_population_training -> RunResult, load_population_run
Uses `beta_ols` and `beta_train` through imports only; nothing there imports this package.
"""

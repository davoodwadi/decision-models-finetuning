"""CE + lambda x population beta loss: the respondents' choices (CE) plus a published-table term (population betas).

Loss of one sample from one training dataset:
    total = CE(y, p) + beta_lambda * cross loss,
where CE is the mean cross-entropy against the respondents' one-hot choices (the KL term of `beta_ols.unified_loss`)
and the cross loss is the split-sample population beta loss of `beta_population` (frozen base p0, two half-sample
estimates of the population betas B_full @ p, mean of the product of their errors; unbiased for the population loss).
Every training dataset needs both its respondents' y and its target population betas.

Modules:
    config    CEPopulationTrainConfig (every run setting)
    loss      task_log_probs, FrozenBaseEstimator, ce_loss, combined_loss
    trainer   run_ce_population_training -> RunResult, load_ce_population_run
Copied from `beta_population` (loss, merge of one-task samples, trainer) so that package stays unchanged; uses
`beta_ols` and `beta_train` through imports only. Nothing outside imports this package.
"""

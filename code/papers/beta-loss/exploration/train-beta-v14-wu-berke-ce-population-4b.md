# v14: Wu + Berke, CE + λ × population beta loss (Kev 4B)

**Question.** The Kev 4B version of `train-beta-v14-wu-berke-ce-population-800m`, with every setting unchanged. Does
the population beta loss improve held-out betas when it is added to CE, and does the answer change with model scale?
The 4B comparison is v12, which ran v8's settings on Kev 4B with the sample beta loss (λ = 0, 0.25 and 1 at S = 40),
loaded from its folders in the last cell. The 800M notebook's results are the scale comparison.

**Loss.** For one sample of 40 tasks from one training piece:

    total = CE(y, p) + λ × cross loss

- CE: the mean cross-entropy against the respondents' choices, the same KL term as v8.
- Cross loss: v13's population beta loss (`beta_ce_population/loss.py`, copied from `beta_population`). The target is
  the piece's standardized human population betas, the `std_beta` column of the human OLS replication. The model's
  population betas B_full · p over all of the piece's tasks are estimated on two disjoint halves of the sample, with
  the frozen base probabilities p0 as a control variate. The loss is the mean product of the two halves' errors,
  which is unbiased for the population loss.
- v8's sample beta loss instead compares the model's and the humans' OLS betas fitted on the same 40 tasks.

**Training pieces and targets.** Wu (one piece) and Berke's four delivery tasks (four pieces), as in v8. Each piece
has its own population betas, p0 and estimator.

**Runs.** λ = 0.25 and λ = 1.0, the λ of v12's sample-loss runs. All other settings are v8's (and v12's): S = 40,
peak LR 4e-5 (sqrt rule), one epoch,
OneCycle, weight decay 0.01, clip 1.0, seed 42; best.pt is picked by `kriner2026/val_loss`.

**Validators.** Kriner 2026 and Grummon 2025, held out as in v8. `val_loss` = KL + λ × std beta MSE, the same form as
v8, so `val_loss` is comparable only between runs with the same λ. Compare KL and std beta MSE across λ.

**Reading.**
- `ce_loss` is the training CE, `cross_loss` the population beta estimate (noisy and can be negative per step; read
  its rolling mean), `weighted_cross_loss` = λ × cross loss, `total_loss` their sum. `sample_beta_mse` is the squared
  error of the full-sample estimate; it sits above the true error.
- At λ = 0 this trainer would give v12's CE-only run: the same CE, the same sample order (seed 42) and the same LR
  schedule, apart from one detail below. That is why v12's λ = 0 run serves as the baseline instead of a new run.
- v12's λ = 0.25 run was interrupted. Its row in the comparison is its best validation before the interruption, and
  Grummon may be missing there (Grummon validated every 160 steps).
- The micro-chunk search runs again: v12 used 4 on Kev 4B. Set `GRADIENT_CHECKPOINTING = True` if no size fits.

**Differences from v8 and v12 beyond the loss.**
- Berke v1 has 11,481 tasks, and 11,481 mod 40 = 1. The cross loss cannot split a one-task sample, so that task is
  added to another Berke v1 sample (one sample of 41). The epoch has one step fewer than v8's and v12's, and the
  samples after the merge point shift by one step.
- The validators use the default CUDA-graph limits, as v13's 4B notebook does. v12 raised them for speed. The limits
  change only how batches are split, not the predictions.
- Each run starts with one no-grad eval-mode pass over every training task to compute p0 (about 55,000 tasks).
- The frozen p0 is not refreshed. CE moves p away from p0 on every task, and the estimator's noise grows with that
  distance (each task's p − p0 is scaled by N / 20). Watch `grad_norm` and the `cross_loss` rolling mean. A p0
  refresh is the planned fix if the noise grows.

**Caveats.** One seed per λ, so only large gaps count. Wu and Berke provide both y and the table, so the beta term
carries no information beyond y (the table is computed from y). This tests whether the population form is a
better-weighted auxiliary loss than v8's sample form, not the table-only use case.

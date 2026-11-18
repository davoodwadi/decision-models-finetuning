# v13: Wu, population betas only (Kev 800M)

**Question.** Can the model learn from a published regression table alone? Published conjoint papers rarely release
respondent choices y. They release the design and the population betas. v13 is a best-case test of that use: Wu's
real inputs (persona and choice task), with Wu's population betas as the only target. The respondents' y never enters
the gradient.

**Loss.** Per regression: the mean squared error between the model's standardized population betas and the human ones
(the `std_beta` column of the human OLS replication). The model's population betas are B_full · p over all 18,591 Wu
tasks, where B_full is fixed by Wu's full design (`beta_population/loss.py`).

**Why a probability cache.** A step sees 40 tasks, but the loss needs all of them. If the 40-task sample betas were
compared with the population target instead, the expected loss would add the variance of a 40-task beta estimate.
That variance grows with how much p varies across personas, so the loss would push the model to ignore the persona.
Instead, every task's probability is stored (filled at step 0 by one no-grad pass over Wu). Each step recomputes the
40 sampled tasks, computes the loss on the full vector, and writes the 40 fresh values back. Only the fresh entries
carry gradients. Their gradient is scaled by N / S, so it estimates the full population gradient without bias,
apart from the staleness of the cached entries.

**Runs.**
- **beta_only**: `beta_population.run_population_training`, the population beta loss only.
- **kl_only** (control): the existing `run_training` at λ = 0 on the same Wu items. It uses y and no beta loss. This
  is the comparison the question needs: v8's λ = 0 run trained on Wu + Berke, so it is not the same data.

Both runs use v8's S = 40 regime: peak LR 4e-5 (sqrt rule), OneCycle, one epoch, weight decay 0.01, clip 1.0, seed 42.
Both pick best.pt by Kriner's beta MSE.

**Validators.**
- **wu2025** (in-sample): does beta-only training reduce Wu's population beta error? It is expected to: 8 numbers to
  fit. This is the feasibility check of the gradient path, not evidence that the idea is useful.
- **kriner2026, grummon2025** (held out): do betas learned from a table transfer as well as KL learned from microdata?
  This is the question that decides whether collecting published tables is worth it.
- **Wu KL** (reported by the wu2025 validator, and per step as `monitor_kl`): does fitting the population betas keep
  individual choices sensible? It is never in the gradient.

**Reading.**
- `val_loss` in a beta-only run equals that validator's std beta MSE (the training loss form has no KL). The shared
  live figure still titles the panel "KL + lambda x std beta MSE".
- `population_loss` per step is computed from the cache, so it lags the exact Wu validation slightly (stale entries,
  dropout on).
- One seed per run, so only a large gap between the runs counts.

**Caveats.**
- Persona sampling here is the real Wu sample. A real table-only study would have to guess the persona mix, so this
  is a best case.
- The gradient carries no persona-specific signal: for a task, d loss / d p depends only on its attributes. Persona
  differences can drift without being corrected.
- The learning rate was tuned for KL training. The beta-only gradient has a different scale; Adam mostly removes the
  scale, but clipping at 1.0 may act differently (see `grad_norm` in steps.csv).

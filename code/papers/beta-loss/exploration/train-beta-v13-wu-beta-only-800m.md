# v13: Wu, population betas only (Kev 800M)

**Question.** Can the model learn from a published regression table alone? Published conjoint papers rarely release
respondent choices y. They release the design and the population betas. v13 is a best-case test of that use: Wu's
real inputs (persona and choice task), with Wu's population betas as the only target. The respondents' y never enters
the gradient.

**Loss.** Per regression: the mean squared error between the model's standardized population betas and the human ones
(the `std_beta` column of the human OLS replication). The model's population betas are B_full · p over all 18,591 Wu
tasks, where B_full is fixed by Wu's full design (`beta_population/loss.py`).

**Estimator: frozen base p0 and a split-sample cross loss.** A step sees 40 tasks, but the loss needs all 18,591.
- p0 = the untrained model's probabilities of every Wu task, computed once at the start (eval mode, no gradients) and
  frozen; β0 = B_full · p0.
- Each step splits its shuffled sample into two disjoint halves (20 + 20; the epoch's last sample 15 + 16) and
  estimates the population betas on each: β̂ = β0 + (N / half size) · B_full[:, half] · (p(θ) − p0). Its expected
  value is the model's current population betas. Every term uses the current weights or the frozen p0, so nothing is
  stale, and p0 only reduces the variance.
- Loss = mean over coefficients of (β̂₁ − β*) · (β̂₂ − β*). Squaring one estimate would add its variance, which grows
  with how much p varies across personas, so the loss would push the model to ignore the persona. The cross product
  of two independent halves has the population loss as its expected value, and its gradient is unbiased. A single
  step's value can be negative.

**First attempt: a probability cache (failed, runs `beta_only_lr=...`).** Every task's probability was stored, and
each step refreshed only its 40 tasks (0.2% of the cache). The weights change every task at once, so the cached
residual lagged the model by up to an epoch and kept pushing after the true betas had passed the target. In both runs
(peak LR 4e-5 and 1e-5, 5 epochs) the cached loss fell to about 1e-4 while Wu's true std beta MSE rose above its
step-0 value (0.205 → 0.441 at step 400 with 4e-5), Spearman ρ turned negative, and Wu KL rose from 1.07 to about 12.
Lowering the LR did not fix it. These runs used the cache code, which the estimator above replaced.

**Run and baseline.**
- **beta_only**: `beta_population.run_population_training`, the population beta loss only. Settings are declared in
  the notebook's Train cell (S = 40, sqrt-rule peak LR, OneCycle, weight decay 0.01, clip 1.0, seed 42; the LR,
  epochs and best metric are set there).
- **Baseline: the base model** (step 0, untrained). The use case is a paper without microdata, so the alternative to
  training on its table is not training on it at all. A KL run on Wu's microdata would answer a different question
  (table vs microdata) that does not arise when there is no microdata.

**Validators.**
- **wu2025** (in-sample): does beta-only training reduce Wu's population beta error? It is expected to: 8 numbers to
  fit. This is the feasibility check of the gradient path, not evidence that the idea is useful.
- **kriner2026, grummon2025** (held out; berke2023 in the held-out cells at the end): does training on one table
  improve other studies' population betas over the base model? This is the question that decides whether collecting
  published tables is worth it.
- **Wu KL** (reported by the wu2025 validator, and per step as `monitor_kl`): does fitting the population betas keep
  individual choices sensible? It is never in the gradient.

**Reading.**
- `val_loss` in a beta-only run equals that validator's std beta MSE (the training loss form has no KL). The shared
  live figure still titles the panel "KL + lambda x std beta MSE".
- `cross_loss` (= `total_loss`, the training objective) is noisy and can be negative per step; read its rolling mean.
  `sample_beta_mse` is the squared error of the full 40-task estimate: readable, but its expected value sits slightly
  above the true error. The Wu validator is the exact measure.
- p0 comes from eval mode and the training forwards run with dropout, so the estimate targets the dropout-on
  population betas; the Wu validator measures eval mode. The gap is the usual small dropout effect.
- One seed, so only a large gap from the base model counts.

**Caveats.**
- Persona sampling here is the real Wu sample. A real table-only study would have to guess the persona mix, so this
  is a best case.
- The gradient carries no persona-specific signal: for a task, d loss / d p depends only on its attributes. Persona
  differences can drift without being corrected.
- Saturation: the cache runs pushed most tasks to probabilities near 0 or 1 (Wu KL about 12). The 8 beta equations do
  not pin down individual probabilities, and this loss has no term that does. A KL anchor to p0 is the planned second
  fix.
- Gradient noise: the half-sample estimate multiplies each task's (p − p0) by N / 20 ≈ 930. Near p0 that is small; as
  the model moves away from p0 the gradient gets noisier.
- The learning rate was tuned for KL training. The beta-only gradient has a different scale; Adam mostly removes the
  scale, but clipping at 1.0 may act differently (see `grad_norm` in steps.csv).

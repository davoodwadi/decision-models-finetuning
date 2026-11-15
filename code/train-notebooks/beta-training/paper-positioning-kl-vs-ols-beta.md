# Counteracting KL uniform hedging via differentiable sample OLS moment supervision

## Core positioning and research problem

**Econometric estimands versus token-level likelihood.** When fine-tuning decision models such as `Kev` to simulate human participants in conjoint experiments, randomized trials, and policy surveys, the primary scientific target is rarely per-response log-likelihood in isolation. Downstream researchers in marketing, economics, and political science evaluate synthetic cohorts through structural and reduced-form treatment effects, most notably ordinary least squares (`OLS`) average marginal component effects (`AMCE`) and implied willingness-to-pay ratios. A training objective that achieves low per-trial cross-entropy while systematically compressing estimated treatment slopes toward zero undermines counterfactual demand estimation and policy inference.

**Uniform hedging under standard KL fine-tuning.** Standard respondent-level fine-tuning minimizes per-question Kullback-Leibler (`KL`) divergence, equivalently cross-entropy against observed one-hot choices. Although `KL` is a strictly proper scoring rule when the model class contains the exact data-generating conditional probability without representation error, practical fine-tuning always operates under residual epistemic and representation uncertainty (e.g., finite LoRA rank, unobserved respondent tastes omitted from demographic personas, and wording or domain shift on unseen surveys). Because negative log-likelihood penalizes confident mispredictions unboundedly while bounding the loss of a uniform prediction `[0.5, 0.5]` at $\log 2$, any representation noise makes shrinking logits toward zero strictly optimal under expected `KL`. In practice, the model learns to hedge toward a uniform distribution whenever contextual certainty weakens, systematically attenuating every downstream regression coefficient toward zero, especially under zero-shot transfer to a new experiment.

**Differentiable sample OLS as an anti-collapse regularizer.** Augmenting per-response `KL` with a single-pass differentiable sample `OLS` loss resolves this attenuation from first principles. By projecting both predicted choice probabilities and observed human choices onto the experimental design matrix across a multi-respondent sample within the same forward pass, the `OLS` penalty vanishes only when predicted attribute slopes match empirical human slopes, exerts its strongest logit gradient right at the uniform `[0.5, 0.5]` boundary, and averages out unstructured task-level error before squaring. Across progressive experiments in [train-beta-v1-800m.ipynb](./train-beta-v1-800m.ipynb), [train-beta-v2-800m.ipynb](./train-beta-v2-800m.ipynb), and [train-beta-v3-800m.ipynb](./train-beta-v3-800m.ipynb) (building on the initial proposal in [train-beta-v1-800m.md](./train-beta-v1-800m.md)), this mechanism improves both in-experiment `AMCE` recovery and zero-shot cross-experiment generalization with less than `1 ms` of linear-algebra overhead per optimization step.

## Mathematical mechanism of uniform hedging and moment restoration

**Latent choice setup under representation uncertainty.** Consider a sample of $S$ human respondents answering $T_s = \sum_{r=1}^{S} T_r$ binary forced-choice tasks ($2T_s$ stacked option rows). On task $t \in \{1, \dots, T_s\}$, let $\pi_t \in (0, 1)$ denote the true conditional probability of selecting Option 1 given the respondent persona and displayed profiles, with true log-odds $z_t^* = \log(\pi_t / (1 - \pi_t))$ so that $\pi_t = \sigma(z_t^*)$. Suppose a finite-capacity or domain-shifted decision model forms a noisy utility feature $u_t = z_t^* + \tilde{\delta}_t$, where $\tilde{\delta}_t$ represents zero-mean specification or domain-transfer error satisfying $\mathbb{E}[\tilde{\delta}_t \mid z_t^*] = 0$ and $\operatorname{Var}(\tilde{\delta}_t \mid z_t^*) = \sigma_\delta^2 > 0$. Scaling this feature by readout sharpness $\gamma > 0$ produces logit difference $z_t = \gamma u_t$ and predicted Choice 1 probability $p_t = \sigma(\gamma u_t)$.

**Proposition on KL logit damping and AMCE attenuation.** Because binary `KL` evaluates the strictly convex log-partition function $\log(1 + e^{z_t})$ on each task prior to averaging across the dataset, a second-order expansion around $\gamma z_t^*$ (using $\frac{d^2}{dz^2}\log(1 + e^z) = \sigma(z)(1 - \sigma(z))$) yields expected per-task cross-entropy:

$$
\mathbb{E}_{\tilde{\delta}}\!\left[\mathcal{L}_{\mathrm{KL}}(\gamma u_t, y_t)\right] \approx \mathcal{L}_{\mathrm{KL}}(\gamma z_t^*, \pi_t) + \frac{1}{2}\gamma^2 \sigma_\delta^2 \, \sigma(\gamma z_t^*)\big(1 - \sigma(\gamma z_t^*)\big).
$$

Whereas the noiseless first term is uniquely minimized at unattenuated scale $\gamma = 1$, the second term is strictly positive and quadratic in $\gamma$ whenever $\sigma_\delta^2 > 0$. Consequently, the `KL`-optimal sharpness satisfies

$$
\gamma^*_{\mathrm{KL}} \approx \frac{\operatorname{Var}(z^*)}{\operatorname{Var}(z^*) + \sigma_\delta^2} < 1.
$$

Log-loss risk aversion therefore forces even an optimally trained model within its hypothesis class to damp logits toward zero, pulling predicted probabilities $p_t$ toward `0.5` and shrinking every model-implied regression slope $|\hat{\beta}_{p}|$ below the ground-truth human effect $|\hat{\beta}_{y}|$. Furthermore, when transferring zero-shot to an unseen survey where unfamiliar stimuli increase $\sigma_\delta^2$, this shrinkage intensifies automatically.

**Sample OLS projection and anti-collapse gradient geometry.** Let $X_s \in \mathbb{R}^{2T_s \times (P + 1)}$ denote the respondent sample's stacked experimental design matrix, where column `0` is the constant intercept $\mathbf{1}_{2T_s}$ and columns `1` through $P$ are binary indicators for randomized non-reference attribute levels. Whenever $X_s$ has full column rank $P + 1$, define the fixed linear `OLS` projector $A_s = (X_s^\dagger)_{1:,:} \in \mathbb{R}^{P \times 2T_s}$ via the Moore-Penrose pseudoinverse, dropping row `0` (the intercept), and define the paired task-sensitivity matrix $A_{\mathrm{diff}} = A_s[:, 0::2] - A_s[:, 1::2] \in \mathbb{R}^{P \times T_s}$. Stacking model option probabilities $p(z) \in \mathbb{R}^{2T_s}$ and one-hot human choices $y_s \in \{0, 1\}^{2T_s}$ from the exact same forward pass defines the differentiable sample `OLS` loss:

$$
\mathcal{L}_{\beta}(z) = \frac{1}{P}\left\lVert A_s p(z) - A_s y_s \right\rVert_2^2, \qquad \mathcal{L}_{\mathrm{total}}(z) = \mathcal{L}_{\mathrm{KL}}(z, y_s) + \lambda \mathcal{L}_{\beta}(z).
$$

Three exact algebraic properties make $\mathcal{L}_{\beta}(z)$ a targeted counterweight to `KL` hedging. First, because $\mathbf{1}_{2T_s}$ is column `0` of $X_s$, $X_s^\dagger \mathbf{1}_{2T_s} = \mathbf{e}_0$, which implies $A_s(0.5 \cdot \mathbf{1}_{2T_s}) = \mathbf{0}$ identically; collapsing toward a uniform `[0.5, 0.5]` distribution zeroes out all $P$ predicted `AMCE` terms and incurs the full signal penalty $\frac{1}{P}\lVert \hat{\beta}_{y,s} \rVert_2^2$. Second, differentiating $\mathcal{L}_{\beta}$ with respect to task $t$'s logit difference $z_t$ yields

$$
\frac{\partial \mathcal{L}_{\beta}}{\partial z_t} = \frac{2}{P}\, p_t(1 - p_t) \left[A_{\mathrm{diff}}^\top \big(\hat{\beta}_{p,s} - \hat{\beta}_{y,s}\big)\right]_t.
$$

Notice that the Bernoulli variance factor $p_t(1 - p_t)$ attains its global maximum of `0.25` precisely at $p_t = 0.5$. Right where `KL` pulls predictions toward uniformity, $\mathcal{L}_{\beta}$ applies maximum restoring leverage along the $P$ experimental contrast directions $\operatorname{row}(A_{\mathrm{diff}})$ without amplifying orthogonal prompt noise. Third, because $A_s p(z)$ averages across all $T_s$ tasks *before* applying the quadratic loss, unstructured per-task errors $\tilde{\delta}_t$ cancel inside $A_s p(z)$ at rate $\mathcal{O}(\sigma_\delta^2 / T_s)$, eliminating the per-task curvature tax that causes $\gamma^*_{\mathrm{KL}} < 1$.

## Operating conditions for sample size and regularization weight

**Sample-size threshold and target-noise scaling.** Because human choices $y_{t,1} \in \{0, 1\}$ are noisy Bernoulli realizations around $\pi_t$ with variance $v_t = \pi_t(1 - \pi_t)$, the human sample `OLS` vector $\hat{\beta}_{y,s} = A_s y_s$ fluctuates around its noiseless conditional expectation $\beta^*_s = A_s \pi_s$ with exact expected mean squared error

$$
\operatorname{NoiseMSE}(S) = \frac{1}{P}\operatorname{Tr}\!\left(A_{\mathrm{diff}}\operatorname{diag}(v) A_{\mathrm{diff}}^\top\right) = \frac{1}{P}\sum_{k=1}^{P}\sum_{t=1}^{T_s} A_{\mathrm{diff}}[k, t]^2 \, \pi_t(1 - \pi_t).
$$

On the Wu 2025 electricity conjoint (`18,591` paired tasks across `1,861` respondents, $P = 8$ `AMCE` slopes, population signal mean square $\frac{1}{8}\lVert \hat{\beta}_{\mathrm{pop}} \rVert_2^2 = 0.04746$), single-respondent steps ($S = 1$, $T_s \approx 10$) leave `136` respondents rank-deficient and yield $\operatorname{NoiseMSE}(1) = 0.14934$ (`SNR = 0.32`), three times noisier than the true signal power. Consequently, in [train-beta-v1-800m.ipynb](./train-beta-v1-800m.ipynb), single-respondent beta supervision failed to improve population `OLS` error (`0.0016` to `0.0021` for $\lambda \in \{1, 5, 20\}$ versus `0.0015` for $\lambda = 0$). Increasing respondent `sample_size` to $S = 16$ ($T_s \approx 159$ tasks, `318` option rows per step) eliminates rank deficiency completely (`0 / 117` steps) and reduces Bernoulli target noise thirtyfold to `0.00504` (`SNR = 9.42`, total empirical batch variance `0.00819` around $\hat{\beta}_{\mathrm{pop}}$ when including 16-person taste heterogeneity).

| Respondent `sample_size` ($S$) | Steps per epoch | Mean option rows ($2T_s$) | Rank-deficient steps | Population signal mean square | Expected target noise MSE | Signal-to-noise ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1861 | 19.98 | 136 | 0.04746 | 0.14934 | 0.32 |
| 2 | 931 | 39.94 | 0 | 0.04746 | 0.05083 | 0.93 |
| 4 | 466 | 79.79 | 0 | 0.04746 | 0.02190 | 2.17 |
| 8 | 233 | 159.58 | 0 | 0.04746 | 0.01021 | 4.65 |
| 16 | 117 | 317.79 | 0 | 0.04746 | 0.00504 | 9.42 |
| 32 | 59 | 630.20 | 0 | 0.04746 | 0.00269 | 17.66 |
| 64 | 30 | 1239.40 | 0 | 0.04746 | 0.00176 | 27.03 |

**Why excessive lambda reverses performance.** Even when $S = 16$ supplies high-SNR targets, $\mathcal{L}_{\beta}$ constrains only $P = 8$ aggregate linear combinations out of roughly `159` task probabilities per step, leaving an approximately `151`-dimensional nullspace governing individual baseline rates and persona-by-attribute interactions that only $\mathcal{L}_{\mathrm{KL}}$ supervises. When $\lambda$ becomes too large (e.g., $\lambda = 5$ or $\lambda = 20$), three coupled distortions arise. First, once population beta error drops below `0.001`, aggressive updates along global attribute tokens over-step onto individual 16-respondent batch fluctuations (`0.00819` variance around $\hat{\beta}_{\mathrm{pop}}$), increasing late-epoch batch beta error rather than averaging smoothly across steps. Second, because global gradient norm climbs from `1.81` ($\lambda = 0$) to `1.91` ($\lambda = 1$), `2.75` ($\lambda = 5$), and `6.35` ($\lambda = 20$), unit-norm gradient clipping rescales the combined gradient by `1 / 6.35` at $\lambda = 20$, throttling per-respondent `KL` gradients by roughly `3.5x` and degrading choice calibration (`pop_kl_loss` rises from `0.5328` to `0.5427`). Third, because $\nabla_z \mathcal{L}_{\beta}$ weights tasks by $p_t(1 - p_t)$ whereas population `OLS` weights all tasks equally, starving the linear $\nabla_z \mathcal{L}_{\mathrm{KL}} = p - y$ anchor causes the model to satisfy sample `OLS` equations mainly by over-pushing uncertain mid-probability tasks, overshooting steep renewable slopes in the unweighted population average. Moderate weighting ($\lambda = 1$ at $S = 16$, where gradient norm rises by only `5%`) preserves individual `KL` calibration while removing uniform attenuation.

## Empirical evidence from electricity and aviation conjoints

**In-experiment generalization on Wu 2025 (`v2`).** Training `Kev 0.8B` (`bf16` backbone with rank-64 LoRA) for one epoch on Wu 2025 electricity plan choices (`1,861` respondents, `S = 16`, `117` steps) in [train-beta-v2-800m.ipynb](./train-beta-v2-800m.ipynb) directly confirms both uniform `KL` shrinkage and the U-shaped response to $\lambda$. Defining the empirical projection slope $\hat{\gamma} = \langle \hat{\beta}_{\mathrm{model}}, \hat{\beta}_{\mathrm{human}} \rangle / \lVert \hat{\beta}_{\mathrm{human}} \rVert_2^2$, pure `KL` ($\lambda = 0$) shrinks $|\hat{\beta}_{\mathrm{model}}|$ below $|\hat{\beta}_{\mathrm{human}}|$ on all `8 / 8` terms ($\hat{\gamma} = 0.940$, `pop_beta_mse = 0.00023`, `MAPE = 7.46%`). Adding $\lambda = 1$ lifts $\hat{\gamma}$ to `0.961`, cuts population beta `MSE` by `2.6x` to `0.00009`, and nearly halves `MAPE` to `3.89%` (e.g., `$110 vs $105` improves from `-0.0822` to `-0.0960` against human `-0.0969`, and `$115 vs $105` improves from `-0.1612` to `-0.1778` against human `-0.1779`), before large weights reverse at $\lambda = 5$ (`MSE = 0.00017`) and $\lambda = 20$ (`MSE = 0.00075`, $\hat{\gamma} = 1.026$).

| Benchmark and run (`S = 16`) | Choice KL loss | Population beta MSE | Population beta MAE | Population beta MAPE (%) | Spearman rho | Attenuation slope ($\hat{\gamma}$) | Mean grad norm | Train time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wu 2025 in-experiment, step 0 | 1.06993 | 0.05118 | 0.20925 | 114.13 | -0.214 | 0.055 | n/a | 0.0 |
| Wu 2025 in-experiment, $\lambda = 0$ | 0.53275 | 0.00023 | 0.01258 | 7.46 | 1.000 | 0.940 | 1.81 | 252.0 |
| Wu 2025 in-experiment, $\lambda = 1$ | 0.53481 | 0.00009 | 0.00742 | 3.89 | 1.000 | 0.961 | 1.91 | 261.0 |
| Wu 2025 in-experiment, $\lambda = 5$ | 0.53746 | 0.00017 | 0.00982 | 4.72 | 1.000 | 0.996 | 2.75 | 251.0 |
| Wu 2025 in-experiment, $\lambda = 20$ | 0.54274 | 0.00075 | 0.02044 | 11.30 | 0.976 | 1.026 | 6.35 | 264.5 |
| Kriner 2026 zero-shot, step 0 | 1.08491 | 0.01776 | 0.11429 | 133.53 | -0.266 | -0.028 | n/a | 0.0 |
| Kriner 2026 zero-shot, $\lambda = 0$ | 0.62682 | 0.00389 | 0.05537 | 75.40 | 0.937 | 0.592 | 1.90 | 250.2 |
| Kriner 2026 zero-shot, $\lambda = 1$ | 0.62384 | 0.00294 | 0.04329 | 63.41 | 0.895 | 0.724 | 1.95 | 238.8 |
| Kriner 2026 zero-shot, $\lambda = 5$ | 0.63163 | 0.00357 | 0.05084 | 67.46 | 0.860 | 0.621 | 2.78 | 238.1 |
| Kriner 2026 zero-shot, $\lambda = 20$ | 0.63135 | 0.00329 | 0.04682 | 66.31 | 0.839 | 0.686 | 6.38 | 238.0 |

**Zero-shot cross-experiment generalization to Kriner 2026 (`v3`).** In [train-beta-v3-800m.ipynb](./train-beta-v3-800m.ipynb), the exact same Wu 2025 training runs are evaluated strictly out-of-experiment on Goldfarb, Kriner et al. (2026) sustainable aviation fuel (`SAF`) flight choices (`5,000` paired tasks across `1,000` US respondents, `12` Table 2 `AMCE` coefficients). Under pure `KL` ($\lambda = 0$), domain shift induces pronounced uniform hedging across all `12 / 12` coefficients, compressing the overall 12-term projection slope to $\hat{\gamma} = 0.592$ (`kriner_beta_mse = 0.00389`, `MAPE = 75.40%`). Supervising Wu 2025 sample betas with $\lambda = 1$ transfers directly across domains, lowering held-out `kriner_beta_mse` by `24.4%` (`0.00389` to `0.00294`), cutting `MAPE` by `12.0` percentage points (`75.40%` to `63.41%`), lifting the 12-term slope to $\hat{\gamma} = 0.724$, and simultaneously improving held-out `kriner_kl_loss` (`0.62682` to `0.62384`). Focusing on the five substantively transferable economic, environmental, and quality attributes (`Price: $275`, `Price: $300`, `Price: $325`, `Fuel: Sustainable aviation fuel`, and `Space: Extra legroom`, excluding aviation-specific carrier, seat, and baggage conventions absent from Wu 2025), $\lambda = 1$ raises the zero-shot slope scale from `0.728` to `0.894` and reduces cross-experiment `MSE` by `4.6x` (`0.00271` to `0.00059`).

| Kriner 2026 Table 2 term (reference level in parentheses) | Human beta | Step 0 | $\lambda = 0$ (KL only) | $\lambda = 1$ (KL + OLS) | $\lambda = 5$ | $\lambda = 20$ |
| --- | --- | --- | --- | --- | --- | --- |
| Airline: Frontier (`Delta`) | -0.0854 | 0.0344 | -0.0042 | -0.0097 | -0.0113 | -0.0106 |
| Airline: Southwest (`Delta`) | -0.0238 | 0.0267 | -0.0014 | -0.0053 | -0.0065 | -0.0028 |
| Airline: United (`Delta`) | -0.0220 | 0.0010 | 0.0080 | 0.0048 | 0.0049 | 0.0036 |
| Fuel: Sustainable aviation fuel (`Petroleum`) | 0.0691 | -0.0574 | 0.0232 | 0.0299 | 0.0292 | 0.0272 |
| Baggage: Checked bag+carry-on (`Carry-on`) | 0.0561 | -0.0435 | 0.0051 | -0.0004 | -0.0002 | -0.0031 |
| Baggage: Personal item only (`Carry-on`) | -0.1393 | -0.0030 | -0.0171 | -0.0219 | -0.0177 | -0.0157 |
| Space: Extra legroom (`Standard 31"`) | 0.0653 | -0.0813 | 0.0087 | 0.0473 | 0.0400 | 0.0365 |
| Seat: Middle (`Aisle`) | -0.0906 | -0.0057 | -0.0104 | -0.0043 | 0.0030 | 0.0018 |
| Seat: Window (`Aisle`) | 0.0384 | 0.0086 | 0.0038 | 0.0046 | 0.0084 | 0.0098 |
| Price: $275 (`$250`) | -0.0853 | 0.0032 | -0.0719 | -0.0849 | -0.0711 | -0.0838 |
| Price: $300 (`$250`) | -0.1953 | -0.0133 | -0.1335 | -0.1711 | -0.1463 | -0.1671 |
| Price: $325 (`$250`) | -0.3040 | -0.0201 | -0.2387 | -0.2813 | -0.2422 | -0.2676 |

## Positioning in the manuscript

**Three-part methodological contribution.** For the methodology and theory sections of the manuscript, these findings support a cohesive three-part narrative that distinguishes decision-model fine-tuning for social and market research from standard instruction tuning. First, we formalize why proper point-prediction losses (`KL` and cross-entropy) are structurally biased toward uniform `[0.5, 0.5]` hedging under representation uncertainty and domain shift, yielding systematic attenuation ($\gamma^*_{\mathrm{KL}} < 1$) on econometric treatment effects even when choice rankings (`Spearman rho`) already look high. Second, we introduce single-pass differentiable sample `OLS` projection ($A_s = (X_s^\dagger)_{1:,:}$) as a parameter-free, `0.5 ms` moment-matching regularizer whose Jacobian peaks at $p = 0.5$ and whose task-averaging structure cancels idiosyncratic representation error before squaring. Third, we characterize the two theoretical regime boundaries required in practice, showing that multi-respondent sampling ($S \ge 16$, so $T_s \gg P$) is necessary to push target Bernoulli noise well below population signal power, while moderate gradient weighting ($\lambda \approx 1$) prevents nullspace starvation and softmax curvature overshoot, delivering sharper treatment-effect recovery both within the training study and zero-shot on unseen experimental designs.

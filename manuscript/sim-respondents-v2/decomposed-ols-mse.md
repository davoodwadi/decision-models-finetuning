# Decomposing OLS slope MSE into scale and shape error

## Why raw OLS MSE conflates two distinct economic errors

Evaluating synthetic choice data by the raw mean squared error between model-implied and human OLS slopes ($\mathrm{MSE}_\beta$) mixes two fundamentally different properties of a discrete-choice model, namely relative preferences and choice-noise scale. In random utility theory, choice probabilities follow $P(y = k \mid x) = \mathrm{softmax}(v_k(x) / \tau)$, where $\tau$ governs idiosyncratic error variance (or equivalently the decision-head temperature $T$). Multiplying all latent utilities by a positive constant shifts the overall amplitude of OLS average marginal component effects ($\boldsymbol{\beta}$) roughly uniformly across attributes, yet leaves willingness-to-pay ratios ($\beta_j / \beta_{\text{price}}$), attribute trade-offs, and effect orderings nearly unchanged.

Because raw $\mathrm{MSE}_\beta$ sums squared deviations directly, a model that recovers exact human preference trade-offs with sharper choice probabilities can suffer a much worse raw $\mathrm{MSE}_\beta$ than a weak model whose probabilities stay near uniform. This distortion is severe when comparing across model sizes and training regimes. Larger and choice-trained models often learn accurate relative preferences while exhibiting higher logit sharpness on unseen choice formats, inflating raw $\mathrm{MSE}_\beta$ through uniform over-reaction rather than directional error.

## Exact orthogonal decomposition of OLS slope MSE

Let $\mathbf{h} \in \mathbb{R}^d$ denote the standardized human slope benchmark ($\hat{\boldsymbol{\beta}}^{(g)} / \sigma_y$) and let $\mathbf{m} \in \mathbb{R}^d$ denote the corresponding model-implied standardized slope vector. Define the nonnegative optimal scalar rescaling of the model vector onto the human vector as

$$a^* = \max\!\left(0,\; \frac{\mathbf{m}^\top \mathbf{h}}{\|\mathbf{m}\|_2^2}\right)$$

and define the implied scale ratio as $\rho = 1 / a^*$ whenever $a^* > 0$, so $\rho > 1$ indicates uniform over-reaction relative to humans and $\rho < 1$ indicates uniform attenuation. Whenever $a^* > 0$, the residual $\mathbf{h} - a^*\mathbf{m}$ is orthogonal to $\mathbf{m}$, yielding an exact Pythagorean decomposition of raw $\mathrm{MSE}_\beta$ into scale error and shape error:

$$\mathrm{MSE}_\beta(\mathbf{m}, \mathbf{h}) = \frac{1}{d}\|\mathbf{m} - \mathbf{h}\|_2^2 = \underbrace{\frac{(1 - a^*)^2}{d}\|\mathbf{m}\|_2^2}_{\text{Scale MSE}} + \underbrace{\frac{1}{d}\|\mathbf{h} - a^*\mathbf{m}\|_2^2}_{\text{Shape MSE}}.$$

When $a^* > 0$, Shape MSE also simplifies to $\frac{\|\mathbf{h}\|_2^2}{d}(1 - \cos^2 \theta)$, where $\cos \theta = (\mathbf{m}^\top \mathbf{h}) / (\|\mathbf{m}\|_2 \|\mathbf{h}\|_2)$ is cosine similarity. Shape MSE therefore measures the portion of human treatment-effect variance that cannot be recovered by any positive scalar rescaling of the model's slopes. It isolates directional fidelity, relative attribute weights, and willingness-to-pay structure, and it is strictly invariant to any uniform rescaling of $\mathbf{m}$. Scale MSE isolates the remaining penalty from uniform amplitude mismatch ($a^* \neq 1$), which reflects choice-noise calibration rather than preference structure.

## Empirical evidence from v1 post-hoc analysis

Applying this decomposition to the v1 held-out test datasets (Consumer rights, AI and offshoring, Food labels, and Vaccines) resolves the apparent paradox in which the off-the-shelf KEV-0.8B model achieved lower raw $\mathrm{MSE}_\beta$ than the choice-trained KEV-9B model at raw temperature $T = 1.0$. The table below reports the four-dataset mean metrics (MSE values multiplied by $10^3$) from the v1 post-hoc temperature analysis.

| Model | Regime | Temperature rule | T | Raw MSE | Shape MSE | Scale MSE | Share from scale (%) | Scale ratio | Cosine sim |
|--------|--------|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| KEV-0.8B | Off-the-shelf | Raw logits | 1.00 | 14.95 | 13.05 | 1.90 | 15.4 | 1.46 | 0.470 |
| KEV-0.8B | Off-the-shelf | Shipped KEV T | 2.35 | 13.45 | 12.39 | 1.06 | 14.2 | 0.94 | 0.494 |
| KEV-0.8B | Drones + Electricity | Raw logits | 1.00 | 9.38 | 8.68 | 0.70 | 12.0 | 0.73 | 0.759 |
| KEV-4B | Off-the-shelf | Raw logits | 1.00 | 33.56 | 8.85 | 24.72 | 63.2 | 1.96 | 0.763 |
| KEV-4B | Drones + Electricity | Raw logits | 1.00 | 17.39 | 4.89 | 12.50 | 46.5 | 1.35 | 0.903 |
| KEV-9B | Off-the-shelf | Raw logits | 1.00 | 30.36 | 6.31 | 24.05 | 64.1 | 2.07 | 0.855 |
| KEV-9B | Drones + Electricity | Raw logits | 1.00 | 20.55 | 3.15 | 17.41 | 68.2 | 1.74 | 0.933 |
| KEV-9B | Drones + Electricity | Flights NLL T\* | 1.26 | 15.22 | 3.11 | 12.11 | 58.5 | 1.54 | 0.934 |
| KEV-9B | Drones + Electricity | Shipped KEV T | 2.19 | 5.82 | 3.06 | 2.76 | 34.8 | 1.04 | 0.935 |

**Raw MSE hides monotonic gains in preference recovery.** At $T = 1.0$, raw $\mathrm{MSE}_\beta$ makes off-the-shelf KEV-0.8B (14.95) look better than choice-trained KEV-9B (20.55). Shape MSE reveals the exact opposite pattern. Choice-trained KEV-9B cuts Shape MSE by more than fourfold relative to off-the-shelf KEV-0.8B (3.15 versus 13.05) and by half relative to off-the-shelf KEV-9B (3.15 versus 6.31), raising mean cosine similarity from 0.470 to 0.933. Across every model size (0.8B, 4B, and 9B), Shape MSE drops monotonically with both model scale and choice training.

**Temperature acts almost purely on Scale MSE.** Sweeping the decision-head temperature $T$ barely moves Shape MSE for choice-trained KEV-9B (3.15 at $T = 1.00$, 3.11 at $T = 1.26$, and 3.06 at $T = 2.19$). Instead, 68.2% of choice-trained KEV-9B's raw $\mathrm{MSE}_\beta$ at $T = 1.0$ comes from Scale MSE alone, reaching 82.0% on Vaccines, 84.0% on AI and offshoring, and 94.8% on Food labels, where cosine similarity is already 0.980. Raising $T$ to 2.19 collapses four-dataset mean raw $\mathrm{MSE}_\beta$ from 20.55 to 5.82 (scale ratio 1.04), outperforming off-the-shelf KEV-0.8B and off-the-shelf KEV-9B on every single held-out test dataset without changing a single backbone or LoRA weight.

**Why post-hoc temperature alone could not fix v1.** Although a single temperature near $T = 2.2$ eliminates the scale mismatch on the v1 test datasets, calibrating $T$ on the binary-only Flights validation dataset selected $T^* = 1.26$ by NLL and $T^* = 0.85$ by $\mathrm{MSE}_\beta$. Choice training on binary-only datasets (Drones and Electricity) had already matched the sharpness of binary forced choice (Flights), leaving Scale MSE unresolved on multi-option and Likert test datasets.

## Implications for reporting and analysis in v2

**Report the full decomposition alongside raw MSE.** Main evaluation tables in sim-respondents-v2 should report raw $\mathrm{MSE}_\beta$, Shape MSE, Scale MSE, and the scale ratio ($\rho = 1 / a^*$) together (alongside cosine similarity or Spearman rank correlation and choice NLL). Reporting only raw $\mathrm{MSE}_\beta$ conflates whether a model learned the wrong consumer preferences with whether its choice probabilities are simply sharper or flatter than human responses on an unseen format.

**Separate what choice training fixes from what scale calibration fixes.** Shape MSE is the primary measure of whether choice training improves structural preference recovery (relative attribute weights and WTP trade-offs) on unseen studies, because no scalar temperature adjustment can reduce Shape MSE meaningfully. Scale MSE diagnoses whether the model's response sharpness transfers across choice formats (binary forced choice, multi-option opt-out choice, and ordinal Likert scales), directly testing whether the multi-format training and validation design in v2 prevents binary over-sharpening.

**Guard against conflating rescaling with post-inference data requirements.** Decomposing $\mathrm{MSE}_\beta$ into Shape MSE and Scale MSE is strictly a diagnostic evaluation lens, not a deployment method that uses target-study human data. In unseen-study deployment, the practitioner needs low raw $\mathrm{MSE}_\beta$ as well as low Shape MSE, which means both components matter. A well-designed multi-format training and validation split aims to drive down Shape MSE through choice training while keeping Scale MSE low out of the box without target-study human labels.
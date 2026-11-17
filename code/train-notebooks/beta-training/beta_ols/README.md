# `beta_ols`: dataset OLS modules and the unified beta loss

Used by the notebooks in `train-notebooks/beta-training/`. Their first cell adds this folder to `sys.path`.
Data and OLS only: no model or tokenizer. The training side (items, gradient step, validation, trainer) is `../beta_train/`.

## Layout

| File | Contents |
|---|---|
| `core.py` | Shared types (`RegressionSpec`, `OutcomeRow`, `RespondentBetaPart`), `fit_human_ols`, `wide_ols_table`, `ols_projector`, `beta_noise_by_sample_size` |
| `unified_loss.py` | `build_sample_beta_operators`, `compute_unified_loss` |
| `evaluation.py` | `build_population_eval`, `flat_probs_from_predictions`, `compare_population_betas`: model vs human population betas with the module's own coding |
| `wu2025.py`, `berke2023.py`, `grummon2023_calories.py`, `grummon2023_labels.py`, `kriner2026.py`, `covid.py`, `grummon2025.py` | One module per dataset (`grummon2023_calories` and `grummon2023_labels` are two codings of Grummon 2023's data) |
| `__init__.py` | `DATASET_MODULES`: every dataset module by name |

## Dataset module interface

Every dataset module has the same sections, in the same order:

1. **Levels.** The paper's attributes or arms; the reference level comes first.
2. **Design.** `DESIGN_COLUMNS` (with `"Intercept"` first), `TARGET_COLUMNS`, and `design_row(...)`.
3. **Choice rows.** `load_choice_df()` returns one row per answered task, with each respondent's tasks in prompt order.
4. **Respondent beta part.** `respondent_beta_part(r_df)` returns option counts per task, flat one-hot choices, and outcome rows.
5. **Regression table and specs.** `regression_rows(choice_df)` and `regression_specs(rows)`.

Plus `NAME` and `PERSONA_COLS`. Which datasets are trained on, validated on or tested on is decided in the notebook.

## The one rule

`respondent_beta_part` is the only place a dataset's regression is coded. The human replication (`regression_rows` → `fit_human_ols`), each regression's σ, the training loss, and the population evaluation (`evaluation.py`) are all built from it. So the betas the model is trained toward are exactly the betas the replication reports.

## How the loss works

For each regression, a sample's betas are a fixed linear map of its option probabilities:
`betas = B @ p`, with `B = pinv(X)[target rows] @ W / sigma`.
- `X`: the sample's design rows for that regression.
- `W`: each outcome row's weights over options (an option's own probability, expected kcal, a person mean, ...).
- `sigma`: the outcome SD on the full human data.

Human betas use the same `B` with one-hot choices, so the beta error is `B @ (p - y)`.

**Loss = KL (mean over tasks) + λ × mean over regressions of each regression's mean squared standardized beta error.** Every regression counts equally, and a sample comes from one dataset only. A regression that is rank-deficient in a sample is skipped for that step and reported.

## What each dataset regresses

| Dataset | Regressions | Rows | Outcome | SEs |
|---|---|---|---|---|
| Wu 2025 | 1 | each displayed plan | that plan's choice probability | clustered by respondent |
| Berke 2023 | 4 (one per vendor) | stacked Ground/Drone rows, own − other option, plus a drone constant | that option's probability | clustered |
| Grummon 2023 calories | 2 (beverage, snack), one shared design row | respondent | calories of the chosen product as printed on its label (Σ p·kcal; see data/nutrition/grummon2023/DECISIONS.md) | HC1 |
| Grummon 2023 labels | 2 (beverage, snack) | each of the 16 displayed products; product fixed effects, arm dummies (nuisance) and 5 label targets (see data/nutrition/grummon2023/DECISIONS.md) | that product's probability | clustered |
| Kriner 2026 | 1 | each displayed flight | that flight's probability | clustered |
| COVID (Kreps 2020) | 1 | profiles A and B ("neither" has no row) | that profile's probability | clustered |
| Grummon 2025 | 1 | each of the 6 items (the paper's unit of analysis) | p(picked the healthier product) | clustered |

Each module's docstring states the paper's estimand and how ours compares. All six human replications were checked against the earlier notebook cells (betas, SEs and σ agree to about 1e-13 or better). Kriner, COVID and Grummon 2023 also match the published tables. Grummon 2025 is 0.01 below 3 of 5 published effects because the paper used mixed logit.

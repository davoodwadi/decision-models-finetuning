# Grummon 2023 decisions

## 2026-11-17: beta-loss outcome uses the printed calories, not the true calories

**Observation.** Every product in every arm shows a white front-of-package calorie label ("240 CALORIES PER
SERVING"). The beta OLS outcome (calories of the chosen product, `beta_ols/grummon2023.py`, now `grummon2023_calories.py`) used the authors' true
calories (`true_cal` / `item_cal_*`). The two differ for one of the 32 products:

| task | product | printed label | true calories |
|---|---|---|---|
| beverage | Diet Mountain Dew - 20oz | 0 | 10 |

All 16 snack labels match their true calories, and the printed labels are identical across the 5 arms.

**Decision (user).** Use the calories printed on the label, because that is the value the respondent saw and what
could affect their choice. `load_choice_df` now reads each option's kcal from its calorie label in
`option_descriptions` and raises if an option does not show exactly one calorie label.

**Effect on the human betas (HC1 OLS, n = 7,943 per task).**

| regression | term | true kcal (paper) | printed kcal (now) |
|---|---|---|---|
| grummon2023/bev | Green vs Calorie | -34.1531 | -34.1732 |
| grummon2023/bev | TrafficLight vs Calorie | -31.4715 | -31.4913 |
| grummon2023/bev | PhysicalActivity vs Calorie | -39.0136 | -39.1213 |
| grummon2023/bev | Warning vs Calorie | -28.1985 | -28.2941 |
| grummon2023/snack | all 4 terms | unchanged | unchanged |

Beverage sigma_y: 115.08 -> 115.42 kcal. Our betas therefore no longer reproduce the paper's published beverage
estimates exactly (paper: -34.2, -31.5, -39.0, -28.2 kcal); the difference is at most 0.11 kcal.

**Consequences.** The Grummon 2023 training items and step-0 caches rebuild on the next run (the module code
changed). Runs before 2026-11-17 used the true-calorie outcome.

## 2026-11-17: second coding `grummon2023_labels`: choice on the label printed on each product

**Why (user).** The calorie outcome measures how labels shift calories, not how they shift choices. The aim is to
teach the model how the treatments change what people pick, so we regress the choice itself, regardless of the
paper's published estimand. The calorie coding (then `grummon2023`, now `grummon2023_calories`) is kept unchanged as its own dataset.

**Observation.** Within an arm, the labels differ between products and are the same for every respondent:

| arm | labels on products (beverage / snack) |
|---|---|
| Calorie | none (calorie label only) |
| Green | CHOOSE OFTEN on 4 / 4 products |
| TrafficLight | every product: OFTEN 4 / 4, SOMETIMES 6 / 7, RARELY 6 / 5 |
| PhysicalActivity | every product: walking minutes (0 to 130) |
| Warning | octagons: HIGH IN SUGAR on 6 beverages; every snack has 1 or more (HIGH IN CALORIES on all 16; SUGAR 8, SATURATED FAT 6, SODIUM 5) |

An arm-level regressor alone cannot move choice shares (each respondent picks exactly one of 16 products, so the mean
of a product-row 0/1 outcome is 1/16 in every arm), so the arm must act through what each product shows.

**Coding (`beta_ols/grummon2023_labels.py`).** One regression per task on product rows (y = 1 if that product was
chosen; the model's y = its probability). Design: intercept, 15 product dummies (product fixed effects), 4 arm dummies
(nuisance: they absorb the share unlabelled products lose as a group), and 5 targets:
`Green: CHOOSE OFTEN`, `TrafficLight: CHOOSE OFTEN` and `TrafficLight: CHOOSE SOMETIMES` (reference: RARELY),
`PhysicalActivity: minutes / 100`, `Warning: octagons` (count). SEs clustered by respondent. sigma = SD of the 0/1
product rows = 0.2421. Labels are parsed from `option_descriptions`; any label an arm should not show raises.

**What it does not capture.** The poster is the same for every product in an arm, and every treatment arm shows both
its poster and its labels, so each label beta is "this label, after that arm's poster". A label is assumed to have
the same effect on every product that carries it (e.g. a poster naming soda could move sodas beyond their minutes
label; arm x product regressions would show that).

**Human betas (n = 7,943 respondents x 16 products per task).**

| term | beverage | snack |
|---|---|---|
| Green: CHOOSE OFTEN | 0.0375 (se 0.0046) | 0.0353 (0.0044) |
| TrafficLight: CHOOSE OFTEN | 0.0578 (0.0053) | 0.0507 (0.0050) |
| TrafficLight: CHOOSE SOMETIMES | 0.0093 (0.0039) | 0.0214 (0.0040) |
| PhysicalActivity: minutes / 100 | -0.0338 (0.0038) | -0.0386 (0.0047) |
| Warning: octagons | -0.0240 (0.0039) | -0.0313 (0.0026) |

**Target noise at sample size 40 tasks** (Monte Carlo, `beta_noise_by_sample_size`): signal / noise = 0.32 for both
tasks (0.33% of samples rank-deficient, i.e. an arm missing; those are skipped). The calorie coding has 0.30 / 0.17,
Wu 1.8, Berke 2.3 to 2.8. At 160 tasks: 1.4 / 1.5.

**Use.** Train on `grummon2023_labels` or `grummon2023_calories`, not both: they are the same respondents and choices, so both
together would count Grummon 2023's KL twice.

## 2026-11-17: the calorie coding is renamed `grummon2023` -> `grummon2023_calories`

**Decision (user).** To avoid mixing up the two codings, the calorie coding's module is now
`beta_ols/grummon2023_calories.py` with `NAME = "grummon2023_calories"`. Its regressions are
`grummon2023_calories/bev` and `grummon2023_calories/snack` (pseudo datasets `grummon2023_calories/bev_choice` and
`.../snack_choice`). The coding itself is unchanged. The data loaders (`data.grummon2023`,
`data/nutrition/grummon2023/`) and the prompts keep the name `grummon2023` (each row's `year` is still
`grummon2023`, so prompts are identical). Older run folders and notebooks (v5 to v8) refer to the old name
`grummon2023`; `beta_ols.DATASET_MODULES["grummon2023"]` no longer exists, so rerunning them with that name raises.

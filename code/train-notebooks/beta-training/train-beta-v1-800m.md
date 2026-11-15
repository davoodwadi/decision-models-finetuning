# train-beta-v1-800m: train on KL plus regression betas

Companion to `train-beta-v1-800m.ipynb` (currently a copy of the v5 notebook). Nothing below is implemented yet.

## Goal

The model should reproduce the regression betas of an experiment, not only each person's answer.
v3 and v5 trained on answers only, and the two did not always move together.
Here the betas become part of the training loss.

## Terms

| Term | Meaning |
|---|---|
| Row | One option shown to one respondent in one choice task. A task with two options gives two rows. |
| `y` | 1 if the respondent chose that option, else 0. |
| `p` | The model's probability that this respondent chooses that option. |
| `X` | The option's attribute levels written as 0/1 columns, plus a constant. |
| Beta | The OLS coefficients of an outcome on `X`. Human betas use `y`, model betas use `p`. |
| Beta gap | Model beta minus human beta, fitted on the same rows. |
| KL loss | The loss used today: one term per question, comparing the model's answer distribution with the human answer. |
| λ | How much the beta loss counts next to the KL loss. `λ = 0` is KL only. |
| Step | One optimizer update: 12 respondents with up to 80 questions each, as in v5. |

## The loss

In every step, for each experiment, two regressions are fitted on that step's rows: one on the human choices and
one on the model's probabilities. The loss adds the squared difference of their betas to the KL loss.

    loss = KL  +  λ · Σ over experiments of mean((model beta − human beta)²)

The KL term asks: is each answer right?
The beta term asks: for the people in this step, does price (or privacy, or renewables) move choices by the right amount?

**Why a regression on a few people is a valid target.** The betas of the 2 or 3 people in a step are not the
population betas, and they are not meant to be. The model sees those people's personas, so their own betas are the
right target for their rows. Each 0/1 choice is also a noisy draw, but that noise is zero on average, and with a
squared gap it does not bias the update. It only makes single steps jumpy.

**Why this needs no special machinery.** OLS is a fixed linear recipe: `beta = A · outcome`, with
`A = (XᵀX)⁻¹Xᵀ`. `A` depends only on which options were shown, so the beta gap is simply `A · (p − y)`.
That is an ordinary differentiable function of the model's probabilities.

## What changes from v5

| | v5 | beta-v1 |
|---|---|---|
| Trainer | `ChunkedMarginalTrainer` | new `BetaChunkedTrainer` |
| Training target | Sub-persona marginals | Each respondent's own answer |
| Persona shown | 1 to 5 attributes, or full | Always the full persona |
| Loss | KL | KL + λ · beta loss |
| Data, splits, eval sets, lr, epochs, step size | | Unchanged |

Two things change at once, so a `λ = 0` run is needed. Without it, a difference from v5 could come from dropping
marginals and not from the beta term.

## Which questions get a beta term

| Study | Experiment | Options per task | Randomized attributes | In v1 |
|---|---|---|---|---|
| wu2025 | `plan_choice`, 10 tasks per respondent | Choice 1, Choice 2 | Monthly bill, percentage of renewables, locally produced renewables | Yes |
| berke2023 | `delivery_choice_v0..v3`, 12 tasks per respondent or 3 per product type | Ground vehicle, Drone | Fee and wait time for both, privacy for the drone | Yes |
| grummon2023 | 5 label arms, one per respondent | | The arm | No, KL only |

Grummon 2023 is left out of v1 because an arm regression needs a defined outcome for its multi-product and 1 to 5
scale questions. All other questions keep the KL loss only.

Each formula is built from the study's own attributes and levels. Nothing is copied from the Kriner 2026 or
Grummon 2025 formulas, since those are the test sets.

## Implementation

### 1. The trainer: `BetaChunkedTrainer`

A subclass of `ChunkedGroupedTrainer` in a new file `training/beta_trainer.py`, in the same way
`ChunkedMarginalTrainer` extends it. Grouping, schedule, evaluation, checkpoints and resume are inherited unchanged.

It adds:

- Two required arguments: `beta_lambda` and `beta_min_rows`.
- An overridden `train_one_step` (below).
- `beta_lambda`, `beta_min_rows` and the regression names in `describe()`, `_run_signature()` and `_build_extra()`.

It reads one extra field from a training record, written by the notebook:

    rec["beta"] = {"group": "wu2025:plan_choice", "x": [[...], [...]]}

`group` names the regression the record belongs to. `x` holds one 0/1 row per option, in the order of
`option_texts`, with the constant as the first column. Records without `"beta"` are KL only.
The trainer never parses text; it only does the linear algebra.

### 2. The training step

A beta needs all rows of an experiment in the step, but the trainer backpropagates one small chunk at a time and
then frees it. So the step runs in two passes.

**Pass 1, no gradients.** For each respondent group, forward only its records that carry `"beta"`, with the same
persona text that pass 2 will use. This gives `p` for every row. Then, per regression:

    gap = A · (p − y)                  # model beta minus human beta, constant dropped
    beta_loss = mean(gap²)
    w = (2 / K) · Aᵀ · gap             # one number per row, K = number of betas

`w` tells each row how far its probability should move to close the gap.

**Pass 2, the existing chunked pass.** Unchanged, except that each chunk's loss gets one more term for its rows
that carry `"beta"`:

    chunk_loss = KL part (as today)  +  λ · Σ wᵢ · pᵢ

`w` is treated as a fixed number here. Summed over the step, this gives exactly the gradient of `λ · beta_loss`.

Details:

- **Constant.** It stays in `X` but is left out of the gap. The overall choice level is the KL term's job, and the
  evaluators also compare only the attribute betas.
- **Skipped regressions.** A regression is left out of a step when it has fewer than `beta_min_rows` rows or when
  `XᵀX` cannot be inverted (a level that nobody in the step saw). Those rows still get the KL loss.
- **Dropout.** Pass 1 runs without dropout, pass 2 with it (LoRA dropout 0.05). The gap is therefore measured on the
  predictions the model would give at inference. The mismatch between the passes is small and accepted.
- **Logging.** Per step: the beta loss and how many regressions were used or skipped. Kept on the trainer and written
  to the checkpoint metadata. The plotter is not changed.

### 3. The notebook

1. Replace `ChunkedMarginalTrainer` with `BetaChunkedTrainer`. Remove `marginal_lookup`, `k_schedule` and the
   v1/v2 marginal-grouping inspection cells, which no longer apply. Set a new `save_dir`.
2. Add one design cell that declares, per experiment: which `v` it covers, the attributes, the levels of each
   attribute and the reference level. The same cell writes `rec["beta"]` onto the matching training records.
   - wu2025: levels are read from `option_descriptions` (`"Monthly bill: …; Percentage of renewables: …; …"`).
   - berke2023: levels are read from the `qid` (`gc._gt._dc._dt._dp.`). Columns: drone (yes/no), fee, wait time,
     drone privacy.
3. Declare `BETA_LAMBDA` and `BETA_MIN_ROWS` next to `lr` and `epochs`, and pass them to the trainer.

### 4. Runs and read-out (yours)

1. One `λ = 0` run and one `λ > 0` run, same seed, same everything else.
2. Compare KL and OLS metrics of the two runs at the same steps on Kriner 2026 and Grummon 2025.
   Look at `OLS MAE` before `OLS MAPE`, because MAPE inflates when a human beta is close to zero.

## How many rows a step really has

Groups are respondents, and the three studies have about 7,900, 3,600 and 1,900 respondents. A step of 12
respondents therefore holds on average about 7 from Grummon 2023, 3 from berke2023 and 2 from wu2025:

| Regression | Tasks per step | Rows per step | Betas to fit |
|---|---|---|---|
| wu2025 `plan_choice` | about 17 | about 34 | 8 (1 local, 3 renewables, 4 bill) |
| berke2023, one product type | about 8 to 10 | about 16 to 20 | 8 (drone, 3 fee, 3 wait, privacy) |

These are averages from respondent counts, not measured on actual steps. For berke2023 per product type, rows and
betas are close in number, so many steps will be skipped or will rest on a nearly exact fit.
This is the reason for decision D1.

## Decisions for you

**D1. berke2023: how to group the product types?**
Take-out food, liquor and groceries share the same fee and wait levels. Prescription medications have their own.
1. Two regressions: the three same-level product types together, medications alone (recommended). The pooled one has
   about three times the rows, so it is skipped far less often. Its betas are the average effect over the three
   types, for both humans and model.
2. Four regressions, one per product type. Cleaner, but the beta term will often be inactive for berke2023.

**D2. `beta_min_rows`: when is a regression too small to use?**
1. At least 3 rows per beta (recommended, a judgment and not derived from data).
2. Use every regression that can be inverted, however small.

**D3. Value of λ.**
I cannot predict a good value. Rough arithmetic only: KL per question is around 0.5, and the squared beta gap on
about 34 rows is around 0.03 to 0.1 per beta from choice noise alone, so the two terms are of similar size
around λ = 5 to 20.
1. Start with λ = 10 (recommended), next to the `λ = 0` run.
2. Give me the value you want.

## Known limits

- **The beta term is mostly noise per step.** It is correct on average, but with 2 or 3 people per regression a
  single step's gap is dominated by choice noise.
- **It may crowd out KL.** The trainer clips the gradient at norm 1.0. A noisy beta term with a large λ uses part of
  that budget. Reasoned from the code, not measured; the `λ = 0` run shows it if it happens.
- **Speed.** Pass 1 adds one forward pass over the conjoint records of each step. Not measured.
- **Whether it helps is empirical.** The mechanism says the beta term targets errors that KL underweights.
  Whether that transfers to the unseen Kriner and Grummon 2025 experiments is what the two runs test.
- **Not checked yet:** that the model returns option probabilities in the order of `option_texts`. This is checked
  when the trainer is written.

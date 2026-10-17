# Plan: switchable record construction

Goal: change **what data** we train on and **how a training record is built from it** by writing a recipe file, never
by editing the trainer. Adding a new way of building records = one new class + a recipe.

## Three layers

| Layer | Job | Knows about |
|---|---|---|
| **Source** | Load respondent-level data; hold out the calibration slots; build a stored record for any set of respondents | the files on disk |
| **Construction** | Decide what one epoch passes over (`slots`) and how one record is drawn for a slot (`draw`) | a source, a random generator |
| **Trainer** (`train_issp.py`) | Model, loss, epochs, validation, run folder | only `slots`, `draw`, calibration records, test sets |

The record format between them is the stored format that already exists (persona, questions with target, n).

## Interfaces

```python
class IsspSource:                       # from IsspSampler.__init__, pool, make, calibration_records, group
    units                               # one per (year, country): attributes, answers, questions, asked_of
    slots                               # all (unit, question) slots minus the held-out ones
    held_out                            # calibration slots (hash of year/country/question, calibration_frac)
    def pool(unit, questions)           # respondents who answered
    def make(unit, questions, chosen, combo, match)   # the stored record of respondents `match`
    def calibration_records()           # fixed group-level records of the held-out slots
    def group(name, code, persona_values, questions)  # one fixed persona (notebooks)
    def files(), summary()              # for run_info.json

class Construction(Protocol):
    def slots(self, source) -> list                 # one epoch = one pass over these
    def draw(self, source, rng, slot) -> (record, info)

GroupShares(min_n=10)        # today's default branch of IsspSampler.record(); slots = those with >= min_n respondents
SingleRespondent()           # today's single_respondent branch; all source slots
FixedSubset(n, of=...)       # today's --train_slots: n slots, each drawn once by `of` with Random("train_slots")

class TrainingData:          # what the trainer gets: TrainingData(recipe)
    slots, draw(rng, slot), calibration_records(), test_sets(), summary(), files(), recipe
```

`IsspSampler` disappears; its code is split between `IsspSource` (loading, `pool`, `make`, calibration) and the
construction classes (`record()`'s two branches). Every construction gives records in the same format, so the trainer
and the loss are unchanged (a one-hot target is a distribution with all its weight on one answer).

## Recipes: `configs/data/*.json`

```json
{"source": {"issp": {"years": "all", "calibration_frac": 0.01}},
 "construction": {"group_shares": {"min_n": 10}}}
```
```json
{"source": {"issp": {"years": "all", "calibration_frac": 0.01}},
 "construction": {"single_respondent": {}}}
```
```json
{"source": {"issp": {"years": "all", "calibration_frac": 0.01}},
 "construction": {"fixed_subset": {"n": 128, "of": {"group_shares": {"min_n": 10}}}}}
```

The trainer takes `--data configs/data/<recipe>.json`. These flags leave the trainer: `--train`, `--min_n`,
`--single_respondent`, `--calibration_frac`, `--train_slots`. Training flags stay: `--base`, `--lora_rank`, `--epochs`,
`--lr`, `--lr_schedule`, `--batch`, `--augment` (the view of a record; `--p_reverse` until 2026-10-14), `--eval_batch`, `--validate_every`,
`--checkpointing`, `--seed`, `--no_eval`, `--out`. `run_info.json` stores the full recipe.

## Rules kept

- **Epoch = one pass over the construction's slots.** No construction rescales batches or epochs.
- **Calibration is a property of the source, not of the construction**: the same held-out ISSP slots and the same
  group-level calibration records whatever the construction, so runs with different constructions stay comparable.
- **SubPOP is never training data.** The JSONL training path added in `training_data.py` (only offered `subpop100`) is
  removed; no source reads SubPOP. A JSONL source for other prepared datasets is added when
  one exists.
- **Old runs stay readable**: `recipe_from_settings(settings)` translates an old `run_info.json` (flags) into its recipe,
  so the notebook replay and `show_train_samples.py` rebuild any run's data the same way.

## Steps

1. `IsspSource` + `GroupShares` + `SingleRespondent`, moved out of `IsspSampler` with the generator calls in the same
   order (old runs must replay exactly).
2. `FixedSubset`.
3. `training_data.py`: registry {name: class}, `recipe_from_settings`, `TrainingData(recipe)`; remove the
   JSONL/SubPOP training path and the dataset flags from `TrainingDataset`.
4. `train_issp.py`: `--data`; drop the dataset flags; `run_info` from `TrainingData.summary()/files()/recipe`; remove
   unused imports (`hashlib`, `SystemOneRequest`, `to_record`).
5. Callers: notebook replay cell (`TrainingData(recipe_from_settings(...))`), notebook gender cells (`IsspSource.group`),
   `show_train_samples.py` (keeps its own `--year`, `--min_n`, `--single_respondent` for viewing one year), `commands.md` (recipes instead of flags), `data/README.md`.
6. Recipes for what we run today: `issp-all-group-min10.json`, `issp-all-single.json`,
   `issp-all-group-min10-fixed128.json`.

## Checks (CPU only, scripts in the scratchpad, not in notebooks)

- Replay `issp-all-0.8b-r64-v3` (group shares) and `issp-all-0.8b-test-r64-v7` (fixed 128) through `TrainingData(recipe)`:
  every step's mean target entropy equals its training log (0 mismatches).
- Single respondent: 1,000 draws from `SingleRespondent` equal the old `IsspSampler(single_respondent=True)` draws
  with the same generator.
- Slot lists per construction equal the old ones; calibration records byte-identical (47,324 records).

## Adding a construction later

Write one class with `slots(source)` and `draw(source, rng, slot)`, register its name, write a recipe. No change to
the trainer, the source, or the other constructions.

## Decisions (2026-10-14)

- **A. Calibration**: fixed by the source for every construction.
- **B. Choosing data**: recipe files.
- **C. Script name**: keep `train_issp.py` for now.

## Status

Done 2026-10-14 (steps 1-6); all checks above pass (`data/issp2020/DECISIONS.md`, decision 29).

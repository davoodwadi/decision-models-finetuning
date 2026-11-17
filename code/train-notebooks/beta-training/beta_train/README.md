# `beta_train`: training, validation, and diagnostics for beta-training notebooks

This is the model side; `../beta_ols/` is the data and OLS side. Nothing in this package reads notebook globals: the
model, items, specs, validators, weight snapshot and `TrainConfig` are always passed in, and every `RunResult`
records the config that produced it.

## Layout, in the order a notebook uses it

| File | Use |
|---|---|
| `config.py` | `TrainConfig`: every setting that varies between runs (validated on creation); `scale_lr_for_sample_size(base_lr, S)` |
| `model_state.py` | `snapshot_trainable`, `reset_trainable` (every run starts from the snapshot), `set_checkpointing` |
| `items.py` | `build_items(prepare_fn, modules, choice_dfs)`: tokenizes each respondent and checks Kev's targets against their `beta_ols` beta part; `build_items_cached(model, modules, choice_dfs, cache_dir)` (same, cached on disk per dataset); `items_summary` |
| `cache.py` | Disk cache keyed by everything a result depends on (data fingerprint, source files of the code, settings), so stale results are never loaded |
| `validation.py` | Two validators with one interface (`.validate(model) -> (metrics, beta_df)`), both scoring population betas via `beta_ols.evaluation`: `PopulationValidator(module, choice_df, specs, max_questions, max_chars)` ("fused": LoRA merged, fused kernels, CUDA graphs, shared persona states; the two budgets set the outer batches) and `TrainingPathValidator(module, choice_df, specs, items, eval_chunk)` ("training_path": the training forward in eval mode, LoRA unmerged; `eval_chunk` items (rows) per pass sets memory). `validate_all`, `validate_all_cached` (untrained step-0 validation, cached on disk per validator and kind; resets to the snapshot first), `validation_metrics_frame` |
| `diagnostics.py` | `gradient_check` (two-pass against one big graph), `find_max_micro_chunk` (VRAM search over items per micro-chunk: a size fits only if every worst-case chunk fits, ranked by padded tokens = items × longest state + questions × longest branch, from one dataset at a time) |
| `trainer.py` | `run_training(...) -> RunResult`: writes the run folder at every validation. An existing run in `run_dir` is resumed (a finished one is loaded), a missing one created; `from_scratch=True` deletes it and trains again. `load_run(run_dir) -> RunResult` reads a saved run back |
| `runs.py` | Run folders: validation loss, stats CSVs, betas per validation, `best.pt`, `last.pt` (resume state); `load_best(run_dir)` |
| `reports.py` | `comparison_table`, `per_dataset_table`, `eval_beta_table`, `mse_decomposition` (final step); `best_table`, `best_beta_table` (best step) |
| `batching.py`, `step.py`, `live_display.py` | Internals: samples (items of one dataset; an item is one data row = one answered task), micro-chunks of M items, the two-pass gradient step, the live table and plots |

## A training run

1. **Start.** Reset the weights to the snapshot, seed the RNGs, and set gradient checkpointing.
2. **Each epoch.** Shuffle every training dataset's respondents and cut them into samples of that dataset's size, then shuffle the order of all samples. Every respondent is seen once per epoch, and each sample comes from one dataset.
3. **Each step:**
   - build the sample's beta operators;
   - compute the exact full-sample gradient by two-pass micro-chunking (`step.py` explains why it's exact);
   - clip, take an AdamW step, and advance the LR schedule (OneCycle or constant).
4. **Validation.** Each validator runs on its own interval (`eval_every_by_validator`, e.g. `{"kriner2026": 20, "grummon2025": 100}`), and all of them run at step 0 and the last step. There's one row per step where any validator ran; a validator that didn't run at that step has NaN in its columns, and the live plots connect only each validator's own points. The best checkpoint can only change at steps where the best metric's validator ran. Rows go to the live table in this column order: train loss, each validator's val_loss / std beta MSE / KL / beta MSE / MAE / MAPE / ρ, lr, eval times, train time.
5. **Run folder.** After each validation the run folder is written (below), and `best.pt` is replaced when the best metric improves.
6. **Held-out check of `best.pt`.** When training ends, `best.pt` is loaded and validated on every validator that didn't run at its step, and then the final weights are put back. This costs one extra validation per such validator. `best_validation` / `best_row` (and `best_table`, `best_beta_table`) then hold exact values for every validator.
7. **Cleanup.** A `try/finally` frees gradients, the optimizer and GPU memory, also after an error or interrupt, and marks an unfinished run `interrupted`.

## Validation loss

Each validator's `val_loss` uses the run's training-loss form: KL + `beta_lambda` × std beta MSE. Std beta MSE is computed per regression as the mean of ((model β − human β) / σ)², then averaged over regressions, in the same units and weighting as the training beta loss. Validators are never pooled. The best checkpoint is picked on one validator: `best_metric` must be one of that validator's columns, e.g. `"kriner2026/val_loss"` or `"kriner2026/beta_mse"`, with `best_mode` `"min"` or `"max"`. The other validators are not used for selection, so their values at the best step are a held-out check.

## Run folder (`run_dir`, declared in the notebook)

```
run.json                 identity (config, run_metadata, item and validation-data fingerprints, trainable shapes,
                         best_metric / best_mode), git commit, status (running / interrupted / finished),
                         resume history, final summary
steps.csv                one row per optimization step
val_history.csv          one row per validation step (step 0 included); NaN for a validator that didn't run there
betas/step_000124.csv    the beta table of every validator that ran at that step
betas/best_step_000120.csv  every validator's betas for best.pt (end-of-run check for the ones that didn't run there)
best_validation.csv      one row: every validator's columns for best.pt
checkpoints/best.pt      trainable weights at the best validation so far, with {metric, mode, step, value}
checkpoints/last.pt      resume state at the last validation: weights, AdamW, LR scheduler, RNG states (Python,
                         torch, CUDA), the current epoch's sample plan, all rows so far, best so far
```

- **Default: resume or create.** If `run_dir` holds a run (a `run.json`), it restarts from `last.pt`, i.e. the last validation, and re-runs the steps after it with the same samples and dropout. A finished run is loaded instead of trained. If `run_dir` doesn't exist, the run is created. Everything in the identity must match for a resume. The only exceptions are `micro_chunk` and `gradient_checkpointing`, which may change (e.g. a smaller micro-chunk after an OOM); the change is logged in `resumes`.
- **`from_scratch=True`:** deletes the run in `run_dir` and trains it again from the snapshot weights. Only a folder with a `run.json` is ever deleted; a folder without one raises an error.
- `last.pt` is about 3× the trainable weights (weights + two AdamW moments), and `best.pt` about 1×. Checkpoints are git-ignored.

## Notebook pattern

```python
INIT_TRAINABLE_STATE = snapshot_trainable(kev_model)
validators = {name: PopulationValidator(beta_ols.DATASET_MODULES[name], choice_dfs[name], specs_by_dataset[name],
                                       max_questions=VALIDATION_MAX_QUESTIONS, max_chars=VALIDATION_MAX_CHARS)
              for name in VALIDATION_DATASETS}
step0_validation = validate_all(kev_model, validators)
unified_items = build_items(kev_model.prepare, {name: beta_ols.DATASET_MODULES[name] for name in ITEM_DATASETS}, choice_dfs)
BASE_CONFIG = TrainConfig(...)
runs["lambda=5"] = run_training(model=kev_model, items_by_dataset=unified_items, specs_by_dataset=regression_specs_by_dataset,
                                validators=validators, step0_validation=step0_validation,
                                init_snapshot=INIT_TRAINABLE_STATE, config=replace(BASE_CONFIG, beta_lambda=5.0),
                                device=device,
                                run_dir=RUNS_DIR / "lambda=5", best_metric="kriner2026/val_loss", best_mode="min",
                                run_metadata=RUN_METADATA)   # resumes RUNS_DIR/lambda=5 if it exists; from_scratch=True restarts
display(comparison_table(runs, step0_validation))
display(best_table(runs))
runs = {run_dir.name: load_run(run_dir) for run_dir in sorted(RUNS_DIR.iterdir())}   # later, from disk
```

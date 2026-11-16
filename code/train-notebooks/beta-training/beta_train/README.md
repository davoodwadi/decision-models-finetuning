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
| `validation.py` | Two validators with one interface (`.validate(model) -> (metrics, beta_df)`), both scoring population betas via `beta_ols.evaluation`: `PopulationValidator(module, choice_df, specs, max_questions, max_chars)` ("fused": LoRA merged, fused kernels, CUDA graphs, shared persona states; the two budgets set the outer batches) and `TrainingPathValidator(module, choice_df, specs, items, eval_chunk)` ("training_path": the training forward in eval mode, LoRA unmerged; `eval_chunk` respondents per pass sets memory). `validate_all`, `validate_all_cached` (untrained step-0 validation, cached on disk per validator and kind; resets to the snapshot first), `validation_metrics_frame` |
| `diagnostics.py` | `gradient_check` (two-pass against one big graph), `find_max_micro_chunk` (VRAM search: a size fits only if every worst-case chunk fits, ranked by padded tokens = states × longest state + question rows × longest branch, from one dataset at a time) |
| `trainer.py` | `run_training(...) -> RunResult` |
| `reports.py` | `comparison_table`, `per_dataset_table`, `eval_beta_table`, `mse_decomposition` |
| `batching.py`, `step.py`, `live_display.py` | Internals: samples and chunks, the two-pass gradient step, the live table and plots |

## A training run

1. **Start.** Reset the weights to the snapshot, seed the RNGs, and set gradient checkpointing.
2. **Each epoch.** Shuffle every training dataset's respondents and cut them into samples of that dataset's size, then shuffle the order of all samples. Every respondent is seen once per epoch, and each sample comes from one dataset.
3. **Each step:**
   - build the sample's beta operators;
   - compute the exact full-sample gradient by two-pass micro-chunking (`step.py` explains why it's exact);
   - clip, take an AdamW step, and advance the LR schedule (OneCycle or constant).
4. **Validation.** Every `eval_every` steps and at the last step, every validator runs. Rows go to the live table in this column order: train loss, each validator's KL / beta MSE / MAE / MAPE / ρ, lr, eval times, train time.
5. **Cleanup.** A `try/finally` frees gradients, the optimizer and GPU memory, also after an error or interrupt.

## Notebook pattern

```python
INIT_TRAINABLE_STATE = snapshot_trainable(kev_model)
validators = {name: PopulationValidator(beta_ols.DATASET_MODULES[name], choice_dfs[name], specs_by_dataset[name],
                                       max_questions=VALIDATION_MAX_QUESTIONS, max_chars=VALIDATION_MAX_CHARS)
              for name in VALIDATION_DATASETS}
step0_validation = validate_all(kev_model, validators)
unified_items = build_items(kev_model.prepare, {name: beta_ols.DATASET_MODULES[name] for name in ITEM_DATASETS}, choice_dfs)
BASE_CONFIG = TrainConfig(...)
runs["lambda=5"] = run_training(model=kev_model, items_by_dataset=unified_items, specs=unified_specs,
                                validators=validators, step0_validation=step0_validation,
                                init_snapshot=INIT_TRAINABLE_STATE, config=replace(BASE_CONFIG, beta_lambda=5.0),
                                device=device)
display(comparison_table(runs, step0_validation))
```

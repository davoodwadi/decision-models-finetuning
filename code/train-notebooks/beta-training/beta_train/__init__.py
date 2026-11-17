"""Beta-training training package (model side; `beta_ols` is the data/OLS side).

Modules, in the order a notebook uses them:
    config        TrainConfig (every run setting), scale_lr_for_sample_size
    model_state   snapshot_trainable, reset_trainable, set_checkpointing
    items         build_items, build_items_cached (tokenize data rows into training items), items_summary
    validation    PopulationValidator (fused) / TrainingPathValidator (training forward), validate_all, validate_all_cached, validation_metrics_frame
    diagnostics   gradient_check, find_max_micro_chunk
    trainer       run_training -> RunResult (writes the run folder; resume=True continues it), load_run
    runs          run folders: validation loss, stats CSVs, betas per validation, best.pt, last.pt (resume state)
    reports       comparison_table, per_dataset_table, eval_beta_table, mse_decomposition, best_table, best_beta_table
Internals: cache (disk cache keyed by data + code + settings), batching (samples, chunks), step (predict_chunk, two_pass_gradients), live_display (live table + plots).
Nothing reads notebook globals: the model, items, specs, validators, snapshot, and config are passed in.
"""

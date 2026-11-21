# LLM vs. Untrained Decision Model Baseline (`Vaccines`, 200-Respondent Subsample)

Side analysis supporting the motivating argument in the Introduction of `manuscript-training.qmd`: off-the-shelf generative LLMs misestimate human conjoint AMCEs, while switching to an untrained single-pass decision model (`KEV`) slashes econometric error dramatically (with continuous probability readouts driving the gain).

## Inputs (Read-Only)
- `code/proprietary-llms/gemini/data_sample1000/*.jsonl`: 8 saved Gemini runs on the 200-respondent (`n = 1,000` choice sets, `seed = 42`) subsample of Kreps et al. (2020), matching Wang (2026)'s sample size.
- `beta_ols.covid`: exact unstandardized Table 3 Model 1 OLS AMCE operator (11 treatment slopes, natural probability-point units).
- Pinned untrained KEV checkpoints (`jaredpalmer/kev-{0.8b,4b,9b}` at `temperature = 1.0`, `bf16`), identical to `03-analysis-paper-microdata.ipynb`.

## Workflow (Inference Separated from Analysis)
1. **Base KEV GPU inference (run once):** `predict_kev.py` runs the released base checkpoints (`jaredpalmer/kev-{0.8b,4b,9b}`) at their shipped calibrated temperatures (`T_cal` from `head.pt`) and at raw `T = 1.0`, saving per-task predictions to `preds/kev-{0.8b,4b,9b}_covid_sample200.parquet` and `preds/meta.json`.
2. **Qwen3.5 LLM GPU inference (run once):** `predict_qwen.py` runs autoregressive generation (`model.generate`, no logit masking) for `Qwen/Qwen3.5-{0.8B,4B,9B}` using the exact `data_sample1000/` Gemini Argon system/user prompt structure (`--thinking off` and/or `on`), progressively persisting JSONL files (`preds/{slug}_{THINKING_OFF|ON}_{persona}.jsonl`) and `preds/qwen_meta.json` in the same schema as Gemini so the analysis notebook applies the exact same A/B/C parser and invalid/refusal accounting to both LLM families.
3. **CPU analysis:** `01-gemini-vs-kev-covid-ols.ipynb` loads only `preds/` and `code/proprietary-llms/gemini/data_sample1000/` (no GPU or model loading), scores all models with `beta_ols.covid`, and writes `results_summary.csv`, `betas_per_term.csv`, and `fig_llm_vs_kev_covid.png`.

## Outputs
- `predict_kev.py`: dedicated GPU inference script for base KEV models.
- `predict_qwen.py`: dedicated GPU inference script for Qwen3.5 LLMs with the exact Gemini Argon prompt structure.
- `preds/`: saved per-task probability/logit parquets and metadata JSONs.
- `01-gemini-vs-kev-covid-ols.ipynb`: pure CPU analysis notebook.
- `results_summary.csv`: OLS $\beta$ MSE ($\times 10^3$), MAE, MAPE (%), Spearman $\rho$, sign accuracy, opt-out share, and refusal count for every model and ablation.
- `betas_per_term.csv`: all 11 OLS AMCE slopes per model vs. full-sample human ground truth.
- `fig_llm_vs_kev_covid.png`: motivating comparison figure (OLS $\beta$ MSE across models + per-attribute AMCE recovery).

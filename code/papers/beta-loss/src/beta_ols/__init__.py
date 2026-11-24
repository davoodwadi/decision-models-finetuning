"""Beta-training OLS package: one module per dataset, a shared core, and the unified training loss.

Every dataset module exposes the same interface:
    NAME, PERSONA_COLS
    load_choice_df() -> task rows (one row per answered task, each respondent's tasks in prompt order)
    respondent_beta_part(r_df) -> RespondentBetaPart   (the one coding used by the human OLS and the loss)
    regression_rows(choice_df) -> one row per regression observation (CASEID, group, design columns, y)
    regression_specs(rows) -> {group: RegressionSpec}
Shared helpers (core): fit_human_ols, wide_ols_table, beta_noise_by_sample_size, ols_projector.
Training loss (unified_loss): build_sample_beta_operators, compute_unified_loss.
No model or tokenizer dependency: tokenizing respondents into training items lives in `beta_train.items`.
Population evaluation (evaluation): build_population_eval, flat_probs_from_predictions, compare_population_betas.
"""

from beta_ols import berke2023, covid, grummon2023_calories, grummon2023_labels, grummon2025, kantorowicz2026, kriner2026, magistro2025, wu2025

# Every dataset module, by name. Which ones are trained on, validated on, or tested on is decided in the notebook.
DATASET_MODULES = {
    module.NAME: module
    for module in (
        wu2025,
        berke2023,
        grummon2023_calories,
        grummon2023_labels,
        kriner2026,
        covid,
        grummon2025,
        kantorowicz2026,
        magistro2025,
    )
}

# test-best: best.pt of every paper run on Grummon 2025 and COVID

One notebook per model (`04-test-best-800m`, `04-test-best-4b`, `04-test-best-9b`), run locally on the runs pulled from
Rorqual. Each loads the base model once, then for every finished run in `runs-rorqual/<model>/` loads `checkpoints/best.pt` (LoRA + pointer head, picked during training by
`kriner2026/beta_mse`) and evaluates it on the test datasets in `TEST_DATASETS`, with the same fused population
validation used during training.

- Test datasets: `grummon2025` and `covid` (Kreps et al. 2020 vaccine conjoint; 1,971 respondents, 9,855 tasks, 11 Table 3
  terms). Neither is trained on, and neither picks the checkpoint.
- Grummon 2025 at best.pt is also in each run's `best_validation.csv` (checked at the end of training); this notebook
  computes it again from best.pt with the same validator, next to COVID.
- Row 1 (untrained model): `runs-rorqual/<model>/row1_base_untrained/test_best/` (step-0 results, cached in `code/.cache/beta_training`).
- Output per run: `runs-rorqual/<model>/<run>/test_best/`
  - `metrics.csv`: one row per test dataset (`kl_loss`, `beta_mse`, `beta_mae`, `beta_mape_pct`, `spearman_rho`, timings).
  - `betas.csv`: model and human population betas per term, stacked with a `validator` column.
  - `test.json`: notebook, model, paper row, checkpoint step, best metric and value, test datasets.
- Unfinished runs are skipped. A run already tested at the same checkpoint step on the same test datasets is skipped, so
  rerunning a notebook tests only the runs that finished since.
- Before running: `./pull-rorqual-runs.sh` (repository root) copies each run's files and `checkpoints/best.pt` (not
  `last.pt`) into `runs-rorqual/`. A later pull does not touch `test_best/` (it exists only locally).

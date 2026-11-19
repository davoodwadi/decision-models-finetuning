# test-best: best.pt of every paper run on Grummon 2025 and COVID

One notebook per model (`04-test-best-800m`, `04-test-best-4b`, `04-test-best-9b`), run locally on the runs pulled from
Rorqual and Nibi. Each loads the base model once, then for every finished run in its runs folders (`RUNS_DIR`, or
`RUNS_DIRS` for 9B) loads `checkpoints/best.pt` (LoRA + pointer head, picked during training by
`kriner2026/beta_mse`) and evaluates it on the test datasets in `TEST_DATASETS`, with the same fused population
validation used during training.

- Test datasets: `grummon2025` and `covid` (Kreps et al. 2020 vaccine conjoint; 1,971 respondents, 9,855 tasks, 11 Table 3
  terms). Neither is trained on, and neither picks the checkpoint.
- Grummon 2025 at best.pt is also in each run's `best_validation.csv` (checked at the end of training); this notebook
  computes it again from best.pt with the same validator, next to COVID.
- Row 1 (untrained model): `runs-rorqual/<model>/row1_base_untrained/test_best/` (step-0 results, cached in `code/.cache/beta_training`).
- Output per run: `<runs folder>/<model>/<run>/test_best/`
  - `metrics.csv`: one row per test dataset (`kl_loss`, `beta_mse`, `beta_mae`, `beta_mape_pct`, `spearman_rho`, timings).
  - `betas.csv`: model and human population betas per term, stacked with a `validator` column.
  - `test.json`: notebook, model, paper row, checkpoint step, best metric and value, test datasets.
- Unfinished runs are skipped. A run already tested at the same checkpoint step on the same test datasets is skipped, so
  rerunning a notebook tests only the runs that finished since.
- Runs folders: 0.8B and 4B in `runs-rorqual/<model>/`. 9B in two: Rows 2 and 2 ablation in `runs-rorqual/kev-9b/`
  (trained on Rorqual), Rows 2b to 5 in `runs-nibi/kev-9b/` (trained on Nibi). A run name found in both stops the
  9B notebook, since it would be two copies of the same regime.
- Before running: `./pull-rorqual-runs.sh` and, for 9B, `./pull-nibi-runs.sh` (repository root) copy each run's files and
  `checkpoints/best.pt` (not `last.pt`) into `runs-rorqual/` and `runs-nibi/`. A later pull does not touch `test_best/`
  (it exists only locally).

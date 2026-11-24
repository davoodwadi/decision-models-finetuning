# Manuscript-training assets

One notebook per figure or table in `manuscript/sim-respondents/manuscript-training.qmd`, named by its Quarto label: `<label>.ipynb`
writes one asset into this folder:

- figures: `<label>.png` (what the manuscript shows) and `<label>.pdf`
- tables: `<label>.md`, pulled into the manuscript with `{{< include ... >}}`

Nothing is copied into `manuscript/`. `manuscript/sim-respondents/manuscript-training-assets` is a symlink to this
folder, and the manuscript reads `manuscript-training-assets/<label>.png` (or `.md`) through it, so `quarto preview`
(which serves only `manuscript/sim-respondents/`) can show the figures. Run a notebook from this folder; it finds `microdata_analysis.py` in `../src/`.

## Where to edit what

| What | Where |
|---|---|
| Wording, labels, colours, sizes, row/column order | the "Labels and formatting" cell of the notebook |
| Which runs are read (run folders, models, regimes, validators) | the "Run design" cell of the notebook |
| Caption and table note | `manuscript-training.qmd` (not in the notebook) |
| How runs and `test_best/` evaluations are loaded | `../src/microdata_analysis.py` (numbers only, no labels) |

## Assets

| Label | Kind | Reads |
|---|---|---|
| `fig-intro-off-the-shelf` | figure | `../analysis/llm-baseline/results_summary.csv` |
| `tbl-datasets` | table | dataset files through `beta_ols` (respondents, tasks, $d$) |
| `tbl-results-mse` | table (grid) | saved runs |
| `fig-results-overall` | figure | saved runs |
| `fig-results-laptops-slopes` | figure | saved runs |
| `fig-results-vaccines-slopes` | figure | saved runs |
| `apptbl-llm-baseline` | table | `../analysis/llm-baseline/results_summary.csv` |
| `apptbl-covariates` | table | dataset modules through `beta_ols` (`PERSONA_COLS`, country excluded) |
| `apptbl-hyperparameters` | table | saved runs (`run.json`) and fixed settings in `beta_train` / `kev` |
| `appfig-flights-slopes` | figure | saved runs |
| `appfig-training-curves` | figure | saved runs (`steps.csv`, `val_history.csv`) |
| `apptbl-vaccines-slopes` | table | saved runs |
| `apptbl-food-slopes` | table | saved runs |
| `apptbl-consumer-rights-slopes` | table | saved runs |
| `apptbl-ai-offshoring-slopes` | table | saved runs |
| `apptbl-flights-slopes` | table | saved runs |
| `apptbl-magistro-subsets` | table | saved runs |

`fig-pipeline` and `appfig-input-vaccines` are TikZ drawings. The manuscript points at
`manuscript/sim-respondents/diagrams-tikz/fig_pipeline_v8.pdf` and `manuscript/sim-respondents/diagrams-tikz/fig_input_vaccines.png`; the build command
is in each `.tex` header.

`results_summary.csv` comes from `../analysis/llm-baseline/01-gemini-vs-kev-covid-ols.ipynb`; rerun that first when the LLM
scoring changes.

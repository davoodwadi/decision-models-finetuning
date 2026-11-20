# gemini-4-carbon-beta-test

Off-the-shelf baseline for the paper: does a frontier LLM, given the same persona and choice task as KEV, reproduce
the human population OLS betas? Scored with the same `beta_ols` operators as KEV's `test_best` results, so the
numbers sit next to the KEV rows in `03-analysis-paper`.

## Data

- Flights (`kriner2026`, 5,000 tasks), Vaccines (`covid`, 9,855 tasks), Food labels (`grummon2025`, 33,798 tasks).
  Every task of every respondent: `build_population_eval` needs one prediction per task.
- Prompts are built from each dataset's long parquet (`kriner2026_US_long.parquet`, `covid_US_long.parquet`,
  `grummon2025_US_long.parquet`): the rows `load_choice_df` keeps, one row per task. No KEV record or view code.
  The human choice (`answer_code`, `answer_text`) is never shown.

## Prompt

- System instruction: one framing sentence, then one line per persona column (`<label>: <value>`), in the dataset's
  persona-column order, skipping missing values. Labels come from `data.persona.PERSONA_NAMES` (Flights, Vaccines)
  and `data.grummon2025.GRUMMON2025_PERSONA_NAMES` (Food labels). The country column is left out.
- User message: the parquet `question`, then one line per option: `<option_code>. <option_text>: <option_description>`,
  or `<option_code>. <option_text>` when the description is missing (opt-outs).
- Text is used exactly as stored, including typos ("Sutainable") and Grummon's "Front-of-package label: None" for
  products without a badge.
- No answer instruction in the prompt: the answer format is set only by the response schema.
- One call per task. No memory of the respondent's other tasks.
- A task a respondent saw twice is asked twice (keys carry the occurrence number).

## Answer and scoring

- The answer is restricted by a JSON schema whose `choice` field is an enum of the task's option codes ("1", "2",
  "3"), so no answer needs parsing or dropping.
- One sample per task, coded one-hot. Population betas are linear in the choice probabilities (β = B·p), so this is an
  unbiased estimate of Gemini's population β.
- It adds sampling noise that KEV's exact probabilities do not have, which raises β MSE. For a like-for-like
  comparison, KEV should also be scored from one sampled choice per task (not in this notebook).
- Choice KL is not reported: with one-hot predictions it is infinite whenever Gemini's choice differs from the human's.
- Sampling settings are the API defaults; the thinking level is set per cell (None = model default). Nothing is tuned.

## Platform

- Calls go to Gemini Enterprise Agent Platform (formerly Vertex AI), global endpoint, not Google AI Studio (too slow).
- The client signs in with Application Default Credentials (`gcloud auth application-default login`) and bills the
  Google Cloud project in `GOOGLE_CLOUD_PROJECT`, which must be set in the kernel's environment.
- `run.json` records the platform and location.

## Cost

- `PRICES_PER_MILLION_TOKENS` in the design cell: list price on the global endpoint (standard pay-as-you-go, not launch promotions) in USD per
  1M tokens: gemini-4-carbon 4.00 / 20.00, gemini-4-neon 3.00 / 15.00, gemini-4-hydrogen 1.50 / 7.50. Taken from
  Google's Neon launch post and third-party pricing pages on 2026-11-19; check Google's pricing page before running.
- Each answer stores `cost_usd` = prompt tokens × input price + (output + thinking tokens) × output price, together
  with the two prices used. This is the list-price cost, not what a launch promotion actually bills.
- The "Estimate cost" cell reads every run folder so far and reports, per run and dataset: mean tokens, cost so far,
  mean cost per task, per respondent, and that mean times the task count of the 200 Argon respondents (Vaccines,
  1,000 tasks) and of the full dataset.
  With few answers the estimate is rough: thinking tokens vary a lot between tasks.
- A task whose answer is invalid (not one of the codes, or an incomplete response) is billed by Google but not saved,
  so it is not in the recorded cost; it is reported and asked again on the next run.

## Running in steps

- Every query and scoring cell passes its own settings: `model`, `thinking_level` (None = model default, or
  "MINIMAL", "LOW", "MEDIUM", "HIGH") and, for queries, `caseids` and `max_concurrency`.
- Each (model, thinking level) has its own folder, so answers from different settings never mix.
- A query asks every task of the respondents in `caseids` (the parquet's `CASEID` values). For Vaccines the cell
  before the query supplies the 200 respondents of the earlier Gemini 4 Argon runs (`data_sample1000/`): seed 42 draws
  200 of the sorted respondent ids, as in `gemini-3.5-flash-lite.ipynb`; the parquet's `respondent` column is the same
  id (`CASEID = covid_<respondent>`), with the same tasks in the same order. This gives Carbon and Argon the same
  respondents and tasks, but the prompt wording differs (Argon used Wang (2026) wording built from the Stata file).
- Every query reads `<dataset>.jsonl` first and only asks tasks not in it, so rerunning a cell (or the whole notebook after a restart) never asks a task twice.
- A failed task does not stop the others; every finished answer is saved, then the cell raises with the failed keys.
- A scoring cell reads the answers from disk and needs every task of its dataset answered; until then it prints how
  many are missing and skips.

## Timing

- `call_seconds` (per answer): time of the one request that returned the answer, from sending it to receiving the full
  response, measured with `time.perf_counter()` (monotonic). This is the time per sample to report.
- `task_seconds` (per answer): from the moment the task gets a concurrency slot until its answer arrives, so it adds
  failed tries and the waits between retries but not queueing behind other tasks. `attempts` counts the tries.
- `started_at_utc` and `max_concurrency` are stored with every answer: speed depends on load and time of day.
- `<dataset>_query_runs.jsonl` (one line per query cell run): wall-clock `run_seconds`, tasks asked, answered and
  failed, and `max_concurrency`, i.e. what a researcher actually waits.
- No HTTP timeout is set: a very long call (high thinking) is waited for, not cut off.
- The "Estimate cost" cell also reports, per run: timed answers, mean / median / 90th percentile / max `call_seconds`,
  mean `task_seconds`, and estimated hours for the 200 Argon respondents and the full dataset: mean `call_seconds` x
  tasks, the summed time of the calls. The mean, not the median, is used because total time is a sum over tasks.
- Answers saved before timing was added (the Argon runs, the first Carbon runs) have no timings.

## Outputs (`data_beta_full/<model>_<thinking>/`)

- `run.json`: model, thinking level, price, prompt source, answer format (written by every query).
- `<dataset>.jsonl`: one answer per task (choice, raw response, thought summary, token counts, prices, cost, timings),
  appended as calls finish.
- `<dataset>_query_runs.jsonl`: one line per query run (wall-clock time, tasks asked/answered/failed, concurrency).
- `<dataset>_metrics.csv`: β MSE, MAE, MAPE, Spearman ρ, model, thinking level, task and token counts, total cost.
- `<dataset>_betas.csv`: Gemini and human population betas per term.

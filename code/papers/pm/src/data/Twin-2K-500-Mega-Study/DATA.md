# Twin-2K-500-Mega-Study data notes

## Source and sample

Li et al. (2025), "LLM Digital Twin 2K-500 Mega-Study", hosted at <https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study> (Apache-2.0). The dataset re-surveys $N = 1,788$ US Prolific panelists (`PID` values `pid_1` through `pid_2058`) from the original Twin-2K-500 cohort ($N = 2,058$) across 19 independent follow-up behavioral studies, yielding 13,816 respondent-study completions (median 8 studies per respondent, range 1 to 18). Unlike Twin-2K-500, this release omits pre-built catalog JSONs and flat response CSVs, shipping raw per-respondent Qualtrics survey flow trees in 19 split Parquet files under `hf/data/*.parquet`. Respondent demographics are omitted in 18 of 19 splits (`promiscuous_donors` alone re-asks six demographic items) because 100% of Mega-Study `PID`s link directly to Twin-2K-500 waves 1 to 3 (`CASEID = "twin_<pid>"`), providing identical 14-attribute global personas across both datasets.

## Raw files and survey flow schema

| Artifact | Shape | Content |
|---|---|---|
| `hf/data/*.parquet` | 19 files (13,816 rows $\times$ 3 columns) | Per-study `PID`, `survey_json_with_human_response`, and `survey_text` |
| `question_audit.csv` | 554 audit rows (1,748 `(study, QuestionID)` slots) | Manual verification decisions (`include`, `kev_type`, `context_source`, `context_sep`, `drop_options`, `review_note`) |
| `question_overrides.json` | 10 override entries | Per-question stem, sub-item, and option-label overrides (reconstructing all 10 flattened HTML product tables in `context_effects` from `survey.qsf`, including `"Molte"` to `"Molter"` in `QID544`) |
| `question_tags.json` | 61 tagged entries | Shared editable audit and reference tags (`has-dont-know`, `html-table-flat`, `odd-col-nums`, `prior-priming`, `prior-task-dep`, `backref`, `img-ref`, `empty-stem`, plus custom reviewer tags) |
| `twin2k_megastudy_US_long.parquet` | $287,535 \times 24$ | Exported respondent-question long table matching the `twin2k_US_long.parquet` schema |

Each `survey_json_with_human_response` string parses to a dictionary with exact top-level keys `Elements`, `EmbeddedData`, and `Metadata`. Unlike the flat block lists in Twin-2K-500, `Elements` is a recursive Qualtrics flow tree containing `Block` leaves alongside `Branch`, `Group`, and `Randomizer` container nodes whose child elements reside in `Elements`. Recursive traversal across all 13,816 rows recovers 1,748 unique `(study, QuestionID)` catalog entries with zero duplicate `QuestionID`s per respondent. All text fields normalize HTML entities and `<br>` line-break tags into single spaces.

## Catalog to response mapping by question type

| Family | `QuestionType` | `Selector` | Catalog slots | Included slots | Exported `v` count | Mapping and inclusion notes |
|---|---|---|---:|---:|---:|---|
| `1_Matrix_Bipolar` | `Matrix` | `Bipolar` | 12 | 12 | 1,019 | Single-row 6-point or 7-point ordinal rating scales (`Columns` holds the scale labels, `kev_type = "score"`), expanding across randomized conjoint donor vignettes in `promiscuous_donors`. |
| `2_Matrix_Likert` | `Matrix` | `Likert` | 31 | 29 | 190 | Multi-row ordinal Likert batteries (`kev_type = "score"`). Uses `RowsID` when present and 1-based row indices on the 22 batteries where Qualtrics omits `RowsID`. Excludes 2 items (`affective_priming:QID56` and `digital_certification:QID121`) that reference prior open-ended essays. |
| `3_MC_IdeaEval` | `MC` | `SAVR` | 1,200 | 1,200 | 1,200 | Six experimental arms (`1..200_QID7`, `QID13`, `QID14`, `QID23`, `QID24`, `QID25`) rating 200 smartphone app ideas (`kev_type = "score"`). Strips piped serial order (`Idea #k out of 20:` to `Idea:`) and prepends the arm prompt. |
| `4_MC_Single` | `MC` | `SAVR`, `SAHR`, `DL` | 218 | 208 | 458 | Single-select items across 15 studies, expanding randomized vignettes into 106 `choice` and 352 `score` variants. Excludes 10 items referencing missing images (`consumer_minimalism:QID18`, `QID20`), prior continuous tasks, or unspecified earlier stimuli. |
| `5_MC_Multi` | `MC` | `MAVR` | 18 | 0 | 0 | Multi-select check-all-that-apply items, incompatible with single-distribution Kev targets. |
| `0_Consent_Check` | `MC` | `SAVR`, `SAHR` | 19 | 0 | 0 | Seven study consent items and 12 attention, comprehension, or honesty checks. |
| `0_Demographics` | `MC` | `SAVR`, `DL` | 6 | 0 | 0 | Re-asked demographics in `promiscuous_donors`; canonical Wave 1 to 3 personas are used instead. |
| `0_DB`, `0_TE`, `0_Slider`, `0_CS` | `DB`, `TE`, `Slider`, `CS` | Various | 244 | 0 | 0 | Descriptive stimulus blocks (177), open text entries (49, including `political_persuasion`), continuous sliders (10), and constant-sum percentage allocations (8). |

## Manual verification and Kev export

Every discrete slot is controlled by `question_audit.csv`, which maps `context_source` values (`""`, `"in_block_db"`, `"qid:<QID>[+<QID2>]"`, or `"prefix:<text>"`) and optional `|`-separated `drop_options` into self-contained question stems. Because several studies randomize stimuli inside the same Qualtrics `QuestionID` (e.g., `obedient_twins`, `junk_fees`, `story_beliefs`, `targeting_fairness`, and `promiscuous_donors`), `data.ipynb` groups observations by `(base_slot, question, option_texts)` so every exported `v` identifier (`<study-slug>-<QID>[_<subitem>][_v<NNN>]`) has strictly invariant question wording and options across respondents, while `v.split("_")[0]` groups sibling matrix rows, loop ideas, and randomized variants into 254 base `QuestionID` families. Items tagged in `question_tags.json` (such as `prior-priming` in `affective_priming`, `odd-col-nums` in `promiscuous_donors`, `html-table-flat` in `context_effects`, and `has-dont-know` in `preference_redistribution`) can be added, edited, or dismissed by both automated checks and human reviewers in the shared Audit UI, while inclusion and modeling decisions live in `question_audit.csv` and clean text replacements for stems, matrix sub-items, and option labels live non-destructively in `question_overrides.json`.

## Deferred multi-stage and condition-dependent questions

In the current single-turn audit pass, we exclude questions whose meaning or experimental stimulus depends on an earlier manipulated setup, an interactive priming task, or an endogenous choice made by the participant earlier in the survey (tracked in `question_tags.json` under tags such as `prior-task-dep`, `prior-priming`, and `backref`). A representative case is `digital_certification:QID121`, where each respondent first selects one of six luxury sweaters in `QID199`, views either the digital-passport manipulation (`QID141`) or the control block (`QID177`) whose Qualtrics `${q://QID199/ChoiceGroup/SelectedChoices}` pipe was stripped from the upstream Parquet export, and writes an open-ended night-out essay (`QID142` or `QID176`) before rating the sweater. Although incompatible with standalone single-turn prompts as currently exported, these multi-stage and condition-dependent questions remain valuable behavioral signal for decision-model training, so we plan to revisit them in a future pass and package them into separate condition-resolved or multi-turn datasets reconstructed from each respondent's full survey flow and `.qsf` templates.

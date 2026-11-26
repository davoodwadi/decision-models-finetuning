# WVS Wave 7 United States decision log

This log documents the finalized rebuild of the WVS7 US training data from official WVSA sources in `WVS-Wave7/` (2026-11-25). Initial `include` and `kev_type` proposals were generated from the official sources and verified by the user in the audit app. Persona mappings, data-source selection, and final inclusion and type decisions belong to the user.

## Sources

| File | Used for |
|---|---|
| `WVS_Wave_7_United_States_Stata_v5.1.dta` | Answers (2,596 respondents, `D_INTERVIEW` = CASEID), variable labels (`data_label`), and value labels for answer options and codes. |
| `F00010562-World_Values_Survey_Wave_7_2017-2020_United_States_v3.0.pdf` | Question wording (`QuestionText`), parsed via `pdftotext -layout` and independently checked with PyMuPDF in the audit app. Each variable block is "`Qn- <wording>`" followed by a weighted percentage table of response categories. |
| `F00008646-WVS7_Questionnaire_USA_2017_English.pdf` | Not a text source. Consulted only to explain disagreements because it is the master English form with `[COUNTRY]`-style placeholders. |

Question wording comes from the US codebook PDF, while options and codes come from the Stata value labels. Where the two disagree, the Stata labels are used and the disagreement is recorded in the row's `review_note` ("[PDF vs Stata: ...]") and summarized below. The legacy WorldValuesBench repository, `WorldValuesBench-main.zip`, and `data-check.ipynb` were never read by this pipeline and have been removed from `WVS-Wave7/`.

## Files

`build_question_catalog.py` parses the PDF and the `.dta` file, writing `wvs7_questions.json` (per Q-variable section, Stata label, codes, labels, valid count, PDF text, PDF categories, and PDF-only or Stata-only categories) and the starting `question_audit.csv`. It never replaces an existing audit unless run with `--overwrite`. `question_audit.csv` contains the columns `catalog_idx`, `QuestionID`, `family`, `BlockName`, `n_answered`, `include`, `kev_type`, `drop_options`, `review_note`, `QuestionText`, `Options_preview`, `data_label`, and `codes`, where `family` is the WVSA questionnaire section, `BlockName` is the variable name, and `codes` holds Stata codes in `Options_preview` order. `question_overrides.json` stores the verified stem and option text edits (`Q145` stem and `Q152` option 3). `code/data/wvs_export.py` runs the exporter (`python -m data.wvs_export [--dry-run]` from `code/`, or via the audit app's Parquet button), and `code/data/wvs.py` loads the exported standardized parquet directly.

The standing project rule (2026-11-14) omits the `reversed` column from new audits, exporters, and parquets because training does not read it. The audit app handles its absence cleanly.

## Catalog of 305 Q-variables

The catalog covers all `Q*` variables in the `.dta` file, including Q1 through Q259 (Q215 does not exist in the US file), the derived variants Q33_3, Q34_3, Q35_3, Q82, Q82_NAFTA, Q223_ABREV, and Q223_LOCAL, and the demographics block Q260 through Q290 with its recodes. Out of 305 rows, 304 have a PDF block; Q82 (cross-national combination), Q223_ABREV, and Q223_LOCAL do not. Missing codes such as Don't know, No answer, and Not asked are stored as Stata extended missing values (value labels at codes >= 2,147,483,621), which read as `NaN`, are excluded from options, and never reach the parquet.

## Finalized include and kev_type decisions (252 included: 176 score, 76 choice)

Included `score` items (176) are traditional Likert and ordered numeric scales listed in scale order, covering agree-disagree, importance, frequency, confidence, trust, 1 through 10 scales, and closeness scales. Included `choice` items (76) cover binary items (Yes/No, Important/Not mentioned, Mentioned/Not mentioned), unordered categories, scales with an off-scale option kept so no answers are dropped (Q254 "I am not American", Q119 "Hard to say" coded 0 first), Q56 ("Better off / Worse off / Or about the same", non-scale order), second-choice national-aims items (Q153, Q155, Q157, retained with original questionnaire wording), and three-option non-Likert items that the user classified as `choice` during review (Q43 through Q45 "Good thing / Don't mind / Bad thing", Q94 through Q105 membership "Don't belong or Not a member / Inactive member / Active member", Q200 "Frequently / Occasionally / Never", and Q209 through Q220 "Have done / Might do / Would never do").

The 53 excluded rows consist of 48 format-skipped persona columns (Q260 through Q290 with recodes plus Q97 union membership, Q171 religious attendance, Q222 national voting, Q223 party with its two label variants, and Q240 left-right placement, which the exporter also enforces via `WVS_PERSONA_RAW_COLS`), four derived duplicates (Q33_3, Q34_3, Q35_3 three-point recodes of Q33 through Q35, and Q82 identical to Q82_NAFTA), and Q221 ("Vote in elections: Local level", excluded by the user during review because the codebook prints only a variable label rather than a question stem).

The codebook is treated as the source of truth, so defective codebook stems and follow-up items keep their original questionnaire wording rather than being rewritten. Rows tagged in `question_tags.json` (`review-codebook-stem` on Q144, Q221, Q254, Q256, Q257, and Q259; `backref` on Q154 and Q156) were all reviewed in the audit app, with Q221 excluded and the remaining tagged rows kept verbatim.

## Question text repairs (all add words; none remove or replace codebook words)

| Rows | PDF | QuestionText | Why |
|---|---|---|---|
| Q1-Q6 | "...or not important at all Family" | "...or not important at all: Family" | The PDF lost the stem/item separator. |
| Q241-Q249 | "...“an essential characteristic of democracy Governments tax..." | "...democracy”: Governments tax..." | Same, plus the missing closing quote. |
| Q145 | "And what about your immediate family" | "Has your immediate family been the victim of a crime during the past year?" (via `question_overrides.json` stem override) | Only makes sense after Q144; clarified cleanly via stem override. |

Everything else preserves the PDF wording verbatim, including minor quirks such as Q110 ("chose any number"), Q113 ("corruption?."), Q163's repeated label, Q144's trailing "Respondent", Q254's phrasing, and Q256, Q257, and Q259's codebook stems.

## Placeholders filled at export

The audit keeps the master questionnaire's placeholders exactly as the codebook prints them, matching the ISSP audit convention for `{country}`. `wvs_export.py` (`US_FILLS`) fills them at export time in question and option text before `drop_options` and overrides, and fails if any `[...]` remains in an included question or option.

| Placeholder | Filled with | Rows |
|---|---|---|
| `[your country]` | the United States | Q112, Q121, Q122-Q129 |
| `[my country]` | my country (matches the Stata label "There is no corruption in my country") | Q112 |
| `[Country]` | the United States | Q257 |
| `[nationality]` | American | Q254 option code 5 ("I am not [nationality]") |
| `[Continent; e.g. Europe, Asia etc.]` | North America | Q258 |
| `[village, town or city]` | village, town or city (brackets removed) | Q255 |
| `[county, region, district]` | county, region, district (brackets removed) | Q256 |

## Option text repairs (`question_overrides.json`)

| Row | Stata label | Exported text |
|---|---|---|
| Q152 code 3 | "Seeing that people have more say about how  are done at their jobs and in their communities" | "...about how things are done..." (word missing in both Stata and PDF) |

Other Stata spelling and spacing quirks are kept as data (e.g., Q153 missing "things", Q157 "more humane  society", Q253 "Fairly much  respect", and Q39 through Q40 "Neither agree or disagree" next to "Neither agree nor disagree" elsewhere). Scales whose interior points are labeled only with digits ("2" through "9") are written "1. None at all", "2.", ..., "10. A great deal" (same rule as ISSP), while labels that already start with their code ("1 There is no corruption in my country") keep their text.

## PDF vs Stata disagreements (questions)

| Row | PDF | Stata (used) |
|---|---|---|
| Q33, Q34, Q35 | 5 categories; Agree strongly / Disagree strongly at 0% | 3 labels, codes 2-4 (the US fielded 3 points) |
| Q106 | endpoints "Incomes more equal" / "Larger income differences" | "Incomes should be made more equal" / "There should be greater incentives for individual effort" |
| Q152, Q153 | lists "Trying to make our cities and countryside more beautiful" at 0% | no label and no answers for it (3 options) |
| Q241-Q249 | lists "It is against democracy (spontaneous)" at 0% | no label, no answers |
| Q121 | code 1 "Rather bad" (agrees with Stata) | "Rather bad"; the master questionnaire says "Very bad" |
| Q142, Q143, Q147, Q148 | (agrees with Stata) | "A great deal" as the 2nd point; Q146 says "A good deal" |

PDF "INAP" categories (Q217 through Q220, Q264 through Q265, Q281) are missing codes and ignored. Persona-block disagreements (Q270, Q274, Q275 through Q280, Q282 through Q283, Q289) concern persona columns only and do not affect the questions. The PDF tables report weighted percentages and weighted counts (e.g., Q260 Male 48.4%, N 1,256), while the `.dta` counts are unweighted (1,390 male). The parquet uses unweighted respondent answers, matching the other datasets.

## Finalized export and loader (`code/data/wvs_export.py` and `code/data/wvs.py`)

`code/data/wvs_export.py` reads the `.dta` answers and labels, `question_audit.csv`, and `question_overrides.json`, fills `US_FILLS` placeholders, attaches personas via `standardize_dataframe(load_raw_wvs7_personas(), dataset="wvs7")` (keeping the 29 populated `persona_spec.yaml` attributes), and writes `wvs7_US_long.parquet` and `questions_verified.csv` atomically. It raises on any question-level error in included rows (missing column, persona overlap, more than 15 options, unfilled placeholder, invalid audit value, or invalid `drop_options`).

The finalized export on disk (`2026-11-25`) contains 642,775 rows, 2,596 respondents, 252 questions (176 score, 76 choice), and 29 standardized persona columns, with every included question answered in the US sample and zero nulls or code/text mismatches. `code/data/wvs.py` loads `WVS-Wave7/wvs7_US_long.parquet` directly without legacy WorldValuesBench persona remapping.

## Source tab codebook verification

The audit app Source tab compares each question against the codebook PDF independently using `audit-app/scripts/source_check.py` (PyMuPDF) on the source side and `wvs_final_questions()` in `code/data/wvs_export.py` on the final side, with unified review status stored in `.question_audit.review_progress.json` (the earlier `.source_check_progress.json` was merged into it and removed). All 75 Suspect rows and all 38 Codebook diff rows have been reviewed (115 of 257 substantive rows reviewed total, with 0 blocking export issues). The independent PyMuPDF parse also caught four excluded persona rows (Q266, Q267, Q268, Q289CS9) where the `pdftotext` parser had previously run into table headers; those were fixed in `build_question_catalog.py`. For all included questions, catalog wording matches the independent PyMuPDF parse.

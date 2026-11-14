# Twin-2K-500 data notes

## Source and sample

Toubia, Gui, Peng, Merlau, Li, and Chen (2025), "Twin-2K-500: A data set for building digital twins of over 2,000 people based on their answers to over 500 questions", Marketing Science 44(6), pp. 1446 to 1455 (`doi:10.1287/mksc.2025.0262`), hosted at <https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500> (CC BY 4.0). The dataset covers $N = 2,058$ US respondents on Prolific, quota-sampled on age, sex, and ethnicity across four waves from January to March 2025. Waves 1 to 3 were collected one week apart, and wave 4 was collected two weeks after wave 3 as a test-retest repetition of the behavioral economics experiments and pricing study under the same respondent conditions.

## Files used (`hf/question_catalog_and_human_response_csv/`)

| File | Shape | Content |
|---|---|---|
| `question_catalog.json` | 256 catalog entries (246 unique `QuestionID`s) | Question text, `QuestionType`, `Settings`, options or scale labels, block name, and expanded `csv_columns` |
| `wave1_3_response_label.csv` | $2,058 \times 761$ (`pid` + 760 columns) | Waves 1 to 3 human responses as text labels |
| `wave4_response_label.csv` | $2,058 \times 127$ (`pid` + 126 columns) | Wave 4 retest human responses as text labels |

Both response CSVs are loaded with `dtype=str` (with `pid` cast to `int` as the index) so that numeric scale labels (e.g., `'1'` and `'2'` in Bipolar matrices or `'2'` through `'8'` in 9-point Likert matrices) remain strings matching the catalog definitions. The 760 columns across `catalog_df["csv_columns"]` form a 1-to-1 bijection with the 760 response columns in `wave1_3_response_label.csv`, and all 126 columns in `wave4_response_label.csv` are a subset of the waves 1 to 3 columns. Ten `QuestionID`s (`QID268` through `QID272` and `QID275` through `QID279`) are reused across two blocks with different question types, but their expanded `csv_columns` are disjoint.

## Catalog to response mapping by question type

| `QuestionType` | `Selector` | Catalog entries | CSV columns | Column pattern | Mapping from catalog to `wave1_3_response_label.csv` |
|---|---|---:|---:|---|---|
| `MC` | `SAVR`, `SAHR` | 154 | 154 | `{QID}` | Single-select choice (14 demographics, 40 pricing, 100 other). Each column stores one label from `Options`. |
| `MC` | `MAVR`, `MAHR` | 21 | 84 | `{QID}_{1..4}` | Multi-select (20 BDI items plus `QID221` Wason selection). `csv_columns[i]` corresponds to `Options[i]` and stores `Options[i]` if selected or `NaN` otherwise. |
| `Matrix` | `Likert` | 23 | 290 | `{QID}_{RowID}` | Single-answer rating scale per row. `csv_columns[i]` corresponds to sub-question `Rows[i]`, and each response is one scale label from `Columns`. |
| `Matrix` | `Bipolar` | 13 | 158 | `{QID}_{RowID}` | Two-endpoint choice per row in economic preference multiple price lists. `Columns` is `['1', '2']` and `Rows[i]` is formatted as `<Option 1>:<Option 2>`, with responses recorded as `'1'` (left option) or `'2'` (right option). |
| `Slider` | `HSLIDER` | 3 | 12 | `{QID}` or `{QID}_{StatementsID}` | Continuous numeric slider (`QID154`, `QID156`, and 10 statements in `QID290`). `csv_columns[i]` corresponds to `Statements[i]` with numeric responses in `[Range['Min'], Range['Max']]` (`[0, 100]`). |
| `TE` | `SL`, `ML`, `FORM` | 28 | 62 | `{QID}_TEXT` or `{QID}_{RowID}` | Open numeric or text entry (`SL` has 19 `ValidNumber` and 2 text entries, `ML` has 3 paragraph entries, and `FORM` has 4 multi-row text entries expanding to 38 `Rows` columns). |
| `DB` | (none) | 14 | 0 | `[]` | Descriptive instruction blocks (`is_descriptive == True`) with no `Settings` and no response columns. |

## Pricing study (`QID9_1` to `QID9_40`)

Each respondent saw each of the 40 products at a randomly drawn price (Gui and Toubia 2023). The response CSVs record only `"Yes, I would purchase the product"` or `"No, I would not purchase the product"`, whereas `question_catalog.json` stores the `QuestionText` (and thus the price) shown to a single respondent. Per-respondent prices live in each respondent's question JSON inside `wave_split/chunks/*.parquet`.

# Question Catalog (`question_catalog.json`)

A catalog of all unique questions from the HuggingFace dataset `LLM-Digital-Twin/Twin-2K-500` (`wave_split` config), extracted from:
- `wave1_3_persona_json` (171 QuestionIDs) — demographics, personality, cognitive tests, economic preferences
- `wave4_Q_wave1_3_A` (85 QuestionIDs) — behavioral economics / heuristics & biases questions

## QuestionIDs vs. Questions Asked vs. CSV Columns

The catalog contains **256 unique QuestionIDs** (Qualtrics-level question blocks), but participants answered approximately **500 individual questions** as reported in the paper. This is because many QuestionIDs contain multiple sub-items:

- A single **Matrix** QuestionID (e.g., Big Five personality QID25) contains up to 44 row items, each of which is a separate question the participant answers.
- A single **Slider** QuestionID (e.g., QID290) can contain 10 separate slider statements.
- A single **TE FORM** QuestionID (e.g., Forward Flow QID10) contains 20 text entry fields.
- **Between-subject experiments** list all conditions as separate QuestionIDs (e.g., both "Anchoring - African countries high" and "low"), but each participant only saw one condition.

These three levels of counting are:

| Level | Count | Description |
|-------|-------|-------------|
| **QuestionIDs** | 256 | Unique Qualtrics question blocks in the catalog (242 answerable + 14 descriptive) |
| **Questions asked** | ~500 | Individual items each participant answered (expanding Matrix rows, Slider statements, etc., but counting only one condition per between-subject experiment) |
| **CSV columns** | 760 | Total data columns across all CSVs (further expands multi-select MAVR/MAHR into one binary column per option) |

The paper's ~500 question count maps to the catalog as follows:

| Paper Category | QuestionIDs | Questions Asked | CSV Columns |
|----------------|-------------|-----------------|-------------|
| Demographics | 14 | 14 | 14 |
| Personality (19 tests, 26 constructs) | 49 | ~279 | 338 |
| Cognitive ability tests (incl. Forward Flow) | 70 | 85 | 88 |
| Economic preferences (10 tests) | 34 (+2 intro) | 34 | 194 |
| Non-experimental heuristics & biases | 5 | 5 | 20 |
| Behavioral economics experiments (11 between + 5 within) | 39 | ~48 | 66 |
| Pricing study (40 products) | 40 (+1 intro) | 40 | 40 |
| **Total** | **256** | **~500** | **760** |

Note: Personality and cognitive tests are counted at the item level (each Matrix row = 1 question). Economic preferences and experiments are counted at the QuestionID level (each Matrix = 1 question). Between-subject experiments contribute fewer questions-per-participant than QuestionIDs because each participant sees only one condition.

## Schema

Each entry in the catalog is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `QuestionID` | string | Qualtrics question identifier (e.g., `"QID11"`, `"QID9_1"`) |
| `QuestionText` | string | The question text shown to participants. Empty for BDI items (see MAVR below) |
| `QuestionType` | string | One of: `MC`, `Matrix`, `Slider`, `TE`, `DB` |
| `Options` | string[] | (MC only) Available answer choices |
| `Rows` | string[] | (Matrix/TE FORM) Row labels |
| `RowsID` | string[] | (Matrix/TE FORM) Row identifiers for column expansion |
| `Columns` | string[] | (Matrix) Column labels (scale points) |
| `Statements` | string[] | (Slider) Statement labels for multi-slider questions |
| `StatementsID` | string[] | (Slider) Statement identifiers for column expansion |
| `Range` | object | (Slider) `{Min, Max, Ticks}` |
| `Settings` | object | Qualtrics settings: `Selector`, `SubSelector`, `ForceResponse`, etc. |
| `BlockName` | string | The survey block this question belongs to |
| `source` | string | Which JSON directory this question was first found in |
| `csv_columns` | string[] | The CSV column name(s) this question expands to |
| `is_descriptive` | boolean | (DB only) Marks instructional text blocks with no answer |

## Question Types

### MC — Multiple Choice (175 QuestionIDs)

Single or multiple answer selection from a list of options.

**Selectors:**

| Selector | Count | Description | Answer Format |
|----------|-------|-------------|---------------|
| `SAVR` | 129 | Single Answer Vertical | `SelectedByPosition` (int), `SelectedText` (string) |
| `SAHR` | 25 | Single Answer Horizontal | Same as SAVR |
| `MAVR` | 20 | Multiple Answer Vertical | `SelectedByPosition` (int[]), `SelectedText` (string[]) |
| `MAHR` | 1 | Multiple Answer Horizontal | Same as MAVR |

**SAVR/SAHR** — Standard single-select. Each produces 1 CSV column named `{QID}`.
- Numeric CSV: stores the 1-based position index (e.g., `3` = third option)
- Label CSV: stores the option text (e.g., `"South (TX, OK, ...)"`)

**MAVR/MAHR** — Multi-select (used for Beck Depression Inventory items and Wason Selection). Each produces N CSV columns (one per option): `{QID}_1`, `{QID}_2`, ..., `{QID}_N`, matching the binary encoding used in `wave3_anonymized.csv`.
- Numeric CSV: `1` if the option was selected, empty/NaN if not
- Label CSV: the option text if selected (e.g., `"I feel sad"`), empty/NaN if not
- All 21 MAVR/MAHR questions have exactly 4 options, producing 4 columns each (84 columns total)

Note: 20 MAVR questions (QID126-QID147) are BDI items with empty `QuestionText`. The instructional text is in a preceding DB question (QID127/QID240). The `Options` array contains the self-describing answer choices (e.g., `"I don't feel sad"`, `"I feel sad"`, ...).

### Matrix — Matrix/Likert Scale (36 QuestionIDs)

Multi-row rating scales. Each row produces a separate CSV column: `{QID}_{RowID}`.

**Selectors:**

| Selector | SubSelector | Count | Description |
|----------|-------------|-------|-------------|
| `Likert` | `SingleAnswer` | 23 | Standard Likert scale (e.g., "Disagree strongly" to "Agree strongly") |
| `Bipolar` | (empty) | 13 | Two-endpoint scale (e.g., choosing between two delayed payment options) |

- `Rows` / `RowsID`: Define the row items. RowsID values become the column suffix (e.g., QID287 with RowsID `["1","2","3",...,"12"]` produces `QID287_1`, `QID287_2`, ..., `QID287_12`). Note: RowsID values may not be contiguous (e.g., `["1","2","3","4","5","6","7","10","11","12"]`).
- `Columns`: The scale labels
- Numeric CSV: 1-based column position (e.g., `4` = "Somewhat support" on a 5-point scale)
- Label CSV: column text (e.g., `"Somewhat support"`)
- Column count per question: 3 to 44 (Big Five personality has 44 items)

### Slider — Horizontal Slider (3 QuestionIDs)

Numeric input on a continuous scale.

| QID | Statements | Range | CSV columns |
|-----|-----------|-------|-------------|
| `QID156` | 1 (single) | 0-100 | `QID156` |
| `QID154` | 1 (single) | 0-100 | `QID154` |
| `QID290` | 10 (multi) | 0-100 | `QID290_1` through `QID290_12` |

- Single-statement sliders produce 1 CSV column: `{QID}`
- Multi-statement sliders produce N columns: `{QID}_{StatementsID}`
- Both numeric and label CSVs store the raw numeric value (sliders have no text labels)

### TE — Text Entry (28 QuestionIDs)

Free-text or numeric input fields.

**Selectors:**

| Selector | Count | ContentType | CSV column pattern |
|----------|-------|-------------|-------------------|
| `SL` | 21 | `ValidNumber` (19), empty (2) | `{QID}_TEXT` |
| `ML` | 3 | empty | `{QID}_TEXT` |
| `FORM` | 4 | empty | `{QID}_{RowID}` (one per row) |

- `SL` (Single Line): Most are `ValidNumber` (numeric input like "How many African countries...?"). Produces 1 column: `{QID}_TEXT`
- `ML` (Multi Line): Free-text paragraphs. Produces 1 column: `{QID}_TEXT`
- `FORM`: Multiple labeled text fields. Produces N columns: `{QID}_{RowID}` (e.g., QID10 "Forward Flow" has 20 word slots → `QID10_1` through `QID10_20`)
- Both CSVs store the raw text value

### DB — Descriptive Block (14 QuestionIDs)

Instructional text shown to participants (no answer collected). Marked with `is_descriptive: true`. Produces 0 CSV columns.

## Duplicate QIDs

10 QuestionIDs are reused across different survey blocks with different question types:

| QID | Block 1 (MC) | Block 2 (TE/Matrix) |
|-----|-------------|---------------------|
| QID268-270 | Cognitive tests (MC) | Personality (TE) |
| QID271-272, 275 | Cognitive tests (MC) | Economic preferences (TE FORM) |
| QID276-279 | Cognitive tests (MC) | Economic preferences (Matrix) |

These produce non-colliding CSV columns: the MC version → `{QID}`, the TE version → `{QID}_TEXT` or `{QID}_{RowID}`, the Matrix version → `{QID}_{RowID}`.

## Survey Blocks (43 blocks)

| Block | QuestionIDs | ~Questions Asked | Description |
|-------|-------------|-----------------|-------------|
| Cognitive tests | 69 | 65 | Syllogistic reasoning, Wason selection, CRT |
| Forward Flow | 1 | 20 | Creative word generation (20 word slots) |
| Personality | 49 | ~279 | Big Five (BFI-44), Need for Cognition, BDI, Empathy |
| Product Preferences - Pricing | 41 | 40 | 40 purchase decisions + 1 instruction (QID8) |
| Economic preferences | 36 | 34 | Time preferences, trust game, risk aversion (2 are intro blocks) |
| Demographics | 14 | 14 | Region, sex, age, education, race, religion, politics, income, household, employment |
| Non-experimental heuristics | 5 | 5 | Risk/benefit perception, omission bias, denominator neglect, false consensus (others) |
| Behavioral economics experiments | 39 | ~48 | Anchoring, framing, conjunction fallacy, sunk cost, Allais paradox, etc. |

## Generated CSV Files

| File | Rows | Columns | Content |
|------|------|---------|---------|
| `wave1_3_response.csv` | 2058 | 761 | Numeric answers (persona + wave4_Q_wave1_3_A) |
| `wave1_3_response_label.csv` | 2058 | 761 | Text label answers |
| `wave4_response.csv` | 2058 | 127 | Numeric answers (wave4_Q_wave4_A) |
| `wave4_response_label.csv` | 2058 | 127 | Text label answers |

All CSVs have `pid` as the first column (participant ID, 1-2058). The remaining columns are named by `{QID}` or `{QID}_{SubID}` as described above.

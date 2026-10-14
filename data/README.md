# Data

One folder per source. An ISSP year's training data is respondent-level (`issp<year>_respondents.parquet` +
`issp<year>_questions.json`); `train_issp.py` takes the years as `--train issp<year>` (default: all) and draws
training records from them on the fly (`issp_sampler.py`, below). The SubPOP calibration and test sets are fixed
records (`subpop/<name>.jsonl`). Raw downloads sit in `data/<source>/raw/`; build scripts and intermediate files sit
next to the data. Each folder has a README with its source, how to rebuild it, and its own decisions.

**Roles (2026-10-12):** all ISSP data is training data, used whole; SubPOP is only for calibration and testing.

| Folder | Dataset | Topic | Size | Role |
|---|---|---|---|---|
| `issp2015/` | `issp2015` | Work Orientations IV | 51,668 respondents, 94 questions | training |
| `issp2016/` | `issp2016` | Role of Government V | 48,720 respondents, 63 questions | training |
| `issp2017/` | `issp2017` | Social Networks and Social Resources | 44,492 respondents, 66 questions | training |
| `issp2018/` | `issp2018` | Religion IV | 46,267 respondents, 78 questions | training |
| `issp2019/` | `issp2019` | Social Inequality V | 44,975 respondents, 58 questions | training |
| `issp2020/` | `issp2020` | Environment IV | 44,100 respondents, 55 questions | training |
| `issp2021/` | `issp2021` | Health and Health Care II | 44,549 respondents, 68 questions | training |
| `issp2022/` | `issp2022` | Family and Changing Gender Roles V | 45,762 respondents, 64 questions | training |
| `issp2023/` | `issp2023` | National Identity and Citizenship | 43,517 respondents, 68 questions | training |
| `issp2024/` | `issp2024` | Digital Societies I | 44,703 respondents, 78 questions | training |
| `subpop/` | `subpop100` | Pew American Trends Panel (SubPOP training split) | 41,200 targets | not used |
| `subpop/` | `subpop_eval` | US General Social Survey (SubPOP evaluation split) | 2,332 targets | test |

Training on all ISSP years: 458,753 respondents in 310 country-years. The calibration set is held out of the ISSP years
trained on: 1% of each year's (country, question) slots (`train_issp.py --calibration_frac`; 204 slots of all ten
years), never trained on; per held-out slot, the whole country and every one-attribute group, with the answer shares of
all matching respondents (`issp_sampler.IsspSampler.calibration_records`). The SubPOP sets leave out every question the
ISSP data also asks (`subpop/README.md`); the test set is set in `train_issp.py` (`TEST_DATASET`).

## ISSP training years

Every year (2020 included, since 2026-10-13) is built by the shared builder **`issp_common.py`**; a year's folder
holds only what differs. Project-wide decisions are logged in `issp2020/DECISIONS.md`.

- `build_question_bank.py`: the wording of every item (`SOURCE`), which items are `choice`, `noul` or excluded, which
  `score` scales are printed high → low, and answers to keep that the data codes as non-answers. Writes
  `question_bank.csv`.
- `build_dataset.py`: which questionnaire blocks were asked of only part of the sample (`ASKED_OF`), and year-specific
  placeholders. Writes `issp<year>_respondents.parquet` (one row per respondent: country, every attribute, every
  question's answer code) and `issp<year>_questions.json` (per question and country: wording, options, the map from
  answer code to option; per block and country: who was asked).

Rebuild a year from the project root, in this order (e.g. 2018):
```bash
kev/.venv/bin/python data/issp2018/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2018/build_dataset.py
```

### Rules shared by all ISSP training years
1. **Wording** is transcribed from the year's English source questionnaire (`raw/*_bq.pdf`), the version every
   country translated from. Form instructions ("PLEASE TICK ONE BOX") are left out. Battery items are written as the
   battery's stem plus the item. Placeholders `{country}`, `{nationality}`, `{parliament}` are filled per country from
   `issp_countries.csv`.
2. **Answer options** come from the data file's value labels, cleaned: number prefixes removed, country notes removed
   from the master wording ("About once or twice a year; KR: About once a year" → "About once or twice a year").
   A country note gives that country's own option (`country_options` in the question bank, used only for that
   country's records: Korea gets "About once a year"), since such notes mark a category that differs there; a note
   starting with "+" or "also" extends the master text ("Father unknown + father already passed away"); "Not used"
   removes the option for that country (`country_dropped`; for a `score` item only at the ends of the scale). Notes
   on answers given by hand (`EXTRA_ANSWERS`) are not used. Scales labelled only at their
   ends get their middle points as numbers, and labelled points keep their number ("1 Not at all important", "2", …,
   "7 Very important").
3. **Non-answers** (can't choose, don't know, no answer, refused, not asked, not available in the country) are not
   answers: the codes listed in the year's `raw/*_missing.txt`, plus every negative code. Answers that only some
   countries offered (labels such as "TW: Women should decide") are dropped too.
4. **Printed answers that are not on the scale are kept as answers** ("No access", "Does not apply", "Did not need
   medical treatment", "I do not have any brothers or sisters"): the target is the distribution of everyone who
   answered. Such an item becomes `choice`, since the extra answer has no place on the ordered scale. Where the data
   codes such an answer as a non-answer, the year lists it in `EXTRA_ANSWERS`.
5. **Question types**: ordered scales are `score` with levels low → high (scales printed high → low are reversed;
   the question bank's `reversed` column says which); unordered answer lists are `choice`; yes/no is `noul`.
6. **Questions asked of only some people** (filters such as "working for pay", "used the Internet") keep the
   questionnaire's wording; who was asked is added to the persona of those blocks' records (`ASKED_OF`), so it
   appears in the state, e.g. `Currently works for pay: Yes`.
7. **Excluded** only when the answer is not a choice among options: open amounts, counts, dates and measurements
   (earnings, hours, number of children, height), open occupations, and spouse's education (a background variable).
   Each exclusion is listed with its reason in the year's `question_bank.csv` (`note`).
8. **Attributes** (`issp_common.PERSONA_KEY`), the same in every year: sex, age, education, religion, attendance of
   religious services, party voted for, the voted party's left-right position, whether voted, household income,
   region, place of living, marital status, main activity, trade union membership, self-placed social position,
   ancestry. Each is the respondent's raw value: age in years, the data file's value label otherwise (education is
   `EDULEVEL` from 2020 on, `DEGREE` up to 2019; party, income, region and ancestry are each country's own variable,
   e.g. `US_PRTY`). Missing values are the codes in `raw/*_missing.txt` plus every negative code. Every respondent is
   kept, whatever they miss. The respondent table stores each attribute as the data code; `issp<year>_attributes.json`
   gives, per attribute and country, the text of each code **as that country's questionnaire had it**: a country note
   in the value label ("2 or 3 times a month; DE: 1-3 times a month", "89 years; US: 89 years or more") gives that
   country's own text, read as for answer options (rule 2; a note starting with "incl" extends the master text: "No,
   incl. was not eligible"); notes the data abbreviates are written out in the year's `build_dataset.py`
   (`ATTRIBUTE_NOTES`, from the codebook's per-country lists). Notes name countries by two letters, so "GB" applies to
   the data's `GB-GBN`. Only the label notes are used (2026-10-13); the codebooks' fuller per-country lists (2015–2018)
   and free-text notes are not yet.
9. **Training records are drawn, not stored** (`issp_sampler.py`, 2026-10-13): a random (year, country, question),
   one question per record; a random number of attributes (uniform from 0 to all, no cap) chosen at random; a value
   combination uniformly among those that occur among the respondents to the question (every persona counts the same,
   however many people it matches); as target the answer shares of the matching respondents, with no minimum (one
   respondent gives a one-hot target). Persona = Country, the drawn attributes, and the `ASKED_OF` lines of the
   question's block. No survey weights. A block is the question number from the data label (`Q12` for Q12a–Q12h).
   One question per record rather than one block (2026-10-13): Kev's questions cannot read each other, so a block's
   questions in one record get the same predictions as apart; apart, every target gets its own drawn persona instead
   of the block sharing one, at the cost of encoding the persona once per question. An epoch is one pass through every
   (year, country, question) slot in random order, each with a freshly drawn persona.
   Until 2026-10-13 the records were fixed groups (country × sex × age band × education, at least 10 answers),
   stored as `issp<year>.jsonl`; re-aggregating the respondent data that way reproduces those files exactly.
10. **Training view** (`issp_augment.augment`): each drawn record's persona is written in a random presentation
    (layout, attribute names, value wording, line order), `score` scales are reversed half the time and `choice`
    options shuffled. Calibration and test use the fixed canonical view.
11. **All of it is training data** (decided 2026-10-12): every suitable question and every respondent. Overlap with
    the calibration and test sets is handled on the SubPOP side: SubPOP questions that any ISSP year also asks are left
    out of them (`subpop/README.md`).

### Cut-off value labels
The .dta files limit the length of value labels (60 characters in 2015 and 2017, about 120 from 2020 on), so some
answer options and attribute values end mid-word ("…decided at s"). `codebook_label_fixes.py` finds every label that
looks cut and adds only its missing end, from the GESIS codebook (`raw/*_cdb.pdf`) or, where the codebook cuts it
too, from another year's data file (listed in the script, with the source). The data's own wording is kept: "incl."
stays "incl." where the codebook writes "including", because wording variation is wanted in training data and is never
unified. The result, `issp<year>/label_fixes.csv`, is applied by `issp_common.read_dta()` whenever a data file is
read, so question banks, datasets and attributes all get the completed text. A completion is rejected when the label's ending is
found only earlier in the codebook label (it would repeat text), when it would leave unbalanced brackets (the label was
worded differently, not cut), or when it contains characters the PDF text garbles; the codebook's frequency columns
(count, percentages, with a decimal point or comma) are never taken as text, and its full label list is preferred over
its frequency table. Every fix was read by hand on 2026-10-13 (`label_fixes.csv` per year, with its source). Done for 2015, 2017, 2020 and 2021 (the
years with a codebook in `raw/` and cut labels). 2016, 2018, 2019 and 2023
have no cut labels in the questions or attributes used. 2022 and 2024 have 6 cut party names (`<CC>_PRTY`), to be
fixed once their codebooks are downloaded.

### `issp_countries.csv`
Country name, nationality word and parliament name for every country in any ISSP year, with their source: the ISSP
2020 table (checked against the 2020 national questionnaires; the printed names are listed in `issp2020/README.md`)
where available, the 2023/2024 national questionnaires for Canada, Greece, Israel, the Netherlands and Poland, and
standard English names otherwise (not checked against national questionnaires). The Philippines uses "Filipino" as
nationality word (ISSP 2020 overrides it with "Philippine" for "the Philippine government").

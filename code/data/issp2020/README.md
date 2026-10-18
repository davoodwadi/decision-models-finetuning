# ISSP 2020 — Environment IV

Training data (`--train issp2020`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder since 2026-10-13: the rules every ISSP year follows are in `data/README.md`; this file
lists only what is specific to 2020. Until 2026-10-12 this year also held the calibration and test sets (see
"Former split" below).

## Source
- GESIS study **ZA7650**, data file version **v2.0.0**: <https://search.gesis.org/research_data/ZA7650>
- 44,100 respondents, 28 countries.
- Files in `raw/` (downloaded from the GESIS page):
  - `ZA7650_v2-0-0.dta` (+ `.zip`): the data; `ZA7650_v2-0-0_missing.txt`: missing-value codes
  - `ZA7650_bq.pdf`: final source questionnaire (May 2019), which every country translated from; the question wording comes from here
  - `ZA7650_q_all.zip`, unpacked in `questionnaires/`: the national questionnaires (used to check parliament names)
  - `ZA7650_cdb.pdf`: codebook
  - `ZA7650_q_us-en.pdf` / `.md` (+ `extracted_images_ZA7650_q_us-en/`), `ZA7650_backgroundvar_us.pdf`: the US questionnaire and background variables

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/codebook_label_fixes.py 2020
```
```bash
kev/.venv/bin/python data/issp2020/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2020/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `label_fixes.csv` | `data/codebook_label_fixes.py` | value labels the data file cuts off, with their missing end from the codebook (`raw/*_cdb.pdf`); applied whenever the data file is read |
| `question_bank.csv` | `build_question_bank.py` | one row per item v1–v60: question number, wording, type, data codes, answer options, why excluded |
| `issp2020_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2020_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2020_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |
| `DECISIONS.md` | — | the decision log: this dataset (1–9) and project-wide decisions on training, evaluation and data layout |

Result: 44,100 respondents (all kept), 16 attributes, 55 questions in 21 blocks, 28 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2020.jsonl`, used until 2026-10-13): 13,981 records, 36,518 question targets (29,845 `score`, 3,990 `choice`, 2,683 `noul`), 672 persona groups.

## Data format
As every ISSP year: respondent-level files that training records are drawn from (`data/README.md`, rules 8–10).

## Decisions specific to 2020
See `DECISIONS.md`, entries 1–9. In addition:
1. **Moved onto the shared builder (2026-10-13).** Its own scripts were replaced by calls to `data/issp_common.py`;
   the rebuilt `issp2020.jsonl` is byte-identical to the one the old scripts made.
2. **Nationality word for the Philippines**: Q2e "the [NATIONALITY] government" uses "Philippine", as the Philippine
   questionnaire does (`COUNTRY_FIELDS` in `build_dataset.py`); `data/issp_countries.csv` holds the demonym "Filipino"
   used by ISSP 2023 ("truly Filipino").
3. **Parliament names** (`{parliament}`, Q5d) come from `data/issp_countries.csv`; for the 2020 countries they were
   checked against the national questionnaires, which print:

| Country | `{parliament}` | Printed in the national questionnaire |
|---|---|---|
| AT | Parliament | das Parlament |
| AU | federal parliament | The federal parliament |
| CH | Swiss Parliament (National Council and Council of States) | das Schweizer Parlament (National- und Ständerat) |
| CN | National People's Congress | 全国人民代表大会 |
| DE | German Bundestag | Deutscher Bundestag |
| DK | Danish Parliament (Folketing) | Folketinget |
| ES | Spanish Parliament | el Parlamento español |
| FI | Finnish Parliament (Eduskunta) | Eduskuntaan |
| FR | French Parliament | Le Parlement français |
| HR | Croatian Parliament (Sabor) | Hrvatski sabor |
| HU | Hungarian National Assembly (Országgyűlés) | a magyar Országgyűlésben |
| IN | Indian Parliament | The INDIAN PARLIAMENT |
| IS | Icelandic Parliament (Althingi) | Alþingi |
| IT | Italian Parliament | Il parlamento italiano |
| JP | National Diet | 国会 |
| KR | National Assembly | National Assembly |
| LT | Lithuanian Parliament (Seimas) | Seimas |
| NO | Norwegian Parliament (Storting) | Stortinget |
| NZ | New Zealand Parliament | The New Zealand Parliament |
| PH | Philippine Congress | The Philippine Congress |
| RU | State Duma of Russia | Государственной Думе России |
| SE | Swedish Parliament (Riksdag) | Riksdagen |
| SI | Slovenian National Assembly (Državni zbor) | Državnemu zboru |
| SK | National Council of the Slovak Republic | Národná rada Slovenskej republiky |
| TH | Parliament of Thailand | รัฐสภาของประเทศไทย |
| TW | Legislative Yuan | 立法院 |
| US | U.S. Congress | The U.S. Congress |
| ZA | South African Parliament | the South African Parliament |

## Former split (used until 2026-10-12)
Runs before 2026-10-12 trained and tested on a fixed split of this dataset (`split.json`, made once by
`make_split.py`; both deleted on 2026-10-13, as nothing uses them any more). Sizes in question targets:

| Set | Personas | Questions | Targets |
|---|---|---|---|
| train | train (70% of groups in each country) | all but the 10 test questions | 21,066 |
| calibration | calibration (15%) | all but the test questions | 4,423 |
| test A (new persona) | test (15%) | all but the test questions | 4,396 |
| test B (new question) | train | the 10 test questions | 4,680 |
| test C (new persona + question) | calibration + test | the 10 test questions | 1,953 |

Test questions, one per multi-item block: v5, v13, v19, v24, v27, v33, v42, v45, v56, v59. Persona groups: 474 train,
99 calibration, 99 test.

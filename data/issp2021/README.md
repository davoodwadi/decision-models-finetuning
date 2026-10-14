# ISSP 2021 — Health and Health Care II

Training data (`--train issp2021`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2021.

## Source
- GESIS study **ZA8000**, data file version **v2.0.0**: <https://search.gesis.org/research_data/ZA8000>
- 44,549 respondents, 30 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA8000_v2-0-0.dta` (+ `.zip`), `ZA8000_v2-0-0_missing.txt` (non-answer
  codes), `ZA8000_bq.pdf` (source questionnaire, November 2020: the wording comes from here), plus the codebook and
  national questionnaires as downloaded.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/codebook_label_fixes.py 2021
```
```bash
kev/.venv/bin/python data/issp2021/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2021/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `label_fixes.csv` | `data/codebook_label_fixes.py` | value labels the data file cuts off, with their missing end from the codebook (`raw/*_cdb.pdf`); applied whenever the data file is read |
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options, why excluded |
| `issp2021_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2021_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2021_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 44,549 respondents (all kept), 16 attributes, 68 questions in 33 blocks, 30 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2021.jsonl`, used until 2026-10-13): 22,248 records, 46,092 question targets (39,956 `score`, 5,418 `choice`, 718 `noul`), 718 persona groups.
Items: 68 of 70 items: 59 `score`, 8 `choice`, 1 `noul`.
All of it is training data.

## Decisions specific to 2021
1. **Excluded**: v53, v54 (height and weight: open measurements).
2. Each country named its own alternative and conventional medicine; the wording uses "alternative/traditional/folk" and "mainstream/Western conventional".
3. **Printed answers kept** (`EXTRA_ANSWERS`): Q11 "I do not have access to the internet", Q18 "Did not need medical treatment", Q21 "Does not apply"; these items are `choice`.
4. **Filter** (`ASKED_OF`): Q12–Q13 were asked only of people with internet access: `Has access to the internet: Yes`.
5. Suriname's and Taiwan's country versions (`SR_v42`, `TW_v68a`, `TW_v68b`) are not used.

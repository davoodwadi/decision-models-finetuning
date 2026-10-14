# ISSP 2017 — Social Networks and Social Resources

Training data (`--train issp2017`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2017.

## Source
- GESIS study **ZA6980**, data file version **v2.0.0**: <https://search.gesis.org/research_data/ZA6980>
- 44,492 respondents, 30 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA6980_v2-0-0.dta` (+ `.zip`), `ZA6980_v2-0-0_missing.txt` (non-answer
  codes), `ZA6980_bq.pdf` (source questionnaire, July 2016: the wording comes from here), plus the codebook and
  national questionnaires as downloaded.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/codebook_label_fixes.py 2017
```
```bash
kev/.venv/bin/python data/issp2017/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2017/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `label_fixes.csv` | `data/codebook_label_fixes.py` | value labels the data file cuts off, with their missing end from the codebook (`raw/*_cdb.pdf`); applied whenever the data file is read |
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options, why excluded |
| `issp2017_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2017_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2017_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 44,492 respondents (all kept), 16 attributes, 66 questions in 36 blocks, 30 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2017.jsonl`, used until 2026-10-13): 25,685 records, 47,747 question targets (26,108 `score`, 21,639 `choice`), 770 persona groups.
Items: 66 of 67 items: 37 `score`, 29 `choice`.
All of it is training data.

## Decisions specific to 2017
1. **Excluded**: v67 (spouse's education: a background variable).
2. Q1 (who you know in each job), Q3–Q4, Q7–Q8 (whom you would turn to) are `choice`: the answers are kinds of people, not a scale.
3. **Printed answers kept**: Q21–Q25 and Q36 include "I do not have any brothers and sisters", "…lives with me" etc. (codes 95, 96), Q26 includes "I do not use any of these devices"; these items are `choice`.
4. Q34's printed answer "It does not apply" shares code 0 with "not asked in this country", so it cannot be kept and drops out with the non-answers.
5. **Filter** (`ASKED_OF`): Q18 (making new friends when going out) was asked only of people who go out with friends: `Goes out to eat or drink with friends or acquaintances: Yes`.

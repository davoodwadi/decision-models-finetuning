# ISSP 2015 — Work Orientations IV

Training data (`--train issp2015`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2015.

## Source
- GESIS study **ZA6770**, data file version **v2.1.0**: <https://search.gesis.org/research_data/ZA6770>
- 51,668 respondents, 37 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA6770_v2-1-0.dta` (+ `.zip`), `ZA6770_v2-1-0_missing.txt` (non-answer
  codes), `ZA6770_bq.pdf` (source questionnaire, July 2014: the wording comes from here), plus the codebook and
  national questionnaires as downloaded.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/codebook_label_fixes.py 2015
```
```bash
kev/.venv/bin/python data/issp2015/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2015/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `label_fixes.csv` | `data/codebook_label_fixes.py` | value labels the data file cuts off, with their missing end from the codebook (`raw/*_cdb.pdf`); applied whenever the data file is read |
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options, why excluded |
| `issp2015_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2015_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2015_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 51,668 respondents (all kept), 16 attributes, 94 questions in 50 blocks, 37 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2015.jsonl`, used until 2026-10-13): 30,120 records, 56,969 question targets (47,650 `score`, 5,581 `choice`, 3,738 `noul`), 916 persona groups.
Items: 94 of 97 items: 79 `score`, 8 `choice`, 7 `noul`.
All of it is training data.

## Decisions specific to 2015
1. **Excluded**: v62, v63 (year and month the last job ended: open dates), v85 (age of youngest child: open count).
2. **Split questionnaire** (`ASKED_OF`): Q11–Q31 were asked only of people working for pay (`Currently works for pay: Yes`), Q32, Q36, Q43, Q44 only of people not working for pay (`Currently works for pay: No`); Q34–Q35 add `Has had a paid job for one year or more: Yes`, Q37–Q42 add `Would like to have a paid job: Yes`, Q31 adds `Did other paid work besides the main job in the past 12 months: Yes`. Q6 (main reason for discrimination) adds `Discriminated against with regard to work in the past five years: Yes`. Optional O1 was asked only of people not working; O2 adds `Worked for pay in the past five years: Yes`.
3. **Printed answers kept** (`EXTRA_ANSWERS`): Q5 "Did not work or did not seek work" and Q7 "Does not apply (I do not have a job/ superior/ co-worker)"; both items are `choice`.
4. Q7 (harassment) is stored in the data as `V16` (capital V).
5. Q35's "Other" answer was offered only by some countries (its label starts with country codes) and is dropped.
6. Answer texts with national values in square brackets ("A full-time job [30 hours or more per week]") use round brackets.

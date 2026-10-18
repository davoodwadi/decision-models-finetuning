# ISSP 2022 — Family and Changing Gender Roles V

Training data (`--train issp2022`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2022.

## Source
- GESIS study **ZA10000**, data file version **v2.0.0**: <https://search.gesis.org/research_data/ZA10000>
- 45,762 respondents, 32 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA10000_v2-0-0.dta` (+ `.zip`), `ZA10000_v2-0-0_missing.txt` (non-answer
  codes), `ZA10000_bq.pdf` (source questionnaire, August 2021: the wording comes from here), plus the codebook and
  national questionnaires as downloaded.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/issp2022/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2022/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options, why excluded |
| `issp2022_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2022_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2022_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 45,762 respondents (all kept), 16 attributes, 64 questions in 28 blocks, 32 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2022.jsonl`, used until 2026-10-13): 18,797 records, 42,835 question targets (28,350 `score`, 12,857 `choice`, 1,628 `noul`), 767 persona groups.
Items: 64 of 72 items: 39 `score`, 20 `choice`, 5 `noul`.
All of it is training data.

## Decisions specific to 2022
1. **Excluded**: v19 (ideal number of children), v26 (months of paid leave), v34–v37 (hours of household work and care), v63 (number of children): open counts; v62 (spouse's education): a background variable.
2. **Printed answers kept** (`EXTRA_ANSWERS`): Q20 "Doesn't apply/no job"; Q18 also has "Is done by a third person", Q24 "You cannot rely on either of them"; these items are `choice`.
3. **Filters** (`ASKED_OF`): Q9 (dividing paid leave) only of people who think there should be paid leave (`Thinks there should be paid parental leave: Yes`); Q17–Q19 only of people living with a spouse or partner (`Lives with a spouse or partner: Yes`); optional O34 only of people with children (`Has children: Yes`).

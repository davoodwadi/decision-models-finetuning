# ISSP 2016 — Role of Government V

Training data (`--train issp2016`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2016.

## Source
- GESIS study **ZA6900**, data file version **v2.0.0**: <https://search.gesis.org/research_data/ZA6900>
- 48,720 respondents, 35 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA6900_v2-0-0.dta` (+ `.zip`), `ZA6900_v2-0-0_missing.txt` (non-answer
  codes), `ZA6900_bq.pdf` (source questionnaire, June 2015: the wording comes from here), plus the codebook and
  national questionnaires as downloaded.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/issp2016/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2016/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options, why excluded |
| `issp2016_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2016_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2016_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 48,720 respondents (all kept), 16 attributes, 63 questions in 23 blocks, 35 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2016.jsonl`, used until 2026-10-13): 19,421 records, 53,106 question targets (46,299 `score`, 6,807 `choice`), 872 persona groups.
Items: all 63 items: 55 `score`, 8 `choice`.
All of it is training data.

## Decisions specific to 2016
1. **All 63 items kept.** Q9b (second most influential group) is kept too, although it depends on the answer to Q9a (ISSP 2020 excluded such items).
2. Q9's list of groups is written into the question text, since the answers name the groups.
3. Hungary's and New Zealand's country versions of Q13 and Q15 (`HU_v41`, `HU_v42`, `NZ_v46`) are not used; those countries have no targets for these items.
4. Every question was asked of everyone: no persona lines added.

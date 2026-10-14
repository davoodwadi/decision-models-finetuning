# ISSP 2019 — Social Inequality V

Training data (`--train issp2019`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2019.

## Source
- GESIS study **ZA7600**, data file version **v3.0.0**: <https://search.gesis.org/research_data/ZA7600>
- 44,975 respondents, 29 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA7600_v3-0-0.dta` (+ `.zip`), `ZA7600_v3-0-0_missing.txt` (non-answer
  codes), `ZA7600_bq.pdf` (source questionnaire, July 2018: the wording comes from here), plus the codebook and
  national questionnaires as downloaded.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/issp2019/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2019/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options, why excluded |
| `issp2019_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2019_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2019_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 44,975 respondents (all kept), 16 attributes, 58 questions in 23 blocks, 29 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2019.jsonl`, used until 2026-10-13): 14,737 records, 37,607 question targets (33,169 `score`, 4,438 `choice`), 740 persona groups.
Items: 58 of 70 items: 49 `score`, 9 `choice`.
All of it is training data.

## Decisions specific to 2019
1. **Excluded**: v11–v20 (what people in five jobs earn and should earn: open amounts), v59–v60 (father's and mother's occupation: open answers coded to ISCO, hundreds of codes).
2. "When you were [14-15-16] years old" is written "when you were about 15 years old", the age the questionnaire recommends.
3. Q15 (type of society) shows diagrams; the question text describes the five types in words.
4. Denmark's and Norway's country versions (`DK_v44`–`DK_v47`, `NO_v25`) are not used.
5. Every question was asked of everyone: no persona lines added.
6. **Q20a/b (v57, v58) in Sweden**: code 6 "Father/Mother unknown" is "Don't know/ Don't remember" there (value
   label note), so it is a non-answer for Swedish records (`COUNTRY_NOT_ANSWERS` in `build_question_bank.py`); Israel
   and Venezuela's code 6 also includes a parent who had passed away, worded so for their records.

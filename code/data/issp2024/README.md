# ISSP 2024 — Digital Societies I

Training data (`--train issp2024`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2024.

## Source
- GESIS study **ZA10020**, data file version **v1.0.0**: <https://search.gesis.org/research_data/ZA10020>
- 44,703 respondents, 27 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA10020_v1-0-0.dta` (+ `.zip`), `ZA10020_bq.pdf` (source
  questionnaire, August 2023: the wording comes from here), `ZA10020_q_all.zip` (national questionnaires, used only to
  check parliament names). There is no `_missing.txt` for this year: negative codes are the non-answers.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/issp2024/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2024/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options |
| `issp2024_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2024_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2024_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 44,703 respondents (all kept), 16 attributes, 78 questions in 36 blocks, 27 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2024.jsonl`, used until 2026-10-13): 20,279 records, 40,067 question targets (28,008 `score`, 11,192 `choice`, 867 `noul`), 647 persona groups.
Items: all 78: 48 `score`, 18 `choice`, 12 `noul`.
All of it is training data.

## Decisions specific to 2024
1. **Questions asked of only some people**: Q1 splits the sample (`ASKED_OF`):

   | Blocks | Asked of | Extra persona line |
   |---|---|---|
   | Q2–Q5 | people who did not use the Internet | `Internet use in the last 12 months: Did not use the Internet` |
   | Q6–Q8, Q10–Q16 | people who used the Internet | `Internet use in the last 12 months: Used the Internet` |
   | Q9 | Internet users who deliberately cut down | the line above + `Deliberately reduced time on the Internet: Yes` |
   | Q1, Q17–Q36 | everyone | none |

   The non-user items keep few targets (about 80 groups reach 10 answers).
2. **"No access" (Q6, code 9) is a real answer and is kept**; Q6a–c are `choice`.
3. **Items not asked in some countries** simply have no targets there: Q2–Q5 in Canada; Q2k in Japan; Q17b in Hungary
   (Hungary asked a different wording, stored as `HU_v43`, not used); the optional Q36c–e in Sweden, the US and South
   Africa.
4. **Parliament names** (Q26a): see `data/README.md`; for Canada, Greece, Israel, the Netherlands and Poland taken from
   their 2024 questionnaires. Czech Republic: not checked (its questionnaire PDF is a scanned image).
5. Moving this year onto the shared builder changed nothing: `issp2024.jsonl` is byte-identical to the earlier build.

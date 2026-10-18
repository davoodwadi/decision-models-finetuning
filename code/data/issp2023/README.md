# ISSP 2023 — National Identity and Citizenship

Training data (`--train issp2023`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2023.

## Source
- GESIS study **ZA10010**, data file version **v1.0.0**: <https://search.gesis.org/research_data/ZA10010>
- 43,517 respondents, 29 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA10010_v1-0-0.dta` (+ `.zip`), `ZA10010_v1-0-0_missing.txt`
  (non-answer codes), `ZA10010_bq.pdf` (source questionnaire, August 2022: the wording comes from here),
  `ZA10010_q_all.zip` (national questionnaires, used for the regional association names).

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/issp2023/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2023/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options |
| `issp2023_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2023_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2023_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 43,517 respondents (all kept), 16 attributes, 68 questions in 44 blocks, 29 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2023.jsonl`, used until 2026-10-13): 26,549 records, 42,111 question targets (40,215 `score`, 1,213 `choice`, 683 `noul`), 685 persona groups.
Items: all 68: 62 `score`, 5 `choice`, 1 `noul`.
All of it is training data.

## Decisions specific to 2023
1. **All 68 items are used** (since 2026-10-12; earlier builds left out v3, v20 and v61–v67).
2. **Q1c "to be a [religion]"** (v3): each country filled in its own dominant religion; the wording here is "To be an
   adherent of the dominant religion in {country}".
3. **Optional Q38–Q43** (v61–v67) ask about the country's regional association. `{association}` is filled per country
   (`ASSOCIATION` in `build_dataset.py`): USMCA (Canada, the United States), SAARC (India), ASEAN (the Philippines),
   checked in their national questionnaires; the European Union for the European countries (Switzerland, Germany,
   Finland, Greece, Croatia, Hungary, Italy, the Netherlands, Norway, Sweden), not checked. Members answered Q39a and Q43, non-members Q39b and Q42. Q39a/b include the printed
   answer "Have never heard of …", so they are `choice`. Q39–Q43 were asked only of people who had heard of the
   association: `Has heard or read about <association>: Yes`.
4. **v20 (Q6a, limiting imports) is the same question as ISSP 2020 v5**; both are training data.
5. **Nationality words** from `issp_countries.csv` ("truly Filipino", "New Zealand citizenship").
6. Q4 (statement A ↔ statement B) keeps the questionnaire's order. Q44 (optional) was asked in 12 countries.

# ISSP 2018 — Religion IV

Training data (`--train issp2018`), used whole. Calibration and test sets are SubPOP's (`data/subpop/README.md`).
Built by the shared builder: the rules every ISSP training year follows are in `data/README.md`; this file lists
only what is specific to 2018.

## Source
- GESIS study **ZA7570**, data file version **v2.1.0**: <https://search.gesis.org/research_data/ZA7570>
- 46,267 respondents, 33 countries.
- Files in `raw/` (downloaded from the GESIS page): `ZA7570_v2-1-0.dta` (+ `.zip`), `ZA7570_v2-1-0_missing.txt` (non-answer
  codes), `ZA7570_bq.pdf` (source questionnaire, June 2017: the wording comes from here), plus the codebook and
  national questionnaires as downloaded.

## Rebuild
From the project root, in this order:
```bash
kev/.venv/bin/python data/issp2018/build_question_bank.py
```
```bash
kev/.venv/bin/python data/issp2018/build_dataset.py
```

| File | Made by | Contents |
|---|---|---|
| `question_bank.csv` | `build_question_bank.py` | one row per item: question number, wording, type, data codes, answer options, why excluded |
| `issp2018_respondents.parquet` | `build_dataset.py` | one row per respondent: country, every attribute's data code, every question's answer code |
| `issp2018_attributes.json` | `build_dataset.py` | per attribute and country: the text of each code, as that country had it |
| `issp2018_questions.json` | `build_dataset.py` | per question and country: wording, options, answer code → option; who was asked each block |

Result: 46,267 respondents (all kept), 16 attributes, 78 questions in 45 blocks, 33 countries. Training records are drawn from them by `issp_sampler.py` (`data/README.md`). The former fixed-group dataset (`issp2018.jsonl`, used until 2026-10-13): 32,261 records, 54,227 question targets (45,102 `score`, 7,080 `choice`, 2,045 `noul`), 846 persona groups.
Items: all 78 v-numbered items: 64 `score`, 10 `choice`, 4 `noul`.
All of it is training data.

## Decisions specific to 2018
1. **Parents' and own religion** (Q18–Q20): the data's harmonised religious groups (v35, v37, v39: Catholic, Protestant, Orthodox, …) are used; the country-specific lists (`XX_v34`, …) are not.
2. Where the questionnaire asks for country-specific examples (Q27 "icon, retablo, mezuzah, menorah, or crucifix", Q28 "shrine/temple/church/mosque"), the source questionnaire's examples are used.
3. **Printed answers kept**: Q21–Q22 "No mother/mother not present", "No father/father not present"; these items are `choice`.
4. Q7a asks about confidence in the national parliament: `{parliament}` from `issp_countries.csv`.
5. Every question was asked of everyone: no persona lines added.

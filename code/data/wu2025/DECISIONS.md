# wu2025 decision log

Wu et al. (2025), Energy Research & Social Science: conjoint experiment on Community Choice Aggregation (CCA)
electricity plans, Lucid, April-May 2022. Each entry records what was observed, the options considered, and what was
decided. All decisions are the user's.

## Source files (in `dataverse_files.zip`)

| File | What it is |
|---|---|
| `data/Conjoint data.dta` | Authors' cleaned final sample: **1,862 respondents** x 20 rows (10 choice tasks x 2 profiles) = 37,240 rows. Written by `1. Data Cleaning.do`. |
| `CCA_Data.xlsx` | Raw Qualtrics export; row 2 holds each column's question text. |
| `1. Data Cleaning.do` | Authors' cleaning: drop non-consent, under 18, "I do not reside in the United States", failed attention check 2 ("Red,Green"), then failed attention check 1 ("Somewhat disagree"). |

Correction: an earlier quick look read only the first 20,000 rows of the `.dta` and reported 1,000 respondents;
the full file has 1,862 (all passed both checks; 2 are marked `Finished = False`).

Personal data: both files contain `IPAddress`, `LocationLatitude` and `LocationLongitude`. These must never be copied
into any derived file.

The Lucid panel profile columns (`LH`, `gender`, `hhi`, `LK`, `hispanic`, `LM`, `political_party`, `region`) are numeric
codes without labels in the data, and duplicate the survey's own demographic questions.

## Persona mapping (`persona.py`, dataset key `wu2025`)

Persona values that map to NaN are dropped from that respondent's persona (`base.py`). Question text from row 2 of
`CCA_Data.xlsx`; counts from the `.dta` (1,862 respondents).

| Attribute | Source | Decision |
|---|---|---|
| `SEX` | `sex` ("What is your sex?") | Female → `Female` (973), Male → `Male` (877), blank → NaN (12; no answer recorded, not "Prefer not to say"). |
| `AGE` | `age` ("How old are you?", 8 brackets) | Union with shared levels: the five ranges identical to berke2023's reuse its levels (18 - 24 → `18 - 24 years`, 25 - 34 → `25 - 34 years`, 35 - 44 → `35 - 44 years`, 45 - 54 → `45 - 54 years`, 55 - 64 → `55 - 64 years`); three new global levels in the same format: `65 - 74 years` (234), `75 - 84 years` (75), `85 or older` (10). Not bucketed: 25-34 and 45-54 (572 respondents) cross `AGE_BINS`. `65 or older` (berke2023) overlaps the three new levels. |
| `RACE` | `race` (select-all, comma-separated labels; no Hispanic option) | Union: White → `White`, Black or African American → `Black`, American Indian or Alaska Native → `American Indian or Alaska Native`, Other → `Other`, Asian/Pacific Islander → new global level `Asian/Pacific Islander` (69; combines two groups the global set keeps apart, so not mapped to `Asian`), Multi-racial (an answer option) → `Two or more races`; two or more boxes → `Two or more races` (73 + 24 = 97); blank → NaN (10). |
| `HISPANIC` | `ethnicity` ("Are you Spanish, Hispanic, or Latino or none of these?") | Separate attribute, as in Grummon (race keeps its own answers): Spanish, Hispanic, or Latino → `Yes` (199), None of these → `No` (1,629), blank → NaN (34). `HISPANIC` was moved into `persona.py` (global levels `Yes` / `No`, one map per dataset) so Grummon 2023, Grummon 2025 and wu2025 share it; Grummon's persona output is unchanged. |

Implemented so far: `SEX`, `AGE`, `RACE`, `HISPANIC` (`DATASET_PERSONA_COLS["wu2025"]`, `WU2025_COL_TO_PERSONA`, `WU2025_RACE_LABELS`, a `"wu2025"` entry in each attribute block, and the `apply_persona_mapping` branch, which turns "" into NaN). Still to decide: education, income, party, ideology, region and country, marital status, main activity, residence, and possible new attributes.

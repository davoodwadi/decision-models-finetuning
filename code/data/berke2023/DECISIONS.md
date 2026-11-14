# berke2023 decision log

Berke, Ding, Chin, Gopalakrishnan, Larson, Balakrishnan, Li (2023), "Drone delivery and the value of customer privacy:
A discrete choice experiment with U.S. consumers", Transportation Research Part C 157, 104391. Prolific, Qualtrics.
Each entry records what was observed, the options considered, and what was decided. All decisions are the user's.

## Source files (in `drones-consumer-privacy-master.zip`)

| File | Rows x columns | What it is |
|---|---|---|
| `data/sample-preprocessed.csv` | 3,715 x 831 | Authors' cleaned public data, one row per respondent, original answer codes. |
| `data-analysis/choice-model-data-long.csv` | 36,297 x 37 | Same choices in long format (3,625 respondents) with the authors' recoded variables; no consumer questions. |
| `data/sample-fake.csv` | 77 x 847 | Fake responses used to probe the survey logic; not real respondents. |
| `data/v0-icrat-sample.csv` | 22 x 145 | Earlier "ICRAT" conference version of the survey (different instrument). |

Question text sources: `survey-questions/generate-questions.py` + `attributes.py` (exact choice-question HTML and levels),
`survey.pdf` (privacy explanation, scenario introductions, consumer and demographic questions), `data/fields.csv`
and `data-analysis/codebook.py` (labels). The raw Qualtrics export is not public.

Authors' cleaning (`preprocess-survey-data.ipynb`): kept Prolific participants who finished, dropped 2 test responses
and anyone failing any of the 3 attention checks; removed Prolific IDs and Qualtrics location metadata.

**Decision (data file).** Use `sample-preprocessed.csv` only. Not used: the long file (recoded demographics, no
consumer questions), `sample-fake.csv` (not real people), the ICRAT pilot (different survey).

## Choice questions

**Observation (200 columns per product type).** Each of the 100 generated questions per product type appears twice in
`sample-preprocessed.csv`, as `CHOICE_..._1` and `CHOICE_..._1.1` (pandas renames a duplicate column name). The two
copies are the two survey paths in `survey.pdf`: `..._1` = 268 respondents who would order only one product type
(12 questions from that type); `..._1.1` = 3,357 respondents who would order 2-4 types (3 questions per type).
90 respondents would order none and made no choices. No respondent answered both copies of the same question.
Total: 36,297 choices from 3,625 respondents.

| Product type (`v`) | Choices | Respondents |
|---|---:|---:|
| 0 take-out food | 9,528 | 3,014 |
| 1 prescription medications | 11,481 | 3,323 |
| 2 liquor store items | 7,269 | 2,381 |
| 3 last-minute groceries | 8,019 | 2,577 |

Product types differ in the question's product wording, the introduction screen shown before that type's questions,
and the levels (medications: fees $0/$1/$5/$10, waits 30 minutes/2 hrs/6 hrs/1 day; others: $0/$1/$3/$5,
15/20/30/45 minutes). The drone-privacy explanation was shown once, before all choices.

**Decision (audit rows).** Four choice rows in the audit app, one per product type, each with its own wording,
introduction and levels.

## Audit catalog (`stimulus.py` -> `question_audit.csv`)

**Decision (starting values).** The user asked Claude to judge each question's `include` and `kev_type`; the user
verifies them in the audit app. Each row's `review_note` gives the reason.

**Text sources.** Consumer and demographic question text: `fields.csv` (Qualtrics header). Choice stem and levels:
`generate-questions.py` / `attributes.py`. Privacy explanation and product introductions: `survey.pdf` (checked
line by line against the PDF text; the PDF is the authors' abbreviated version and has no images). Answer labels:
`codebook.py`; the "reasons for never ordering" options come from `survey.pdf`, and their codes 1-3 assume the
PDF's listing order (code 4 = Other, from the `_4_TEXT` column name).

| Row(s) | Starting value | Reason |
|---|---|---|
| `delivery_choice_v0..v3` | include, `choice` | The experiment's outcome. Context starts as `qid:privacy_explainer+intro_v<k>`, because "NO/YES privacy" is defined only in the explanation screens. |
| `Q_online_shopping`, `Q_takeout_food`, `Q_online_groceries` | include, `score` | Ordered 5-level frequency scale; everyday behaviour a persona can inform. |
| `Q_v_1` (medications), `Q_v_2` (liquor) | include, `choice` | Yes / No / I don't know; not scored because "I don't know" is not a scale point. Tagged `has-dont-know`. |
| `Q_v_*_why_no` | exclude | Select-all-that-apply, shown only to "Never"/"No" answerers; option codes partly assumed. |
| `*_TEXT`, `Q_comments` | exclude | Free text (comment samples not shown). |
| Demographics | exclude | Used for the persona. |
| `privacy_explainer`, `intro_v0..v3` | exclude | Information screens (`0_DB`), used as context. |

## Persona mapping (`persona.py`, dataset key `berke2023`)

Persona values that map to NaN are dropped from that respondent's persona (`base.py`).

| Attribute | Source | Decision |
|---|---|---|
| `SEX` | `Q_gender` ("What is your gender?") | 1 Male → `Male`, 2 Female → `Female`, 3 Other → new global level `Other` (68; union rule, rather than `Non-binary` or NaN), 4 Prefer not to answer → `Prefer not to say` (12; first NaN, changed 2026-11-14 when `Prefer not to say` became a global level for every dataset that offers it). |
| `AGE` | `Q_age` ("What is your age?", 6 brackets) | All six brackets added as new global levels, verbatim from `codebook.py` (`18 - 24 years`, `25 - 34 years`, `35 - 44 years`, `45 - 54 years`, `55 - 64 years`, `65 or older`; chosen over survey.pdf's "65 and older" because the codebook was written against the Qualtrics export). Two brackets (25-34: 1,253; 45-54: 457) cross the global `AGE_BINS` boundaries, so mapping into the four standard bins was not possible without guessing; NaN for them would have made a missing age line mean "25-34 or 45-54". Mapped with a dict, not `bucket_age_series`. |
| `RACE` | `Q_race` (select-all, comma-separated codes) | Single box → category: White → `White`, Black or African American → `Black`, Asian → `Asian`, Hispanic or Latino → `Hispanic`, Other → `Other`, and two new global levels: `American Indian or Alaska Native` (14, verbatim) and Native Hawaiian or Pacific Islander (5) → `Native Hawaiian or Other Pacific Islander` (the official OMB wording, chosen when Grummon's identical-meaning "Native Hawaiian or Other Pacific Islander" was mapped to the same level; rule: same meaning → one level, standard wording). Two or more boxes → new global level `Two or more races` (273; 244 include White, 119 include Hispanic or Latino). Label "Two or more races" is constructed (not shown to respondents). Covid and Kriner2026 were then changed to the same multi-box rule (2+ boxes → `Two or more races`); their "American Indian" option became its own level `American Indian`, separate from this dataset's "American Indian or Alaska Native" because the wording differs. |
| `US_INC` | `Q_income` ("What was your entire household income in the previous year before taxes? Please give your best guess.") | One global level per bracket, from the income union (one level per distinct range across all datasets, format `$X - $Y`): 1 → `Less than $25,000` (existing), 2 → `$25,000 - $49,999` (new), 3 "$50,000 to $74,999" → `$50,000 - $75,000` and 4 "$75,000 to $99,999" → `$75,000 - $100,000` (same ranges as Twin-2K's), 5–7 → `$100,000 - $149,999`, `$150,000 - $199,999`, `$200,000 or more` (new, shared with Grummon). The same union split ISSP's and Grummon's previously merged brackets. |

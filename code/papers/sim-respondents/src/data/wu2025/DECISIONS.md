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
| `c_alphan` | none | All respondents `United States`: the authors' cleaning removed "I do not reside in the United States". |
| `US_REG` | `state` (dropdown "50 States, D.C. and Puerto Rico"; data holds names) | Hand-written state name → Census region map (51 entries, same regions as Covid's `STATE_CODE_TO_REGION`); chosen over the Lucid panel's unlabelled `region` code. All 50 states and DC appear; Puerto Rico never does (mapping fails if a value outside the 51 names appears). |
| `SEX` | `sex` ("What is your sex?") | Female → `Female` (973), Male → `Male` (877), blank → NaN (12; no answer recorded, not "Prefer not to say"). |
| `AGE` | `age` ("How old are you?", 8 brackets) | Union with shared levels: the five ranges identical to berke2023's reuse its levels (18 - 24 → `18 - 24 years`, 25 - 34 → `25 - 34 years`, 35 - 44 → `35 - 44 years`, 45 - 54 → `45 - 54 years`, 55 - 64 → `55 - 64 years`); three new global levels in the same format: `65 - 74 years` (234), `75 - 84 years` (75), `85 or older` (10). Not bucketed: 25-34 and 45-54 (572 respondents) cross `AGE_BINS`. `65 or older` (berke2023) overlaps the three new levels. |
| `RACE` | `race` (select-all, comma-separated labels; no Hispanic option) | Union: White → `White`, Black or African American → `Black`, American Indian or Alaska Native → `American Indian or Alaska Native`, Other → `Other`, Asian/Pacific Islander → new global level `Asian/Pacific Islander` (69; combines two groups the global set keeps apart, so not mapped to `Asian`), Multi-racial (an answer option) → `Two or more races`; two or more boxes → `Two or more races` (73 + 24 = 97); blank → NaN (10). |
| `HISPANIC` | `ethnicity` ("Are you Spanish, Hispanic, or Latino or none of these?") | Separate attribute, as in Grummon (race keeps its own answers): Spanish, Hispanic, or Latino → `Yes` (199), None of these → `No` (1,629), blank → NaN (34). `HISPANIC` was moved into `persona.py` (global levels `Yes` / `No`, one map per dataset) so Grummon 2023, Grummon 2025 and wu2025 share it; Grummon's persona output is unchanged. |
| `EDULEVEL` | `education` ("What is the highest level of school you have completed or the highest degree you have received?") | The current 4-level harmonised scheme, same as Covid's 8-level mapping: Less than high school degree → `Less than high school` (39); High school graduate, Some college but no degree → `High school or some college` (812); Associate degree → `Associate's degree` (232); Bachelor's, Master's, Professional, Doctoral → `College graduate/some postgrad` (778); blank → NaN (1). Whether education should become a union across all datasets is a separate open decision. |
| `US_INC` | `income` ("In the previous year, what was your total household income (before taxes)? ...") | Income union: one global level per bracket (format `$X - $Y`). Shared with existing levels: Less than $10,000, $40,000 - $49,999, $50,000 - $59,999, $100,000 - $149,999. New: `$10,000 - $19,999`, `$20,000 - $29,999`, `$30,000 - $39,999`, `$60,000 - $69,999`, `$70,000 - $79,999`, `$80,000 - $89,999`, `$90,000 - $99,999`, `$150,000 or more` (global income levels 27 → 35). Blank → NaN (3). |
| `US_PRTY` | `partisan` ("Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent, or something else?") | Question A only, as for Kriner2026: Democrat (776), Republican (486), Independent (510), Something else (89) verbatim; blank → NaN (1). Not used: the strength follow-ups (`democratstrength`, `republicanstrength`) and `leaner` (asked correctly to Independents / Something else), which would have allowed Grummon 2023's 7-point levels. |
| `PARTY_LR` | `ideology` ("When it comes to politics, would you describe yourself as liberal, conservative, or neither liberal nor conservative?", 7 points: Very / Somewhat / Slightly) | Mapped by wording (union), not scale position: Very liberal (223), Somewhat liberal (188), Somewhat conservative (278), Very conservative (323) → existing levels; Moderate; middle of the road → `Moderate` (590); Slightly liberal (130), Slightly conservative (127) → new global levels; blank → NaN (3). Note: wu2025's "Somewhat liberal" is the 2nd step of its scale, while in Covid / Kriner2026 / Twin-2K it is the 3rd; the shared label follows the words respondents chose. |
| `POL_LABEL` (new attribute) | `ideology_original` ("When it comes to politics, would you describe yourself as libertarian, socialist, communist, liberal, conservative, centrist, or something else?") | New persona attribute `POL_LABEL`, named with the question text; levels verbatim: Conservative (672), Liberal (536), Other (299), Centrist (125), Libertarian (122), Socialist (76), Communist (26); blank → NaN (6). Only wu2025 uses it; excluded from ISSP's expected columns. |
| `MARITAL` | `marriage` ("Are you now married, widowed, divorced, separated or never married?") | Married (873), Divorced (236), Widowed (88), Separated (35) → same levels; Never married → `Never been married` (625, same meaning); blank → NaN (5). "Living with a partner" was not offered. |
| `MAINSTAT` | `employment` ("Which statement best describes your current employment status?") | Same meaning → existing level: Working (self-employed) → `Self-employed` (163), Not working (retired) → `Retired` (392), Not working (looking for work) → `Unemployed` (148). New verbatim levels: `Working (paid employee)` (829; narrower than `In paid work`), `Not working (disabled)` (160; not "permanently sick or disabled"), `Not working (temporary layoff from a job)` (23), `Not working (other)` (126; user: its own level, not the catch-all `Other`). Prefer not to answer → new level `Prefer not to say` (20, the sex wording). Blank → NaN (1). |
| `RESTYPE` | `residence` ("Please select the choice that best describes your primary residence (the place where you usually live):") | Apartment → `Apartment` (393, shared with berke2023); new verbatim levels `Single family home` (1,224; narrower than berke2023's `Private home`), `Townhouse` (86), `Mobile home` (80), `Condominium` (51), `Something else` (28; catch-all kept verbatim). |
| `OWNHOME`, `RENTHOME`, `ORIENTATION`, `MILITARY`, `ELECHEAT` (new attributes) | `residenceownership` ("Do you own your primary residence?"), `residencerental` ("Do you rent your primary residence?"); both asked to everyone | Two new persona attributes, each named with its question, levels `Yes` / `No` verbatim. OWNHOME: Yes 1,193, No 667, blank 2 → NaN; RENTHOME: Yes 745, No 1,116, blank 1 → NaN. 239 respondents answered inconsistently (158 own and rent, 81 neither); their answers are kept as given. Chosen over using only the ownership question or a constructed own/rent category. |
| `ORIENTATION` (new attribute) | `orientation` ("Which of the following best describes your sexual orientation?") | New persona attribute named with the question, levels verbatim: Heterosexual (straight) 1,643, Bisexual 110, Homosexual (gay) 59, Other 23, Prefer not to say 24; blank → NaN (3). Chosen over leaving it out (Claude had recommended leaving it out as weakly related and single-dataset). |
| `MILITARY` (new attribute) | `military` ("Have you ever served on active duty in the US Armed Forces?") | New persona attribute named with the question, `Yes` (261) / `No` (1,586) verbatim; blank → NaN (15). |
| `ELECHEAT` (new attribute) | `householdheat` ("Does your household use electricity for heat?") | New persona attribute named with the question, `Yes` (1,140) / `No` (668) verbatim; blank → NaN (54). Kept in the persona (a household fact relevant to electricity bills); the other energy-background questions (electricity in rent, IOU, CCA knowledge / participation) go to the audit catalog as questions. |

Implemented so far: `c_alphan`, `US_REG`, `SEX`, `AGE`, `RACE`, `HISPANIC`, `EDULEVEL`, `US_INC`, `US_PRTY`, `PARTY_LR`, `POL_LABEL`, `MARITAL`, `MAINSTAT`, `RESTYPE`, `OWNHOME`, `RENTHOME`, `ORIENTATION`, `MILITARY`, `ELECHEAT` (`DATASET_PERSONA_COLS["wu2025"]`, `WU2025_COL_TO_PERSONA`, `WU2025_RACE_LABELS`, a `"wu2025"` entry in each attribute block, and the `apply_persona_mapping` branch, which turns "" into NaN). All persona attributes are decided. Not used: the Lucid panel profile columns (unlabelled codes), the party strength / lean follow-ups.

## Audit catalog (`stimulus.py` -> `question_audit.csv`)

**Data.** Both files are read in memory from `dataverse_files.zip` and never extracted (they contain IP addresses and
locations; the catalog drops those columns on load). Answers and counts come from the `.dta` (authors' final sample,
1,862); question text from the export's question row (`CCA_Data.xlsx`, needs `openpyxl`).

**Conjoint.** One row, `plan_choice`: 18,591 answered tasks (10 per respondent; 29 tasks from 10 respondents
unanswered), 8,212 distinct displayed pairs, all previewed. Options `Choice 1` / `Choice 2`; each plan's description
lists its three attributes in the respondent's row order (`.dta` X / Y / Z rows, identified by `Xmoney` / `Xpercent`;
verified against the attribute columns; order fixed per respondent across the 10 tasks). The attribute names
("Percentage of renewables", "Monthly bill", "Locally produced renewables"), the background screen and the bill message
are **not in the export**; they come from the paper (Section 2.2, footnote 2). The stem ("Please carefully review the
options detailed below, then please answer the questions. Which of these choices do you prefer?") is from the export,
without its "Task t/10:" prefix.

**Option order.** The data stores answer labels without codes, so the on-screen order is unknown. Starting rule
(Claude's): scales low → high; unordered answers in the order the question text names them, otherwise as noted.

**Starting values** (Claude's evaluation; the user verifies; reasons in each row's `review_note`):

| Rows | Starting value | Reason |
|---|---|---|
| `plan_choice` | include, `choice` | The experiment's outcome. |
| Opinion / energy items: `cuomopolicyapproval`, `prior_cca*` (3), `pollutioninterest`, `envirospending`, `energyneeds`, `warmingconsensus`, `warmingbelief`, `warmingbelief1/2`, `warmingcause1/2`, `billknowledge`, `billestimate`, `currentiou`, `iousatisfaction`, `mainiou`, `mainiousatisfaction`, `ccaknowledge`, `currentcca`, `polint`, `trust_sci_*` (2), `electricityinrent`, `thirdparty*` (2 arms), the 14 "My ideal third party would ..." items, `nominee`, `thirdpartyideology1/2`, `thirdpartycaveat` | include (`score` for ordered scales, `choice` otherwise) | Complete wording in the export. |
| `ccaUP`, `ccaDOWN`, `ccaNOCHANGE`, `cca_state`, `local_cca`, `personal_cca`, `ccabenefit` | exclude, tag `missing-context` | Asked after the CCA information screens (two randomized versions, `ccatreatment`), whose text is in neither the export nor the paper. |
| 24 harassment items (`generalharassment*`, `cuomoharassment*`) | exclude, tag `truncated-text` | The export cuts the stem ("Please indicate whether you be..."); randomized general vs. Cuomo wording. |
| `consent`, `check1`, `check2`, `cca_attention`, `mix_attention` | exclude | Consent and attention / comprehension checks. |
| `billestimate2`, `employment_7_TEXT`, `comment` | exclude | Free entry (comment samples not shown). |
| Persona and party follow-up columns | exclude | Used for the persona (`persona.py`) or deliberately not used there. |
| `conjoint_intro` | exclude (`0_DB`) | Context screen for `plan_choice`. |

## Parquet export and loaders (`__init__.py`)

- `export_wu2025_parquet` (audit app "This dataset" / "All datasets" buttons) writes `wu2025_US_long.parquet` from the
  rows marked `include=True`, with the audit app's `drop_options`, `context_source`, `context_sep` and
  `question_overrides.json` edits. IP / location columns are never loaded.
- `plan_choice`: one row per answered task; `option_texts = ['Choice 1', 'Choice 2']`, `option_descriptions` = each
  plan's three attributes in the respondent's row order; `qid` = `wu2025:plan_choice:<plan 1>_<plan 2>_<attribute order>`
  (same screen -> same qid). Question = background screen + bill message + stem.
- Other included questions: `option_texts` in the catalog's option order (codes 1..n in that order).
- Loaders: `load_wu2025_df`, `load_wu2025_records`, `load_wu2025_eval` (no dataset-specific evaluator yet; the paper's
  benchmark would be its AMCEs / marginal means and attribute salience scores).
- Dry run with the reviewed audit CSV (identical to the starting values): 90,979 rows from 1,862 respondents,
  46 questions (27 score, 19 choice; plan_choice 18,591 rows), all building into valid training records.

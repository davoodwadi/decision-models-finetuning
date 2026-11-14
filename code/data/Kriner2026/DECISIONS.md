# Kriner2026 decision log

Goldfarb, Kriner et al. (2026), sustainable aviation fuel (SAF) survey, Verasight, 1,000 US respondents.
Each entry records the issue, the options considered, and what was decided. All decisions are the user's.

## Persona mapping (`persona.py`)

Persona values that map to NaN are dropped from that respondent's persona (`base.py`), so no line appears for that attribute.

Implemented as dataset key `kriner2026` in `persona.py` (`DATASET_PERSONA_COLS`, `KRINER2026_COL_TO_PERSONA`,
`KRINER2026_RACE_COLS`, a `"kriner2026"` entry in each attribute block of `PERSONA_MAPS` (region uses the shared `STATE_CODE_TO_REGION`), and the `apply_persona_mapping` branch).
`stimulus.build_persona_input_df` builds its input, including `age_years` decoded from the `age` value labels.
All persona attributes are decided.

| Attribute | Source variable | Decision |
|---|---|---|
| `c_alphan` (Country) | none (US panel) | All respondents `United States`. |
| `SEX` | `gender` | 1 Male → `Male`, 2 Female → `Female`, 3 Prefer not to say → `Prefer not to say` (5; first NaN as in Covid, changed 2026-11-14 together with Covid when it became a global level). |
| `AGE` | `age` | The `.dta` stores codes, not ages (code 1 = 18, …, code 80 = 97). Decode through the value labels, then bucket with `AGE_BINS`. |
| `EDULEVEL` | `education` | Same 8 labels and mapping as Covid: 1 → `Less than high school`; 2, 3 → `High school or some college`; 4 → `Associate's degree`; 5–8 → `College graduate/some postgrad`. |
| `RACE` | `native_american`, `asian`, `black`, `hispanic`, `white`, `race_6` | Covid rule: one box → that category, with American Indian kept verbatim as `American Indian` (10); two or more boxes → `Two or more races` (98, 65 of whom checked Hispanic). Chosen over Hispanic-first (Grummon) so it matches Covid. Revised 2026-11-14 together with Covid, after berke2023 added `Two or more races`: previously American Indian and 2+ boxes → `Other`, which made `Other` mostly mean multi-race. "American Indian" is kept separate from berke2023's "American Indian or Alaska Native" (verbatim wording). |
| `US_INC` | `income` | Same 6 labels as Covid, each mapped to the identically worded global level. |
| `PARTY_LR` | `ideology` | Same 7 labels as Covid, one-to-one to the global levels (persona uses the global lower-case spelling). |
| `US_PRTY` | `party` | Same as Covid: 1 → `Republican`, 2 → `Democrat`, 3 → `Independent`, 4 Other/don't know → `Something else`. `party_lean` (asked only to Independents) is not used. The authors' `party3` is not used (it folds leaners into parties and Other/don't know into Independent). |
| `US_REG` | `state` | State codes are identical to Covid's `STATE_LABELS`; Covid's state → Census region map is reused. Chosen over the Verasight profile `vs_region` (missing for 18, differs for 17). |
| `RELIGGRP` | `relig` | See below. |
| `URBRURAL` | `urban_rural_best` | 1 → `Urban`, 2 → `Suburban`, 3 → `Rural`. `Suburban` (476 respondents) had no global level; it was added to `GLOBAL_PERSONA_LEVELS["URBRURAL"]` so the global levels stay the union of the datasets' levels, instead of mapping it to ISSP's differently worded "The suburbs or outskirts of a big city" or setting it to NaN. `urban_rural_least` is not used. |
| `VOTE_LE` | `vs_vote_2024` (Verasight panel profile, 2024 presidential election; not asked in this survey) | 1 I voted → `Yes`, 4 I did not vote → `No`, 2 → new global level `I was too young to vote` (10), 3 → new global level `I was not registered to vote` (77); 2 missing → NaN. Union rule: the two reasons were added as levels instead of folded into `No`, so they overlap in meaning with other datasets' `No`. |
| (none) | `vs_votechoice_2024` (Verasight panel profile: Kamala Harris 300, Donald Trump 283, Someone else 81; voters only) | Not used. No global attribute records vote choice; adding a new `VOTE_CHOICE` attribute was considered and declined. |

### Religion

**Issue.** The survey offered 6 answers (Protestant, Catholic, Jewish, Muslim, Other, None), while the global levels split
what this survey calls "Other" into `Orthodox`, `Other Christian`, `Buddhist`, `Hindu`, `Other`, and what it calls "None" into
`Atheist`, `Agnostic`, `Nothing in particular`. So:
- "None" (289 respondents) has no single matching global level.
- "Other" (247 respondents) is broader than the global `Other`, which in Covid and Twin-2K excludes the separately listed groups.

**Options considered.** (a) leave religion out of the persona; (b) map Other → `Other` and None → `Nothing in particular`;
(c) add new global levels for this survey's categories; (d) map the matching answers and set None to NaN.

**Decision.** Protestant → `Protestant`, Catholic → `Catholic`, Jewish → `Jewish`, Muslim → `Muslim`, Other → `Other`,
None → `No religion`.
History: first None → NaN (rejected: a missing line would mean "None" here but "not asked" elsewhere), then a new level
`None`, then (2026-11-14) the religion union across all datasets, where "None" shares the level `No religion` with ISSP's
"No religion" and WVS's "Do not belong to a denomination" (same meaning, clearer wording than a bare "None").

**Known consequences.**
- `Other` here includes groups that other datasets list separately (Orthodox, Mormon, Buddhist, Hindu, ...): a
  catch-all answer is only as specific as the list it was offered with.

## Parquet export (`__init__.py`, `export_Kriner2026_parquet`)

- Writes `kriner2026_US_long.parquet` from the rows marked `include=True` in `question_audit.csv`, with the audit app's
  `drop_options`, `context_source`, `context_sep` and `question_overrides.json` edits. Run it from the audit app's export button.
- `flight_choice`: one row per respondent x task; `option_texts = ['Flight 1', 'Flight 2']`, `option_descriptions` = each
  flight's attributes in the respondent's display order. `qid` = `kriner2026:flight_choice:<profile A>_<profile B>_<attribute order>`,
  so identical screens share a `qid`.
- No `reversed` column (decision 2026-11-14): nothing in training or inference reads it (`build_long_records` drops it;
  the model sees only the listed option order), and the repo used two conflicting meanings for it (ISSP: codes run
  downward in the listed order; Covid: the high end is listed first). Excluded from the audit CSV, the UI and the parquet.

## Loaders (`__init__.py`)

- `load_kriner2026_df`: reads the parquet, restores list columns, adds `year = "kriner2026"` and `one_hot`.
- `load_kriner2026_records`: training records via `build_long_records` (optionally state-grouped).
- `load_kriner2026_eval` + `evaluate_kriner2026_ols`: Table 2 replication. Each flight choice gives two rows (one per flight);
  attribute codes are parsed from `option_descriptions` (on-screen text -> the authors' `*_n` codes, `FLIGHT_LEVEL_CODES`).
  Reports OLS MAPE, MAE and Spearman rho between model and human AMCEs. Checked: with human choices as predictions,
  the human AMCEs equal the paper's Table 2 column (1) on all 12 terms.
- Not decided yet: whether Kriner2026 is training data, evaluation data, or both, and if both, how train and eval are kept apart.

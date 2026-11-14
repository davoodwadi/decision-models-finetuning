# Kriner2026 decision log

Goldfarb, Kriner et al. (2026), sustainable aviation fuel (SAF) survey, Verasight, 1,000 US respondents.
Each entry records the issue, the options considered, and what was decided. All decisions are the user's.

## Persona mapping (`persona.py`)

Persona values that map to NaN are dropped from that respondent's persona (`base.py`), so no line appears for that attribute.

Implemented as dataset key `kriner2026` in `persona.py` (`DATASET_PERSONA_COLS`, `KRINER2026_COL_TO_PERSONA`,
`KRINER2026_RACE_COLS`, the `PERSONA_MAPS[...]["kriner2026"]` block, and the `apply_persona_mapping` branch).
`stimulus.build_persona_input_df` builds its input, including `age_years` decoded from the `age` value labels.
All persona attributes are decided.

| Attribute | Source variable | Decision |
|---|---|---|
| `c_alphan` (Country) | none (US panel) | All respondents `United States`. |
| `SEX` | `gender` | 1 Male → `Male`, 2 Female → `Female`, 3 Prefer not to say → NaN (5 respondents). Same as Covid. |
| `AGE` | `age` | The `.dta` stores codes, not ages (code 1 = 18, …, code 80 = 97). Decode through the value labels, then bucket with `AGE_BINS`. |
| `EDULEVEL` | `education` | Same 8 labels and mapping as Covid: 1 → `Less than high school`; 2, 3 → `High school or some college`; 4 → `Associate's degree`; 5–8 → `College graduate/some postgrad`. |
| `RACE` | `native_american`, `asian`, `black`, `hispanic`, `white`, `race_6` | Covid rule: one box → that category (American Indian → `Other`); two or more boxes → `Other` (98 respondents, 65 of whom checked Hispanic). Chosen over Hispanic-first (Grummon) so `Other` means the same as in Covid. |
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
None → a new global level `None` (added to `GLOBAL_PERSONA_LEVELS["RELIGGRP"]` in `persona.py`).
This replaced an earlier decision of None → NaN, which would have left the 289 "None" respondents without a religion
line, so that a missing line meant "None" in Kriner2026 but "not asked or not answered" elsewhere.

**Known consequences.**
- `None` is used only by Kriner2026; other datasets split non-religious respondents into `Atheist`, `Agnostic` and
  `Nothing in particular`.
- `Other` here includes groups that other datasets list separately.
- The level is the string `"None"`, not Python's `None` (which means missing); the mapping code must keep them apart.

## Parquet export (`__init__.py`, `export_Kriner2026_parquet`)

- Writes `kriner2026_US_long.parquet` from the rows marked `include=True` in `question_audit.csv`, with the audit app's
  `drop_options`, `context_source`, `context_sep` and `question_overrides.json` edits. Run it from the audit app's export button.
- `flight_choice`: one row per respondent x task; `option_texts = ['Flight 1', 'Flight 2']`, `option_descriptions` = each
  flight's attributes in the respondent's display order. `qid` = `kriner2026:flight_choice:<profile A>_<profile B>_<attribute order>`,
  so identical screens share a `qid`.
- `reversed` for included score items (`SCORE_REVERSED`), confirmed by the user: `saf_familiar_old` True,
  `saf_safe_old` True, `sci_know_self` False (True = the first listed option is the high end, Covid convention).
  Nothing in this repo currently reads `reversed`.

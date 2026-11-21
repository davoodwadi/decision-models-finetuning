# Magistro 2025 decision log

This document records the source files, persona mapping choices, audit catalog entries, and export design for Magistro, Borwein, Alvarez, Bonikowski, and Loewen (2025), published in the American Journal of Political Science. The study fielded twin preregistered conjoint survey experiments on public attitudes toward generative AI and offshoring across the United States and Canada through YouGov.

## Source files

The Dataverse replication archive in `dataverse_files/` provides `clean_AJPS.csv` (6,056 respondents: 3,025 US and 3,031 Canada, each with a unique `ID` from 1 to 6,056), `Codebook_AJPS.pdf`, `main.R`, `appendix.R`, and `readme.txt`. Respondents were randomly assigned between subjects to either the generative AI arm (3,049 respondents: 1,535 US, 1,514 Canada) or the offshoring arm (3,007 respondents: 1,490 US, 1,517 Canada) and evaluated four randomized smartphone-company scenarios (`qtable_1` through `qtable_4`) on a five-point support scale (1 = Strongly oppose to 5 = Strongly support), with a sixth "Don't know" option coded as `NA` in `clean_AJPS.csv` (3,219 missing out of 24,224 task evaluations, leaving 21,005 answered conjoint tasks: 10,672 AI and 10,333 Offshoring).

## Persona mapping

All seven persona decisions below are implemented in `code/data/persona.py` under dataset key `magistro2025`, and missing values (`NaN`) are omitted from the respondent persona string by `base.py`.

**Country (`c_alphan`).** Both US and Canadian respondents are included, mapping `countr` values `US` to `United States` (3,025) and `Canada` to `Canada` (3,031, added to `GLOBAL_PERSONA_LEVELS["c_alphan"]`).

**Region (`US_REG`, prompt label `Region`).** Dispatched strictly by `countr` using country-prefixed keys (`US:<int>` and `Canada:<int>`) so integer codes 1 through 4 cannot collide. US respondents read `region` (`1` = `Northeast`, `2` = `Midwest`, `3` = `South`, `4` = `West`; 0 missing). Canadian respondents read `GeoPC_Region1` (`1` = `Alberta`, `2` = `British Columbia`, `3` = `Manitoba`, `4` = `New Brunswick`, `5` = `Newfoundland and Labrador`, `6` = `Northwest Territories`, `7` = `Nova Scotia`, `9` = `Ontario`, `10` = `Prince Edward Island`, `11` = `Quebec`, `12` = `Saskatchewan`; 0 missing; codes verified against `Codebook_AJPS.pdf` pp. 9–10). Twenty Canadian rows also contain stale US `region` numbers that are ignored by the country dispatch.

**Sex (`SEX`).** Mapped from `gender` (preferred over `gender_bin`, which drops 79 `Other` respondents) to existing global levels `Female` (3,277), `Male` (2,699), and `Other` (79), with 1 missing value mapped to `NaN`.

**Age (`AGE`).** Mapped from `age_group` (`18 to 29` to `18-29`, `30 to 44` to `30-44`, `45 to 64` to `45-64`, `65 or older` to `65+`; 0 missing), adding `30-44` and `45-64` to `GLOBAL_PERSONA_LEVELS["AGE"]` because both cross the existing `AGE_BINS` boundary.

**Education (`EDULEVEL`).** Mapped from `education` (preferred over binary `education_bin`) to four new global levels preserving the `College grad` versus `Postgrad` distinction: `HS or less` to `High school or less` (2,374), `Some college` to `Some college` (1,920), `College grad` to `College graduate` (1,165), and `Postgrad` to `Postgraduate` (597), with 0 missing values.

**Income (`US_INC`).** Mapped verbatim from `income` to three new global levels (`Low`: 1,102; `Middle`: 2,996; `High`: 1,173; 785 missing mapped to `NaN`), because dollar cutoffs are not documented in the paper, appendix, or codebook.

**Political party (`US_PRTY`).** Mapped from `party` (without US leaners, preferred over `party2` to match `kriner2026` and `wu2025`): `Democratic` to `Democrat` (1,056), `Republican` to `Republican` (755), `CPC` to `Conservative Party of Canada` (786), `LPC` to `Liberal Party of Canada` (738), `NDP` to `New Democratic Party` (472), and `Other` to `Other` (2,249), adding `Other` and the three Canadian party names to `GLOBAL_PERSONA_LEVELS["US_PRTY"]`.

## Audit catalog and stimulus wording

The audit catalog in `question_audit.csv` exposes both conjoint arms (`conjoint_ai` and `conjoint_offshoring`, `include=True`, `kev_type=score`, `context_source=in_block_db`), all six pre-treatment policy items (`policy_ai_offshoregr_1..6`, `include=True`, `kev_type=score`, `context_source=""`, 34,293 rows for future training), all six post-treatment policy repeats (`tables_policygr_1..6`, `include=False` to avoid post-treatment contamination and duplicate prompt strings), the post-conjoint manipulation check (`manipulation_check2`, `include=False`), and all eleven demographic/recode columns (`include=False`).

Conjoint definitions, localized US/Canadian scenario preambles, attribute labels, Before/After/Change values, and outcome stems follow Magistro et al. (2025, pp. 8–10, Tables 1–2) and `Codebook_AJPS.pdf` (pp. 5–6) verbatim. Because per-respondent randomized attribute row order is not saved in `clean_AJPS.csv`, scenarios use the canonical Table 1 order (smartphone price, customer service jobs, factory jobs, computer/data science jobs). All 162 distinct `(country, profile)` variants per arm are included in `.audit_details.json` for direct inspection in the Audit UI.

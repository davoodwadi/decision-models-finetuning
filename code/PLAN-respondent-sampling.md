# Plan: training on personas sampled from respondent-level ISSP data

Status: plan, not built. Written 2026-10-13.

## Why

The current training data fixes one way of grouping respondents: country × sex × age band × education, aggregated
once by the build scripts (`build_dataset`) into `issp<year>.jsonl`. Every other attribute in the ISSP files
(religion, party, income, …) is averaged away. Respondents missing any of the four fields are dropped entirely. The
fine-tuned model ended up predicting nearly the same distribution for every persona (e.g. ISSP 2022 v45, France,
45–59, tertiary: men, women and "sex not given" all ≈ 9 / 24 / 42 / 16 / 10%, while real men and women differ by
28% of the scale).

The ISSP files are respondent-level: one row per person, with all their attributes and answers. So any grouping can
be made from them. New approach: build each training record on the fly from a random subset of attributes.

## Decisions taken (2026-10-13)

1. **Personas are random attribute subsets**, aggregated from respondents when the record is made.
2. **Persona length**: the number of attributes shown besides country is drawn uniformly from 0 to all available;
   no cap.
3. **Every distinct persona counts equally**: no weighting by group size. Values are picked uniformly among the value
   combinations that occur, not by picking a respondent.
4. **No minimum group size**: a persona matching one person gives a one-hot target.
5. **The test set never informs training**: attributes, wording and sampling rules are justified from the ISSP data
   and the task, not from resemblance to SubPOP.
6. **Attributes**: every usable ISSP background attribute (list below), ancestry included, judged on its own merit.

## Data flow, before and after

| | Now | After |
|---|---|---|
| Build scripts write | `issp<year>.jsonl`: fixed groups with answer shares | `issp<year>_respondents.parquet` (one row per person) + `issp<year>_questions.json` (question specs) |
| `train_issp.py` reads | the `.jsonl` records | the two files per year |
| A training record is | a stored group × block, re-presented by `augment` | made fresh by the sampler, then `augment` as now |
| Calibration / test | SubPOP `.jsonl` | unchanged |

## Step 1: respondent table and question file (build scripts)

A new function in `data/issp_common.py`, `build_respondents(here, year, dta, asked_of, country_fields)`, called by
each year's `build_dataset.py`. ISSP 2020 is moved onto the shared builder first (step 1a). It writes:

**`issp<year>_respondents.parquet`**: one row per respondent, every respondent kept.
- `country`: the country code (`c_alphan`).
- One column per attribute: the cleaned label text (e.g. `Catholic`, `45-59`), or empty when missing, refused,
  not asked or not applicable. Categories use the data file's value labels, cleaned with `clean()` (number prefixes
  and country notes removed).
- One column per question (`v1`, `v2`, …): the answer as its option key (`"0"`–`"4"` for `score`, the option text
  for `choice`, `"true"`/`"false"` for `noul`), or empty for a non-answer. This is the mapping `question_spec`
  already makes per group, applied per person.

**`issp<year>_questions.json`**: per question its block, type, criteria and wording template (placeholders left in,
filled per country by the sampler with `fill`); per block its filter lines (`ASKED_OF`); per country the placeholder
values (`issp_countries.csv` plus year fields such as `{association}`).

**Step 1a: ISSP 2020 onto the shared builder.** Its question bank has no `codes` column; its
`build_question_bank.py` adds one (from the value labels, as its `build_dataset.py` does today). Check: the shared
builder's aggregated output for 2020 equals the current `issp2020.jsonl`.

**Check for every year**: aggregating the respondent table by country × sex × age × education with the 10-answer
minimum reproduces that year's current `issp<year>.jsonl` exactly (same groups, same targets). This proves the
respondent table holds the same answers as today's data.

## Step 2: attributes

Every year 2015–2024 has all of these (country counts per year checked). Country is always shown: question wording
depends on it (`{country}`, `{nationality}`, …), so it is part of every persona and not part of the random subset.

| Persona line | ISSP variable | Values |
|---|---|---|
| Sex | `SEX` | Male, Female (as now) |
| Age | `AGE` | 15-29, 30-44, 45-59, 60-74, 75 or older (as now) |
| Education | `EDULEVEL` / `DEGREE` | 3 levels (as now) |
| Religion | `RELIGGRP` | No religion, Catholic, Protestant, Orthodox, Other Christian, Jewish, Islamic, Buddhist, Hindu, Other Asian religions, Other religions |
| Attendance of religious services | `ATTEND` | the 8 categories as labelled |
| Party voted for in the last general election | `<CC>_PRTY` | the country's party labels as the data has them (English and local name) |
| Voted party's left-right position | `PARTY_LR` | Far left, Left / center left, Center / liberal, Right / conservative, Far right |
| Voted in the last general election | `VOTE_LE` | Yes, No |
| Household income | `<CC>_INC` | position within the country and year: lowest fifth … highest fifth |
| Region | `<CC>_REG` | the country's regions as labelled |
| Place of living | `URBRURAL` | big city, suburbs, town or small city, country village, farm or home in the country |
| Marital status | `MARITAL` | the categories as labelled |
| Main activity | `MAINSTAT` | in paid work, unemployed, in education, apprentice, sick or disabled, retired, domestic work, service, other |
| Trade union membership | `UNION` | currently, previously, never |
| Self-placed social position | `TOPBOT` | 1 (bottom) … 10 (top) |
| Ancestry | `<CC>_ETHN1` | the country's categories as labelled (the country of family origin; not race) |

Each label is checked by hand per year when built (category wording varies slightly between years); the table of
final labels goes into `data/README.md`.

## Step 3: the sampler (`train_issp.py`)

Loaded once at start: per year, per country, a matrix of attribute codes and a matrix of answer codes (numpy, -1 =
missing). One training record is made like this:

1. **Slot**: pick a (year, country, question) uniformly among those that exist (one question per record since
   2026-10-13, `DECISIONS.md` 25; first designed as (year, country, block)).
2. **Number of attributes k**: uniform from 0 to the number of attributes available in that country and year.
3. **Which attributes**: k of them, at random.
4. **Values**: uniform among the value combinations of those k attributes that occur among the country's
   respondents who answered at least one question of the block.
5. **Targets**: for each question of the block, the answer shares of the matching respondents who answered it.
   Questions nobody matching answered are left out of the record.
6. **Persona**: `Country` plus the k lines, plus the block's filter lines (`ASKED_OF`). Wording filled for the
   country.
7. The record then goes through `augment` (random line order, reversed scales, shuffled options), the model and the
   loss exactly as now.

**Epoch**: `--records_per_epoch` sampled records (default 224,078, the size of the current training set, so runs
stay comparable in time). Every epoch is freshly sampled; the sampler uses the run's seeded random generator.

**Logged once at start** (and in `run_info.json`), from a sample of 10,000 records: the share of targets from
groups of 1, 2–9 and 10+ respondents, and the distribution of persona lengths.

**Removed**: the `.jsonl` training path (`--train` then names years whose respondent files are read). The existing
`.jsonl` files stay on disk; the post-eval notebook's ISSP example cell will read the respondent file instead.

## Step 3b: wording variation in the training view (decided 2026-10-13)

The training text is too uniform: the state is always "Key: value" lines with fixed keys, the question always the
master questionnaire's short self-completion wording. Wording variation is training signal (never unified, see the
cut-label fix); the training view should show the many ways surveys and personas are written. The target is the range
of real-world formats, not the test set's format. Calibration and evaluation keep the canonical view. Every variant
describes exactly the same people and the same question: wording changes, never who is meant or what is asked.

- **B. Persona presentation** (rule-based, in `issp_augment.py`), first. **Done 2026-10-13** (`persona_text`; `DECISIONS.md` 21):
  layouts (key-value lines, bullets, one line, JSON, prose in third / first / second person; the optional header was removed 2026-10-13, DECISIONS.md 26);
  several names per attribute ("Sex" / "Gender", "Religion" / "Religious affiliation"); value wordings that keep the
  meaning ("Female" / "Woman", "43" / "43 years old", "Yes" / "yes"); line order random (as before).
- **A. Real variants from the sources**, second: English national questionnaires (US, GB, AU, NZ, IE, IN, PH, ZA
  in each year's `q_all.zip`) as alternative wordings for those countries; interviewer / form instructions
  ("PLEASE TICK ONE BOX", "READ OUT") as optional parts; country-specific option wording (**done 2026-10-13**,
  `DECISIONS.md` 22: every note marks a different category, so all are own-country only). Rule: a variant that only
  changes the words may be used for every country (country terms as placeholders); one that changes the meaning only
  for its own country.
- **C. Question and option presentation**, third: options inline in the question text vs listed; numbered /
  lettered / plain options (real label variants only).
- **D. Generated paraphrases**, last: LLM paraphrases of question stems written once to a file, each checked for
  unchanged meaning and scale, sampled at training time.

## Step 4: documentation

- `data/README.md`: the respondent-table format, the attribute table with final labels, the sampling rules
  (decisions 1–6 above). Rules 8–10 (fixed persona, 10-answer minimum, one record per group × block) are replaced.
- Each year's README: its respondent count, attribute availability (countries missing an attribute, e.g. no party
  variable in Croatia 2023), its rebuild commands.
- `data/issp2020/DECISIONS.md`: new entry 19 recording these decisions.

## Verification (CPU only, in the scratchpad)

1. Step 1 check: respondent tables re-aggregate to the current `.jsonl` exactly, every year.
2. Sampler: 10,000 records convert to Kev input (augmented and canonical), no placeholder left, every target sums
   to 1, every persona line has a value, filter lines present on filtered blocks.
3. Hand-read 20 sampled records across years and lengths.

## Order of work

1. ~~ISSP 2020 onto the shared builder (1a), with its check.~~ Done 2026-10-13: `issp2020.jsonl` byte-identical;
   unused files removed (`DECISIONS.md` entry 19).
2. ~~Respondent tables + attribute decoding, all years, with the re-aggregation check.~~ Done 2026-10-13 (all ten
   years reproduce the former `.jsonl` exactly; `DECISIONS.md` 23).
3. Attribute labels reviewed per year; documented.
4. ~~Sampler in `train_issp.py`; verification 2–3.~~ Done 2026-10-13 (`issp_sampler.py`, used by `train_issp.py`
   and `show_train_samples.py`). Still open: how ordered attributes (age, income, social position) are shown —
   raw values for now.
5. Wording variation 3b: ~~B (persona presentation)~~ done 2026-10-13; then A, C, D.
6. Documentation; then the training command handed over.

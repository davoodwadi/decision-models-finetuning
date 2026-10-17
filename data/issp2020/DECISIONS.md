# ISSP 2020 → decision-model dataset: decisions

Goal: fine-tune a decision model to predict the distribution of answers, P(answer | persona, question).
Data: ISSP 2020 Environment IV (GESIS ZA7650 v2.0.0), 44,100 respondents, 28 countries.

1. **One unified questionnaire.** Every country gets the English wording of the ISSP source questionnaire
   (`ZA7650_bq.pdf`), with `{country}`, `{nationality}`, `{parliament}` filled from `countries.csv` (since 2026-10-13:
   `data/issp_countries.csv`, see 19).
   Built by `build_question_bank.py` → `question_bank.csv`.
2. **Items.** 45 `score`, 6 `choice`, 4 `noul`. Excluded: v2, v9 (depend on a previous answer), v48, v49, v51
   (open counts). `score` levels run low → high.
3. **Missing answers.** Negative codes (can't choose, don't know, no answer, not applicable) are dropped.
4. **No survey weights.** They correct population composition; we model answers given the persona.
5. **Targets.** Respondents with identical personas are grouped; the target is the share of each answer in the
   group (Kev `target`), `label` = most common answer. Records are split by question block to fit Kev's limits.
   Kev's calibration and evaluation must be changed to use soft targets.
6. **Persona = country, sex, age band (15-29 / 30-44 / 45-59 / 60-74 / 75+), education (lower secondary or less /
   upper secondary or post-secondary non-tertiary / tertiary).** 820 groups, median 39 respondents; adding main
   status or urban/rural drops typical groups to 3-11 people (group-size table in the notebook).
7. **Dataset.** `build_dataset.py` → `issp2020.jsonl`: one record per persona group × questionnaire block.
   A group's shares on a question are used only if at least 10 of its members answered it.
   Records store the persona as fields (`persona`), not as state text; the state is rendered when a record is used.
8. **Augmentation (training views, `prepare_finetuning.ipynb`).** Persona attributes in random order; `score` scales
   reversed with probability 0.5 (target and label remapped). Kev itself shuffles `choice` options every epoch.
   Split by persona `group` before augmenting.
9. **Fixed split (`make_split.py` → `split.json`, created once, never re-drawn; unused since 2026-10-12, deleted
   2026-10-13, see `README.md`).** Test questions: one item per
   multi-item block — v5, v13, v19, v24, v27, v33, v42, v45, v56, v59 — the rest of each topic stays in training.
   Personas: within each country, groups assigned 70% train / 15% calibration / 15% test (474 / 99 / 99 groups).
   Targets: train 57.7%, calibration 12.1%, test A (new persona) 12.0%, test B (new question) 12.8%,
   test C (new persona + new question) 5.3%.
10. **Training.** Every question target counts equally (no weighting by group size). Model: `kev-0.8b`.
11. **Evaluation.** Metrics compare full distributions: KL divergence (bits) for all questions, Wasserstein
    distance (levels) for `score` (JS divergence dropped from training logs and metrics.json 2026-10-13; the SubPOP
    notebooks still compute it). **No temperature anywhere**: the model's raw distribution is compared with the
    observed shares (saved checkpoints carry temperature 1.0). This includes after training: no post-hoc temperature is fitted or applied in any notebook. Step 0 (the released model before training) is the
    baseline, scored on calibration and all test sets; then the calibration set is validated every
    `--validate_every` steps (default 500) and the weights with the lowest KL are kept.
12. **Extra training data: SubPOP train split.** Training data only. Details and decisions: `data/subpop/README.md`.
13. **Data layout.** One folder per source: `data/<source>/<name>.jsonl` is a training-ready dataset, passed as
    `--train <name>`; a `split.json` next to it means only its split's training part is used. Raw files in
    `data/<source>/raw/`, build scripts and intermediate files alongside. Calibration and tests: always
    `data/issp2020/split.json` (replaced by 18).
14. **Extra training data: ISSP 2024 Digital Societies (ZA10020).** Details and decisions: `data/issp2024/README.md`.
15. **Extra training data: ISSP 2023 National Identity and Citizenship (ZA10010).** Details and decisions:
    `data/issp2023/README.md`.
16. **Each dataset folder has a README** (source and version, how to rebuild, record format, its decisions).
17. **All ISSP years 2015–2024 as training data (2026-10-12).** 2015, 2016, 2017, 2018, 2019, 2021 and 2022 added;
    every year except 2020 is built by one shared builder, `data/issp_common.py`, with the rules in `data/README.md`
    and each year's specifics in its README. Every suitable question is used, without a test-set leakage check
    (questions repeating ISSP 2020 test questions included), and every other year is trained on whole, including
    the persona groups that are ISSP 2020 calibration/test personas (`train_issp.load_sets` no longer drops them).
    Calibration and test sets are to be redesigned.
18. **Calibration and test sets from SubPOP; all ISSP data for training (2026-10-12).** Training: every ISSP year,
    used whole (ISSP 2020 included; `split.json` is no longer used). Calibration: SubPOP's training split
    (`subpop100`); test: SubPOP's evaluation split (`subpop_eval`); the roles can be swapped in `train_issp.py`. Both
    leave out the questions that the ISSP data also asks (29 questions, listed in `data/subpop/README.md`). (Reversed by 34: every question kept.) The
    average-over-personas baseline is empty for these test questions, since none of them is in the training data.
19. **ISSP 2020 on the shared builder; unused files removed (2026-10-13).** `build_question_bank.py` and
    `build_dataset.py` now call `data/issp_common.py` like every other year (the rebuilt `issp2020.jsonl` is
    byte-identical). Deleted as unused: `make_split.py`, `split.json`, `countries.csv` (its names are in
    `data/issp_countries.csv`; its printed parliament names are in `README.md`) and `data/issp2024/show_sample.py`.
    First step of `PLAN-respondent-sampling.md`.
20. **Cut-off value labels completed from the codebooks (2026-10-13).** The .dta files cut long value labels; only
    the missing end is added, from the GESIS codebooks, and the data's own wording is kept (wording variation is
    training signal, never unified) (`data/codebook_label_fixes.py` → `issp<year>/label_fixes.csv`, applied by
    `read_dta()`). Training data changed only in the wording of those options: 2015 v35, 2017 v35 and v48–v52 (targets,
    counts and groups identical). Details in `data/README.md`.
21. **Persona presentation varies in the training view (2026-10-13).** `issp_augment.persona_text` writes each
    training record's persona in a random presentation: layout (key-value lines with `:`, `-` or `=`, bullets, one
    line, JSON, or prose in third / first / second person), an optional header ("Respondent profile:"), one of several
    names per attribute ("Sex" / "Gender", "Religion" / "Religious affiliation"), and value wordings that mean exactly
    the same (sex, age, yes / no only; other values keep the data's text). The canonical view for calibration and
    evaluation is unchanged. Stage B of `PLAN-respondent-sampling.md` step 3b; A, C, D follow.
22. **Country-specific answer options (2026-10-13).** The country notes in value labels give that country's own
    option wording or remove an option it did not offer (`data/README.md`, rule 2); a wording variant is used for other
    countries only if it means the same, and every note so far marks a different category, so all are own-country only.
    Changed training records: 2015 v15 (TW), v97 (HU); 2017 v46 (ES, HU, TW); 2018 v35, v37, v39 (NZ), v40–v44 (KR),
    v43, v44 (US); 2019 v57, v58 (IL, VE, and SE, whose code 6 is a non-answer); 2021 v22 (PL). Stage A of
    `PLAN-respondent-sampling.md` step 3b.
23. **Respondent-level training data, records drawn in the training loop (2026-10-13).** Each ISSP year is stored one row
    per respondent (every respondent kept, 16 attributes as raw values, every answer); `train_issp.py` draws each
    epoch's records with `issp_sampler.IsspSampler` (random country and block, random attribute subset of any size,
    value combination uniform among those that occur, answer shares of the matching respondents as targets, no
    minimum) and shows them through the training view. Re-aggregating the respondent data by the former fixed groups
    reproduces the former `issp<year>.jsonl` exactly (all ten years), after which those files and the fixed-group code
    were removed. The average-over-personas baseline is dropped (it needs fixed training groups and was empty for the
    SubPOP test). Steps 2–4 of `PLAN-respondent-sampling.md`.
24. **Every label fix read by hand (2026-10-13).** All completions in `label_fixes.csv` (2015, 2017, 2020, 2021) were
    read one by one against the codebook. Five were wrong (codebook frequency numbers taken as text, a bracket added to
    a label that was worded differently rather than cut, a repeated half-label, a garbled "½", a space lost in the PDF),
    all in national degree variables that training does not use; each fault now has a rule in
    `codebook_label_fixes.py` (`data/README.md`). One cut label in a training attribute had been missed (2015 Hungary,
    "Jobbik (Magyarorszagert Mozg…"): completed by hand from the codebook, 2015 rebuilt.

25. **One question per training record (2026-10-13).** `issp_sampler.IsspSampler` draws a (year, country, question)
    slot instead of a (year, country, block) slot; the record holds that one question, with its block's `ASKED_OF`
    lines in the persona. Kev's questions in one request cannot read each other, so predictions are the same either
    way; one question per record gives every target its own drawn persona (a block's questions no longer share one),
    at the cost of encoding the persona once per question. An epoch is one pass through every slot in random order (each
    with a freshly drawn persona), replacing the fixed 224,078 random draws (the former fixed-group set's size). `data/README.md`
    decision 9.
26. **Attribute values as each country's questionnaire had them (2026-10-13).** Attributes are stored as data codes,
    with their text per country in `issp<year>_attributes.json`. Country notes in attribute value labels now give that
    country's own text, as they already did for answer options (e.g. German attendance code 3 is "1-3 times a month",
    US age 89 is "89 years or more", New Zealand's "Other Christian" is "Christian"); five abbreviated notes are written
    out from the codebooks' per-country lists (`ATTRIBUTE_NOTES` in the 2015, 2016 and 2018 build scripts). Notes name
    countries by two letters, so they now also reach Great Britain (`GB-GBN`), which they had silently missed. Every
    changed value was read by hand (about 90 per-country values across the ten years); the question data is unchanged.
    The codebooks' fuller per-country lists (2015–2018) and free-text notes are not used yet ("level 2/3").

26. **No persona header (2026-10-13).** The optional header before the persona ("Respondent:", "About the respondent:",
    ... from decision 21) is removed from the training view: a fixed phrase from a short list is boilerplate, not
    variation in how people are described, and would not shift the training distribution. Layout, attribute names,
    value wording and line order still vary.

27. **Calibration set held out of the ISSP training data (2026-10-13).** The weights kept from a run are now chosen on
    ISSP data instead of SubPOP's training split (`subpop100`, no longer used): 1% of each year's (country, question)
    slots (lowest hash of year, country and question, so the same in every run; 204 slots for all ten years) are never
    drawn for training. Each gives fixed calibration records: the whole country, and one record per value of each
    single attribute, with the answer shares of all matching respondents (47,324 records for all ten years). Reason:
    SubPOP's calibration error flattened by step 100 and cannot show whether the model uses the persona (run
    issp-all-0.8b-v8 predicted the same distribution for men and women). The test set stays SubPOP's evaluation split.

28. **Minimum group size for training personas (2026-10-13).** `IsspSampler(min_n=...)`, `train_issp.py --min_n`
    (default 10): a drawn persona's value combination must be shared by at least min_n of the slot's respondents; if
    the drawn attributes have no such combination, k and the attributes are drawn again, and slots with fewer than
    min_n respondents (93 of 19,991 at 10) are not trained on. `min_n=1` gives exactly the former draws. Reason: with
    no minimum, 72% of targets were one respondent's one-hot answer, so a group difference such as men vs women
    (run issp-all-0.8b-v11: the same prediction for both, on a training question and a calibration question) is buried
    in noise. Cost: long personas mostly disappear; at 10, 0-3 attributes are 65% of records and 9+ under 1%
    (formerly uniform from 0 to all). Calibration records are unchanged.

29. **Training data chosen by a recipe; record construction separate from the trainer (2026-10-14).** A run names its
    training data with `train_issp.py --data configs/data/<recipe>.json`: a source (the ISSP years and
    `calibration_frac`, `issp_sampler.IsspSource`) and a construction (how a record is built from a (year, country,
    question) slot: `group_shares` with `min_n`, `single_respondent`, `fixed_subset`; `constructions.py`). The flags
    `--train`, `--min_n`, `--single_respondent`, `--train_slots` and `--calibration_frac` are gone; the full recipe is
    saved in each run's `run_info.json` (`settings.recipe`), and earlier runs' flags translate to their recipe
    (`training_data.recipe_from_settings`). The calibration records belong to the source, so they are the same whatever
    the construction and runs built different ways are compared on the same records. SubPOP is never training data: no
    source reads it. Reason: trying other training data or other ways of building records should be a new recipe (and
    at most a new construction class), never an edit of the trainer. Every construction draws exactly as before:
    replaying runs issp-all-0.8b-r64-v3 and issp-all-0.8b-test-r64-v7 gives every logged step's targets.
30. **Country subset of the ISSP source (2026-10-14).** A recipe's source can name `"countries"`, e.g.
    `configs/data/issp-us-group-min10.json`: `{"issp": {"datasets": "all", "calibration_frac": 0.01, "countries": ["US"]}}`.
    Only those countries are loaded; training and calibration both come from them, and `calibration_frac` is a share of
    their slots per year (US, min_n 10: 0.01 -> 9 slots / 2,055 calibration records, 609 training slots; 0.05 -> 31 /
    7,002, 587; 0.1 -> 63 / 14,133, 555; 0.2 -> 123 / 27,746, 495). Slots are held out in the same fixed hash order, so
    a larger share keeps every slot a smaller one held out. Without `countries` nothing changes (calibration records
    byte-identical; run issp-all-0.8b-full-v1's training batches replay identically). Calibration numbers of a
    country-subset run are on that subset's records, so they are not comparable with all-country runs; the SubPOP test
    set is the same for every run. Reason: train on the US respondents alone (SubPOP's test set is the US General
    Social Survey; the US ISSP data has Democrat / Republican party and US Census divisions).
31. **At most max_attributes in a group_shares persona (2026-10-14).** `group_shares` takes `max_attributes` (default
    none: up to all, as before): k is drawn uniform from `min_attributes` to `max_attributes`. With it unset the draw
    makes the same random calls as before, so earlier recipes give the same records. First recipe:
    `configs/data/issp-us-group-min100-attr2.json` (US, `min_n` 100, exactly 2 attributes besides the country): 594
    training slots (15 of the 609 US slots have no pair of attributes shared by 100 respondents), 134,832
    possible (question, persona) targets, every target from 100+ respondents (sampling noise about 0.007 bits instead of
    about 0.18 for 10-99). Reason: within the US, min_n 10 with up to all 16 attributes gives about 1.44 million targets
    with 1 or 2 attributes alone, nearly all from 10-99 respondents, so each target is mostly noise and seen about once
    a run; the persona effects are small next to that noise. A user-chosen cap: it is not set from SubPOP's format.
32. **Every view variation off by default (2026-10-14).** `issp_augment.view` takes an `Augmentation` that switches
    each variation on separately (persona layouts, attribute names, value wording, punctuation, line order, reversed
    `score` scales, shuffled `choice` options); with none given, the view is 'Key: value' lines in stored order, as
    calibration and test always used. `train.py --augment` sets the training view; calibration and test are never augmented (always
    the canonical view). `run_info.json` records the training view under `train_view`. Until 2026-10-14 training always used every variation, with
    `p_reverse` 0.5 (`--augment full` reproduces it draw for draw); runs from then on train on the plain view unless
    asked. Reason: augmentation is a choice per run, so a run shows what it changed.
33. **Calibration groups of at least 100 respondents in the US min-100 recipes (2026-10-14).** A recipe's source can
    set `calibration_min_n` (default 1 = every group, as before): calibration keeps only the whole-country and
    one-attribute records whose group has at least that many respondents. `issp-us-group-min100-attr2` and
    `issp-us-group-min100-attr1-2` set 100: 388 calibration targets instead of 2,055 (the same 9 held-out US slots),
    so their calibration noise matches training's min_n 100. Calibration KL of runs with these recipes before
    2026-10-14 is on the 2,055-target set and not comparable; the test set is unchanged.
34. **No SubPOP question left out for resembling the ISSP data (2026-10-15).** Reverses the overlap rule of 18: the
    29 SubPOP questions that ISSP also asks are back in calibration and test (`build_subpop.py` has no `ISSP_OVERLAP`).
    Test: 133 questions, 2,926 targets (was 106 and 2,332); `subpop100`: 1,875 questions, 41,244 targets (was 1,873
    and 41,200). Every earlier target is unchanged; test numbers from before are on the smaller set and not comparable.
    Reason: a test set is not filtered by similarity to the training data. `data/subpop/README.md` still lists which
    questions ISSP also asks (now incl. `fefam`, `jobfind`), for reading results.
35. **ISSP 2022 v29 asks only the least desirable option (2026-10-15).** `build_question_bank.py` had appended Q10b
    to the whole of Q10a, so v29 asked for the best and the least desirable family-work arrangement at once ("…what is
    the best way…? Which of these options would be the least desirable?"). Now Q10a's context sentence plus Q10b's own
    wording (raw/ZA10000_bq.pdf): "Consider a family with a child under school age. And, in your opinion, which of
    these options would be the least desirable?" Only v29's text changed (all 32 countries); answers and respondents
    are the same. Runs before 2026-10-15 trained (or were calibrated) on the contradictory text.

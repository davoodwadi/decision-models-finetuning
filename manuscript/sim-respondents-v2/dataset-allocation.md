# Dataset allocation for sim-respondents-v2

## Motivation for the new split

In the first version of the paper, choice training used only binary forced-choice conjoint datasets (Drones and Electricity, both K = 2) and checkpoint selection relied solely on a binary forced-choice validation dataset (Flights, K = 2). Post-hoc temperature analysis showed that this binary-only allocation over-sharpened model logits for two-option forced choices. Because both training and validation saw only K = 2 tasks, early stopping and head-temperature calibration rewarded sharp binary probabilities (T near 1.0) even as predictions degraded on noisier multi-option and ordinal Likert test datasets that required higher temperatures (T between 1.8 and 4.0).

The v2 allocation resolves this mismatch by placing all three response families (binary choice, multi-option choice, and ordinal Likert ratings) into choice training, expanding validation to cover both binary price conjoint and multi-option non-price choice, and holding out three high-signal unseen domains for testing. Every dataset in this allocation is a self-contained, single-turn published choice experiment.

## Dataset allocation table

| Split | Dataset | Source key | Response format | Price attribute | Respondents / tasks | Target slopes | Domain | Loader status |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| Training | Drones | `berke2023` | K = 2 binary conjoint | Yes | 3,625 / 36,297 | 28 | Delivery and privacy | Built |
| Training | EV adoption | `Forsythe2023` | K = 3 multi-option conjoint | Yes | 1,626 / 24,390 | TBD | Automotive (EVs, cars and SUVs) | To build |
| Training | AI and offshoring | `magistro2025` | K = 5 ordinal Likert conjoint | Yes | 5,525 / 21,005 | 32 | Labor and AI | Built |
| Validation | Electricity | `wu2025` | K = 2 binary conjoint | Yes | 1,861 / 18,591 | 8 | Energy and climate | Built |
| Validation | Food labels | `grummon2025` | K = 3 three-product RCT | No | 5,633 / 33,798 | 5 | Nutrition and food | Built |
| Test | Flights | `kriner2026` | K = 2 binary conjoint | Yes | 1,000 / 5,000 | 12 | Travel and aviation | Built |
| Test | Consumer rights | `kantorowicz2026` | K = 2 binary conjoint | Yes | 2,242 / 26,823 | 38 | Consumer durables | Built |
| Test | Vaccines | `covid` (Kreps et al., 2020) | K = 3 opt-out conjoint | No | 1,971 / 9,855 | 11 | Public health | Built (locked) |

## Design principles and coverage

**Training datasets.** Choice training uses three high-signal multi-attribute conjoint datasets, one from each response family (K = 2, K = 3, and K = 5 Likert). Every training dataset varies multiple randomized attributes per profile, avoiding between-subjects single-factor designs during weight updates. During training, batches should be weighted equally across the three response formats rather than proportional to raw task counts so binary choices do not dominate gradient updates.

**Validation datasets.** Checkpoint selection and any post-hoc head-temperature calibration use two already-built datasets that span different choice formats and experimental designs, namely Electricity (K = 2 binary conjoint with price) and Food labels (K = 3 three-product RCT without price). Selecting checkpoints using the unweighted average across both validation datasets prevents binary-only over-sharpening. If evaluating full validation every step is slow, training runs can score a fixed stratified respondent subsample per validation dataset during intermediate steps and evaluate the full validation datasets on candidate checkpoints.

**Held-out test datasets.** Unseen-study evaluation uses three high-signal datasets totaling 61 treatment slopes across three domains that never appear in training or validation, namely travel and aviation (Flights), consumer durables across laptops and laundry detergent (Consumer rights), and public health (Vaccines, locked in the test suite). These test datasets cover binary forced choice with price and three-option opt-out choice without price. One mild thematic adjacency remains between Electricity in validation (never trained on) and Flights in test, as both involve emissions framing, while EV adoption in training and Flights in test belong to distinct consumer categories with different attributes and price scales.

**Open gap for unseen ordinal Likert evaluation.** Because AI and offshoring moves into training and `kreps2021` is kept out of the test suite for now, the three current held-out test datasets cover K = 2 and K = 3 choice tasks rather than ordinal Likert ratings. Adding one external, self-contained ordinal Likert conjoint from an unseen domain (outside delivery, automotive, labor, energy, nutrition, travel, consumer durables, and vaccines) would restore unseen-study Likert evaluation without introducing leakage.

## Excluded candidate datasets

**Twin-2K-500-Mega-Study.** Excluded because several sub-studies separate treatments from dependent variables across distinct survey stages (e.g., anchoring paradigms that ask respondents to write an open-text response before completing a separate choice task), which does not match single-turn choice training.

**General national surveys (ISSP, SHED, HINTS, and WVS-Wave 7).** Excluded because non-experimental attitudinal microdata tends to dilute choice training and worsen treatment-effect recovery on experimental datasets.

**Grummon et al. (2023) vending-machine RCT (`grummon2023`).** Excluded from training because its 16-product vending-machine choice puts most of the cross-entropy signal into static product and brand popularity rather than label manipulations, and its five-arm between-subjects design produces very low minibatch signal-to-noise ratios under regression-slope training.

**Kreps and Kriner (2021) vaccine conjoint (`kreps2021`).** Excluded from training and validation because three of its five attributes (vaccine efficacy at 50%, 70%, and 90%, FDA full approval versus emergency use authorization, and minor side-effect risk) overlap directly with the locked Vaccines (`covid`, Kreps et al., 2020) test dataset. It is also kept out of the test suite for now.

**Musgrave (2026) sanctions willingness-to-pay experiments (`musgrave2026`).** Excluded as too thin for core evaluation or training, since only the first bid stage is single-turn and each respondent completes just one task across two treatment factors (cost and efficacy).

# 03-analysis-paper: story and claims

Prose for `03-analysis-paper.ipynb`; the notebook holds code and one-line headers. Sections are added here as each
step of the notebook is cleaned up. Numbers are read from the notebook's outputs, not typed into the notebook.

## Datasets

- **Wu 2025** (training): electricity-plan conjoint; attributes monthly bill, share of renewables, locally produced
  renewables. Used either as its 18,591 choices ("Wu choices") or as an 8-slope table ("Wu table"): the AMCEs of
  monthly bill $110/$115/$120/$125 vs. $105 (4), renewables 30%/50%/100% vs. 10% (3), and locally produced yes vs. no
  (1). The table is refit from Wu's choices with the paper's OLS specification (`beta_ols/wu2025.py`) and divided by
  the outcome's standard deviation; halved, it matches the paper's Fig. 3 AMCEs to within plotting precision. The
  paper reports these AMCEs only as a figure, not as numbers.
- **Berke 2023** (training): drone-delivery choice experiment; attributes delivery fee, wait time, privacy. 36,297
  choices ("Berke choices").
- **Kriner 2026** (validation; selects each run's checkpoint): flight conjoint, 12 terms (airline, fuel, baggage, seat,
  legroom, price).
- **Grummon 2025** (held-out test): food warning labels, 5 terms.
- **Kreps 2020** (held-out test): COVID-19 vaccine conjoint, 11 terms.

No training study shares a domain with any evaluation set: every evaluation is transfer to a new domain. Kriner differs
from the two test sets in being the selection set and in having a price attribute, as Wu (monthly bill) and Berke
(delivery fee) do.

## Overview — the story at a glance (Table 0, Figure 0)

**Purpose.** The introduction's figure: every main regime on one map, so the reader sees the whole argument before the
steps. KL-anchored runs only; no-anchor runs are in the ablation.

**Axes.** x = % change in β MSE vs. the untrained model on Kriner 2026 (flights; the selection set).
y = the mean of the % changes on the two held-out test sets (Grummon 2025, Kreps 2020). Both axes run so that up and
right are better; the untrained model sits at the origin.

**What it shows.**
- Microdata regimes (Wu choices, Berke choices, Wu + Berke choices) move up and right: better on Kriner and on the
  held-out tests.
- The Wu table alone matches Wu choices on Kriner but falls short of them on the held-out tests: −52% vs. −82% at
  KEV-4B, and worse than untrained (+22%) at KEV-0.8B.
- Berke choices + Wu table is furthest right with Wu + Berke choices, but stays at Berke's height on the held-out tests.

**Definitions.** The test mean averages two percentages, so each test set counts equally whatever its scale. Kriner
selects the checkpoint for every run, so the x-axis is optimistic for all regimes alike. Table 0 lists the same
numbers with the data each regime uses.

## Step 1 — Microdata fine-tuning transfers to unseen domains

**Punchline.** Training on respondents' choices from one study makes the model a better choice simulator in domains it
never saw, and pooling two studies transfers better than either one alone.

**Evidence (Table 1, Figure 1; KEV-0.8B and KEV-4B).**
- Every microdata regime (Wu choices, Berke choices, Wu + Berke choices) lowers population β MSE on every benchmark
  at both model sizes.
- Wu + Berke has the lowest β MSE of the three in every model × dataset cell.
- The transfer evidence is the two held-out test sets: Grummon 2025 (food warning labels) and Kreps 2020 (vaccines).
  Kriner 2026 selects the checkpoint, so its gains are optimistic.

**What this does not show.** Wu + Berke has both more choices (18,591 + 36,297) and more domains than either study
alone, so the gain from pooling cannot be attributed to diversity rather than quantity.

**Why it matters for the paper.** More studies help, but respondent-level microdata is rarely available. Step 3 asks
whether a study's published coefficient table can stand in for its microdata.

**Table layout.** Rows are named by training data; row codes (`Row 2b`, `Row 3`, `Row 5`) stay internal. The untrained
model is the first row of each model block (the former Table 1). "Oracle" wording is introduced only in Steps 3–4,
where a regime is compared against a reference. Choice KL moves in the same direction as β MSE and is in Table A2.
KEV-9B is in every table and figure; its rows are NaN (empty in figures) until its runs finish.

## Step 2 — A published table teaches the study's slopes

**Punchline.** Wu's 8 published OLS slopes, trained with the population β-loss and the KL anchor (α = 0.005), improve
the model on Kriner about as much as Wu's 18,591 respondent choices do. On the held-out tests they carry much less of
the choices' transfer.

**Evidence (Table 2, Figure 2; all three model sizes).**
- On Kriner 2026 (flights), the table lowers β MSE at every size. It is close to the choices at KEV-4B and better
  than the choices at KEV-0.8B and KEV-9B.
- On Kreps 2020 the choices are far ahead at every size. On Grummon 2025 the choices are far ahead at KEV-0.8B and
  KEV-4B; at KEV-9B the table is slightly better (16.82 vs. 18.85), but both stay close to untrained (22.73).
- At KEV-4B and KEV-9B the table improves on the untrained model on both test sets; at KEV-0.8B it is worse than
  untrained on both.

**Why (hypothesis).** 8 numbers constrain 8 directions of the model's choice probabilities: how much monthly bill,
share of renewables, and local production move choice. What can transfer is what those directions share with a new
study; Kriner has a price attribute, Grummon and Kreps do not (Step 4). The KL anchor, α·KL(p₀‖p_θ) on Wu's tasks,
keeps the fit close to the untrained model; its effect is shown in the ablation.

**Why it matters for the paper.** A table alone is a cheap signal that transfers narrowly, not a substitute for
microdata. Step 3 asks whether a table added to microdata you do have gives the best of both.

**Layout.** Figure 2 also shows the no-anchor table run (hollow red) next to the anchored one; Table 2 does not. The
rest of the no-anchor comparison is in the ablation at the end of the notebook. Choice KL is in Table A3.

## Step 3 — Your microdata + another study's table

**Question.** You have respondent choices for your own study (Berke) but only the published coefficient table for
another (Wu). Does adding Wu's table to Berke's choices give you what Wu's microdata would?

**Punchline.** On Kriner, yes: Berke choices + Wu table match or beat Berke + Wu choices. On the held-out tests, no:
adding the table does not bring the transfer that Wu's choices bring, and it costs a little relative to Berke alone.

**Evidence (Table 3, Figure 3; KEV-0.8B and KEV-4B).**
- Kriner 2026 (flights): adding Wu's table to Berke's choices lowers β MSE well below Berke alone at both sizes, to
  the level of Berke + Wu choices (KEV-0.8B) or below it (KEV-4B).
- Held-out test sets: Berke + Wu table is worse than Berke alone in 3 of 4 cells (KEV-0.8B Grummon and Kreps,
  KEV-4B Kreps) and better in one (KEV-4B Grummon). Berke + Wu choices is best on both test sets at both sizes.

**Caveats.**
- Kriner selects each run's checkpoint, so every row's Kriner value is optimistic. The comparison across rows is
  like-for-like, but "matches or beats the reference" is a statement on the selection set.
- The main table uses the KL-anchored hybrid only (the paper's method). At KEV-4B the unanchored hybrid scores better
  on all three benchmarks; at KEV-0.8B the anchored one is better on both test sets. This is reported in the ablation,
  not used to pick the method per row, since that would let the test sets choose.

**Through-line with Steps 1–2.** A published table transfers where a new study shares its attributes (on the evidence
so far, price); microdata carries choice behaviour that transfers more broadly.

**Layout.** Figure 3 also shows the no-anchor hybrid (hollow orange) next to the anchored one; Table 3 does not. The
hybrid training trajectories are in the ablation. Choice KL is in Table A4.
KEV-9B hybrid runs show as NaN until they finish.

## Step 4 — Where the table helps: Kriner 2026 flight slopes by attribute

**Purpose.** An optional illustration of Step 3 at the level of individual attributes, on the set where the hybrid's
gain shows. Figure 4 plots each regime's slope error (model − human β) for the 12 Kriner 2026 flight attributes
(KEV-0.8B and KEV-4B). The numbers are in Table A6; Kreps 2020 slopes for all regimes are in Table A5.

**Why Kriner.** Of the three evaluation sets, adding Wu's table to Berke's choices helps most on Kriner (β MSE vs.
Berke alone: −53% at KEV-0.8B, −55% at KEV-4B; Grummon +9% / −11%; Kreps +8% / +38%). Kriner selects each run's
checkpoint, so the figure is optimistic for every regime; Tables 1–3 report all three sets.

**What it shows.**
- The gain is concentrated in the three price terms ($275, $300, $325). Berke choices alone under-state how much
  higher fares lower choice; adding Wu's table brings those errors close to zero, matching Wu + Berke choices.
- On most other attributes (airline, seat, baggage) the three regimes are close to one another.
- Berke choices + Wu table and Wu + Berke choices sit close together on nearly every term: 8 published numbers move the
  model where Wu's 18,591 choices move it.

**A possible mechanism.** Wu 2025 is an electricity-plan conjoint, not a flight study. Half of its 8 slopes are
monthly-bill effects ($110 to $125 vs. $105; verified against `beta_ols/wu2025.py` and the paper's Fig. 3), the
largest in the table. The hypothesis: the table teaches price sensitivity, which transfers to flight fares in Kriner;
Grummon and Kreps have no price attribute, which fits the table's small effect there. Berke also has a price attribute
(delivery fees of $0–$10), yet Berke choices alone under-state flight price sensitivity, so the price signal would
come from Wu's bill slopes specifically. This is consistent with Figure 4 but not a test of it.

## Ablation — the KL anchor

**What it compares.** The same runs with the KL anchor (α = 0.005) and without it (α = 0): the Wu table alone (Table 4a,
Figure 6) and Berke choices + Wu table (Table 4b, Figure 7), at all three sizes. Table 5 sweeps α for the Wu table
alone at KEV-0.8B and KEV-9B. Choice KL is in Table A7.

**Why Kriner only.** Whether to use the anchor is a design choice. Judging it on Grummon or Kreps would let the test
sets shape the method, so the ablation uses the validation set, Kriner 2026, alone. The test sets report how the
chosen method does (Tables 0–3); they are not evidence for or against the anchor.

**Wu table alone (Table 4a, Figure 6).** The effect depends on size.
- KEV-4B: without the anchor, Kriner gets worse than untrained from step 160 on and ends above it (4.71 vs. 4.10);
  its checkpoint is taken at step 96 (3.29). With the anchor, Kriner is above untrained only once, a spike at step 160, and otherwise stays below it (best.pt
  1.95, end 3.03).
  This is the clearest case of the anchor stabilising training.
- KEV-0.8B: both runs stay below untrained throughout and are about equal at best.pt (6.87 with vs. 6.68 without).
  Over the second half of training the anchored run is lower (mean 7.72 vs. 8.73).
- KEV-9B: the unanchored run is lower at best.pt (1.34 vs. 1.54) and over the second half (mean 1.60 vs. 2.37). Here
  the anchor costs fit on Kriner.
- Choice KL on Kriner (Table A7a) favours the anchor at every size (KEV-0.8B 0.791 vs. 0.815, KEV-4B 0.612 vs. 0.673,
  KEV-9B 0.669 vs. 0.782), including KEV-9B, where β MSE favours no anchor. At KEV-9B, then, the unanchored model fits
  the population slopes better but predicts individual choices worse.

**With microdata (Table 4b, Figure 7).** At KEV-0.8B the anchored hybrid is slightly lower at best.pt (4.55 vs. 4.72),
but both end about equal (5.43 vs. 5.30). At KEV-4B the unanchored hybrid is lower at best.pt (0.30 vs. 0.45) and to
the end (0.33 vs. 0.59). So once Berke's choices are in the mix, Kriner gives no reason to keep the anchor. KEV-9B is
still training.

**α sweep (Table 5).** Drift from the untrained model falls steadily as α grows at both sizes, so the anchor does what it
is built to do. Kriner β MSE at best.pt does not single out α = 0.005: at KEV-0.8B it is lowest at α = 0.005 (3.52 vs.
4.10 at α = 0); at KEV-9B it rises steadily with α, from 0.99 at α = 0 to 2.21 at α = 0.05. The sweep runs differ from
the paper runs: KEV-0.8B used learning rate 1e-5 (paper 5e-6), and both sweeps evaluated Kriner on a different
schedule, so their checkpoints differ from the paper runs'.

**What this means for the paper.** On the validation set, the anchor is not uniformly helpful. The defensible claim: the
anchor keeps the model close to the untrained model (Table 5 drift), and with the table alone it prevents the
unstable training seen at KEV-4B. It costs fit at KEV-9B and gives no benefit once microdata are added. One run per
setting cannot say how much of this is seed noise.

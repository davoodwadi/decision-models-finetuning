# 03-analysis-paper: story and claims

Prose for `03-analysis-paper.ipynb`; the notebook holds code and one-line headers. Sections are added here as each
step of the notebook is cleaned up. Numbers are read from the notebook's outputs, not typed into the notebook.

## Step 1 — Microdata fine-tuning transfers to unseen domains

**Punchline.** Training on respondents' choices from one study makes the model a better choice simulator in domains it
never saw, and pooling two studies transfers better than either one alone.

**Evidence (Table 1, Figure 1; KEV-0.8B and KEV-4B).**
- Every microdata regime (Wu choices, Berke choices, Wu + Berke choices) lowers population β MSE on every benchmark
  at both model sizes.
- Wu + Berke has the lowest β MSE of the three in every model × dataset cell.
- The transfer evidence is the two held-out test sets: Grummon 2025 (food warning labels) and Kreps 2020 (vaccines).
  Kriner 2026 selects the checkpoint, so its gains are optimistic; it is also a flight study, the same domain as Wu.

**What this does not show.** Wu + Berke has both more choices (18,591 + 36,297) and more domains than either study
alone, so the gain from pooling cannot be attributed to diversity rather than quantity.

**Why it matters for the paper.** More studies help, but respondent-level microdata is rarely available. Step 3 asks
whether a study's published coefficient table can stand in for its microdata.

**Table layout.** Rows are named by training data; row codes (`Row 2b`, `Row 3`, `Row 5`) stay internal. The untrained
model is the first row of each model block (the former Table 1). "Oracle" wording is introduced only in Steps 3–4,
where a regime is compared against a reference. Choice KL moves in the same direction as β MSE and is in Table A2.
KEV-9B has no Row 2b, 3, or 5 runs, so it is absent from Step 1; its untrained row appears in Table 2.

## Step 2 — A published table teaches the study's slopes

**Punchline.** Wu's 8 published OLS slopes, trained with the population β-loss and the KL anchor (α = 0.005), teach
the model Wu's own domain about as well as Wu's 18,591 respondent choices. They do not carry the choices'
out-of-domain transfer.

**Evidence (Table 2, Figure 2; all three model sizes).**
- In Wu's domain (Kriner 2026, flights), the table lowers β MSE at every size and is close to the choices at KEV-4B;
  at KEV-0.8B it is better than the choices.
- On the held-out test sets the choices are far ahead wherever both exist. At KEV-4B and KEV-9B the table still
  improves on the untrained model; at KEV-0.8B it is worse than untrained on both test sets.

**Why.** 8 numbers constrain 8 directions of the model's choice probabilities; nothing in the table describes other
domains, so there is nothing for the model to transfer beyond what those slopes imply. The KL anchor,
α·KL(p₀‖p_θ) on Wu's tasks, keeps the fit close to the untrained model; its effect is shown in the ablation.

**What this does not show.** KEV-9B has no Wu-choices run, so the table-vs-choices comparison is 0.8B and 4B only.

**Why it matters for the paper.** A table alone is a cheap, in-domain signal, not a substitute for microdata. Step 3
asks whether a table added to microdata you do have gives the best of both.

**Layout.** Figure 2 also shows the no-anchor table run (hollow red) next to the anchored one; Table 2 does not. The
rest of the no-anchor comparison is in the ablation at the end of the notebook. Choice KL is in Table A3.

## Step 3 — Your microdata + another study's table

**Question.** You have respondent choices for your own study (Berke) but only the published coefficient table for
another (Wu). Does adding Wu's table to Berke's choices give you what Wu's microdata would?

**Punchline.** In Wu's domain, yes: Berke choices + Wu table match or beat Berke + Wu choices. Out of domain, no:
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

**Through-line with Steps 1–2.** A published table carries its study's slopes; microdata carries choice behaviour that
transfers to other domains.

**Layout.** Figure 3 also shows the no-anchor hybrid (hollow orange) next to the anchored one; Table 3 does not. The
hybrid training trajectories are in the ablation. Choice KL is in Table A4.
KEV-9B has no hybrid runs.

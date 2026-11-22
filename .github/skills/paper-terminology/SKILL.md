---
name: paper-terminology
description: Use whenever drafting, editing, or reviewing manuscript prose, figure labels, table headers, captions, notes, or analysis notebooks for this Marketing Science paper. Enforces the repository's locked canonical terminology (training, choice training, off-the-shelf, unseen-study, train, reasoning effort, synthetic respondents, synthetic choice data) and forbids ambiguous or non-Marketing Science synonyms such as fine-tuning, untrained, zero-shot, align, post-training, or silicon sampling.
---

# Paper terminology guidelines

**When to use.** Invoke this skill whenever writing, editing, or reviewing [manuscript/manuscript-training.qmd](../../../manuscript/manuscript-training.qmd), any other manuscript or appendix file, or any figure, table, caption, note, or notebook script that produces reader-facing labels for this paper. Consistency across prose, tables, figures, captions, and notes is mandatory; never bounce between synonyms for the same concept.

## Canonical glossary

| Concept | Canonical term | Forbidden or avoided terms |
|---|---|---|
| Updating decision-model weights on human choice microdata | training, choice training, trained, choice-trained | fine-tuning, fine-tune, post-training, calibration (for weight updates) |
| Models prior to our choice training (generative LLMs and released KEV checkpoints) | off-the-shelf (table/figure label: Off-the-shelf) | untrained, bare base (as baseline regime label), zero-shot model |
| Evaluation on held-out studies excluded from training | unseen-study (e.g., unseen-study transfer, unseen-study evaluation) | bare zero-shot (without stating no target-study training data is used) |
| Pre-inference intervention verb | train (e.g., we train decision models before inference) | align, alignment, calibrate |
| Generative LLM thinking or reasoning knobs | reasoning effort (e.g., higher reasoning effort) | reasoning settings, reasoning budgets, stronger reasoning |
| Wang et al. (2026) and Leng et al. (2026) methods | post-inference adjustment, post-inference reweighting (on fixed model outputs using target-study human data) | conflating with training or post-training |
| Simulated participants in surveys and market research (general) | synthetic respondents | silicon sampling, silicon samples |
| Simulated choice outcomes in conjoint and discrete-choice experiments | synthetic choice data | silicon sampling, LLM-generated choice data (when covering decision models) |
| The seven empirical data collections used for training, validation, and testing | dataset, datasets (training datasets, validation dataset, test datasets, held-out test datasets) | test benchmarks, benchmark studies, benchmark designs, validation set, bouncing between studies and datasets |
| Full-sample human OLS reference estimates | human benchmark, human slope benchmark | reusing "benchmark" to mean test datasets |

## Rationale and enforcement rules

**Use training and choice training aggressively.** Marketing audiences frequently mistake "fine-tuning" for adjusting prompts, temperature, or sampling parameters rather than updating neural network weights on data. Never use "fine-tune" or "fine-tuning" anywhere in reader-facing prose, headings, tables, or figure captions. Clarify on first use that choice training updates neural network parameters (LoRA backbone adapters and the decision head) via respondent-level cross-entropy, which is equivalent to conditional maximum likelihood estimation (MLE). Use "training" and "choice training" aggressively to draw a sharp line between our pre-inference weight updates on historical studies and the post-inference adjustment or reweighting methods of Wang et al. (2026) and Leng et al. (2026), which keep model weights fixed and require target-study human data.

**Use synthetic respondents and synthetic choice data.** Marketing Science authors do not use "silicon sampling" in prose; that label comes from psychology and consumer behavior outlets. Use "synthetic respondents" for general discussion of simulated survey or market-research participants, matching Leng et al. (2026), and use "synthetic choice data" when referring specifically to simulated outcomes in conjoint analysis and discrete-choice experiments, echoing Wang et al. (2026) without limiting the term to text-generating LLMs. Never use "silicon sampling" or "silicon samples" in the title, abstract, keywords, prose, tables, or figure notes.

**Avoid post-training and calibration for weight updates.** Do not describe our method as "post-training" because it collides visually and conceptually with "post-inference." When describing how generative LLM providers create chat or reasoning models, prefer wording such as "optimization for deterministic answers and safety compliance." Likewise, do not use "calibrate" or "calibration" to describe updating weights on choice microdata, because marketing readers associate calibration with post-inference reweighting (e.g., HuMCal).

**Use off-the-shelf for every model prior to choice training.** Calling released generative LLMs or released KEV checkpoints "untrained" is technically vulnerable because those checkpoints were pretrained on large corpora or objective single-label benchmarks. Use "off-the-shelf" symmetrically across generative LLMs and released decision models (e.g., "off-the-shelf generative LLMs", "off-the-shelf decision models", "off-the-shelf baseline"). In all main-paper and appendix tables, axis labels, and legends comparing regimes, label the baseline regime `Off-the-shelf` rather than `Untrained` or `Base`.

**Use unseen-study instead of bare zero-shot.** Marketing readers often associate "zero-shot" with zero-shot prompting (no in-context examples in the prompt) rather than out-of-study evaluation. Use "unseen-study" (e.g., "unseen-study transfer", "unseen-study evaluation", "transfer to unseen datasets") and explicitly state that no target-study training data is used.

**Use train instead of align.** In AI research, "alignment" evokes safety and value alignment (RLHF) rather than econometric parameter estimation. Write "we train decision models before inference" rather than "we align models before inference."

**Use reasoning effort for LLM thinking controls.** Refer to generative LLM thinking controls uniformly as "reasoning effort" (e.g., "increasing reasoning effort", "lower (off, min, low) and higher (on, high) reasoning effort") rather than "reasoning settings", "reasoning budgets", or "stronger reasoning."

**Use dataset consistently for the seven empirical datasets.** Match `@tbl-datasets` (`Dataset`) across all prose, headings, tables, and notes: refer to our seven data collections as `training datasets`, `validation dataset`, and `test datasets` (or `held-out test datasets`). Never call test datasets `test benchmarks`, `benchmark studies`, or `benchmark designs`, and never call the validation dataset a `validation set`. Reserve `benchmark` strictly for the full-sample human slope benchmark ($\hat{\boldsymbol{\beta}}^{(g)}$) and pretrained objective single-label benchmarks. Reserve `study` or `studies` for published papers, literature citations, or general conceptual framing (e.g., `choice study`, `unseen-study transfer`).

## Concrete examples

```text
WRONG: Whereas existing post-inference methods adjust fixed model outputs, we align models before inference through fine-tuning supervision.
RIGHT: Whereas existing post-inference methods keep model weights fixed and adjust or reweight outputs after inference using target-study human data, we train decision models before inference by updating their weights on historical human choice microdata (a procedure we refer to as choice training).

WRONG: Untrained decision models and generative LLMs with higher reasoning settings fail to recover human treatment effects in zero-shot evaluation.
RIGHT: Off-the-shelf decision models and off-the-shelf generative LLMs, including higher reasoning effort, fail to recover human treatment effects in unseen-study evaluation.

WRONG: Table regimes: Untrained, Electricity, Drones, Drones + Electricity.
RIGHT: Table regimes: Off-the-shelf, Electricity, Drones, Drones + Electricity.

WRONG: Silicon sampling with off-the-shelf models biases downstream regressions.
RIGHT: Replacing human respondents with synthetic choice data from off-the-shelf models biases downstream regressions.

WRONG: Section heading "## Microdata fine-tuning"
RIGHT: Section heading "## Microdata choice training"

WRONG: Our four unseen-study test benchmarks never enter training; Flights is a dedicated validation set.
RIGHT: Our four held-out test datasets never enter training; Flights is a dedicated validation dataset.
```

## Verification before finishing edits

**Audit edited text and plotting code.** After editing manuscript prose, table headers, figure labels, or notebook scripts that export tables or plots, search the modified files for forbidden terms (`fine-tun`, `untrained`, `zero-shot`, `align`, `post-training`, `reasoning setting`, `reasoning budget`, `silicon`, `test benchmark`, `benchmark study`, `benchmark design`, `validation set`) and verify that every match is either removed or an intentional internal reference (e.g., a checkpoint directory slug, `.bib` filename, or citation key that never appears in rendered text, tables, or plots). Always keep the `TERMINOLOGY GUIDE` comment block at the top of [manuscript/manuscript-training.qmd](../../../manuscript/manuscript-training.qmd) synchronized if the user ever revises a term.

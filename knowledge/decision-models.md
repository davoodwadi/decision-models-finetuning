# Decision models (TypeSafe "System One" / Jev)

Source: https://openrouter.ai/typesafe/jev-1.13 (read 2026-10-04).
TypeSafe docs: https://docs.typesafe.ai/concepts/system-one

## What a decision model is

A **decision model** does not generate text. It takes a **state** plus a set of
named, typed **questions** about that state, and returns **calibrated
probabilities** as structured answers. The calling code owns the workflow and
acts on the answers (thresholds, routing, gating).

TypeSafe calls this family **System One** models (fast and intuitive, by analogy
to dual-process System 1, versus slower, text-generating System 2 LLMs).
**Jev** is the first System One model.

## How a decision model works (general mechanism)

This is a synthesis of the three open-weight models documented below (Intern-Decision,
JevK5, Kev). It describes how decision models work *in the open*. Jev's own internals
are undisclosed and may differ.

**In one sentence:** a decision model is a pretrained causal LM, fine-tuned so that,
in a single prefill-only forward pass, the hidden states or logits at designated
positions produce a softmax over *only the options supplied with each question*.
Nothing is sampled or generated.

### 1. Backbone
- An ordinary pretrained decoder-only LM. All current open examples use
  **Qwen3.5-4B** (0.8B, 2B and 9B variants also exist). Kev's description: 32 layers,
  24 Gated DeltaNet linear-attention + 8 full-attention, hidden size 2,560.
- Adapted by a **full fine-tune** (Intern-Decision) or a **LoRA of rank 16** (JevK5,
  merged into the weights; Kev, with a frozen backbone). Model knowledge comes from
  the backbone: fine-tuning teaches the decision format and calibration, not new facts
  (e.g. Kev's MMLU-Pro is 0.565 vs Jev's 0.840).

### 2. Input compilation (the request is turned into a prompt)
- Request = `state` + named `questions` (type, instructions, criteria/options).
- A deterministic "compiler" renders this into a prompt in the model's chat template:
  the state, then each question with its options listed in order. JSON states are
  rendered as labelled text.
- Each option gets an **identifier the readout can point at**:
  - a single-token symbol (A, B, ..., Z, a, ..., z, 0, ..., 9) for logit readouts, or
  - the option text's own closing token for a pointer head.
- `noul` is just a two-option choice (no, yes). `score` is a choice over ordered
  levels. Readout, training and calibration are the same for all three types; only the
  post-processing of the answer differs.

### 3. Forward pass and readout (where the probabilities come from)
- **One causal forward pass, prefill only.** `generate()` is never called.
- **Readout position(s):** for each question, a specific token position whose hidden
  state is used for that decision.
- Two readout styles:
  - **(A) Restricted next-token logits** (Intern-Decision, JevK5 via SemIf's protocol):
    take the LM-head logits at the position where the answer symbol would come next,
    keep **only the logits of that question's allowed symbols**, and take a softmax over
    them. Intern-Decision places an answer skeleton with one `<decision>` placeholder per
    question in the prompt, so all questions are read in the *same* pass (at the position
    just before each placeholder).
  - **(B) Pointer head** (Kev): a small trained head with two linear projections scores
    the hidden state at each option's closing token against the hidden state at the
    question's final token. A softmax over those scores gives the probabilities. The LM
    head is not used.
- **Multiple questions:** either packed into one sequence (Intern-Decision), or the
  **state's KV cache is computed once** and each question runs as its own row continuing
  from it (Kev; this matches Jev's documented "ingest state once, evaluate questions in
  parallel"). Isolated rows mean questions cannot influence each other.
- **Many options:** limited by the symbol alphabet (16 for JevK5, 62 for Intern-Decision).
  With more options, JevK5 uses a "knockout" scheme: score groups of 16, then a final
  round, combined with a second temperature. Kev's pointer head allows up to 255 options.

### 4. Output post-processing (identical across models, Jev-compatible)
- `probabilities`: the distribution, mapped back to the original option keys.
- `choice` / `decision`: argmax (ties broken lexically).
- `noul`: P(yes).
- `score`: **expected level index** = sum of i times p_i (can fall between levels).
- `confidence`: the maximum probability (Intern-Decision). Jev describes its confidence
  only as a measure of how concentrated the distribution is.
- `usage.output_tokens` counts scored decisions, not generated text.

### 5. Training
- **Loss:** cross-entropy over each question's options at the readout position. When the
  gold answer is itself a distribution, the loss is against that distribution (JevK5).
  TypeSafe calls its own approach RL (RLCD); the open models use supervised CE.
- **Data:** public classification, NLI, QA and tool-use train splits with native labels,
  plus synthetic "teacher" questions (a large model writes documents and typed questions,
  keeping only answers it gives consistently). Also programmatically generated policy,
  rule, date and probability cases, and "missing evidence" cases with a uniform target
  (to teach abstention).
- **Robustness tricks:** shuffle option order, insert distractors and "none of the above",
  and use minimal pairs (the correct option present vs removed).
- **Small compute:** about 1-2 epochs and a few GPU-hours on one H100/H200.

### 6. Known properties and failure modes
- **Option order can still change answers** (Kev states this explicitly).
- Weakest areas: date arithmetic and numbers; knowledge-heavy exams (limited by the
  backbone); long states beyond the trained length (accuracy drops past ~8-16k tokens).
- Level descriptions matter: bare-number levels give diffuse distributions (from Jev docs).
- Latency is ~10-50 ms per request on one GPU for short states, versus ~100-200 ms for Jev.

### Relation to plain LLM token-probability measurement
Readout (A) is the same operation as reading an instruction-tuned LLM's next-token
distribution over answer-option tokens and renormalising over the allowed set. The
differences are: (i) fine-tuning to make that distribution accurate, and (ii) a fixed
compiler and prompt format.
A decision model is therefore "constrained option scoring" made into a product.

## Question types

| type    | criteria format                          | answer                                             |
|---------|------------------------------------------|----------------------------------------------------|
| `noul`  | `{true: "...", false: "..."}`            | `noul`: P(yes) in [0, 1]                           |
| `choice`| `{label: "description", ...}`            | `choice` (argmax label) + `probabilities` over all |
| `score` | ordered list, e.g. `["Calm","Frustrated","Very angry"]` | `score` + full distribution over the rubric |

The state can be a string, an object or an array (context limit: 32K tokens).

## Jev 1.13 facts

- Model id: `typesafe/jev-1.13` (`Jev Latest` always points to the newest Jev)
- Released: 2026-09-18
- Price: $0.042 per 1M input tokens; output is free (it produces no text)
- Latency: ~0.2 s P50, ~0.7 s P99
- Related: **Jev Router**, which uses Jev to pick the model and reasoning effort for each request

## API (OpenRouter Decisions API, alpha)

Not OpenAI-compatible: chat-completions SDKs will not work.

```
POST https://openrouter.ai/api/alpha/decisions
Authorization: Bearer $OPENROUTER_API_KEY
```

```json
{
  "model": "typesafe/jev-1.13",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent":  {"type": "noul",   "instructions": "Does this message convey urgency?",
                   "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds",
                                "technical": "Bugs, outages, integrations",
                                "sales": "Pricing, upgrades, new accounts"}},
    "frustration":{"type": "score",  "instructions": "How frustrated is the customer?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

The response has an `answers` object keyed by question name, e.g.
`answers.is_urgent.noul`, `answers.department.choice`,
`answers.department.probabilities` and `answers.frustration.score`.

TypeScript SDK: `openrouter.alpha.decisions.create({ decisionsRequest: {...} })`
from `@openrouter/sdk`.

## Typical uses

Routing, classification and labelling at scale, gating agent tool calls (e.g.
auto-execute if P(safe) >= 0.8), auto-approving coding-agent permission
prompts, and checking a cheaper model's output in a model cascade.

## How Jev works internally: undisclosed

TypeSafe docs (read 2026-10-04: concepts/system-one, introduction/machine-learning-primer,
models, primitives/score) do NOT reveal the backbone, size, or readout. What they say:
- A pretrained LM post-trained with **RLCD** ("reinforcement learning for calibrated
  decisions"), presented as a third path after RLHF and RLVR (RL with verifiable rewards).
  Outputs decisions and probabilities, not text.
- The same weights serve every account (no per-customer fine-tune or LoRA).
- The state is ingested once and all questions are evaluated against it in parallel
  (context: 64k total, 32k for the state plus the longest question).
- Score levels are "each judged on its own against the state": the model never sees
  the level numbers or neighbouring levels, and bare-number levels ("0","1","2") work poorly.
- Score response: `probabilities` per level (summing to 1), `score` = sum of level index times
  probability, `confidence` = a measure of how concentrated the distribution is.
  `usage.output_tokens` is non-zero (~18) despite no text.

## Open-weight decision models (how they actually work)

All three are Qwen3.5-4B fine-tunes, are read in **one forward pass with no generate()**,
and serve the Jev/TypeSafe `/v1/systemone` request/response shape. None is Jev's real
architecture.

### Readout A: option-letter next-token logits (Intern-Decision, JevK5)
1. Map each question's options to single-token symbols (A, B, C ...).
2. Render the prompt: state + schema of questions + options.
3. One causal forward pass; read the logits at the position where the answer
   letter would come next (Intern-Decision: a JSON answer skeleton with one
   `<decision>` placeholder per question, read at the position just before each
   placeholder, so all questions are scored in one pass).
4. Softmax over **only** that question's allowed letter logits.

- **internlm/Intern-Decision-4B** (https://huggingface.co/internlm/Intern-Decision-4B):
  multimodal (images too), full fine-tune of Qwen3.5-4B, T = 1.992 fitted by NLL on 1,728
  calibration cases. Up to 16 questions x 62 options per request, 8,192-token limit. Also
  sizes 0.8B and 2B. Card reports avg 90.02 vs Jev 88.74 on its 7-benchmark table, with
  Brier 0.347 vs 0.358 and ECE 0.065 vs 0.095; ~44 ms on an RTX 4090 vs Jev ~110 ms.
- **alibiserikbay/JevK5** v0.3 (https://huggingface.co/alibiserikbay/JevK5): LoRA rank 16
  on the attention projections, merged into the weights. Uses the readout protocol of SemIf
  (TheoLeeCJ/SemIf). T = 1.22. For more than 16 options it scores groups of 16 in several
  passes ("knockout") and combines them with a second temperature of 0.93. Trained with
  cross-entropy on the option-letter logits (or against a target distribution when the gold
  answer is a distribution). Data: 17k teacher questions (Qwen3.6-27B, GPT-6 Luna) plus
  30k items from public train splits. No Jev outputs used. ~13 ms per decision on an H100.
  JevBench v1.4: v0.2 ranked #2 of 76 (62.04 vs Jev 63.29).

### Readout B: pointer head (Kev)
- **jaredpalmer/kev-4b** (https://huggingface.co/jaredpalmer/kev-4b, github.com/jaredpalmer/kev):
  the backbone (Qwen3.5-4B-Base: 32 layers = 24 Gated DeltaNet linear-attention
  + 8 full-attention, hidden 2,560) is kept frozen, with a LoRA (r16, alpha 32, on the attention,
  MLP and DeltaNet projections; 33.8M parameters) and a **pointer head**: two projections
  score each option's *closing token* hidden state against the question's *final
  token* hidden state, followed by a softmax over the options. Runs as prefill only.
  - The state's KV cache is computed once. Each question is its own row continuing from
    the cached state, so questions cannot influence each other. Option order can still
    change the answer.
  - Up to 255 options per question. T = 2.41. Loss = cross-entropy over the options,
    with shuffled option order and inserted "none of the above"/distractor options.
  - Trails Jev on held-out breadth (chance-corrected index 38 vs 54) and on
    knowledge (MMLU-Pro 0.565 vs 0.840). Better calibrated on held-out data
    (ECE ~0.02-0.03).

### Common pattern
See "How a decision model works (general mechanism)" above. Benchmarks used in this
ecosystem: JevBench, the Jev Decision Index, bev-decision-150K, SemIf.

## Relevance to this project

The paper measures exact Likert-scale distributions from LLMs via token
probabilities. Readout A (softmax over the allowed option-token logits at the answer
position) is essentially the same measurement, plus decision fine-tuning, so these models are natural comparison points. Kev's pointer
head is a different, non-token readout. Jev's `score` collapses the distribution to
an expected value. Jev itself is a black box, and its "calibrated" claim
comes from the vendor and has not been verified here.

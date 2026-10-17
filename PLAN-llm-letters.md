# Plan: fine-tune an LLM through its own next-token head (option letters)

Status: built 2026-10-14 (models/llm_letters.py; train.py without --kev). Builds on the dataset layer (PLAN-record-construction.md) and the model layer (PLAN-model-layer.md).

## What the trainer will accept

| Flags | Model | Weights trained |
|---|---|---|
| `--kev --base jaredpalmer/kev-0.8b` | Kev: backbone + pointer head (today's model) | released LoRA adapter + head |
| `--kev --lora_rank R` | Kev, new rank-R adapter on the plain base + released head | adapter + head |
| `--kev --full_finetune` | Kev, plain base, no adapter + released head | every weight |
| `--base <any HF causal LM> --prompt prompts/letters-v1.txt` | the LLM with its own vocabulary head | every weight |

Without `--kev` there is no pointer head and no LoRA: the LLM is fully fine-tuned. `--lora_rank` / `--full_finetune`
without `--kev` are refused.

## The letter model, step by step

1. **Pre-processing (prompt).** For one question of one record (the view after augmentation, so reversed scales and
   shuffled options are already in place), the model's own chat template is applied to:
   - system: the prompt file (e.g. "You answer survey questions as the person described. Reply with the letter of one
     option only.");
   - user: the persona text (the view's `state`, the same text Kev reads), the question's instructions, and
     its options in view order, labelled `A.`, `B.`, ... (at most 12 options in ISSP, 9 in SubPOP: A-L suffice);
   - the assistant turn opened, thinking off (`enable_thinking=False` where the template has it), so the next token is
     the answer.
   One prompt per question: a SubPOP record with 7 questions becomes 7 prompts (as Kev's row form does on Qwen3.5).
2. **One forward pass, no generation.** Rows are right-padded (pads come after every real token, so causal and
   DeltaNet layers never see them) and the backbone's hidden state is taken at each row's last real token.
3. **Post-processing (constrain + normalise).** The vocabulary head is applied to those positions only, and of its
   logits only the K option letters' are kept. The model returns these K logits per question, the same shape the Kev
   model returns, so `soft_cross_entropy` (log-softmax over the K logits = restrict to valid letters and renormalise),
   `with_probs`, and every metric stay unchanged. No temperature.
4. **Letter tokens.** Found once per tokenizer by rendering the template and reading which token ids "A".."L" become
   right after the assistant prefix (Qwen: `A` and ` A` are different tokens). Each must be a single token, or the
   model refuses to load.

Why the vocabulary head only at the answer position: logits for every position would be batch x length x vocabulary
(128 x ~300 x 248k floats, ~38 GB). At one position per question it is 128 x 248k.

## Code

| File | What | Size |
|---|---|---|
| `llm_letters.py` (new) | `LetterModel`: load (`AutoModelForCausalLM` + tokenizer + prompt), letter ids, `prepare(view)` -> prompt ids + K letter ids + target in option order, `forward(batch)` -> K logits per question, `save(folder)` (`save_pretrained` + tokenizer + `letters.json`: base, prompt text, letter ids), `load(folder)` | ~150 lines |
| `kev_model.py` (new, moved) | `KevModel`: today's `load_model`, `save_checkpoint`, `to_model_input`, Kev's encode + `forward_batch` behind the same `prepare` / `forward` / `save` methods | ~70 lines, mostly moved |
| `train.py` | `train_epoch` and `predict` call `model.prepare` / `model.forward` instead of `to_model_input` + `model.encode`; load and save dispatch on `--kev`; `--prompt`; log the probability the LLM puts on the valid letters before renormalising (whether it follows the prompt) | ~40 lines changed |
| `prompts/letters-v1.txt` (new) | the system prompt; its text is copied into run_info.json and the checkpoint | a few lines |
| post-eval notebook `cached_predictions` | load either checkpoint kind (Kev `head.pt` or `letters.json`) | ~15 lines |
| `show_train_samples.py` | print the exact prompt (template applied) for the letter model | ~20 lines |

## Checks before any GPU run (CPU, in the scratchpad)

1. Kev path unchanged after the move: the same logits as now, on a tiny random Qwen3.5 (as the `--full_finetune` check).
2. Prompts render as intended for `score`, `choice` and `noul` questions; after reversing a scale or shuffling options,
   letter i always carries the target share of the option shown i-th.
3. Every letter is one token and is the token the template puts at the answer position.
4. Tiny random model: forward + backward reach every weight; save, reload, same logits.

The first real check of the prompt (how much probability lands on valid letters before training) needs the model's
weights: that is the user's run (step 0 of the first training run logs it).

## Decided

Default LLM: `Qwen/Qwen3.5-0.8B`, the post-trained chat version (it follows system prompts; the user downloads it).
Any other causal LM can be passed with `--base`.

Where the code goes: `models/llm_letters.py` and `models/kev_pointer.py` behind one interface (PLAN-model-layer.md), not
the top-level `llm_letters.py` / `kev_model.py` named in the table above.

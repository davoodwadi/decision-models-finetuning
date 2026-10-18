# Plan: a separate model layer

Status: built 2026-10-14 (models/; train_issp.py renamed train.py).

Goal: change **which model** we train (Kev's pointer model, or an LLM through its own next-token head) and **how**
(adapter or every weight) with flags, never by editing the trainer. Adding a new kind of model = one new module in
`models/` that implements the interface below. Companion to PLAN-record-construction.md (the dataset layer) and
PLAN-llm-letters.md (the first new model).

## Three layers

| Layer | Job | Knows about |
|---|---|---|
| **Data** (`training_data.py`, `constructions.py`, `issp_sampler.py`, `issp_augment.py`) | Records and their views (persona, questions, options, target shares keyed by option) | the files on disk |
| **Model** (`models/`) | Turn a view into model inputs and targets in the model's option order; run a batch to K logits per question; save and load checkpoints | a view; its weights and tokenizer |
| **Trainer** (`train.py`) | Loss, optimizer, schedule, epochs, validation, best weights, metrics, run folder | only the interface below |

The boundary between data and model is the **view**: what `issp_augment.view` returns: the run's augmentation in training (default: every variation off), always
the canonical view for calibration and test.
Augmentation stays in the data layer: it is the same for every model.

## Interface (`models/base.py`)

```python
class SurveyModel(torch.nn.Module):
    kind: str                                     # "kev" | "llm_letters": written to run_info.json and the checkpoint
    def prepare(self, view) -> (inputs, questions)
        # inputs: whatever this model's forward needs for one record; questions: per question
        # {v, src, type, target}, target = the observed shares in this model's option order (sums to 1)
    def forward(self, batch: list[inputs]) -> list[list[Tensor]]
        # per record, per question: K logits, one per option, in the same order as the target
    def trainable_parameters(self) -> list[Parameter]
    def enable_checkpointing(self)                # gradient checkpointing, however this model does it
    def save(self, folder, extra: dict)           # a checkpoint that load() reads back
    def describe(self) -> dict                    # for run_info.json: kind, base, what is trained, prompt, ...

def build(args) -> SurveyModel                    # models/__init__.py: from the trainer's model flags
def load(folder_or_hub_id, device) -> SurveyModel # models/__init__.py: any saved checkpoint, by its files
                                                  # (head.pt: Kev; letters.json: llm_letters)
def preview(args, view) -> str                    # the exact text the model reads, tokenizer only (show_train_samples)
```

Everything the trainer does with a model goes through these: `train_epoch` and `predict` become
`model.forward([model.prepare(view)[0] ...])`, and the loss, `with_probs` and every metric read K logits against a
target of K shares, whatever the model. `trainable_state` / `restore` stay in the trainer: they only use
`trainable_parameters`.

## Modules

| Module | What | From |
|---|---|---|
| `models/__init__.py` | `build`, `load`, `preview`: pick the module from the flags or the checkpoint's files | new, ~40 lines |
| `models/base.py` | the interface | new, ~30 lines |
| `models/kev_pointer.py` | `KevPointerModel`: released adapter, `--lora_rank`, `--full_finetune`; `prepare` = today's `to_model_input` + Kev's `encode`; `save` = today's `save_checkpoint` | moved from `train_issp.py` (`load_model`, `save_checkpoint`) and `training_data.py` (`to_model_input`, the only Kev import in the data layer) |
| `models/llm_letters.py` | `LetterModel` (PLAN-llm-letters.md) | new, ~150 lines |
| `prompts/letters-v1.txt` | the letter model's system prompt | new |

Named `kev_pointer.py`, not `kev.py`, so it can never be confused with the vendored `kev` package it imports.

## Trainer flags

```
--kev                       Kev's pointer model; --base is a Kev checkpoint (default jaredpalmer/kev-0.8b)
  [--lora_rank R | --full_finetune]   new adapter / no adapter, as today; default: continue the released adapter
(no --kev)                  an LLM through its next-token head, every weight trained;
                            --base is any Hugging Face causal LM (default Qwen/Qwen3.5-0.8B, the post-trained chat
                            version); --prompt is the system prompt file (default prompts/letters-v1.txt)
```

`--lora_rank` and `--full_finetune` without `--kev` are refused; `--prompt` with `--kev` is refused. The model's
`describe()` goes into run_info.json beside the data recipe, so every run records both layers.

## What else changes

- `subpop-test-post-eval.ipynb`: `cached_predictions` loads with `models.load(run, device)` instead of
  `kev.checkpoint.load`, so released Kev, our Kev runs and letter-model runs sit side by side in `MODELS`. Its saved
  predictions stay valid (they are keyed by checkpoint and rows, not by code).
- `show_train_samples.py`: prints `models.preview(...)`, so it shows what either model reads.
- `commands.md`: every existing command gains `--kev` (without it, a run would now train the LLM).

## Order and checks (CPU, scratchpad; no GPU run needed until the end)

1. **Move Kev behind the interface, no behaviour change.** On a tiny random Qwen3.5 (as the `--full_finetune` check):
   the same logits and the same gradients before and after the move, for the released adapter, `--lora_rank` and
   `--full_finetune`; checkpoints the new code saves load through Kev's loader with the same logits. The data replay
   check (r64-v3's logged entropies) still passes, since the data layer only loses `to_model_input`.
2. **Notebook and `show_train_samples.py` on the new loader.** Same predictions for the tiny model through both paths.
3. **Add `LetterModel`** with PLAN-llm-letters.md's checks. Needs the `Qwen/Qwen3.5-0.8B` tokenizer (the user downloads
   it; a few MB). The first real run's step 0 logs how much probability lands on valid letters.

## Decided

`train_issp.py` is renamed `train.py`: it no longer knows about ISSP (data layer) or Kev (model layer).

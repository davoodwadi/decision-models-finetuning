"""The model layer (PLAN-model-layer.md): which model a run trains, and loading any saved one.

  --kev                 Kev's pointer model (models/kev_pointer.py); --base is a Kev checkpoint. With --lora_rank R a
                        new rank-R adapter on the plain base, with --full_finetune no adapter (every weight trained);
                        by default the released adapter is trained further. The released pointer head is always kept.
  (no --kev)            an LLM through its own next-token head (models/llm_letters.py), every weight trained; --base is
                        any Hugging Face causal LM, --prompt its system prompt file.
  --train_last_layers N only the last N decoder layers trained (LLM, or --kev --full_finetune); with --base a previous
                        run's best/ and a larger N, layers are unfrozen stage by stage.
  --lr_decay M          layer-wise learning rates: --lr for the last decoder layer, divided by M once per layer
                        going down (LLM, or --kev --full_finetune).
  --weights_dtype bf16  --kev with an adapter: the frozen backbone in bf16 (half the memory, for Kev-9B on 32 GB).

Every model implements models/base.py's SurveyModel, which is all the trainer (train.py) uses.
"""
from pathlib import Path

from models import kev_pointer, llm_letters


def add_arguments(parser):
    """The model flags, for train.py and show_train_samples.py."""
    parser.add_argument("--kev", action="store_true",
                        help="Kev's pointer model (--base: a Kev checkpoint). Without it: an LLM through its own "
                             "next-token head, every weight trained (--base: a Hugging Face model, --prompt)")
    parser.add_argument("--base", default=None,
                        help=f"the model to start from (default: {kev_pointer.DEFAULT_BASE} with --kev, "
                             f"{llm_letters.DEFAULT_BASE} without)")
    backbone = parser.add_mutually_exclusive_group()
    backbone.add_argument("--lora_rank", type=int, default=None,
                          help="--kev only: train a new LoRA adapter of this rank (same layers, alpha = 2 x rank) on "
                               "the plain base instead of the released adapter; the released pointer head is kept")
    backbone.add_argument("--full_finetune", action="store_true",
                          help="--kev only: no LoRA: train every weight of the plain base backbone (without Kev's "
                               "adapter) and the released pointer head; use a much lower --lr than for LoRA (~1e-5)")
    parser.add_argument("--prompt", default=None,
                        help="without --kev only: the system prompt file (default: prompts/letters-v1.txt)")
    parser.add_argument("--train_last_layers", type=int, default=None,
                        help="train only the last N decoder layers (and the final norm, and Kev's head); the token "
                             "embeddings and earlier layers stay frozen. Default: every weight. For the LLM, or --kev "
                             "with --full_finetune; for incremental unfreezing, continue with --base <run>/best and a "
                             "larger N")
    parser.add_argument("--weights_dtype", choices=["fp32", "bf16"], default="fp32",
                        help="--kev with an adapter only: the dtype of the frozen backbone weights. bf16 halves their "
                             "memory (Kev-9B: ~36 -> ~18 GB) so a larger base fits; the adapter and pointer head stay "
                             "fp32. The checkpoint records it and loads the same way (kev.checkpoint)")
    parser.add_argument("--lr_decay", type=float, default=1.0,
                        help="layer-wise learning rates: the last decoder layer (and the final norm, and Kev's head) "
                             "train at --lr, each layer below at the rate of the layer above divided by this, the token "
                             "embeddings one step below the first layer (e.g. 2.6: the layer before the last at lr/2.6, "
                             "the one before at lr/2.6^2). Default 1: every weight at --lr. For the LLM, or --kev with "
                             "--full_finetune")


def check_arguments(parser, args):
    """Refuse flags that do not apply to the chosen model; fill in the defaults."""
    if args.kev and args.prompt:
        parser.error("--prompt is for the LLM (without --kev); Kev reads no system prompt")
    if not args.kev and (args.lora_rank or args.full_finetune):
        parser.error("--lora_rank / --full_finetune are for --kev; without --kev every weight of the LLM is trained")
    if args.kev and args.train_last_layers is not None and not args.full_finetune:
        parser.error("--train_last_layers needs --full_finetune with --kev (it freezes backbone layers, not an adapter)")
    if args.kev and args.lr_decay != 1 and not args.full_finetune:
        parser.error("--lr_decay needs --full_finetune with --kev (it sets backbone layers' rates, not an adapter's)")
    if args.weights_dtype == "bf16" and (not args.kev or args.full_finetune):
        parser.error("--weights_dtype bf16 is for --kev with an adapter (released or --lora_rank): it sets the frozen "
                     "backbone's dtype; trained weights stay fp32")
    if args.lr_decay < 1:
        parser.error("--lr_decay must be at least 1 (earlier layers train slower, not faster)")
    args.base = args.base or (kev_pointer.DEFAULT_BASE if args.kev else llm_letters.DEFAULT_BASE)
    base = Path(args.base)
    if not base.exists() and (args.base.startswith(("runs/", "./", "../", "/")) or args.base.count("/") != 1):
        # not a Hugging Face id (namespace/name) either: Kev's loader would report it as an invalid repo id
        versions = sorted(p.name for p in base.parent.parent.glob(f"{base.parent.name}-v*") if (p / base.name).exists())
        parser.error(f"--base {args.base}: no such folder" + (f"; runs with that name and a {base.name}/: {', '.join(versions)} "
                     f"(e.g. --base {base.parent.parent / versions[-1] / base.name})" if versions else ""))
    if args.kev and args.lora_rank and (base / "adapter_config.json").exists():
        parser.error(f"--lora_rank starts a new untrained adapter, so {args.base}'s trained adapter would be dropped "
                     f"(only its pointer head is kept); to continue training it, leave out --lora_rank")
    if not args.kev:
        args.prompt = str(args.prompt or llm_letters.DEFAULT_PROMPT)


def build(args, device):
    """The model a run trains, from the flags (after check_arguments)."""
    if args.kev:
        return kev_pointer.KevPointerModel.build(args.base, device, args.lora_rank, args.full_finetune,
                                                 args.weights_dtype)
    return llm_letters.LetterModel.build(args.base, args.prompt, device)


def encoder(args):
    """The model's tokenizer-only part (prompt format, targets, preview), from the flags, without loading weights."""
    if args.kev:
        return kev_pointer.KevEncoder.for_base(args.base)
    return llm_letters.LetterEncoder.for_base(args.base, args.prompt)


def load_encoder(run):
    """Any saved model's tokenizer-only part (its input format, preview), without loading weights; the same checkpoints
    as load()."""
    if (Path(run) / llm_letters.META_FILE).exists():
        return llm_letters.LetterEncoder.for_checkpoint(run)
    return kev_pointer.KevEncoder.for_base(run)


def load(run, device):
    """Any saved model for evaluation: a letter-model checkpoint (it holds letters.json), otherwise a Kev checkpoint
    (a run folder or a Hub id such as jaredpalmer/kev-0.8b)."""
    path = Path(run)
    if (path / "best").is_dir():   # a run folder, not its checkpoint
        raise ValueError(f"{run} is a run folder; its checkpoint is {path / 'best'}")
    if not path.exists() and (Path("runs") / path).exists():   # a run name without runs/ (would be read as a Hub id)
        raise FileNotFoundError(f"no folder {run}; the run's checkpoint is {Path('runs') / path / 'best'}")
    if (path / llm_letters.META_FILE).exists():
        return llm_letters.LetterModel.load(run, device)
    return kev_pointer.KevPointerModel.load(run, device)

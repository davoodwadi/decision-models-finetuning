"""Kev's pointer model behind the model interface (models/base.py): a Qwen backbone run prefill-only, and a pointer head
that scores each option's closing token against the question's last token (kev/kev/model.py).

Three ways to train it (train.py --kev):
  default          the released Kev: its LoRA adapter and pointer head, both trained further
  lora_rank=R      a new untrained rank-R adapter on the plain base backbone, and the released head
  full_finetune    no adapter: the plain base backbone, every weight trained, and the released head; when the base
                   is a full-weight checkpoint (a previous full_finetune run's best/), its backbone and head instead,
                   so a run continues where the previous one stopped
Checkpoints are Kev's format (adapter, or the whole backbone; head.pt; tokenizer), loadable by kev.checkpoint/kev.serve.
"""
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kev"))
from kev.checkpoint import Checkpoint, LoadOptions, write_meta          # noqa: E402
from kev.checkpoint import load as load_checkpoint                       # noqa: E402
from kev.model import MAX_STATE, DecisionModel, encode, load_tokenizer  # noqa: E402

from models.base import SurveyModel, options_and_targets                 # noqa: E402

DEFAULT_BASE = "jaredpalmer/kev-0.8b"


class KevEncoder:
    """Kev's input format, tokenizer only: the view's state and questions packed with Kev's special tokens."""

    def __init__(self, tok, option_isolation=False):
        self.tok, self.option_isolation = tok, option_isolation

    @classmethod
    def for_base(cls, base):
        """The encoder of a Kev checkpoint (its base's tokenizer), without loading weights."""
        meta = Checkpoint(base).meta
        return cls(load_tokenizer(meta.base, revision=meta.base_revision), meta.option_isolation)

    def prepare(self, view):
        record, questions = options_and_targets(view)
        return encode(self.tok, record, option_isolation=self.option_isolation), questions

    def preview(self, view):
        """The encoded record decoded back to text, for reading: special tokens included, each special token except
        <|box_end|> on a new line (display only: Kev reads no such line breaks)."""
        enc, _ = self.prepare(view)
        cut = f", CUT to {MAX_STATE}" if enc["state_truncated"] else ""
        text = self.tok.decode(enc["ids"])
        for token in ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"):   # state, question, decide: own lines
            text = text.replace(token, f"\n{token}\n")
        text = text.replace("<|box_start|>", "\n<|box_start|>")                 # one option per line
        header = f"{len(enc['ids'])} tokens; state {enc['state_tokens']} of at most {MAX_STATE}{cut}"
        return header, text.strip("\n").replace("\n\n", "\n")


class KevPointerModel(SurveyModel):
    kind = "kev"

    def __init__(self, net, tok, meta, mode):
        super().__init__(KevEncoder(tok, net.option_isolation))
        self.net, self.tok, self.meta, self.mode = net, tok, meta, mode

    @classmethod
    def build(cls, base, device, lora_rank=None, full_finetune=False, weights_dtype="fp32"):
        """The Kev checkpoint `base` ready to train (module docstring: default, lora_rank or full_finetune). weights_dtype:
        the frozen backbone's dtype (bf16 only with an adapter); recorded in head.pt, so kev.checkpoint loads it the same way."""
        checkpoint = Checkpoint(base)
        meta = dataclasses.replace(checkpoint.meta, weights_dtype=weights_dtype)
        if lora_rank:
            meta = dataclasses.replace(meta, lora=lora_rank)   # the rank our checkpoints record
        if full_finetune:   # what Kev's loader reads as a full-weight checkpoint: the whole fp32 backbone, no adapter
            meta = dataclasses.replace(meta, lora=0, weights="full", weights_dtype="fp32")
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        net = DecisionModel(meta.base, tok, device, lora=meta.lora or None, revision=meta.base_revision,
                            head_dim=meta.head_dim, option_isolation=meta.option_isolation,
                            special_embeddings=meta.special_embeddings,
                            lora_targets=meta.extra.get("args", {}).get("lora_targets", "all"),
                            dtype=torch.bfloat16 if weights_dtype == "bf16" else torch.float32)
        net.lm.config.use_cache = False                  # no generation: the cache only costs memory
        # a full-weight checkpoint (one of our full_finetune runs): continue from its backbone, not the plain base
        continued = full_finetune and checkpoint.meta.weights == "full"
        if (lora_rank or full_finetune) and not continued:
            net.head.load_state_dict(meta.head)          # released head only; the backbone starts as the plain base
        else:
            checkpoint.warm_start(net, meta)             # loads adapter or backbone + head; refuses a mismatched architecture
        mode = (f"lora_rank {lora_rank}" if lora_rank else f"full_finetune, continued from {base}" if continued
                else "full_finetune" if full_finetune else "released adapter")
        return cls(net, tok, meta, mode)

    @classmethod
    def load(cls, run, device):
        """A trained or released Kev checkpoint for evaluation, raw distribution (temperature 1, no calibration)."""
        tok, net = load_checkpoint(run, device, LoadOptions(temperature=1.0))
        return cls(net, tok, Checkpoint(run).meta, "loaded")

    def forward(self, batch):
        return self.net.forward_batch(batch)

    def decoder(self):
        return self.net.lm

    def enable_checkpointing(self):
        # the same setting as Kev's own trainer (kev/kev/train.py --checkpointing)
        self.net.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def save(self, folder, extra):
        """Kev's checkpoint format (adapter files, or with full_finetune the whole backbone's config.json +
        model.safetensors; + head.pt + tokenizer). Temperature 1.0: Kev's server divides scores by it, so it serves
        the raw distribution."""
        folder.mkdir(parents=True, exist_ok=True)
        self.net.lm.save_pretrained(folder)
        head = {k: v.detach().cpu() for k, v in self.net.head.state_dict().items()}
        write_meta(folder, dataclasses.replace(self.meta, head=head, temperature=1.0, extra=extra))
        self.tok.save_pretrained(folder)

    def describe(self):
        return {"kind": self.kind, "base": self.meta.base, "mode": self.mode,
                "trainable_params": sum(p.numel() for p in self.trainable_parameters())}

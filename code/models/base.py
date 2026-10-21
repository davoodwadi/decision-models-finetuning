"""The model layer's interface (PLAN-model-layer.md): what the trainer (train.py) may do with any model.

The boundary with the data layer is the view: what issp_augment.view returns (training) or canonical (calibration, test),
{"state": persona text, "questions": {v: {type, instructions, criteria, target, src, ...}}}. A model turns a view into
its own inputs and the targets in its own option order (`prepare`), and a batch of inputs into K logits per question
(`forward`), K = the question's options; so the loss, train.with_probs and every metric are the same for every model.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kev"))
from kev.api import SystemOneRequest, to_record          # noqa: E402  view -> options as text + their keys


def options_and_targets(view):
    """(record, questions) of a view: record = {"state", "questions": [{"instr", "options", "label"}]}, every question's
    options as text in the order shown (Kev's rendering, kev.api.to_record, used by every model so all read the same
    option text); questions = per question {v, src, type, target}, target = the observed shares in that order."""
    request = SystemOneRequest.model_validate({"state": view["state"], "questions": {
        v: {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
        for v, q in view["questions"].items()}})
    record, option_keys = to_record(request)
    questions = []
    for (v, q), meta in zip(view["questions"].items(), option_keys):
        target = torch.tensor([q["target"][key] for key in meta["keys"]])
        questions.append({"v": v, "src": q.get("src"), "type": q["type"],
                          "target": target / target.sum()})   # shares were rounded: renormalise
    return record, questions


class SurveyModel(torch.nn.Module):
    """A model the trainer can train and evaluate. `encoder` is the tokenizer-only part (prompt format, targets,
    preview), so a model's input can be shown without loading weights (show_train_samples.py)."""

    kind = None   # "kev" | "llm_letters": written to run_info.json

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def prepare(self, view):
        """(inputs, questions): this model's inputs for one record, and per question {v, src, type, target} with the
        target in the order of the logits forward() returns."""
        return self.encoder.prepare(view)

    def forward(self, batch):
        """Per record of `batch` (a list of prepare()'s inputs), per question: K logits, one per option."""
        raise NotImplementedError

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def decoder(self):
        """The backbone's decoder: token embeddings, the stack of layers (`layers`), the final norm."""
        raise NotImplementedError

    def train_last_layers(self, n):
        """Freeze the token embeddings and every decoder layer but the last n; the last n, the final norm and the rest
        of the model (Kev's pointer head) stay trainable. Qwen3.5 ties its output head to the token embeddings, so the
        letter model's output head is frozen with them."""
        decoder = self.decoder()
        if not 0 <= n <= len(decoder.layers):
            raise ValueError(f"--train_last_layers {n}: the backbone has {len(decoder.layers)} layers")
        decoder.embed_tokens.requires_grad_(False)
        for layer in decoder.layers[:len(decoder.layers) - n]:
            layer.requires_grad_(False)

    def parameter_groups(self, lr, decay):
        """The trainable weights as optimizer groups, the learning rate divided by `decay` once per layer going down
        the backbone: the last decoder layer, the final norm and the rest of the model (Kev's pointer head) get lr,
        the layer before lr / decay, the one before that lr / decay^2, ..., the token embeddings one step below the
        first layer. Frozen weights and empty groups are left out; the first group has the full lr."""
        if decay == 1:   # one group: also for a LoRA adapter, whose backbone has no plain layer stack
            return [{"params": self.trainable_parameters(), "lr": lr}]
        decoder = self.decoder()
        layers = list(decoder.layers)
        depth = {id(p): len(layers) - 1 - i for i, layer in enumerate(layers) for p in layer.parameters()}
        depth.update({id(p): len(layers) for p in decoder.embed_tokens.parameters()})
        groups = {}
        for p in self.trainable_parameters():
            groups.setdefault(depth.get(id(p), 0), []).append(p)
        return [{"params": groups[d], "lr": lr / decay ** d} for d in sorted(groups)]

    def enable_checkpointing(self):
        """Gradient checkpointing: recompute activations in the backward pass instead of storing them."""
        raise NotImplementedError

    def save(self, folder, extra):
        """Write a checkpoint that models.load(folder) reads back; `extra` (run, step, ...) is stored with it."""
        raise NotImplementedError

    def describe(self):
        """What this model is, for run_info.json."""
        raise NotImplementedError

    def diagnostics(self):
        """Measures gathered since the last call (and reset), for the training log; {} if the model has none."""
        return {}

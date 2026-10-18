"""An LLM through its own next-token head, behind the model interface (models/base.py; PLAN-llm-letters.md).

Per question of a record (a record with 7 questions gives 7 prompts), the model's chat template is applied to:
  system     the prompt file (e.g. prompts/letters-v1.txt: reply with the letter of one option only)
  user       the persona text, the question's instructions, and its options in the order shown, labelled A., B., ...
  assistant  opened, thinking off (enable_thinking=False where the template has it): the next token is the answer.
One forward pass, no generation: the backbone's hidden state at each prompt's last token goes through the vocabulary
head (at that position only: every position would be batch x length x vocabulary logits), and of its logits only the
K option letters' are kept. A softmax over those K is the model's answer distribution (no temperature): restricting
to the valid letters and renormalising. Every weight is trained. In evaluation the prompts go through in passes of at
most EVAL_PASS_TOKENS (a SubPOP batch of 256 records is ~1,700 prompts); in training the trainer's --micro_batch sets
the pass size.

Checkpoints: save_pretrained of the model + tokenizer + letters.json (base, system prompt, letter token ids).
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import SurveyModel, options_and_targets

DEFAULT_BASE = "Qwen/Qwen3.5-0.8B"    # post-trained chat version: follows the system prompt
DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "letters-v1.txt"
META_FILE = "letters.json"            # marks a letter-model checkpoint (models.load)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EVAL_PASS_TOKENS = 16384              # padded tokens per evaluation pass (Kev's ROW_PASS_TOKENS, kev/kev/model.py)


class LetterEncoder:
    """The prompt format and the letter tokens, tokenizer only."""

    def __init__(self, tok, system_prompt):
        self.tok, self.system_prompt = tok, system_prompt
        self.letter_ids = self.find_letter_ids()

    @classmethod
    def for_base(cls, base, prompt_file):
        """The encoder of a Hugging Face model and a prompt file, without loading weights."""
        return cls(AutoTokenizer.from_pretrained(base), Path(prompt_file).read_text(encoding="utf-8").strip())

    @classmethod
    def for_checkpoint(cls, folder):
        """The encoder of a checkpoint LetterModel saved (its tokenizer and system prompt), without loading weights."""
        meta = json.loads((Path(folder) / META_FILE).read_text(encoding="utf-8"))
        return cls(AutoTokenizer.from_pretrained(folder), meta["system_prompt"])

    def prompt(self, state, instructions, options):
        """The exact text the model reads for one question, up to where it answers."""
        if len(options) > len(LETTERS):
            raise ValueError(f"{len(options)} options: more than the {len(LETTERS)} letters")
        user = f"{state}\n\n{instructions}\n" + "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options))
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def find_letter_ids(self):
        """The token id of each letter as it comes right after the opened assistant turn (Qwen: "A" and " A" are
        different tokens). Refuses a tokenizer that does not give every letter as one token there."""
        prefix = self.prompt("x", "x", ["x"])
        base = self.tok(prefix, add_special_tokens=False).input_ids
        ids = []
        for letter in LETTERS:
            full = self.tok(prefix + letter, add_special_tokens=False).input_ids
            if full[:len(base)] != base or len(full) != len(base) + 1:
                raise ValueError(f"letter {letter!r} is not one token after the prompt: {full[len(base) - 1:]}")
            ids.append(full[-1])
        return ids

    def prepare(self, view):
        """(rows, questions): one row per question, {ids: prompt token ids, letters: its K letters' token ids}."""
        record, questions = options_and_targets(view)
        rows = [{"ids": self.tok(self.prompt(record["state"], q["instr"], q["options"]), add_special_tokens=False).input_ids,
                 "letters": self.letter_ids[:len(q["options"])]} for q in record["questions"]]
        return rows, questions

    def preview(self, view):
        """(header, text): the prompt of each question as the model reads it."""
        record, _ = options_and_targets(view)
        prompts = [self.prompt(record["state"], q["instr"], q["options"]) for q in record["questions"]]
        lengths = [len(self.tok(p, add_special_tokens=False).input_ids) for p in prompts]
        return f"{' + '.join(map(str, lengths))} tokens; the answer is the next token", "\n\n".join(prompts)


class LetterModel(SurveyModel):
    kind = "llm_letters"

    def __init__(self, lm, tok, system_prompt, base, device):
        super().__init__(LetterEncoder(tok, system_prompt))
        self.lm, self.tok, self.base, self.device = lm, tok, base, device
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.letter_mass = []   # per question: the probability on its valid letters before renormalising

    @staticmethod
    def load_lm(name, device):
        # fp32 weights (the trainer's autocast computes in bf16); SDPA on CUDA, eager elsewhere, as Kev does
        attn = "sdpa" if str(device).startswith("cuda") else "eager"
        lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation=attn).to(device)
        lm.config.use_cache = False                      # no generation: the cache only costs memory
        return lm

    @classmethod
    def build(cls, base, prompt_file, device):
        """A Hugging Face causal LM ready to train, every weight trainable, with the system prompt in `prompt_file`."""
        system_prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
        return cls(cls.load_lm(base, device), AutoTokenizer.from_pretrained(base), system_prompt, base, device)

    @classmethod
    def load(cls, folder, device):
        """A checkpoint this class saved."""
        meta = json.loads((Path(folder) / META_FILE).read_text(encoding="utf-8"))
        return cls(cls.load_lm(folder, device), AutoTokenizer.from_pretrained(folder), meta["system_prompt"],
                   meta["base"], device)

    def forward(self, batch):
        rows = [row for record_rows in batch for row in record_rows]
        logits = []
        for part in self.passes(rows):
            logits += self.letter_logits(part)
        out, start = [], 0
        for record_rows in batch:                       # back to per record, per question
            out.append(logits[start:start + len(record_rows)])
            start += len(record_rows)
        return out

    def passes(self, rows):
        """Training: all rows in one pass (the trainer's --micro_batch sets its size). Evaluation: consecutive groups
        of rows whose padded size (rows x longest) stays within EVAL_PASS_TOKENS."""
        if self.training:
            return [rows]
        groups, group, longest = [], [], 0
        for row in rows:
            if group and (len(group) + 1) * max(longest, len(row["ids"])) > EVAL_PASS_TOKENS:
                groups.append(group)
                group, longest = [], 0
            group.append(row)
            longest = max(longest, len(row["ids"]))
        return groups + [group]

    def letter_logits(self, rows):
        """One pass: per row, the K letters' logits at the prompt's last token."""
        length = max(len(row["ids"]) for row in rows)
        # right-padded: pads come after every real token, so causal attention and recurrent (DeltaNet) layers never see them
        ids = torch.full((len(rows), length), self.pad_id, device=self.device)
        mask = torch.zeros((len(rows), length), dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            ids[i, :len(row["ids"])] = torch.tensor(row["ids"], device=self.device)
            mask[i, :len(row["ids"])] = 1
        hidden = self.lm.get_decoder()(input_ids=ids, attention_mask=mask).last_hidden_state
        last = hidden[torch.arange(len(rows), device=self.device), mask.sum(1) - 1]     # each prompt's last token
        vocab = self.lm.get_output_embeddings()(last).float()                         # [questions, vocabulary]
        logits = [vocab[i, row["letters"]] for i, row in enumerate(rows)]
        with torch.no_grad():   # probability on the valid letters = exp(logsumexp(letters) - logsumexp(vocabulary))
            whole = torch.logsumexp(vocab, dim=-1)
            self.letter_mass += [(torch.logsumexp(z, dim=-1) - whole[i]).exp().item() for i, z in enumerate(logits)]
        return logits

    def decoder(self):
        return self.lm.get_decoder()

    def enable_checkpointing(self):
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def save(self, folder, extra):
        folder.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(folder)
        self.tok.save_pretrained(folder)
        meta = {"kind": self.kind, "base": self.base, "system_prompt": self.encoder.system_prompt,
                "letter_ids": self.encoder.letter_ids, "extra": extra}
        (folder / META_FILE).write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    def describe(self):
        return {"kind": self.kind, "base": self.base, "mode": "full_finetune", "system_prompt": self.encoder.system_prompt,
                "letter_ids": self.encoder.letter_ids,
                "trainable_params": sum(p.numel() for p in self.trainable_parameters())}

    def diagnostics(self):
        """letter_mass: the average probability the model puts on the valid letters of a question before
        renormalising, over the questions since the last call: near 1 = it follows the prompt's answer format."""
        mass, self.letter_mass = self.letter_mass, []
        return {"letter_mass": sum(mass) / len(mass)} if mass else {}

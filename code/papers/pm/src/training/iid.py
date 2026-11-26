"""Ungrouped i.i.d. trainer (`shared_prefix=False`)."""

import random

from data.records import make_view, record_chars, sample_persona_state
from training.base import BaseTrainer, MicroBatch, StepBatch, validate_positive_int


def _iid_row_chars(rec: dict) -> int:
    persona_chars = sum(len(str(k)) + len(str(v)) + 3 for k, v in rec["personas"].items())
    return 16 + persona_chars + record_chars(rec)


class IIDTrainer(BaseTrainer):
    """Flat i.i.d. trainer drawing `questions_per_step` random `(persona, question)` records per step."""

    trainer_name = "iid"

    def __init__(
        self,
        *,
        questions_per_step: int = 128,
        micro_batch_size: int = 32,
        max_chars_per_microbatch: int | None = None,
        sort_within_step: bool = True,
        **kwargs,
    ):
        self.questions_per_step = validate_positive_int(questions_per_step, "questions_per_step")
        self.micro_batch_size = validate_positive_int(micro_batch_size, "micro_batch_size")
        self.max_chars_per_microbatch = (
            validate_positive_int(max_chars_per_microbatch, "max_chars_per_microbatch")
            if max_chars_per_microbatch is not None
            else None
        )
        self.sort_within_step = bool(sort_within_step)
        super().__init__(**kwargs)

    def build_epoch_macro_batches(self, epoch_idx: int, rng: random.Random) -> list[list[dict]]:
        recs = list(self.train_records)
        if len(recs) > 1:
            rng.shuffle(recs)
        batches: list[list[dict]] = []
        for start in range(0, len(recs), self.questions_per_step):
            chunk = recs[start : start + self.questions_per_step]
            if self.sort_within_step and len(chunk) > 1:
                chunk.sort(key=_iid_row_chars)
            batches.append(chunk)
        return batches

    def _split_microbatch(self, records: list[dict]) -> list[list[dict]]:
        if self.max_chars_per_microbatch is None or len(records) <= 1:
            return [records]
        subs: list[list[dict]] = []
        cur: list[dict] = []
        cur_max = 0
        for rec in records:
            r_chars = _iid_row_chars(rec)
            nxt_max = max(cur_max, r_chars)
            if cur and (len(cur) + 1) * nxt_max > self.max_chars_per_microbatch:
                subs.append(cur)
                cur = []
                cur_max = 0
                nxt_max = r_chars
            cur.append(rec)
            cur_max = nxt_max
        if cur:
            subs.append(cur)
        return subs

    def materialize_step(self, macro_batch: list[dict], rng: random.Random) -> StepBatch:
        micro_batches: list[MicroBatch] = []
        for start in range(0, len(macro_batch), self.micro_batch_size):
            slice_recs = macro_batch[start : start + self.micro_batch_size]
            for sub in self._split_microbatch(slice_recs):
                views = [make_view(r, prompt_map=self.prompt_map, rng=rng) for r in sub]
                micro_batches.append(MicroBatch(views=views, shared_prefix=False))
        return StepBatch(micro_batches=micro_batches, total_questions=len(macro_batch))

    def _probe_param_name(self) -> str:
        return "micro_batch_size"

    def _max_probe_size(self) -> int:
        return min(self.questions_per_step, len(self.train_records))

    def _prepare_probe_context(self) -> dict:
        tok = self.model.tok
        state_toks: dict[tuple[tuple[str, str], ...], int] = {}
        q_toks: dict[str, int] = {}

        for rec in self.train_records:
            p_key = tuple(sorted(rec["personas"].items()))
            if p_key not in state_toks:
                state = sample_persona_state(rec["personas"], prompt_map=self.prompt_map)
                state_toks[p_key] = len(tok.encode(state, add_special_tokens=False))
            qid = rec["qid"]
            if qid not in q_toks:
                enc, _ = self.model.prepare(make_view(rec, prompt_map=self.prompt_map))
                q_toks[qid] = len(enc["ids"]) - enc["state_tokens"]

        worst_rec = max(
            self.train_records,
            key=lambda r: state_toks[tuple(sorted(r["personas"].items()))] + q_toks[r["qid"]],
        )
        worst_view = make_view(worst_rec, prompt_map=self.prompt_map, rng=random.Random(self.seed))
        enc, _ = self.model.prepare(worst_view)
        seq_tokens = len(enc["ids"])
        return {
            "worst_view": worst_view,
            "seq_tokens": seq_tokens,
            "summary": f"worst_seq={seq_tokens} tok (qid={worst_rec['qid']})",
        }

    def _build_probe_micro_batch(self, candidate_size: int, ctx: dict) -> MicroBatch:
        return MicroBatch(views=[ctx["worst_view"]] * candidate_size, shared_prefix=False)

    def _set_probed_micro_batch_size(self, size: int) -> None:
        self.micro_batch_size = size

    def describe(self) -> dict:
        return {
            **super().describe(),
            "questions_per_step": self.questions_per_step,
            "micro_batch_size": self.micro_batch_size,
            "max_chars_per_microbatch": self.max_chars_per_microbatch,
            "sort_within_step": self.sort_within_step,
        }

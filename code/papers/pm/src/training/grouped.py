"""Persona-grouped shared-prefix trainer with decoupled VRAM micro-chunking."""

from collections import defaultdict
import math
import random

from data.records import make_state_chunk_view, make_view, record_chars, sample_persona_state
from training.base import BaseTrainer, MicroBatch, StepBatch, validate_positive_int


def group_persona_logical(
    records: list[dict],
    *,
    questions_per_group: int,
    rng: random.Random | None = None,
) -> list[list[dict]]:
    """Partition `records` by exact full persona into balanced logical groups of `<= questions_per_group`."""
    validate_positive_int(questions_per_group, "questions_per_group")
    if not records:
        raise ValueError("records must not be empty")

    by_state: dict[tuple[tuple[str, str], ...], list[dict]] = defaultdict(list)
    for rec in records:
        personas = rec["personas"]
        if not personas:
            raise ValueError(f"Expected non-empty personas dict for qid={rec['qid']!r}")
        by_state[tuple(sorted(personas.items()))].append(rec)

    groups: list[list[dict]] = []
    for state_recs in by_state.values():
        items = list(state_recs)
        if rng is not None and len(items) > 1:
            rng.shuffle(items)
        n_groups = math.ceil(len(items) / questions_per_group)
        base, rem = divmod(len(items), n_groups)
        offset = 0
        for g_idx in range(n_groups):
            size = base + (1 if g_idx < rem else 0)
            groups.append(items[offset : offset + size])
            offset += size

    if rng is not None and len(groups) > 1:
        rng.shuffle(groups)
    return groups


def split_group_into_microchunks(
    logical_group: list[dict],
    *,
    micro_max_questions: int,
    micro_max_chars: int,
) -> list[list[dict]]:
    """Deterministically pack `logical_group` into length-sorted shared-prefix VRAM micro-chunks."""
    if not logical_group:
        raise ValueError("logical_group must not be empty")
    validate_positive_int(micro_max_questions, "micro_max_questions")
    validate_positive_int(micro_max_chars, "micro_max_chars")

    first_personas = logical_group[0]["personas"]
    sized = []
    for idx, rec in enumerate(logical_group):
        if rec["personas"] != first_personas:
            raise ValueError("All records in logical_group must share the exact same personas dict")
        sized.append((idx, record_chars(rec), rec))
    sized.sort(key=lambda item: (item[1], item[0]))

    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for _, q_chars, rec in sized:
        if cur and (len(cur) >= micro_max_questions or (len(cur) + 1) * q_chars > micro_max_chars):
            chunks.append(cur)
            cur = []
        cur.append(rec)
    if cur:
        chunks.append(cur)
    return chunks


class GroupedTrainer(BaseTrainer):
    """Shared-prefix persona-grouped trainer decoupling logical batching from VRAM micro-chunks."""

    trainer_name = "grouped"

    def __init__(
        self,
        *,
        questions_per_group: int = 64,
        personas_per_step: int = 12,
        micro_max_questions: int = 32,
        micro_max_chars: int = 20_000,
        micro_batch_size: int = 1,
        **kwargs,
    ):
        self.questions_per_group = validate_positive_int(questions_per_group, "questions_per_group")
        self.personas_per_step = validate_positive_int(personas_per_step, "personas_per_step")
        self.micro_max_questions = validate_positive_int(micro_max_questions, "micro_max_questions")
        self.micro_max_chars = validate_positive_int(micro_max_chars, "micro_max_chars")
        self.micro_batch_size = validate_positive_int(micro_batch_size, "micro_batch_size")
        super().__init__(**kwargs)

    def _epoch_groups(self, epoch_idx: int, rng: random.Random) -> list[list[dict]]:
        return group_persona_logical(
            self.train_records,
            questions_per_group=self.questions_per_group,
            rng=rng,
        )

    def build_epoch_macro_batches(self, epoch_idx: int, rng: random.Random) -> list[list[list[dict]]]:
        groups = self._epoch_groups(epoch_idx, rng)
        return [
            groups[start : start + self.personas_per_step]
            for start in range(0, len(groups), self.personas_per_step)
        ]

    def materialize_step(self, macro_batch: list[list[dict]], rng: random.Random) -> StepBatch:
        total_questions = sum(len(g) for g in macro_batch)
        chunk_views: list[tuple[int, int, dict]] = []
        for group in macro_batch:
            state = sample_persona_state(group[0]["personas"], prompt_map=self.prompt_map, rng=rng)
            chunks = split_group_into_microchunks(
                group,
                micro_max_questions=self.micro_max_questions,
                micro_max_chars=self.micro_max_chars,
            )
            for chunk in chunks:
                max_c = max(record_chars(r) for r in chunk)
                chunk_views.append((max_c, len(chunk_views), make_state_chunk_view(state, chunk)))

        if self.micro_batch_size > 1 and len(chunk_views) > 1:
            chunk_views.sort(key=lambda item: (item[0], item[1]))

        micro_batches = [
            MicroBatch(
                views=[item[2] for item in chunk_views[start : start + self.micro_batch_size]],
                shared_prefix=True,
            )
            for start in range(0, len(chunk_views), self.micro_batch_size)
        ]
        return StepBatch(micro_batches=micro_batches, total_questions=total_questions)

    def _probe_param_name(self) -> str:
        return "micro_max_questions"

    def _max_probe_size(self) -> int:
        unique_qids = len({rec["qid"] for rec in self.train_records})
        return min(self.questions_per_group, unique_qids)

    def _prepare_probe_context(self) -> dict:
        tok = self.model.tok
        state_toks: dict[tuple[tuple[str, str], ...], tuple[int, dict[str, str]]] = {}
        by_qid: dict[str, tuple[int, dict]] = {}

        for rec in self.train_records:
            p_key = tuple(sorted(rec["personas"].items()))
            if p_key not in state_toks:
                state = sample_persona_state(rec["personas"], prompt_map=self.prompt_map)
                state_toks[p_key] = (len(tok.encode(state, add_special_tokens=False)), rec["personas"])
            qid = rec["qid"]
            if qid not in by_qid:
                enc, _ = self.model.prepare(make_view(rec, prompt_map=self.prompt_map))
                by_qid[qid] = (len(enc["ids"]) - enc["state_tokens"], rec)

        _, longest_personas = max(state_toks.values(), key=lambda x: x[0])
        longest_state = sample_persona_state(
            longest_personas, prompt_map=self.prompt_map, rng=random.Random(self.seed)
        )
        sorted_q_entries = sorted(by_qid.values(), key=lambda x: x[0], reverse=True)
        sorted_qs = [{**rec, "personas": longest_personas} for _, rec in sorted_q_entries]

        sample_enc, _ = self.model.prepare(make_state_chunk_view(longest_state, sorted_qs[:1]))
        state_tokens = sample_enc["state_tokens"]
        max_branch_tokens = len(sample_enc["ids"]) - state_tokens
        return {
            "longest_state": longest_state,
            "sorted_qs": sorted_qs,
            "state_tokens": state_tokens,
            "max_branch_tokens": max_branch_tokens,
            "summary": f"largest_chunk: state={state_tokens} tok, max_branch={max_branch_tokens} tok",
        }

    def _build_probe_micro_batch(self, candidate_size: int, ctx: dict) -> MicroBatch:
        view = make_state_chunk_view(ctx["longest_state"], ctx["sorted_qs"][:candidate_size])
        return MicroBatch(views=[view] * self.micro_batch_size, shared_prefix=True)

    def _set_probed_micro_batch_size(self, size: int) -> None:
        self.micro_max_questions = size

    def describe(self) -> dict:
        return {
            **super().describe(),
            "questions_per_group": self.questions_per_group,
            "personas_per_step": self.personas_per_step,
            "micro_max_questions": self.micro_max_questions,
            "micro_max_chars": self.micro_max_chars,
            "micro_batch_size": self.micro_batch_size,
        }

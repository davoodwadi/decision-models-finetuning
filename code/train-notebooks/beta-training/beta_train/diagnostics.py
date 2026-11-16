"""Checks to run before training: two-pass gradient correctness, and the largest micro-chunk that fits in VRAM."""

from __future__ import annotations

import random

import pandas as pd
import torch

from training import free_gpu

from kev.model import rows_of

from beta_ols.unified_loss import build_sample_beta_operators, compute_unified_loss

from beta_train.batching import split_into_chunks
from beta_train.model_state import reset_trainable, set_checkpointing
from beta_train.step import predict_chunk


def gradient_check(
    model,
    dataset_items: list[dict],
    specs: dict,
    init_snapshot: dict,
    sample_size: int,
    micro_chunk: int,
    beta_lambda: float,
    seed: int,
    device: str,
) -> dict:
    # Does the two-pass micro-chunked gradient (method B, used in training) equal one backward through the whole
    # sample (method A)? Method A is run twice to measure the GPU noise floor. Uses checkpointing so A fits.
    reset_trainable(model, init_snapshot)
    set_checkpointing(model, True)
    params = list(model.trainable_parameters())

    sample = random.Random(seed).sample(dataset_items, sample_size)
    sample_ops = build_sample_beta_operators(sample, specs, device)
    if sample_ops["skipped_groups"]:
        raise ValueError(f"Check sample is rank-deficient for {sample_ops['skipped_groups']}; raise sample_size")
    chunks = split_into_chunks(sample, micro_chunk)

    def collect_weight_gradients() -> torch.Tensor:
        # Flatten every trainable gradient into one vector, then clear the gradients
        flat = torch.cat([p.grad.flatten().float() for p in params])
        for p in params:
            p.grad = None
        return flat

    def method_b() -> tuple[float, torch.Tensor, list[torch.Tensor]]:
        # Pass 1 without a graph (saving RNG states), loss, then pass 2 per chunk with its gradient slice
        saved_states = []
        logits_parts = []
        with torch.no_grad():
            for chunk in chunks:
                saved_states.append(torch.cuda.get_rng_state())
                logits_parts.append(predict_chunk(model, chunk))
        all_logits = torch.cat(logits_parts, dim=0)
        all_logits.requires_grad_(True)
        loss = compute_unified_loss(all_logits, sample_ops, beta_lambda)["total_loss"]
        loss.backward()
        options_per_chunk = [int(part.shape[0]) for part in logits_parts]
        grad_per_chunk = all_logits.grad.split(options_per_chunk, dim=0)
        for chunk, rng_state, chunk_grad in zip(chunks, saved_states, grad_per_chunk, strict=True):
            torch.cuda.set_rng_state(rng_state)
            predict_chunk(model, chunk).backward(chunk_grad)
        return float(loss.detach()), collect_weight_gradients(), saved_states

    def method_a(saved_states: list[torch.Tensor]) -> tuple[float, torch.Tensor]:
        # One graph over the whole sample, replaying method B's RNG states (same dropout masks)
        logits_parts = []
        for chunk, rng_state in zip(chunks, saved_states, strict=True):
            torch.cuda.set_rng_state(rng_state)
            logits_parts.append(predict_chunk(model, chunk))
        loss = compute_unified_loss(torch.cat(logits_parts, dim=0), sample_ops, beta_lambda)["total_loss"]
        loss.backward()
        return float(loss.detach()), collect_weight_gradients()

    try:
        loss_b, grad_b, saved_rng_states = method_b()
        loss_a1, grad_a1 = method_a(saved_rng_states)
        loss_a2, grad_a2 = method_a(saved_rng_states)
        return {
            "dataset": sample[0]["dataset"],
            "respondents": len(sample),
            "tasks": len(sample_ops["num_options"]),
            "micro_chunk": micro_chunk,
            "beta_lambda": beta_lambda,
            "loss_A": loss_a1,
            "loss_A_again": loss_a2,
            "loss_B": loss_b,
            "rel_diff_A_vs_A_again (noise floor)": float((grad_a1 - grad_a2).norm() / grad_a1.norm()),
            "rel_diff_A_vs_B (two-pass)": float((grad_a1 - grad_b).norm() / grad_a1.norm()),
            "cos_A_vs_B": float(torch.nn.functional.cosine_similarity(grad_a1, grad_b, dim=0)),
        }
    finally:
        reset_trainable(model, init_snapshot)


def respondent_shape(item: dict) -> dict:
    # What sets a respondent's memory in the shared-prefix forward: its state length and its question-branch lengths
    state_ids, _, rows = rows_of(item["enc"])
    row_lens = [len(row["ids"]) for row in rows]
    return {"state_len": len(state_ids), "num_rows": len(rows), "max_row_len": max(row_lens), "item": item}


def padded_tokens(shapes: list[dict]) -> int:
    # Tokens one chunk processes after padding: states are padded to the chunk's longest state, every question branch
    # to the chunk's longest branch (kev/kev/shared_prefix.py). Memory grows with this, not with unpadded tokens.
    max_state = max(shape["state_len"] for shape in shapes)
    max_row = max(shape["max_row_len"] for shape in shapes)
    total_rows = sum(shape["num_rows"] for shape in shapes)
    return len(shapes) * max_state + total_rows * max_row


# Sort keys for respondent shapes and candidate chunks
def shape_num_rows(shape: dict) -> int:
    return shape["num_rows"]


def shape_state_len(shape: dict) -> int:
    return shape["state_len"]


def shape_max_row_len(shape: dict) -> int:
    return shape["max_row_len"]


def shape_own_padding(shape: dict) -> int:
    return padded_tokens([shape])


def candidate_padded_tokens(candidate: dict) -> int:
    return candidate["padded_tokens"]


def worst_case_chunks(items: list[dict], chunk_size: int, num_candidates: int) -> list[dict]:
    # Candidate chunks of `chunk_size` respondents from ONE dataset (training chunks never mix datasets), built to
    # maximize padding: the respondent with the longest state, the one with the longest branch, and the respondents
    # with the most questions, plus simpler orderings. Returns the `num_candidates` with the most padded tokens.
    shapes_by_dataset: dict[str, list[dict]] = {}
    for item in items:
        if item["dataset"] not in shapes_by_dataset:
            shapes_by_dataset[item["dataset"]] = []
        shapes_by_dataset[item["dataset"]].append(respondent_shape(item))

    candidates = []
    for dataset_name, shapes in shapes_by_dataset.items():
        if len(shapes) < chunk_size:
            continue
        by_rows = sorted(shapes, key=shape_num_rows, reverse=True)
        by_state = sorted(shapes, key=shape_state_len, reverse=True)
        by_row_len = sorted(shapes, key=shape_max_row_len, reverse=True)
        by_own_padding = sorted(shapes, key=shape_own_padding, reverse=True)
        orderings = {
            "longest state + longest branch + most questions": [by_state[0], by_row_len[0], *by_rows],
            "longest state + most questions": [by_state[0], *by_rows],
            "longest branch + most questions": [by_row_len[0], *by_rows],
            "most questions": by_rows,
            "most padded tokens each": by_own_padding,
        }
        for rule, ordering in orderings.items():
            chosen = []
            seen_caseids = set()
            for shape in ordering:
                if shape["item"]["caseid"] not in seen_caseids:
                    chosen.append(shape)
                    seen_caseids.add(shape["item"]["caseid"])
                if len(chosen) == chunk_size:
                    break
            candidates.append(
                {
                    "dataset": dataset_name,
                    "rule": rule,
                    "padded_tokens": padded_tokens(chosen),
                    "items": [shape["item"] for shape in chosen],
                    "caseids": frozenset(seen_caseids),
                }
            )

    # Keep distinct chunks, most padded tokens first
    candidates.sort(key=candidate_padded_tokens, reverse=True)
    distinct = []
    seen_chunks = set()
    for candidate in candidates:
        if candidate["caseids"] not in seen_chunks:
            distinct.append(candidate)
            seen_chunks.add(candidate["caseids"])
    return distinct[:num_candidates]


def find_max_micro_chunk(
    model,
    items: list[dict],
    init_snapshot: dict,
    max_chunk: int,
    gradient_checkpointing: bool,
    candidates_per_size: int,
) -> tuple[pd.DataFrame, int]:
    # Largest micro-chunk for which one pass-2 forward + backward + AdamW step fits in VRAM for EVERY one of the
    # `candidates_per_size` worst-case chunks (most padded tokens, see `worst_case_chunks`). Doubles the size until a
    # candidate runs out of memory, then binary-searches. Weights are reset after every trial.
    set_checkpointing(model, gradient_checkpointing)
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    trial_rows = []

    def candidate_fits(chunk_size: int, candidate: dict) -> bool:
        reset_trainable(model, init_snapshot)
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=1e-8, fused=True)
        chunk_logits = None
        fits = True
        try:
            chunk_logits = predict_chunk(model, candidate["items"])
            chunk_logits.sum().backward()
            optimizer.step()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            fits = False
        finally:
            chunk_logits = None
            optimizer = None
            free_gpu()
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        reset_trainable(model, init_snapshot)
        trial_rows.append(
            {
                "chunk_size": chunk_size,
                "dataset": candidate["dataset"],
                "rule": candidate["rule"],
                "padded_tokens": candidate["padded_tokens"],
                "fits": fits,
                "peak_gib": round(peak_gib, 2) if fits else float("nan"),
                "peak_pct_of_vram": round(100.0 * peak_gib / total_vram_gib, 1) if fits else float("nan"),
            }
        )
        return fits

    def size_fits(chunk_size: int) -> bool:
        # A size fits only if every worst-case candidate fits; stop at the first that does not
        candidates = worst_case_chunks(items, chunk_size, candidates_per_size)
        if not candidates:
            raise ValueError(f"No dataset has {chunk_size} respondents to build a chunk from")
        for candidate in candidates:
            if not candidate_fits(chunk_size, candidate):
                return False
        return True

    # Phase 1: double the chunk size until it does not fit or reaches `max_chunk`
    largest_fit = 0
    smallest_fail = None
    chunk_size = 1
    while chunk_size <= max_chunk:
        if not size_fits(chunk_size):
            smallest_fail = chunk_size
            break
        largest_fit = chunk_size
        chunk_size *= 2

    # Phase 2: binary search between the largest fit and the smallest failure
    if smallest_fail is not None:
        low = largest_fit
        high = smallest_fail
        while high - low > 1:
            middle = (low + high) // 2
            if size_fits(middle):
                low = middle
            else:
                high = middle
        largest_fit = low

    if largest_fit < 1:
        raise RuntimeError("Even micro_chunk=1 ran out of VRAM; try gradient_checkpointing=True")
    return pd.DataFrame(trial_rows).sort_values(["chunk_size", "padded_tokens"], ascending=[True, False]).reset_index(drop=True), largest_fit

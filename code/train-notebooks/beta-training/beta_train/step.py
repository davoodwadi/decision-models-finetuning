"""Forward pass and the exact two-pass micro-chunked gradient for one sample.

Why two passes: the beta loss is a function of the WHOLE sample's predictions (one regression per sample), so it
cannot be split into independent per-chunk losses. Instead:
  pass 1  forward every micro-chunk without a graph, saving each chunk's CUDA RNG state (dropout masks)
  loss    compute the full-sample loss on the detached logits; backward gives d loss / d logit for every option
  pass 2  re-run each chunk with its saved RNG state (same dropout), and backpropagate its slice of d loss / d logit
By the chain rule the accumulated weight gradients equal one backward through the whole sample, while VRAM only
ever holds one chunk's graph.
"""

from __future__ import annotations

import time

import torch
from tqdm.auto import tqdm

from beta_ols.unified_loss import compute_unified_loss

from beta_train.batching import split_into_chunks


def predict_chunk(model, chunk_items: list[dict]) -> torch.Tensor:
    # Shared-prefix forward for one micro-chunk -> flat float64 logits: every option of every task, in item order
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits_per_respondent = model.forward([item["enc"] for item in chunk_items], shared_prefix=True)
    flat_parts = []
    for item, respondent_logits in zip(chunk_items, logits_per_respondent, strict=True):
        option_counts = [int(question_logits.shape[0]) for question_logits in respondent_logits]
        if option_counts != item["num_options"]:
            raise ValueError(f"{item['caseid']}: logits option counts {option_counts} != expected {item['num_options']}")
        for question_logits in respondent_logits:
            flat_parts.append(question_logits.to(dtype=torch.float64))
    return torch.cat(flat_parts, dim=0)


def two_pass_gradients(
    model,
    sample_items: list[dict],
    sample_ops: dict,
    beta_lambda: float,
    micro_chunk: int,
    show_chunk_progress: bool,
) -> tuple[dict, dict]:
    # Fill every trainable parameter's .grad with the exact full-sample gradient. Returns (loss_out, timings_ms).
    # The caller zeroes gradients before and steps the optimizer after.
    # show_chunk_progress: a bar over the 2 x chunks passes of this step (cleared when the step ends). It waits for
    # the GPU after every chunk so it shows finished work, not queued work.
    chunks = split_into_chunks(sample_items, micro_chunk)
    chunk_bar = tqdm(total=2 * len(chunks), desc="pass 1 (forward)", leave=False, disable=not show_chunk_progress)
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # 1. Pass 1: forward every chunk without a graph; save the RNG state right before each chunk
    chunk_rng_states = []
    chunk_logits_list = []
    with torch.no_grad():
        for chunk in chunks:
            chunk_rng_states.append(torch.cuda.get_rng_state())
            chunk_logits_list.append(predict_chunk(model, chunk))
            if show_chunk_progress:
                torch.cuda.synchronize()
                chunk_bar.update(1)
    flat_logits = torch.cat(chunk_logits_list, dim=0)
    torch.cuda.synchronize()
    t_after_forward = time.perf_counter()

    # 2. Full-sample loss on the detached logits; backward fills flat_logits.grad
    flat_logits.requires_grad_(True)
    loss_out = compute_unified_loss(flat_logits, sample_ops, beta_lambda)
    loss_out["total_loss"].backward()
    loss_out["total_loss"] = float(loss_out["total_loss"].detach())
    options_per_chunk = [int(chunk_logits.shape[0]) for chunk_logits in chunk_logits_list]
    grad_per_chunk = flat_logits.grad.split(options_per_chunk, dim=0)
    torch.cuda.synchronize()
    t_after_loss = time.perf_counter()

    # 3. Pass 2: replay each chunk with its saved RNG state and push its gradient slice into the weights
    chunk_bar.set_description("pass 2 (forward + backward)")
    for chunk, rng_state, chunk_grad in zip(chunks, chunk_rng_states, grad_per_chunk, strict=True):
        torch.cuda.set_rng_state(rng_state)
        chunk_logits = predict_chunk(model, chunk)
        chunk_logits.backward(chunk_grad)
        del chunk_logits
        if show_chunk_progress:
            torch.cuda.synchronize()
            chunk_bar.update(1)
    torch.cuda.synchronize()
    t_after_backward = time.perf_counter()
    chunk_bar.close()

    timings_ms = {
        "fwd_ms": (t_after_forward - t_start) * 1000.0,
        "loss_ms": (t_after_loss - t_after_forward) * 1000.0,
        "bwd_ms": (t_after_backward - t_after_loss) * 1000.0,
    }
    return loss_out, timings_ms

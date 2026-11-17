"""Forward pass and the exact two-pass micro-chunked gradient for one sample.

Why two passes: the beta loss is a function of the WHOLE sample's predictions (one regression per sample), so it
cannot be split into independent per-chunk losses. Instead:
  pass 1  forward every micro-chunk without a graph, saving each chunk's CUDA RNG state (dropout masks)
  loss    compute the full-sample loss on the detached logits; backward gives d loss / d logit for every option
  pass 2  re-run each chunk with its saved RNG state (same dropout), and backpropagate its slice of d loss / d logit
By the chain rule the accumulated weight gradients equal one backward through the whole sample, while VRAM only
ever holds one chunk's graph.

Optional gradient split (`TrainConfig.gradient_split`, for debugging): the logit gradient is split exactly into its
KL (CE) part and its beta part, d loss / d logit = d KL / d logit + lambda * d beta / d logit, and pass 2 runs once per
part (same saved dropout masks), filling one weight-gradient buffer each. The step's gradient is then
grad CE + lambda * grad beta, the same gradient as without the split (up to floating-point rounding). Costs a second
pass 2 (forward + backward per chunk).
"""

from __future__ import annotations

import time

import torch

from beta_ols.unified_loss import compute_unified_loss

from beta_train.batching import split_into_chunks


def predict_chunk(model, chunk_items: list[dict]) -> torch.Tensor:
    # Shared-prefix forward for one micro-chunk -> flat float64 logits: every option of every task, in item order
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits_per_item = model.forward([item["enc"] for item in chunk_items], shared_prefix=True)
    flat_parts = []
    for item, item_logits in zip(chunk_items, logits_per_item, strict=True):
        option_counts = [int(question_logits.shape[0]) for question_logits in item_logits]
        if option_counts != item["num_options"]:
            raise ValueError(f"{item['caseid']}: logits option counts {option_counts} != expected {item['num_options']}")
        for question_logits in item_logits:
            flat_parts.append(question_logits.to(dtype=torch.float64))
    return torch.cat(flat_parts, dim=0)


def per_loss_logit_gradients(flat_logits: torch.Tensor, sample_ops: dict) -> tuple[torch.Tensor, torch.Tensor]:
    # d KL / d logit and d beta loss / d logit (beta loss unweighted, i.e. lambda = 1), on detached logits.
    # The loss is KL + lambda * beta, so the beta part is the lambda = 1 gradient minus the KL gradient.
    kl_logits = flat_logits.detach().requires_grad_(True)
    kl_total = compute_unified_loss(kl_logits, sample_ops, 0.0)["total_loss"]
    (kl_grad,) = torch.autograd.grad(kl_total, kl_logits)
    if len(sample_ops["operators"]) == 0:
        # Every regression is rank-deficient in this sample: no beta loss, no beta gradient
        return kl_grad, torch.zeros_like(kl_grad)
    unit_logits = flat_logits.detach().requires_grad_(True)
    unit_total = compute_unified_loss(unit_logits, sample_ops, 1.0)["total_loss"]
    (unit_grad,) = torch.autograd.grad(unit_total, unit_logits)
    return kl_grad, unit_grad - kl_grad


def backprop_chunks(model, chunks: list[list[dict]], chunk_rng_states: list, grad_per_chunk, chunk_bar) -> None:
    # Pass 2: replay each chunk with its saved RNG state (same dropout) and push its logit-gradient slice into .grad
    for chunk, rng_state, chunk_grad in zip(chunks, chunk_rng_states, grad_per_chunk, strict=True):
        torch.cuda.set_rng_state(rng_state)
        chunk_logits = predict_chunk(model, chunk)
        chunk_logits.backward(chunk_grad)
        del chunk_logits
        torch.cuda.synchronize()
        chunk_bar.update(1)


def take_gradients(params: list[torch.Tensor]) -> list[torch.Tensor]:
    # Each parameter's accumulated gradient (zeros if it got none), then clear .grad for the next pass
    gradients = []
    for param in params:
        if param.grad is None:
            gradients.append(torch.zeros_like(param))
        else:
            gradients.append(param.grad)
        param.grad = None
    return gradients


def gradient_split_stats(kl_grads: list[torch.Tensor], beta_grads: list[torch.Tensor], beta_lambda: float) -> dict:
    # Norms over all trainable parameters (float64 sums): CE, beta, lambda * beta, and their sum (the step's gradient
    # before clipping); cosine between CE and lambda * beta; each part's projection onto the total
    # (<part, total> / |total|; the two projections add up to |total|)
    kl_sq = torch.zeros((), dtype=torch.float64, device=kl_grads[0].device)
    beta_sq = torch.zeros_like(kl_sq)
    kl_dot_beta = torch.zeros_like(kl_sq)
    for kl_grad, beta_grad in zip(kl_grads, beta_grads, strict=True):
        kl64 = kl_grad.to(torch.float64)
        beta64 = beta_grad.to(torch.float64)
        kl_sq += (kl64 * kl64).sum()
        beta_sq += (beta64 * beta64).sum()
        kl_dot_beta += (kl64 * beta64).sum()
    kl_norm = float(kl_sq.sqrt())
    beta_norm = float(beta_sq.sqrt())
    lambda_beta_norm = beta_lambda * beta_norm
    lambda_kl_dot_beta = beta_lambda * float(kl_dot_beta)
    total_sq = kl_norm**2 + 2.0 * lambda_kl_dot_beta + lambda_beta_norm**2
    total_norm = max(total_sq, 0.0) ** 0.5
    if kl_norm > 0.0 and lambda_beta_norm > 0.0:
        cosine = lambda_kl_dot_beta / (kl_norm * lambda_beta_norm)
    else:
        cosine = float("nan")   # undefined when one part is zero (e.g. lambda = 0)
    if total_norm > 0.0:
        kl_projection = (kl_norm**2 + lambda_kl_dot_beta) / total_norm
        beta_projection = (lambda_kl_dot_beta + lambda_beta_norm**2) / total_norm
    else:
        kl_projection = float("nan")
        beta_projection = float("nan")
    return {
        "grad_ce_norm": kl_norm,
        "grad_beta_norm": beta_norm,
        "grad_lambda_beta_norm": lambda_beta_norm,
        "grad_total_norm": total_norm,
        "grad_cos_ce_lambda_beta": cosine,
        "grad_ce_projection": kl_projection,
        "grad_lambda_beta_projection": beta_projection,
    }


def two_pass_gradients(
    model,
    sample_items: list[dict],
    sample_ops: dict,
    beta_lambda: float,
    micro_chunk: int,
    chunk_bar,                                     # the run's one tqdm bar, reset here every step (no new output per step)
    gradient_split: bool,                          # True: also split the gradient into its CE and beta parts (2nd pass 2)
) -> tuple[dict, dict, dict | None]:
    # Fill every trainable parameter's .grad with the exact full-sample gradient. Returns (loss_out, timings_ms,
    # split_stats), split_stats = gradient_split_stats(...) when gradient_split, else None.
    # The caller zeroes gradients before and steps the optimizer after.
    # `chunk_bar` counts the 2 x chunks passes of this step. It is created once per run and reset here, because a new
    # notebook bar per step adds one cell output per step and the notebook hides outputs past its limit (VS Code: 500).
    # It waits for the GPU after every chunk so it shows finished work, not queued work.
    chunks = split_into_chunks(sample_items, micro_chunk)   # micro_chunk = items (data rows) per chunk
    if gradient_split:
        chunk_bar.reset(total=3 * len(chunks))
    else:
        chunk_bar.reset(total=2 * len(chunks))
    chunk_bar.set_description("pass 1 (forward)")
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # 1. Pass 1: forward every chunk without a graph; save the RNG state right before each chunk
    chunk_rng_states = []
    chunk_logits_list = []
    with torch.no_grad():
        for chunk in chunks:
            chunk_rng_states.append(torch.cuda.get_rng_state())
            chunk_logits_list.append(predict_chunk(model, chunk))
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
    split_stats = None
    if gradient_split:
        # Once per loss part, each into its own gradient buffer; then .grad = CE part + lambda * beta part
        kl_logit_grad, beta_logit_grad = per_loss_logit_gradients(flat_logits, sample_ops)
        combined_logit_grad = kl_logit_grad + beta_lambda * beta_logit_grad
        if not torch.allclose(combined_logit_grad, flat_logits.grad, rtol=1e-9, atol=1e-12):
            max_diff = float((combined_logit_grad - flat_logits.grad).abs().max())
            raise AssertionError(f"CE + lambda * beta logit gradient differs from the loss gradient (max {max_diff:.3e})")
        params = list(model.trainable_parameters())
        for param in params:
            if param.grad is not None:
                raise AssertionError("gradient_split expects cleared gradients (the caller zeroes them before the step)")
        chunk_bar.set_description("pass 2a (CE gradient)")
        backprop_chunks(model, chunks, chunk_rng_states, kl_logit_grad.split(options_per_chunk, dim=0), chunk_bar)
        kl_weight_grads = take_gradients(params)
        chunk_bar.set_description("pass 2b (beta gradient)")
        backprop_chunks(model, chunks, chunk_rng_states, beta_logit_grad.split(options_per_chunk, dim=0), chunk_bar)
        beta_weight_grads = take_gradients(params)
        for param, kl_weight_grad, beta_weight_grad in zip(params, kl_weight_grads, beta_weight_grads, strict=True):
            param.grad = kl_weight_grad + beta_lambda * beta_weight_grad
        split_stats = gradient_split_stats(kl_weight_grads, beta_weight_grads, beta_lambda)
        kl_weight_grads = None
        beta_weight_grads = None
    else:
        chunk_bar.set_description("pass 2 (forward + backward)")
        backprop_chunks(model, chunks, chunk_rng_states, grad_per_chunk, chunk_bar)
    torch.cuda.synchronize()
    t_after_backward = time.perf_counter()

    timings_ms = {
        "fwd_ms": (t_after_forward - t_start) * 1000.0,
        "loss_ms": (t_after_loss - t_after_forward) * 1000.0,
        "bwd_ms": (t_after_backward - t_after_loss) * 1000.0,
    }
    return loss_out, timings_ms, split_stats

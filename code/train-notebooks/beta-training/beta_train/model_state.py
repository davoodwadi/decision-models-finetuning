"""Trainable-weight snapshots and the gradient-checkpointing switch, so every run starts from the same weights."""

from __future__ import annotations

import torch

from training import free_gpu


def snapshot_trainable(model) -> dict[str, torch.Tensor]:
    # CPU copy of every trainable tensor (LoRA + pointer head)
    snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[name] = param.detach().cpu().clone()
    return snapshot


def reset_trainable(model, snapshot: dict[str, torch.Tensor]) -> None:
    # Copy the snapshot back into the model, clear gradients, return to train mode, and free GPU memory
    current_names = {name for name, param in model.named_parameters() if param.requires_grad}
    if current_names != set(snapshot):
        raise ValueError("Trainable parameter names do not match the snapshot")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))
                param.grad = None
    model.train()
    free_gpu()


def set_checkpointing(model, enabled: bool) -> None:
    # Turn gradient checkpointing of the backbone on or off
    if enabled:
        model.enable_checkpointing()
    else:
        model.net.lm.gradient_checkpointing_disable()

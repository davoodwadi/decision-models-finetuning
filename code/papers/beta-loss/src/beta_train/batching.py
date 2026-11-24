"""Samples and micro-chunks. One sample = items (data rows) from ONE dataset; an epoch covers every item once."""

from __future__ import annotations

import math
import random

from beta_train.config import TrainConfig


def split_into_chunks(items: list, chunk_size: int) -> list[list]:
    # Consecutive chunks of at most `chunk_size` items, in order
    chunks = []
    for start in range(0, len(items), chunk_size):
        chunks.append(items[start : start + chunk_size])
    return chunks


def steps_per_epoch(items_by_dataset: dict[str, list[dict]], config: TrainConfig) -> int:
    # Sum over training datasets of ceil(items / that dataset's sample size)
    total = 0
    for dataset_name in config.train_datasets:
        total += math.ceil(len(items_by_dataset[dataset_name]) / config.sample_size_by_dataset[dataset_name])
    return total


def build_epoch_samples(
    rng: random.Random, items_by_dataset: dict[str, list[dict]], config: TrainConfig
) -> list[tuple[str, list[dict]]]:
    # 1. Within each dataset: shuffle its items and cut them into samples of that dataset's sample size
    epoch_samples = []
    for dataset_name in config.train_datasets:
        dataset_items = list(items_by_dataset[dataset_name])
        rng.shuffle(dataset_items)
        for sample_items in split_into_chunks(dataset_items, config.sample_size_by_dataset[dataset_name]):
            epoch_samples.append((dataset_name, sample_items))
    # 2. Shuffle the order of all samples so datasets are interleaved across the epoch
    rng.shuffle(epoch_samples)
    return epoch_samples

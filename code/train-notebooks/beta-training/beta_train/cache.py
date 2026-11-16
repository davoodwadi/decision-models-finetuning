"""Disk cache for slow, deterministic results (tokenized items, untrained-model validation).

A cached result is stored under a key that lists everything it depends on: the input data's fingerprint, the source
files of the code that produced it, and any settings (tokenizer, model). Change any of them and the key changes, so
a stale result is never loaded; the old file simply stays unused. Delete the cache folder to clear it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from pathlib import Path

import pandas as pd


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    # Content hash of every cell (as text, so list-valued columns work), column names, and row order
    text_df = df.astype(str)
    row_hashes = pd.util.hash_pandas_object(text_df, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(df.columns)).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def source_fingerprint(*modules) -> str:
    # Hash of the source files of the given modules (any code change invalidates results they produced)
    digest = hashlib.sha256()
    for module in modules:
        source_path = Path(inspect.getsourcefile(module))
        digest.update(source_path.name.encode())
        digest.update(source_path.read_bytes())
    return digest.hexdigest()


def cache_file(cache_dir: Path, name: str, key: dict) -> Path:
    # <cache_dir>/<name>_<first 16 hex of sha256(key)>.pkl
    key_digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{name}_{key_digest}.pkl"


def load_or_compute(cache_dir: Path, name: str, key: dict, compute_fn):
    # Return (value, loaded_from_cache). On a miss, compute, then write the value together with its key.
    path = cache_file(cache_dir, name, key)
    if path.exists():
        stored = pickle.loads(path.read_bytes())
        if stored["key"] != key:
            raise ValueError(f"Cache file {path} holds a different key (hash collision?)")
        return stored["value"], True
    value = compute_fn()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_bytes(pickle.dumps({"key": key, "value": value}, protocol=pickle.HIGHEST_PROTOCOL))
    temporary_path.replace(path)   # atomic: an interrupted write never leaves a broken cache file
    return value, False

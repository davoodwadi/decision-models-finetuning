"""NCI HINTS 7 (2024 US) dataset loader and evaluation set builder."""

from pathlib import Path
import pandas as pd

from data.base import (
    DATA_DIR,
    DEFAULT_CACHE_DIR,
    add_one_hot,
    build_eval_set,
    filter_by_qids,
)

HINTS7_PARQUET_PATH = DATA_DIR / "HINTS" / "hints7_long.parquet"


def load_hints7_df(qids=None, parquet_path=HINTS7_PARQUET_PATH):
    """Load `hints7_long.parquet` (already standardized with `persona_spec.yaml` columns),
    attach `year`/`qid`/`one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"HINTS 7 parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {"CASEID", "country", "gender", "v", "option_codes", "option_texts", "answer_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"HINTS 7 parquet {path} is missing expected standardized columns: {sorted(missing)}"
        )
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    df["year"] = "hints7"
    df["qid"] = "hints7:" + df["v"]
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_hints7_eval(
    *,
    attr_sets,
    include_country,
    qids=None,
    min_n=None,
    grouped=True,
    name="hints7",
    valid=False,
    parquet_path=HINTS7_PARQUET_PATH,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build a HINTS 7 evaluation set for the given `qids`, `attr_sets`, and `include_country`."""
    df = load_hints7_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        grouped=grouped,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )

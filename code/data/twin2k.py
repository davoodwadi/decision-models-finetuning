"""Twin-2K-500 dataset loader and persona record builder."""

from pathlib import Path
import pandas as pd

from data.grummon import add_one_hot, group_records_by_state, make_long_view
from data.issp import (
    DEFAULT_CACHE_DIR,
    PAIR_PERSONA_ATTRS,
    SINGLE_PERSONA_ATTRS,
    build_persona_dataset,
)

DATA_DIR = Path(__file__).resolve().parent
TWIN2K_PARQUET_PATH = DATA_DIR / "Twin-2K-500" / "twin2k_US_long.parquet"


def load_twin2k_df(parquet_path=TWIN2K_PARQUET_PATH):
    """Load `twin2k_US_long.parquet` and attach `year`, `qid`, and `one_hot` columns."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Twin-2K-500 parquet not found: {path}")
    twin_df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        twin_df[col] = twin_df[col].map(list)
    twin_df["year"] = "twin2k"
    twin_df["qid"] = "twin2k:" + twin_df["v"]
    add_one_hot(twin_df)
    return twin_df


def load_twin2k_records(
    attr_sets=None,
    min_n=100,
    grouped=False,
    parquet_path=TWIN2K_PARQUET_PATH,
    desc="twin2k_records",
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build (or load from cache) Twin-2K-500 persona records, optionally grouped by shared state."""
    twin_df = load_twin2k_df(parquet_path=parquet_path)
    records = build_persona_dataset(
        twin_df,
        attr_sets=attr_sets,
        min_n=None if attr_sets is None else min_n,
        desc=desc,
        cache_dir=cache_dir,
    )
    if grouped:
        return group_records_by_state(records, view_fn=make_long_view, name="twin2k")
    return records

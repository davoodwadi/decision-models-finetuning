"""Build the ISSP 2019 training data (data/issp2019/issp2019_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py.
Every question was asked of everyone, so no block adds persona lines."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2019/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

if __name__ == "__main__":
    build_dataset(HERE, 2019, HERE / "raw" / "ZA7600_v3-0-0.dta")

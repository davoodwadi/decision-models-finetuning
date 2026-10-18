"""Build the ISSP 2017 training data (data/issp2017/issp2017_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2017/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# blocks asked of only part of the sample -> extra persona lines for that block's records
ASKED_OF = {"Q18": {"Goes out to eat or drink with friends or acquaintances": "Yes"}}   # filter: Q17 not "never"

if __name__ == "__main__":
    build_dataset(HERE, 2017, HERE / "raw" / "ZA6980_v2-0-0.dta", ASKED_OF)

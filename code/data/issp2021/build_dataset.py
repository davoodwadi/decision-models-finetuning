"""Build the ISSP 2021 training data (data/issp2021/issp2021_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2021/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# blocks asked of only part of the sample -> extra persona lines for that block's records
ASKED_OF = {q: {"Has access to the internet": "Yes"} for q in ["Q12", "Q13"]}   # filter: Q11 not "no access"

if __name__ == "__main__":
    build_dataset(HERE, 2021, HERE / "raw" / "ZA8000_v2-0-0.dta", ASKED_OF)

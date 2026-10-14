"""Build the ISSP 2022 training data (data/issp2022/issp2022_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2022/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# blocks asked of only part of the sample -> extra persona lines for that block's records
PARTNER = {"Lives with a spouse or partner": "Yes"}
ASKED_OF = {"Q9": {"Thinks there should be paid parental leave": "Yes"},            # filter: Q8 not "no paid leave"
            **{q: PARTNER for q in ["Q17", "Q18", "Q19"]},                         # filter: living with spouse/partner
            "O34": {"Has children": "Yes"}}                                         # filter: Q30 more than 0 children

if __name__ == "__main__":
    build_dataset(HERE, 2022, HERE / "raw" / "ZA10000_v2-0-0.dta", ASKED_OF)

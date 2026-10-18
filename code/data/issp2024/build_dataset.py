"""Build the ISSP 2024 training data (data/issp2024/issp2024_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2024/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# blocks asked of only part of the sample (filters from Q1 and Q8) -> extra persona lines for that block's records
USER = {"Internet use in the last 12 months": "Used the Internet"}
ASKED_OF = {**{q: {"Internet use in the last 12 months": "Did not use the Internet"} for q in ["Q2", "Q3", "Q4", "Q5"]},
            **{q: USER for q in ["Q6", "Q7", "Q8", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16"]},
            "Q9": {**USER, "Deliberately reduced time on the Internet": "Yes"}}

if __name__ == "__main__":
    build_dataset(HERE, 2024, HERE / "raw" / "ZA10020_v1-0-0.dta", ASKED_OF)

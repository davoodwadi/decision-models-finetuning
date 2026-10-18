"""Build the ISSP 2020 training data (data/issp2020/issp2020_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2020/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# Q2e "the [NATIONALITY] government": the Philippine national questionnaire's adjective, not the demonym "Filipino"
# that data/issp_countries.csv holds for "truly [NATIONALITY]" (ISSP 2023)
COUNTRY_FIELDS = {"PH": {"nationality": "Philippine"}}

if __name__ == "__main__":
    build_dataset(HERE, 2020, HERE / "raw" / "ZA7650_v2-0-0.dta", country_fields=COUNTRY_FIELDS)

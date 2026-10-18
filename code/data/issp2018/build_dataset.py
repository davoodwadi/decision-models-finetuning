"""Build the ISSP 2018 training data (data/issp2018/issp2018_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py.
Every question was asked of everyone, so no block adds persona lines."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2018/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# attribute values whose country note in the data is abbreviated, written out as the codebook's per-country list has them
ATTRIBUTE_NOTES = {("MARITAL", 4, "AT"): ("Divorced/ Separated",
                                         "ZA7570_cdb.pdf, MARITAL per-country list (label note 'AT: Div./Sep.')")}

if __name__ == "__main__":
    build_dataset(HERE, 2018, HERE / "raw" / "ZA7570_v2-1-0.dta", attribute_notes=ATTRIBUTE_NOTES)

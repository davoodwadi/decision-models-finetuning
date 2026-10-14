"""Build the ISSP 2016 training data (data/issp2016/issp2016_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py.
Every question was asked of everyone, so no block adds persona lines."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2016/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# attribute values whose country note in the data is abbreviated, written out as the codebook's per-country list has them
ATTRIBUTE_NOTES = {("ATTEND", 7, "DE"): ("Less frequently than several times a year",
                                        "ZA6900_cdb.pdf, ATTEND per-country list (label note 'DE:L.f.than sev. times a year')"),
                   ("ATTEND", 2, "GB"): ("Once a week or more",
                                        "ZA6900_cdb.pdf, ATTEND per-country list (label note 'GB:O.a w.or more')")}

if __name__ == "__main__":
    build_dataset(HERE, 2016, HERE / "raw" / "ZA6900_v2-0-0.dta", attribute_notes=ATTRIBUTE_NOTES)

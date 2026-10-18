"""Build the ISSP 2015 training data (data/issp2015/issp2015_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2015/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# blocks asked of only part of the sample -> extra persona lines for that block's records
WORKING = {"Currently works for pay": "Yes"}                                  # Q11-Q31: filter Q10 = yes
NOT_WORKING = {"Currently works for pay": "No"}                               # Q32-Q44: filter Q10 = no
HAD_JOB = {**NOT_WORKING, "Has had a paid job for one year or more": "Yes"}   # Q33-Q35: filter Q32 = yes
WANTS_JOB = {**NOT_WORKING, "Would like to have a paid job": "Yes"}          # Q37-Q42: filter Q36 = yes
ASKED_OF = {"Q6": {"Discriminated against with regard to work in the past five years": "Yes"},   # filter Q5 = yes
            **{f"Q{i}": WORKING for i in range(11, 31)},
            "Q31": {**WORKING, "Did other paid work besides the main job in the past 12 months": "Yes"},
            **{q: NOT_WORKING for q in ["Q32", "Q36", "Q43", "Q44"]},
            **{q: HAD_JOB for q in ["Q34", "Q35"]},
            **{f"Q{i}": WANTS_JOB for i in range(37, 43)},
            "O1": NOT_WORKING,                                                 # asked only of those not working now
            "O2": {"Worked for pay in the past five years": "Yes"}}            # workers, and non-workers with O1 = yes

# attribute values whose country note in the data is abbreviated, written out as the codebook's per-country list has them
ATTRIBUTE_NOTES = {("ATTEND", 7, country): ("Less than several times a year",
                                           "ZA6770_cdb.pdf, ATTEND per-country list (label note 'AT,DE:than sev times a year')")
                   for country in ("AT", "DE")}

if __name__ == "__main__":
    build_dataset(HERE, 2015, HERE / "raw" / "ZA6770_v2-1-0.dta", ASKED_OF, attribute_notes=ATTRIBUTE_NOTES)

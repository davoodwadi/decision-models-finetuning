"""Build the ISSP 2023 training data (data/issp2023/issp2023_respondents.parquet, _questions.json, _attributes.json). Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2023/
sys.path.insert(0, str(HERE.parent))
from issp_common import build_dataset  # noqa: E402

# the regional association each country's optional Q38-Q43 ask about (from the national questionnaires); the other
# countries did not ask these questions
EU, USMCA = "the European Union", "the United States-Mexico-Canada Agreement (USMCA)"
ASSOCIATION = {**{c: {"association": EU} for c in ["CH", "DE", "FI", "GR", "HR", "HU", "IT", "NL", "NO", "SE"]},
               "CA": {"association": USMCA}, "US": {"association": USMCA},
               "IN": {"association": "the South Asian Association for Regional Cooperation (SAARC)"},
               "PH": {"association": "the Association of Southeast Asian Nations (ASEAN)"}}

# Q39-Q43 were asked only of people who had heard or read something about the association (Q38 not "nothing at all")
ASKED_OF = {q: {"Has heard or read about {association}": "Yes"} for q in ["Q39", "Q40", "Q41", "Q42", "Q43"]}

if __name__ == "__main__":
    build_dataset(HERE, 2023, HERE / "raw" / "ZA10010_v1-0-0.dta", ASKED_OF, ASSOCIATION)

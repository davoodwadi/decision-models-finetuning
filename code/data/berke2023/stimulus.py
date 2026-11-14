"""Survey catalog and Audit UI indexers for Berke et al. (2023), drone delivery and customer privacy.

Data: `data/sample-preprocessed.csv` inside `drones-consumer-privacy-master.zip` (read in place; decisions in DECISIONS.md).

Question text sources, in this order:
1. `data/fields.csv` (Qualtrics export header row) for consumer and demographic questions.
2. `survey-questions/generate-questions.py` + `attributes.py` for the choice questions (exact generated HTML text).
3. `survey.pdf` for screens not in the data (privacy explanation, product introductions). The authors call it
   "an abbreviated set of the questions shown"; images are not included.
Answer labels: `data-analysis/codebook.py`, or `survey.pdf` where the codebook has none (noted per row).

Starting `include` / `kev_type` values are Claude's per-question evaluation (reason in each row's `review_note`);
the user verifies them in the audit app.
"""

from __future__ import annotations

from collections import OrderedDict
import io
import json
from pathlib import Path
import re
import zipfile

import pandas as pd

BERKE_DIR = Path(__file__).resolve().parent
RAW_ZIP = BERKE_DIR / "drones-consumer-privacy-master.zip"
ZIP_PREFIX = "drones-consumer-privacy-master/"
DATA_MEMBER = ZIP_PREFIX + "data/sample-preprocessed.csv"
FIELDS_MEMBER = ZIP_PREFIX + "data/fields.csv"

N_RESPONDENTS = 3_715
N_CHOICES = 36_297

# --- attributes.py (survey-v1), verbatim ---
VENDOR_TYPES = [
    ("take-out food", "take-out food"),
    ("prescription medications", "prescription medications"),
    ("liquor store", "items from a liquor store"),
    ("last-minute groceries", "last-minute groceries"),
]
COST_LEVELS = [
    ["$0", "$1", "$3", "$5"],
    ["$0", "$1", "$5", "$10"],
    ["$0", "$1", "$3", "$5"],
    ["$0", "$1", "$3", "$5"],
]
TIME_LEVELS = [
    ["15 minutes", "20 minutes", "30 minutes", "45 minutes"],
    ["30 minutes", "2 hrs", "6 hrs", "1 day"],
    ["15 minutes", "20 minutes", "30 minutes", "45 minutes"],
    ["15 minutes", "20 minutes", "30 minutes", "45 minutes"],
]
DRONE_PRIVACY_LEVELS = ["NO privacy", "YES privacy"]

# --- generate-questions.py QUESTION_TEMPLATE (text content), verbatim ---
CHOICE_STEM_TEMPLATE = "Suppose you order {v_value} for delivery to your home.\nGiven the following delivery options, which would you prefer?"
# Qualtrics recode: 0 = ground vehicle, 1 = drone
CHOICE_OPTIONS: list[tuple[int, str]] = [(0, "Ground vehicle"), (1, "Drone")]
CHOICE_COL_RE = re.compile(r"^CHOICE_v(\d)_gc(\d)_gt(\d)_dc(\d)_dt(\d)_dp(\d)_1(\.1)?$")

# --- survey.pdf screens (participant-shown text; page breaks joined by blank lines) ---
PRIVACY_EXPLAINER_TEXT = (
    "Delivery alternatives\n"
    "In the next series of questions we would like you to compare two hypothetical options where you order items "
    "for delivery to your home.\n"
    "While the scenarios are hypothetical, please answer them as if you were truly presented with the scenario.\n\n"
    "Delivery option 1: Ground vehicle\n"
    "This option is however you typically have items delivered in your neighborhood currently (e.g., delivery by "
    "car or bike).\n\n"
    "Delivery option 2: Drone\n"
    "In addition, companies offer delivery via drone.\n"
    "Privacy for drone deliveries\n"
    "Drone deliveries sometimes introduce privacy concerns because safety regulations require drones to broadcast "
    "their locations.\n"
    "What this means for you as a consumer\n"
    "This means that third parties can see that drones went from specific storefronts to your home when making "
    "deliveries. This can potentially allow other people and companies to learn which kinds of products were sent "
    "to you.\n"
    "For example, if a drone visits a store that sells apples and then visits your address, someone can conclude "
    "that you ordered from the store that sells apples.\n\n"
    "On the other hand, if the drone makes other stops before your address, only the delivery company and/or the "
    "store you purchased from can know your order for sure. However, this added privacy may come at additional cost "
    "or require additional delivery time.\n\n"
    "In the following questions, we use “NO privacy” to refer to the first case and “YES privacy” "
    "to refer to the second case."
)
VENDOR_INTRO_TEXTS = [
    (
        "Take-out food delivery\n"
        "Imagine you order take-out food for delivery to your home. You can have it delivered by ground vehicle or "
        "drone. Using drone delivery with no privacy may allow a third party to learn that you ordered take-out from "
        "a specific restaurant."
    ),
    (
        "Medications delivery\n"
        "Imagine you order medications from a company that specializes in delivering prescription medications. You "
        "can have the delivery made by ground vehicle or drone. Using drone delivery with no privacy may allow a third "
        "party to learn that you ordered medications from this service."
    ),
    (
        "Liquor store delivery\n"
        "Imagine you order items from a liquor store for delivery to your home. You can have the delivery made by "
        "ground vehicle or drone. Using drone delivery with no privacy may allow a third party to learn that you "
        "ordered items from a liquor store."
    ),
    (
        "Last-minute groceries delivery\n"
        "Imagine you need last-minute groceries and order them from a service specializing in last-minute grocery "
        "delivery. For example, companies like Gorillas and Getir operate in the U.S. and advertise grocery delivery "
        "within minutes of ordering.\n"
        "You can have the delivery made by ground vehicle or drone. Using drone delivery with no privacy may allow a "
        "third party to learn that you used a grocery delivery service."
    ),
]

# --- answer labels (codebook.py unless noted) ---
FREQ_OPTIONS = [
    (1, "More than once a week"),
    (2, "Multiple times a month"),
    (3, "About once a month"),
    (4, "Once in a few months or longer"),
    (5, "Never"),
]
YES_NO_DK_OPTIONS = [(1, "Yes"), (2, "No"), (3, "I don't know")]
# survey.pdf lists these in this order; code 4 = Other (from the `_4_TEXT` column name); codes 1-3 assume pdf order.
WHY_NO_OPTIONS = [
    (1, "I prefer the in-store experience"),
    (2, "I want to avoid potential delivery fees or added cost"),
    (3, "I want to avoid potential delays"),
    (4, "Other"),
]
STATE_NAMES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "I do not reside in the United States",
]
DEMOGRAPHIC_OPTIONS: dict[str, list[tuple[int, str]]] = {
    "Q_gender": [(1, "Male"), (2, "Female"), (3, "Other"), (4, "Prefer not to answer")],
    "Q_age": [
        (1, "18 - 24 years"), (2, "25 - 34 years"), (3, "35 - 44 years"),
        (4, "45 - 54 years"), (5, "55 - 64 years"), (6, "65 or older"),
    ],
    "Q_income": [
        (1, "Less than $25,000"), (2, "$25,000 to $49,999"), (3, "$50,000 to $74,999"),
        (4, "$75,000 to $99,999"), (5, "$100,000 to $149,999"), (6, "$150,000 to $199,999"),
        (7, "$200,000 or more"),
    ],
    "Q_urban_rural": [(1, "Urban"), (2, "Suburban"), (3, "Rural"), (4, "I don't know")],
    "Q_residence_type": [(1, "Private home"), (2, "Apartment"), (3, "Other")],
    "Q_state": [(i, name) for i, name in enumerate(STATE_NAMES, start=1)],
}
RACE_OPTIONS = [
    (1, "White"),
    (2, "Black or African American"),
    (3, "American Indian or Alaska Native"),
    (4, "Asian"),
    (5, "Native Hawaiian or Pacific Islander"),
    (6, "Hispanic or Latino"),
    (7, "Other"),
]


def _read_zip_csv(member: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(RAW_ZIP) as zf:
        return pd.read_csv(io.BytesIO(zf.read(member)), **kwargs)


def load_raw_df() -> pd.DataFrame:
    """`sample-preprocessed.csv` (3,715 respondents x 831 columns) with validation."""
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"berke2023 zip not found: {RAW_ZIP}")
    df = _read_zip_csv(DATA_MEMBER, low_memory=False)
    if len(df) != N_RESPONDENTS or df["ResponseId"].nunique() != N_RESPONDENTS:
        raise ValueError(f"Expected {N_RESPONDENTS} unique respondents, got {len(df)}")
    return df


def load_field_texts() -> dict[str, str]:
    """Qualtrics header text per column (`fields.csv`)."""
    f = _read_zip_csv(FIELDS_MEMBER, index_col=0)
    return {str(k): str(v) for k, v in f["field"].items()}


def extract_choices_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One row per answered choice: respondent, product type, design levels (codes), survey path, answer code."""
    records = []
    for col in raw_df.columns:
        if not col.startswith("CHOICE"):
            continue
        m = CHOICE_COL_RE.match(col)
        if m is None:
            raise ValueError(f"Unexpected choice column name: {col!r}")
        v, gc, gt, dc, dt, dp = (int(x) for x in m.groups()[:6])
        path = "multi_type" if m.group(7) else "single_type"
        answered = raw_df[col].dropna()
        for rid, ans in answered.items():
            if ans not in (0, 1):
                raise ValueError(f"{col}: unexpected answer {ans!r}")
            records.append(
                {"ResponseId": raw_df.at[rid, "ResponseId"], "v": v, "gc": gc, "gt": gt, "dc": dc, "dt": dt,
                 "dp": dp, "path": path, "column": col, "answer": int(ans)}
            )
    out = pd.DataFrame(records)
    if len(out) != N_CHOICES:
        raise ValueError(f"Expected {N_CHOICES} choices, got {len(out)}")
    if out.duplicated(["ResponseId", "v", "gc", "gt", "dc", "dt", "dp"]).any():
        raise ValueError("A respondent answered the same design twice")
    return out


def choice_stem(v: int) -> str:
    return CHOICE_STEM_TEMPLATE.format(v_value=VENDOR_TYPES[v][1])


def option_descriptions(v: int, gc: int, gt: int, dc: int, dt: int, dp: int) -> tuple[str, str]:
    """(Ground vehicle, Drone) descriptions in the on-screen row order: fee, wait time, privacy."""
    ground = f"Delivery fee: {COST_LEVELS[v][gc]}; Delivery wait time: {TIME_LEVELS[v][gt]}; Privacy: N/A"
    drone = f"Delivery fee: {COST_LEVELS[v][dc]}; Delivery wait time: {TIME_LEVELS[v][dt]}; Privacy: {DRONE_PRIVACY_LEVELS[dp]}"
    return ground, drone


def _count_codes(series: pd.Series, codes: list[int]) -> list[int]:
    vc = series.dropna().astype(int).value_counts()
    return [int(vc[c]) if c in vc.index else 0 for c in codes]


def _count_multi(series: pd.Series, codes: list[int]) -> list[int]:
    counts = {c: 0 for c in codes}
    for val in series.dropna().astype(str):
        for part in val.split(","):
            code = int(part)
            if code not in counts:
                raise ValueError(f"Unexpected multi-select code {code!r} in {series.name}")
            counts[code] += 1
    return [counts[c] for c in codes]


def build_survey_audit_artifacts() -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_by_key, details_payload)` for the drone delivery survey."""
    raw_df = load_raw_df()
    fields = load_field_texts()
    choices_df = extract_choices_df(raw_df)

    rows: list[dict[str, str]] = []
    tags_by_key: dict[str, list[str]] = {}
    items_detail: dict[str, dict] = {}
    study_contexts: list[dict] = []

    def add_entry(
        *, qid, csv_cols, family, block, n_answered, include, kev_type, context_source, review_note,
        question_text, in_block_db_text, options, tags, paired_variants=None, q_type_meta="MC",
        selector_meta="SAVR", sample_texts=None,
    ) -> None:
        idx = len(rows)
        row_key = f"{idx}:{qid}"
        paired_variants = paired_variants or []
        rows.append(
            {
                "catalog_idx": str(idx), "QuestionID": qid, "csv_cols": ";".join(csv_cols),
                "n_cols": str(len(csv_cols)), "family": family, "BlockName": block, "n_subitems": "1",
                "n_answered": str(n_answered), "include": "True" if include else "False",
                "kev_type": kev_type if include else "", "context_source": context_source, "context_sep": "",
                "drop_options": "", "review_note": review_note, "QuestionText": question_text,
                "Context_preview": in_block_db_text if context_source == "in_block_db" else "",
                "Rows_preview": "",
                "Options_preview": f"[{len(options)}] " + " | ".join(o["option"] for o in options) if options else "",
            }
        )
        if tags:
            tags_by_key[row_key] = list(tags)
        items_detail[row_key] = {
            "QuestionType": q_type_meta, "Selector": selector_meta, "rows": [], "options": options,
            "row_breakdowns": [], "qtext_variants": [], "db_variants": [], "opt_variant_sets": [],
            "paired_variants": paired_variants, "paired_variants_total": len(paired_variants),
            "sample_texts": sample_texts or [], "in_block_db_text": in_block_db_text,
        }
        if family == "0_DB":
            study_contexts.append(
                {"qid": qid, "block": block, "is_question": False, "n_variants": 1,
                 "preview": question_text[:140], "full_text": question_text}
            )

    def add_db(qid: str, block: str, text: str, note: str) -> None:
        add_entry(
            qid=qid, csv_cols=[], family="0_DB", block=block, n_answered=N_RESPONDENTS, include=False,
            kev_type="", context_source="", review_note=note, question_text=text, in_block_db_text="",
            options=[], tags=[], q_type_meta="DB", selector_meta="",
        )

    def add_mc(qid, block, family, opts, include, kev_type, note, tags=(), context_source="", context_text=""):
        s = pd.to_numeric(raw_df[qid], errors="raise")
        codes = [c for c, _ in opts]
        unexpected = set(s.dropna().astype(int)) - set(codes)
        if unexpected:
            raise ValueError(f"{qid}: unexpected codes {unexpected}")
        cnts = _count_codes(s, codes)
        add_entry(
            qid=qid, csv_cols=[qid], family=family, block=block, n_answered=int(s.notna().sum()),
            include=include, kev_type=kev_type, context_source=context_source,
            review_note=note, question_text=fields[qid], in_block_db_text=context_text,
            options=[{"option": t, "count": n} for (_, t), n in zip(opts, cnts, strict=True)], tags=list(tags),
        )

    def add_multi(qid, block, family, opts, note):
        cnts = _count_multi(raw_df[qid], [c for c, _ in opts])
        add_entry(
            qid=qid, csv_cols=[qid], family=family, block=block, n_answered=int(raw_df[qid].notna().sum()),
            include=False, kev_type="", context_source="", review_note=note, question_text=fields[qid],
            in_block_db_text="", options=[{"option": t, "count": n} for (_, t), n in zip(opts, cnts, strict=True)],
            tags=[], q_type_meta="MC_Multi", selector_meta="MAVR",
        )

    def add_te(qid, block, note, show_samples: bool):
        vals = raw_df[qid].dropna().astype(str)
        add_entry(
            qid=qid, csv_cols=[qid], family="0_TE", block=block, n_answered=int(len(vals)), include=False,
            kev_type="", context_source="", review_note=note, question_text=fields[qid], in_block_db_text="",
            options=[], tags=[], q_type_meta="TE", selector_meta="",
            sample_texts=vals.head(8).tolist() if show_samples else [],
        )

    freq_note = (
        "Included as score: ordered 5-level frequency scale (More than once a week ... Never); everyday shopping "
        "behaviour that personas can plausibly inform. Labels: codebook.py."
    )
    yn_note = (
        "Included as choice: Yes / No / I don't know. Not scored, because 'I don't know' is not a point on a "
        "Yes-No scale. Labels: codebook.py."
    )
    why_note = (
        "Excluded: select-all-that-apply (one answer can hold several codes), and shown only to respondents who "
        "answered 'Never'/'No' to the previous question. Option text from survey.pdf; codes 1-3 assume the pdf's "
        "listing order (code 4 = Other, from the `_4_TEXT` column)."
    )
    demo_note = "Excluded: demographic question, used for the persona instead. Labels: codebook.py."

    # --- Consumer behaviour (survey.pdf order) ---
    add_mc("Q_online_shopping", "Consumer behavior", "2_Ordinal_Scale", FREQ_OPTIONS, True, "score", freq_note)
    add_mc("Q_takeout_food", "Consumer behavior", "2_Ordinal_Scale", FREQ_OPTIONS, True, "score", freq_note)
    add_mc("Q_online_groceries", "Consumer behavior", "2_Ordinal_Scale", FREQ_OPTIONS, True, "score", freq_note)
    add_multi("Q_v_0_why_no", "Consumer behavior", "1_Choice_Multi", WHY_NO_OPTIONS, why_note + " Shown if take-out frequency = Never.")
    add_te("Q_v_0_why_no_4_TEXT", "Consumer behavior", "Excluded: free-text 'Other' reason.", True)
    add_mc("Q_v_1", "Consumer behavior", "1_Choice_Multi", YES_NO_DK_OPTIONS, True, "choice", yn_note, tags=["has-dont-know"])
    add_multi("Q_v_1_why_no", "Consumer behavior", "1_Choice_Multi", WHY_NO_OPTIONS, why_note + " Shown if Q_v_1 = No.")
    add_te("Q_v_1_why_no_4_TEXT", "Consumer behavior", "Excluded: free-text 'Other' reason.", True)
    add_mc("Q_v_2", "Consumer behavior", "1_Choice_Multi", YES_NO_DK_OPTIONS, True, "choice", yn_note, tags=["has-dont-know"])
    add_multi("Q_v_2_why_no", "Consumer behavior", "1_Choice_Multi", WHY_NO_OPTIONS, why_note + " Shown if Q_v_2 = No.")
    add_te("Q_v_2_why_no_4_TEXT", "Consumer behavior", "Excluded: free-text 'Other' reason.", True)

    # --- Delivery choice experiment ---
    add_db(
        "privacy_explainer", "Delivery alternatives", PRIVACY_EXPLAINER_TEXT,
        "Screens explaining the two delivery options and 'NO privacy' / 'YES privacy', shown once before all choice "
        "blocks (survey.pdf; abbreviated document, images not included).",
    )
    for v in range(4):
        intro_qid = f"intro_v{v}"
        add_db(
            intro_qid, f"Choice block v{v}", VENDOR_INTRO_TEXTS[v],
            f"Introduction shown before the {VENDOR_TYPES[v][0]} choice questions (survey.pdf).",
        )
        sub = choices_df[choices_df["v"] == v]
        designs: OrderedDict[tuple, list[int]] = OrderedDict()
        for r in sub.itertuples(index=False):
            designs.setdefault((r.gc, r.gt, r.dc, r.dt, r.dp), []).append(r.answer)
        ranked = sorted(designs.items(), key=lambda kv: -len(kv[1]))
        stem = choice_stem(v)
        variants = []
        for (gc, gt, dc, dt, dp), answers in ranked:
            g_desc, d_desc = option_descriptions(v, gc, gt, dc, dt, dp)
            variants.append(
                {
                    "n": len(answers), "stem": stem, "in_block_db_qid": intro_qid,
                    "in_block_db_text": VENDOR_INTRO_TEXTS[v], "qid_texts": {},
                    "options": [
                        {"option": "Ground vehicle", "description": g_desc, "count": answers.count(0)},
                        {"option": "Drone", "description": d_desc, "count": answers.count(1)},
                    ],
                    "row_counts": [],
                }
            )
        (gc, gt, dc, dt, dp), _ = ranked[0]
        g_desc, d_desc = option_descriptions(v, gc, gt, dc, dt, dp)
        n_paths = sub["path"].value_counts().to_dict()
        add_entry(
            qid=f"delivery_choice_v{v}",
            csv_cols=sorted(sub["column"].unique()),
            family="1_Choice_Multi",
            block=f"Choice block v{v}",
            n_answered=len(sub),
            include=True,
            kev_type="choice",
            context_source=f"qid:privacy_explainer+{intro_qid}",
            review_note=(
                f"Included as choice: the experiment's outcome for {VENDOR_TYPES[v][0]} ({len(sub):,} choices from "
                f"{sub['ResponseId'].nunique():,} respondents; {len(designs)} designs, all previewed; "
                f"single-type path {n_paths.get('single_type', 0):,}, multi-type path {n_paths.get('multi_type', 0):,}). "
                "Context starts as the privacy explanation + this type's introduction, since 'NO/YES privacy' is "
                "only defined there. Stem and levels: generate-questions.py / attributes.py; options are Qualtrics "
                "recodes 0 = Ground vehicle, 1 = Drone."
            ),
            question_text=stem,
            in_block_db_text=VENDOR_INTRO_TEXTS[v],
            options=[
                {"option": "Ground vehicle", "description": g_desc, "count": int((sub["answer"] == 0).sum())},
                {"option": "Drone", "description": d_desc, "count": int((sub["answer"] == 1).sum())},
            ],
            tags=[],
            paired_variants=variants,
        )

    # --- Demographics (survey.pdf order) ---
    add_mc("Q_gender", "Demographics", "0_Demographics", DEMOGRAPHIC_OPTIONS["Q_gender"], False, "", demo_note)
    race_cnts = _count_multi(raw_df["Q_race"], [c for c, _ in RACE_OPTIONS])
    add_entry(
        qid="Q_race", csv_cols=["Q_race"], family="0_Demographics", block="Demographics",
        n_answered=int(raw_df["Q_race"].notna().sum()), include=False, kev_type="", context_source="",
        review_note=demo_note + " Select-all-that-apply.", question_text=fields["Q_race"], in_block_db_text="",
        options=[{"option": t, "count": n} for (_, t), n in zip(RACE_OPTIONS, race_cnts, strict=True)],
        tags=[], q_type_meta="MC_Multi", selector_meta="MAVR",
    )
    for qid in ["Q_age", "Q_income", "Q_urban_rural", "Q_residence_type", "Q_state"]:
        add_mc(qid, "Demographics", "0_Demographics", DEMOGRAPHIC_OPTIONS[qid], False, "", demo_note)
    add_te("Q_residence_type_3_TEXT", "Demographics", "Excluded: free-text 'Other' residence type.", True)
    add_te("Q_comments", "End of survey", "Excluded: open feedback about the survey; samples not shown.", False)

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 2,
        "schema_kind": "question_audit",
        "study_contexts": {"__all__": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload


def write_berke2023_audit_files(out_dir: Path = BERKE_DIR) -> tuple[Path, Path, Path]:
    """Generate `question_audit.csv`, `question_tags.json`, `.audit_details.json`; refuses to overwrite review work."""
    audit_df, tags_by_key, details_payload = build_survey_audit_artifacts()
    csv_path = out_dir / "question_audit.csv"
    tags_path = out_dir / "question_tags.json"
    details_path = out_dir / ".audit_details.json"
    for p in (csv_path, tags_path):
        if p.exists():
            raise FileExistsError(f"{p} exists and may hold review work; move it away first to regenerate.")
    audit_df.to_csv(csv_path, index=False)
    tags_path.write_text(json.dumps(tags_by_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    details_path.write_text(json.dumps(details_payload, ensure_ascii=False), encoding="utf-8")
    return csv_path, tags_path, details_path


if __name__ == "__main__":
    for path in write_berke2023_audit_files():
        print(f"Wrote {path}")

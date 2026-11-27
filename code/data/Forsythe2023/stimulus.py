"""Survey catalog and Audit UI indexers for Forsythe et al. (2023), *PNAS* electric vehicle choice-based conjoint.

Source files in `code/data/Forsythe2023/`:
- `DrivingElectrificationReplication-v1.0.2.zip` ->
  `crforsythe-DrivingElectrificationReplication-72b0e47/Estimation/Data/ChoiceData/Pooled/Pooled.xlsx`:
  - `cbcLong-car`: 33,705 rows = 749 car-buyer respondents x 15 choice tasks x 3 vehicle alternatives (11,235 tasks).
  - `cbcLong-suv`: 39,465 rows = 877 SUV-buyer respondents x 15 choice tasks x 3 vehicle alternatives (13,155 tasks).
  - `demo-car` (749 rows) & `demo-suv` (877 rows): 1,626 unique respondents (disjoint IDs, exact 1-to-1 match with CBC).
- `pnas.2219396120.sapp.pdf` (Supporting Information Figures S1-S3, Table S2, pp. 2-5):
  verbatim participant-facing choice stem, option labels (`Option 1`, `Option 2`, `Option 3`), attribute labels,
  powertrain range descriptions, `$2.636/gallon` MPG-equivalent operating cost conversion, and car acceleration labels.

Non-CBC attitude/vehicle-history variables in `demo-car`/`demo-suv` are stored only as numeric Sawtooth codes without
the full Sawtooth questionnaire text in the public replication archive; they are cataloged as `include=False` alongside
the 10 locked respondent persona source columns.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import io
from pathlib import Path
import zipfile

import pandas as pd

FORSYTHE_DIR = Path(__file__).resolve().parent
REPLICATION_ZIP = FORSYTHE_DIR / "DrivingElectrificationReplication-v1.0.2.zip"
POOLED_XLSX_MEMBER = (
    "crforsythe-DrivingElectrificationReplication-72b0e47/Estimation/Data/ChoiceData/Pooled/Pooled.xlsx"
)

N_CAR_RESPONDENTS = 749
N_SUV_RESPONDENTS = 877
N_TOTAL_RESPONDENTS = N_CAR_RESPONDENTS + N_SUV_RESPONDENTS  # 1,626
N_TASKS_PER_RESPONDENT = 15
N_CONCEPTS_PER_TASK = 3
N_CAR_TASKS = N_CAR_RESPONDENTS * N_TASKS_PER_RESPONDENT  # 11,235
N_SUV_TASKS = N_SUV_RESPONDENTS * N_TASKS_PER_RESPONDENT  # 13,155

MAX_PAIRED_VARIANTS = 200

# --- Verbatim on-screen stems, options, and context (SI Appendix Figs. S2-S3, Table S2) ---
CHOICE_STEM = (
    "Suppose these 3 vehicles below were the only vehicles available for purchase, which would you choose?"
)
CHOICE_OPTIONS: list[tuple[int, str]] = [
    (1, "Option 1"),
    (2, "Option 2"),
    (3, "Option 3"),
]

CAR_CONTEXT_TEXT = (
    "You are shopping for a passenger car. "
    "The average acceleration for cars in the U.S. is 0 to 60 mph in 7.4 seconds."
)
SUV_CONTEXT_TEXT = "You are shopping for an SUV."

VEHICLE_TYPE_LABELS: OrderedDict[str, str] = OrderedDict(
    [
        ("cv", "Conventional, 300 mile range on 1 tank"),
        ("hev", "Hybrid, 300 mile range on 1 tank"),
        ("phev20", "Plug-In Hybrid, 300 mile range on 1 tank (first 20 miles electric)"),
        ("phev40", "Plug-In Hybrid, 300 mile range on 1 tank (first 40 miles electric)"),
        ("bev100", "Electric, 100 mile range on full charge"),
        ("bev150", "Electric, 150 mile range on full charge"),
        ("bev300", "Electric, 300 mile range on full charge"),
        ("bev400", "Electric, 400 mile range on full charge"),
    ]
)
VEHICLE_TYPE_CODES = {k: str(i) for i, k in enumerate(VEHICLE_TYPE_LABELS)}

BRAND_LABELS: OrderedDict[str, str] = OrderedDict(
    [
        ("american", "American"),
        ("chinese", "Chinese"),
        ("german", "German"),
        ("japanese", "Japanese"),
        ("skorean", "South Korean"),
    ]
)
BRAND_CODES = {k: str(i) for i, k in enumerate(BRAND_LABELS)}

CAR_PRICE_LEVELS: OrderedDict[int, str] = OrderedDict(
    [(17, "$17,000"), (20, "$20,000"), (27, "$27,000"), (36, "$36,000"), (56, "$56,000")]
)
SUV_PRICE_LEVELS: OrderedDict[int, str] = OrderedDict(
    [(22, "$22,000"), (28, "$28,000"), (33, "$33,000"), (41, "$41,000"), (56, "$56,000")]
)

# Operating cost with MPG equivalent (`round(263.6 / cents_per_mile)` at $2.636/gal per SI p. 5 fn *;
# matches Figs. S2-S3 verbatim: 9 cents/mi -> 29 MPG equivalent, 10 cents/mi -> 26 MPG equivalent).
CAR_OPCOST_LEVELS: OrderedDict[float, str] = OrderedDict(
    [
        (4.0, "4 cents per mile (66 MPG equivalent)"),
        (8.0, "8 cents per mile (33 MPG equivalent)"),
        (9.0, "9 cents per mile (29 MPG equivalent)"),
        (10.0, "10 cents per mile (26 MPG equivalent)"),
        (12.0, "12 cents per mile (22 MPG equivalent)"),
    ]
)
SUV_OPCOST_LEVELS: OrderedDict[float, str] = OrderedDict(
    [
        (5.0, "5 cents per mile (53 MPG equivalent)"),
        (7.5, "7.5 cents per mile (35 MPG equivalent)"),
        (10.0, "10 cents per mile (26 MPG equivalent)"),
        (12.5, "12.5 cents per mile (21 MPG equivalent)"),
        (15.0, "15 cents per mile (18 MPG equivalent)"),
    ]
)

CAR_ACCEL_LEVELS: OrderedDict[float, str] = OrderedDict(
    [
        (5.5, "5.5 seconds (Faster)"),
        (7.0, "7 seconds (Medium-Fast)"),
        (8.5, "8.5 seconds (Medium-Slow)"),
        (10.0, "10 seconds (Slower)"),
    ]
)
SUV_ACCEL_LEVELS: OrderedDict[float, str] = OrderedDict(
    [
        (7.0, "7 seconds"),
        (8.0, "8 seconds"),
        (9.0, "9 seconds"),
        (10.0, "10 seconds"),
    ]
)

# All 10 locked persona source columns in `_STATIC_PERSONA_SOURCE_COLS["Forsythe2023"]` (excluded with `include=False`).
PERSONA_AUDIT_SPECS: list[tuple[str, str, str]] = [
    (
        "ZIP",
        "What is your 5-digit home ZIP code? (main survey)",
        "Excluded: locked respondent persona attribute (`state`, combined with `Clean ZIP`).",
    ),
    (
        "Clean ZIP",
        "What is your 5-digit home ZIP code? (cleaned screener ZIP)",
        "Excluded: locked respondent persona attribute (`state`, fallback for missing `ZIP`).",
    ),
    (
        "Clean gender",
        "What is your gender identity?",
        "Excluded: locked respondent persona attribute (`gender`).",
    ),
    (
        "Clean Age",
        "What is your age?",
        "Excluded: locked respondent persona attribute (`age`).",
    ),
    (
        "Clean education",
        "What is the highest level of education you have completed?",
        "Excluded: locked respondent persona attribute (`education`).",
    ),
    (
        "Clean Income Entry",
        "What is your annual household income?",
        "Excluded: locked respondent persona attribute (`family_income`).",
    ),
    (
        "Clean livingSit",
        "What is your current marital / living situation?",
        "Excluded: locked respondent persona attribute (`marital_status`).",
    ),
    (
        "Clean householdSize",
        "How many people currently live in your household?",
        "Excluded: locked respondent persona attribute (`household_size`).",
    ),
    (
        "Clean elecVehTypeOwned",
        "Which types of vehicles have you owned? (powertrain history)",
        "Excluded: locked respondent persona attribute (`vehicle_powertrain_owned`).",
    ),
    (
        "Clean covid",
        "How has the COVID-19 pandemic affected your employment or income?",
        "Excluded: locked respondent persona attribute (`covid_employment_impact`).",
    ),
]

# Non-CBC survey columns in `demo-car`/`demo-suv` whose Sawtooth wording is absent from the public replication zip.
UNDOCUMENTED_DEMO_SPECS: list[tuple[str, str, str, str]] = [
    (
        "consideringEV",
        "Vehicle & EV Background",
        "1_Choice_Multi",
        "Excluded: numeric Sawtooth codes (1-4) only; full Sawtooth wording not in replication zip.",
    ),
    (
        "fastChargerAccess",
        "Vehicle & EV Background",
        "1_Choice_Multi",
        "Excluded: numeric Sawtooth codes (1-3) only; full Sawtooth wording not in replication zip.",
    ),
    (
        "DailyDriving",
        "Vehicle & EV Background",
        "2_Ordinal_Scale",
        "Excluded: numeric Sawtooth codes (1-7) only; full Sawtooth wording not in replication zip.",
    ),
    (
        "YearlyDriving",
        "Vehicle & EV Background",
        "2_Ordinal_Scale",
        "Excluded: numeric Sawtooth codes (1-7) only; full Sawtooth wording not in replication zip.",
    ),
    (
        "maintenance",
        "Vehicle & EV Background",
        "2_Ordinal_Scale",
        "Excluded: numeric Sawtooth codes (1-5) only; full Sawtooth wording not in replication zip.",
    ),
    (
        "NewVsUsed",
        "Vehicle & EV Background",
        "1_Choice_Multi",
        "Excluded: numeric Sawtooth codes (1-3) only; full Sawtooth wording not in replication zip.",
    ),
    (
        "NumVehiclesOwned",
        "Vehicle & EV Background",
        "2_Ordinal_Scale",
        "Excluded: numeric Sawtooth codes (1-4) only; full Sawtooth wording not in replication zip.",
    ),
    (
        "LastVehiclePurchase",
        "Vehicle & EV Background",
        "1_Choice_Multi",
        "Excluded: screener recency codes (1-3); full Sawtooth wording not in replication zip.",
    ),
    (
        "FutureVehiclePurchase",
        "Vehicle & EV Background",
        "1_Choice_Multi",
        "Excluded: screener intent codes (1-4); full Sawtooth wording not in replication zip.",
    ),
    (
        "envLifeRatings",
        "Attitude Batteries",
        "2_Ordinal_Scale",
        "Excluded: 4-item environmental/lifestyle Likert battery (`envLifeRatings_r1..r4`); item stems not in replication zip.",
    ),
    (
        "evConsiderationFeature",
        "Attitude Batteries",
        "2_Ordinal_Scale",
        "Excluded: 7-item EV consideration feature rating battery (`evConsiderationFeature_r1..r7`); item stems not in replication zip.",
    ),
    (
        "productUserRatings",
        "Attitude Batteries",
        "2_Ordinal_Scale",
        "Excluded: 6-item product/technology adoption battery (`productUserRatings_r1..r6`); item stems not in replication zip.",
    ),
    (
        "surveyThoughts",
        "Closing",
        "3_Open_Text",
        "Excluded: open-ended closing feedback text.",
    ),
]


def load_raw_pooled_sheets(
    zip_path: Path = REPLICATION_ZIP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load validated `(cbc_car, cbc_suv, demo_car, demo_suv)` sheets from `Pooled.xlsx`."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Forsythe2023 replication archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        xl = pd.ExcelFile(io.BytesIO(zf.read(POOLED_XLSX_MEMBER)))
        cbc_car = xl.parse("cbcLong-car")
        cbc_suv = xl.parse("cbcLong-suv")
        demo_car = xl.parse("demo-car")
        demo_suv = xl.parse("demo-suv")

    if len(demo_car) != N_CAR_RESPONDENTS or demo_car["id"].nunique() != N_CAR_RESPONDENTS:
        raise ValueError(f"Forsythe2023 demo-car: expected {N_CAR_RESPONDENTS} unique respondents, got {len(demo_car)}")
    if len(demo_suv) != N_SUV_RESPONDENTS or demo_suv["id"].nunique() != N_SUV_RESPONDENTS:
        raise ValueError(f"Forsythe2023 demo-suv: expected {N_SUV_RESPONDENTS} unique respondents, got {len(demo_suv)}")
    if set(demo_car["id"].astype(int)) & set(demo_suv["id"].astype(int)):
        raise ValueError("Forsythe2023: overlapping respondent IDs between demo-car and demo-suv")
    if set(cbc_car["ID"].astype(int)) != set(demo_car["id"].astype(int)):
        raise ValueError("Forsythe2023: respondent ID mismatch between cbcLong-car and demo-car")
    if set(cbc_suv["ID"].astype(int)) != set(demo_suv["id"].astype(int)):
        raise ValueError("Forsythe2023: respondent ID mismatch between cbcLong-suv and demo-suv")

    return cbc_car, cbc_suv, demo_car, demo_suv


def _encode_and_format_option(
    *,
    seg_prefix: str,
    veh_type: str,
    brand: str,
    price_k: int,
    bev_fc: int,
    op_cost: float,
    accel: float,
    price_map: OrderedDict[int, str],
    opcost_map: OrderedDict[float, str],
    accel_map: OrderedDict[float, str],
) -> tuple[str, str]:
    """Return `(compact_code, formatted_description)` for one vehicle alternative."""
    if veh_type not in VEHICLE_TYPE_LABELS:
        raise ValueError(f"Forsythe2023 ({seg_prefix}): unexpected Type {veh_type!r}")
    if brand not in BRAND_LABELS:
        raise ValueError(f"Forsythe2023 ({seg_prefix}): unexpected Brand {brand!r}")
    if price_k not in price_map:
        raise ValueError(f"Forsythe2023 ({seg_prefix}): unexpected Nominal Price {price_k!r}")
    if op_cost not in opcost_map:
        raise ValueError(f"Forsythe2023 ({seg_prefix}): unexpected Nominal OpCost {op_cost!r}")
    if accel not in accel_map:
        raise ValueError(f"Forsythe2023 ({seg_prefix}): unexpected Acceleration {accel!r}")
    if bev_fc not in (0, 1):
        raise ValueError(f"Forsythe2023 ({seg_prefix}): unexpected BEVFC {bev_fc!r}")

    is_bev = veh_type.startswith("bev")
    if not is_bev and bev_fc != 0:
        raise ValueError(f"Forsythe2023 ({seg_prefix}): non-BEV {veh_type!r} has BEVFC={bev_fc}")
    fc_text = ("Available" if bev_fc == 1 else "Not Available") if is_bev else "--"
    fc_digit = ("2" if bev_fc == 1 else "1") if is_bev else "0"

    price_digit = str(list(price_map.keys()).index(price_k))
    op_digit = str(list(opcost_map.keys()).index(op_cost))
    acc_digit = str(list(accel_map.keys()).index(accel))
    code = f"{seg_prefix}{VEHICLE_TYPE_CODES[veh_type]}{BRAND_CODES[brand]}{price_digit}{fc_digit}{op_digit}{acc_digit}"

    desc = (
        f"Vehicle Type: {VEHICLE_TYPE_LABELS[veh_type]}; "
        f"Brand: {BRAND_LABELS[brand]}; "
        f"Purchase Price: {price_map[price_k]}; "
        f"Fast Charging Capability: {fc_text}; "
        f"Operating Cost (Equivalent Gas Fuel Efficiency): {opcost_map[op_cost]}; "
        f"0 to 60 mph Acceleration Time: {accel_map[accel]}"
    )
    return code, desc


def extract_segment_tasks_df(
    cbc_df: pd.DataFrame,
    *,
    segment: str,
    expected_tasks: int,
) -> pd.DataFrame:
    """Group 3-concept rows per `(ID, Task)` into one validated choice-task row."""
    if segment == "car":
        seg_prefix, price_map, opcost_map, accel_map = "c", CAR_PRICE_LEVELS, CAR_OPCOST_LEVELS, CAR_ACCEL_LEVELS
    elif segment == "suv":
        seg_prefix, price_map, opcost_map, accel_map = "s", SUV_PRICE_LEVELS, SUV_OPCOST_LEVELS, SUV_ACCEL_LEVELS
    else:
        raise ValueError(f"Unknown Forsythe2023 segment: {segment!r}")

    expected_rows = expected_tasks * N_CONCEPTS_PER_TASK
    if len(cbc_df) != expected_rows:
        raise ValueError(f"Forsythe2023 ({segment}): expected {expected_rows} CBC rows, got {len(cbc_df)}")

    work = cbc_df.sort_values(["ID", "Task", "Concept"]).reset_index(drop=True)
    req_cols = ["ID", "Task", "Concept", "Choice", "ChoiceInd", "Type", "Brand", "Nominal Price", "Nominal OpCost", "Acceleration", "BEVFC"]
    if work[req_cols].isna().any().any():
        raise ValueError(f"Forsythe2023 ({segment}): unexpected NaN in required CBC columns")

    ids = work["ID"].astype(int).to_numpy()
    tasks = work["Task"].astype(int).to_numpy()
    concepts = work["Concept"].astype(int).to_numpy()
    choices = work["Choice"].astype(int).to_numpy()
    choice_inds = work["ChoiceInd"].astype(int).to_numpy()
    types = work["Type"].astype(str).to_numpy()
    brands = work["Brand"].astype(str).to_numpy()
    prices = work["Nominal Price"].astype(int).to_numpy()
    opcosts = work["Nominal OpCost"].astype(float).to_numpy()
    accels = work["Acceleration"].astype(float).to_numpy()
    bevfcs = work["BEVFC"].astype(int).to_numpy()

    out_rows: list[dict[str, object]] = []
    for base in range(0, len(work), 3):
        rid = int(ids[base])
        task_num = int(tasks[base])
        if ids[base + 1] != rid or ids[base + 2] != rid or tasks[base + 1] != task_num or tasks[base + 2] != task_num:
            raise ValueError(f"Forsythe2023 ({segment}): misaligned 3-option block at row {base}")
        if (concepts[base], concepts[base + 1], concepts[base + 2]) != (1, 2, 3):
            raise ValueError(f"Forsythe2023 ({segment}): expected Concepts (1,2,3) for ID={rid} Task={task_num}")
        ch = int(choices[base])
        if ch not in (1, 2, 3) or choices[base + 1] != ch or choices[base + 2] != ch:
            raise ValueError(f"Forsythe2023 ({segment}): inconsistent Choice for ID={rid} Task={task_num}")
        ind_sum = int(choice_inds[base]) + int(choice_inds[base + 1]) + int(choice_inds[base + 2])
        if ind_sum != 1 or int(choice_inds[base + ch - 1]) != 1:
            raise ValueError(f"Forsythe2023 ({segment}): Choice={ch} disagrees with ChoiceInd for ID={rid} Task={task_num}")

        codes: list[str] = []
        descs: list[str] = []
        for offset in range(3):
            idx = base + offset
            c_str, d_str = _encode_and_format_option(
                seg_prefix=seg_prefix,
                veh_type=types[idx],
                brand=brands[idx],
                price_k=int(prices[idx]),
                bev_fc=int(bevfcs[idx]),
                op_cost=float(opcosts[idx]),
                accel=float(accels[idx]),
                price_map=price_map,
                opcost_map=opcost_map,
                accel_map=accel_map,
            )
            codes.append(c_str)
            descs.append(d_str)

        out_rows.append(
            {
                "CASEID": f"forsythe2023_{rid}",
                "id": rid,
                "task": task_num,
                "code_1": codes[0],
                "code_2": codes[1],
                "code_3": codes[2],
                "desc_1": descs[0],
                "desc_2": descs[1],
                "desc_3": descs[2],
                "choice_code": ch,
            }
        )

    out = pd.DataFrame(out_rows)
    if len(out) != expected_tasks:
        raise ValueError(f"Forsythe2023 ({segment}): expected {expected_tasks} tasks, got {len(out)}")
    return out


def extract_all_tasks(
    cbc_car: pd.DataFrame, cbc_suv: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract validated `(car_tasks_df, suv_tasks_df)` 3-alternative discrete-choice tables."""
    car_tasks = extract_segment_tasks_df(cbc_car, segment="car", expected_tasks=N_CAR_TASKS)
    suv_tasks = extract_segment_tasks_df(cbc_suv, segment="suv", expected_tasks=N_SUV_TASKS)
    return car_tasks, suv_tasks


def _build_choice_variants(
    tasks_df: pd.DataFrame,
    context_text: str,
) -> tuple[list[dict], int]:
    """Group 3-option tasks by `(desc_1, desc_2, desc_3)` and return `(top_variants, total_distinct)`."""
    groups: OrderedDict[tuple[str, str, str], list[int]] = OrderedDict()
    for r in tasks_df.itertuples(index=False):
        key = (str(r.desc_1), str(r.desc_2), str(r.desc_3))
        if key not in groups:
            groups[key] = []
        groups[key].append(int(r.choice_code))
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    variants: list[dict] = []
    for (d1, d2, d3), chs in ranked[:MAX_PAIRED_VARIANTS]:
        cnt = Counter(chs)
        variants.append(
            {
                "n": len(chs),
                "stem": CHOICE_STEM,
                "in_block_db_qid": "",
                "in_block_db_text": context_text,
                "qid_texts": {},
                "row_counts": [],
                "options": [
                    {"option": CHOICE_OPTIONS[0][1], "description": d1, "count": int(cnt[1])},
                    {"option": CHOICE_OPTIONS[1][1], "description": d2, "count": int(cnt[2])},
                    {"option": CHOICE_OPTIONS[2][1], "description": d3, "count": int(cnt[3])},
                ],
            }
        )
    return variants, len(groups)


def build_survey_audit_artifacts() -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_by_key, details_payload)` covering the Forsythe et al. (2023) archive."""
    cbc_car, cbc_suv, demo_car, demo_suv = load_raw_pooled_sheets()
    car_tasks, suv_tasks = extract_all_tasks(cbc_car, cbc_suv)
    demo_all = pd.concat([demo_car, demo_suv], ignore_index=True)

    rows: list[dict[str, str]] = []
    tags_by_key: dict[str, list[str]] = {}
    items_detail: dict[str, dict] = {}
    study_contexts: list[dict] = []

    def add_entry(
        *,
        qid: str,
        csv_cols: list[str],
        family: str,
        block: str,
        n_answered: int,
        include: bool,
        kev_type: str,
        context_source: str,
        review_note: str,
        question_text: str,
        in_block_db_text: str,
        options: list[dict],
        tags: list[str],
        paired_variants: list[dict] | None = None,
        paired_variants_total: int | None = None,
        q_type_meta: str = "MC",
        selector_meta: str = "SAVR",
        sample_texts: list[str] | None = None,
    ) -> None:
        idx = len(rows)
        row_key = f"{idx}:{qid}"
        paired_variants = paired_variants or []
        opts_preview = (
            f"[{len(options)}] " + " | ".join(str(o["option"]) for o in options) if options else ""
        )
        rows.append(
            {
                "catalog_idx": str(idx),
                "QuestionID": qid,
                "csv_cols": ";".join(csv_cols),
                "n_cols": str(len(csv_cols)),
                "family": family,
                "BlockName": block,
                "n_subitems": "1",
                "n_answered": str(n_answered),
                "include": "True" if include else "False",
                "kev_type": kev_type if include else "",
                "context_source": context_source,
                "context_sep": "",
                "drop_options": "",
                "review_note": review_note,
                "QuestionText": question_text,
                "Context_preview": in_block_db_text if context_source == "in_block_db" else "",
                "Rows_preview": "",
                "Options_preview": opts_preview,
            }
        )
        if tags:
            tags_by_key[row_key] = list(tags)
        items_detail[row_key] = {
            "QuestionType": q_type_meta,
            "Selector": selector_meta,
            "rows": [],
            "options": options,
            "row_breakdowns": [],
            "qtext_variants": [],
            "db_variants": [],
            "opt_variant_sets": [],
            "paired_variants": paired_variants,
            "paired_variants_total": (
                paired_variants_total if paired_variants_total is not None else len(paired_variants)
            ),
            "sample_texts": sample_texts if sample_texts is not None else [],
            "in_block_db_text": in_block_db_text,
        }

    # --- 1. Included discrete-choice conjoint tasks (`car_choice`, `suv_choice`) ---
    for qid, seg_label, tasks_df, n_resp, ctx_text, sheet_name in [
        (
            "car_choice",
            "Car Conjoint (15 tasks x 3 options)",
            car_tasks,
            N_CAR_RESPONDENTS,
            CAR_CONTEXT_TEXT,
            "cbcLong-car",
        ),
        (
            "suv_choice",
            "SUV Conjoint (15 tasks x 3 options)",
            suv_tasks,
            N_SUV_RESPONDENTS,
            SUV_CONTEXT_TEXT,
            "cbcLong-suv",
        ),
    ]:
        variants, n_distinct = _build_choice_variants(tasks_df, ctx_text)
        vc = tasks_df["choice_code"].value_counts()
        first = tasks_df.iloc[0]
        add_entry(
            qid=qid,
            csv_cols=[
                f"{sheet_name}:Choice",
                "Type",
                "Brand",
                "Nominal Price",
                "BEVFC",
                "Nominal OpCost",
                "Acceleration",
            ],
            family="1_Choice_Multi",
            block=seg_label,
            n_answered=len(tasks_df),
            include=True,
            kev_type="choice",
            context_source="in_block_db",
            review_note=(
                f"Included as choice: 3-alternative discrete-choice vehicle conjoint ({len(tasks_df):,} tasks from "
                f"{n_resp:,} US vehicle buyers x 15 tasks; {n_distinct:,} distinct 3-option choice sets, top "
                f"{len(variants)} previewed). Verbatim prompt, attribute labels, powertrain range text, and "
                "MPG-equivalent operating cost conversion ($2.636/gal) from PNAS SI Appendix Figs. S2-S3 & Table S2."
            ),
            question_text=CHOICE_STEM,
            in_block_db_text=ctx_text,
            options=[
                {
                    "option": CHOICE_OPTIONS[0][1],
                    "description": str(first["desc_1"]),
                    "count": int(vc[1]) if 1 in vc.index else 0,
                },
                {
                    "option": CHOICE_OPTIONS[1][1],
                    "description": str(first["desc_2"]),
                    "count": int(vc[2]) if 2 in vc.index else 0,
                },
                {
                    "option": CHOICE_OPTIONS[2][1],
                    "description": str(first["desc_3"]),
                    "count": int(vc[3]) if 3 in vc.index else 0,
                },
            ],
            tags=[],
            paired_variants=variants,
            paired_variants_total=n_distinct,
        )

    # --- 2. Excluded locked persona source columns (exact match to `_STATIC_PERSONA_SOURCE_COLS["Forsythe2023"]`) ---
    for qid, qtext, note in PERSONA_AUDIT_SPECS:
        col_s = demo_all[qid].dropna()
        n_ans = int(len(col_s))
        vc = col_s.astype(str).value_counts()
        if len(vc) <= 25:
            opts = [{"option": str(k), "count": int(v)} for k, v in vc.items()]
            q_type_meta = "MC"
            selector_meta = "SAVR"
            sample_txts: list[str] = []
        else:
            opts = []
            q_type_meta = "TE"
            selector_meta = "SL"
            sample_txts = [str(k) for k in vc.index[:8]]
        add_entry(
            qid=qid,
            csv_cols=[qid],
            family="4_Demographics_Persona",
            block="Demographics & Persona",
            n_answered=n_ans,
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=qtext,
            in_block_db_text="",
            options=opts,
            tags=[],
            q_type_meta=q_type_meta,
            selector_meta=selector_meta,
            sample_texts=sample_txts,
        )

    # --- 3. Excluded undocumented non-CBC survey items in `demo-car`/`demo-suv` ---
    for qid, block, family, note in UNDOCUMENTED_DEMO_SPECS:
        if qid == "surveyThoughts":
            txts = [str(v).strip() for v in demo_all[qid].dropna() if str(v).strip()]
            add_entry(
                qid=qid,
                csv_cols=[qid],
                family=family,
                block=block,
                n_answered=len(txts),
                include=False,
                kev_type="",
                context_source="",
                review_note=note,
                question_text="Open-ended survey comments (`surveyThoughts`).",
                in_block_db_text="",
                options=[],
                tags=[],
                q_type_meta="TE",
                selector_meta="SL",
                sample_texts=txts[:8],
            )
            continue

        if qid in demo_all.columns:
            col_s = demo_all[qid].dropna()
            vc = col_s.astype(str).value_counts().sort_index()
            opts = [{"option": f"Code {k}", "count": int(v)} for k, v in vc.items()]
            csv_cols = [qid]
            n_ans = int(len(col_s))
        else:
            sub_cols = sorted(c for c in demo_all.columns if c.startswith(f"{qid}_r"))
            if not sub_cols:
                raise KeyError(f"Forsythe2023: expected sub-columns matching {qid}_r*")
            csv_cols = sub_cols
            n_ans = int(demo_all[sub_cols[0]].notna().sum())
            vc = demo_all[sub_cols[0]].dropna().astype(int).value_counts().sort_index()
            opts = [{"option": f"Code {k}", "count": int(v)} for k, v in vc.items()]

        add_entry(
            qid=qid,
            csv_cols=csv_cols,
            family=family,
            block=block,
            n_answered=n_ans,
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=f"Undocumented Sawtooth item `{qid}` (numeric codes only in `Pooled.xlsx`).",
            in_block_db_text="",
            options=opts,
            tags=[],
        )

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 3,
        "schema_kind": "question_audit",
        "study_contexts": {"": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload

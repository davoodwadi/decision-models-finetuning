"""Survey catalog and Audit UI indexers for Kantorowicz-Reznichenko et al. (2026), contractual rights conjoint.

Source files in `dataverse_files/` (read via Rscript, since pyreadr is not installed in the workspace venv):
- `computers.RDS`: 26,842 rows = 13,421 paired laptop choice tasks from 2,238 US respondents (MSI/Savanta quota sample).
- `laundry_machines.RDS`: 26,804 rows = 13,402 paired laundry-machine choice tasks from 2,239 US respondents.
- `manipulation_checks.RDS`: 3,547 rows x 12 columns of 1-7 legal-term comprehension ratings, unlinked (no Response.ID).
- `amces_computers.csv`, `amces_laundry_machines.csv`, `script_final.R`: published OLS AMCEs (cregg::amce).

Row pairing (verified 100% across every respondent, including incomplete respondents):
In both `.RDS` files, the first half of rows (`0..h-1`) are Option A and the second half (`h..2h-1`) are Option B,
with identical `Response.ID` at rows `i` and `i + h` and `selected[i] + selected[i + h] == 1` everywhere.

Text sources:
- Question stems, option names, attribute labels, and participant-facing level text come verbatim from Figure 1 (p. 114)
  and Tables 1-2 (p. 113) of the published paper. The `.RDS` files store shortened labels for Privacy and have leading
  spaces on some Conflict.Resolution / Liability factor levels; the exact maps below restore the Figure 1 on-screen text.
- Randomized attribute row order per respondent is not stored in the `.RDS` files, so profiles use Table 1 / Table 2
  canonical attribute order.

Decisions: DECISIONS.md and `persona.py` (dataset key `kantorowicz2026`).
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import io
import json
from pathlib import Path
import subprocess

import pandas as pd

KANTOROWICZ_DIR = Path(__file__).resolve().parent
DATAVERSE_DIR = KANTOROWICZ_DIR / "dataverse_files"
COMPUTERS_RDS = DATAVERSE_DIR / "computers.RDS"
LAUNDRY_RDS = DATAVERSE_DIR / "laundry_machines.RDS"
MANIPULATION_CHECKS_RDS = DATAVERSE_DIR / "manipulation_checks.RDS"

N_COMPUTERS_ROWS = 26_842
N_COMPUTERS_TASKS = 13_421
N_COMPUTERS_RESPONDENTS = 2_238
N_LAUNDRY_ROWS = 26_804
N_LAUNDRY_TASKS = 13_402
N_LAUNDRY_RESPONDENTS = 2_239
N_TOTAL_RESPONDENTS = 2_242

MAX_PAIRED_CHOICE_VARIANTS = 200

# --- Verbatim on-screen stems and options (Figure 1, p. 114) ---
COMPUTER_CHOICE_STEM = "Which of the laptop computers (A or B) would you prefer to buy?"
COMPUTER_CHOICE_OPTIONS: list[tuple[int, str]] = [(1, "Laptop computer A"), (2, "Laptop computer B")]

LAUNDRY_CHOICE_STEM = "Which of the laundry machines (A or B) would you prefer to buy?"
LAUNDRY_CHOICE_OPTIONS: list[tuple[int, str]] = [(1, "Laundry machine A"), (2, "Laundry machine B")]

# --- Participant-facing attribute labels (Figure 1, p. 114; row explanations in parentheses as in Kreps 2020) ---
PRIVACY_LABEL = "Privacy (what will be done with your personal data which you provide in the process of purchase?)"
COMPUTER_LIABILITY_LABEL = "Liability (who bears the costs in case of a loss arising out of the use of the product?)"
LAUNDRY_LIABILITY_LABEL = "Liability (who bears the costs in case of a loss arising out of the use of the product)"
CONFLICT_LABEL = (
    "Conflict resolution (who decides in which jurisdiction a lawsuit will be heard, "
    "and whether it will be in-front of a court or arbitration)"
)

PRIVACY_RAW_TO_DISPLAY: OrderedDict[str, str] = OrderedDict(
    [
        ("No privacy", "Collected, processed and stored"),
        ("No privacy: marketing", "Collected and processed for marketing purposes"),
        (
            "No privacy: sustainability",
            "Collected and processed for the development of more sustainable products",
        ),
        ("Full privacy", "Not collected nor processed"),
    ]
)

# Each spec is `(rds_col, display_label, OrderedDict[(raw_rds_value, display_text)])`.
# The order inside each OrderedDict matches the `.RDS` factor level order (first entry = published AMCE reference level).
# Attribute order follows Tables 1-2 (p. 113).
COMPUTER_ATTRIBUTE_SPECS: list[tuple[str, str, OrderedDict[str, str]]] = [
    ("Price", "Price", OrderedDict([("$2000", "$2000"), ("$1250", "$1250"), ("$500", "$500")])),
    (
        "Brand",
        "Brand",
        OrderedDict(
            [("Medion", "Medion"), ("Apple MacBook", "Apple MacBook"), ("Lenovo", "Lenovo"), ("HP", "HP")]
        ),
    ),
    ("RAM", "Working memory (RAM)", OrderedDict([("4 GB", "4 GB"), ("10 GB", "10 GB"), ("16 GB", "16 GB")])),
    (
        "Storage.Capacity",
        "Storage capacity",
        OrderedDict([("128 GB", "128 GB"), ("256 GB", "256 GB"), ("512 GB", "512 GB")]),
    ),
    (
        "Processor",
        "Processor",
        OrderedDict([("Qualcomm snapdragon", "Qualcomm snapdragon"), ("AMD", "AMD"), ("Intel", "Intel")]),
    ),
    (
        "Warranty",
        "Warranty",
        OrderedDict([("No warranty", "No warranty"), ("1 year", "1 year"), ("10 years", "10 years")]),
    ),
    ("Privacy", PRIVACY_LABEL, PRIVACY_RAW_TO_DISPLAY),
    (
        "Liability",
        COMPUTER_LIABILITY_LABEL,
        OrderedDict(
            [
                ("The consumer", "The consumer"),
                ("Seller and consumer", "Seller and consumer"),
                ("The seller", "The seller"),
            ]
        ),
    ),
    (
        "Conflict.Resolution",
        CONFLICT_LABEL,
        # `computers.RDS` stores leading spaces on Conflict.Resolution factor levels.
        OrderedDict([(" The seller", "The seller"), (" The consumer", "The consumer")]),
    ),
]

LAUNDRY_ATTRIBUTE_SPECS: list[tuple[str, str, OrderedDict[str, str]]] = [
    ("Price", "Price", OrderedDict([("$1400", "$1400"), ("$900", "$900"), ("$400", "$400")])),
    (
        "Brand",
        "Brand",
        OrderedDict(
            [
                ("Haier", "Haier"),
                ("Hoover", "Hoover"),
                ("General Electric", "General Electric"),
                ("Whirlpool", "Whirlpool"),
            ]
        ),
    ),
    ("Depth", "Depth", OrderedDict([("28 inches", "28 inches"), ("32 inches", "32 inches"), ("36 inches", "36 inches")])),
    ("Noise", "Noise", OrderedDict([("78 dB", "78 dB"), ("75 dB", "75 dB"), ("70 dB", "70 dB")])),
    (
        "Load.Capacity",
        "Load capacity",
        OrderedDict([("6 pounds", "6 pounds"), ("12 pounds", "12 pounds"), ("20 pounds", "20 pounds")]),
    ),
    (
        "Warranty",
        "Warranty",
        OrderedDict([("No warranty", "No warranty"), ("1 year", "1 year"), ("10 years", "10 years")]),
    ),
    ("Privacy", PRIVACY_LABEL, PRIVACY_RAW_TO_DISPLAY),
    (
        "Liability",
        LAUNDRY_LIABILITY_LABEL,
        # `laundry_machines.RDS` stores leading spaces on Liability factor levels.
        OrderedDict(
            [
                (" The consumer", "The consumer"),
                (" Seller and consumer", "Seller and consumer"),
                (" The seller", "The seller"),
            ]
        ),
    ),
    (
        "Conflict.Resolution",
        CONFLICT_LABEL,
        OrderedDict([("The seller", "The seller"), ("The consumer", "The consumer")]),
    ),
]

RESPONDENT_LEVEL_COLS = [
    "gender_dummy",
    "age_dummy",
    "education_dummy",
    "income_dummy",
    "risk_aversion_dummy",
    "analytical_thinking_dummy",
    "consistent_switch",
]


def _read_rds_df(rds_path: Path) -> pd.DataFrame:
    """Read an `.RDS` data frame via Rscript, preserving exact strings and converting only R `NA` to `NaN`."""
    if not rds_path.exists():
        raise FileNotFoundError(f"RDS file not found: {rds_path}")
    r_expr = (
        "args <- commandArgs(trailingOnly = TRUE); "
        "df <- readRDS(args[1]); "
        "write.csv(df, stdout(), row.names = FALSE, na = '')"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_expr, str(rds_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return pd.read_csv(io.StringIO(proc.stdout), dtype=str, keep_default_na=False, na_values=[""])


def load_raw_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate raw `(computers_df, laundry_df)` from `dataverse_files/`."""
    comp_df = _read_rds_df(COMPUTERS_RDS)
    laundry_df = _read_rds_df(LAUNDRY_RDS)
    if len(comp_df) != N_COMPUTERS_ROWS or comp_df["Response.ID"].nunique() != N_COMPUTERS_RESPONDENTS:
        raise ValueError(
            f"computers.RDS: expected {N_COMPUTERS_ROWS} rows and {N_COMPUTERS_RESPONDENTS} respondents, "
            f"got {len(comp_df)} / {comp_df['Response.ID'].nunique()}"
        )
    if len(laundry_df) != N_LAUNDRY_ROWS or laundry_df["Response.ID"].nunique() != N_LAUNDRY_RESPONDENTS:
        raise ValueError(
            f"laundry_machines.RDS: expected {N_LAUNDRY_ROWS} rows and {N_LAUNDRY_RESPONDENTS} respondents, "
            f"got {len(laundry_df)} / {laundry_df['Response.ID'].nunique()}"
        )
    return comp_df, laundry_df


def extract_respondent_df(comp_df: pd.DataFrame, laundry_df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique respondent (`N = 2,242`) with strict cross-row and cross-product consistency checks."""
    cols = ["Response.ID", *RESPONDENT_LEVEL_COLS]
    combined = pd.concat([comp_df[cols], laundry_df[cols]], ignore_index=True)
    if bool(combined["Response.ID"].isna().any()):
        raise ValueError("kantorowicz2026: missing Response.ID")

    # Verify every respondent-level column is constant across all rows of each respondent (both products).
    nunique_per_resp = combined.groupby("Response.ID", sort=False)[RESPONDENT_LEVEL_COLS].nunique(dropna=False)
    inconsistent = [c for c in RESPONDENT_LEVEL_COLS if int(nunique_per_resp[c].max()) != 1]
    if inconsistent:
        raise ValueError(f"kantorowicz2026: respondent-level columns vary within Response.ID: {inconsistent}")

    resp_df = combined.drop_duplicates("Response.ID", keep="first").reset_index(drop=True)
    if len(resp_df) != N_TOTAL_RESPONDENTS:
        raise ValueError(f"Expected {N_TOTAL_RESPONDENTS} unique respondents, got {len(resp_df)}")
    return resp_df


def _encode_and_format_profile(
    row: dict[str, object],
    specs: list[tuple[str, str, OrderedDict[str, str]]],
    product_prefix: str,
) -> tuple[str, str]:
    """Return `(compact_code, formatted_description)` for one displayed product profile."""
    digits: list[str] = []
    parts: list[str] = []
    for rds_col, label, level_map in specs:
        raw_val = row[rds_col]
        if pd.isna(raw_val) or str(raw_val) not in level_map:
            raise ValueError(f"{product_prefix}: unexpected value {raw_val!r} in column {rds_col!r}")
        raw_str = str(raw_val)
        level_keys = list(level_map.keys())
        digits.append(str(level_keys.index(raw_str)))
        parts.append(f"{label}: {level_map[raw_str]}")
    return f"{product_prefix}{''.join(digits)}", "; ".join(parts)


def extract_product_tasks_df(
    raw_df: pd.DataFrame,
    specs: list[tuple[str, str, OrderedDict[str, str]]],
    *,
    product_name: str,
    product_prefix: str,
    expected_tasks: int,
) -> pd.DataFrame:
    """Pair rows `i` (Option A) and `i + h` (Option B) into one row per answered choice task."""
    n_rows = len(raw_df)
    if n_rows != 2 * expected_tasks:
        raise ValueError(f"{product_name}: expected {2 * expected_tasks} rows, got {n_rows}")

    half = expected_tasks
    left_df = raw_df.iloc[:half].reset_index(drop=True)
    right_df = raw_df.iloc[half:].reset_index(drop=True)

    if not bool((left_df["Response.ID"] == right_df["Response.ID"]).all()):
        raise ValueError(f"{product_name}: Response.ID mismatch between Option A half and Option B half")

    sel_a = pd.to_numeric(left_df["selected"], errors="raise")
    sel_b = pd.to_numeric(right_df["selected"], errors="raise")
    if bool(sel_a.isna().any()) or bool(sel_b.isna().any()):
        raise ValueError(f"{product_name}: missing `selected` values")
    if set(sel_a.unique()) - {0, 1} or set(sel_b.unique()) - {0, 1}:
        raise ValueError(f"{product_name}: non-binary `selected` values")
    if not bool((sel_a + sel_b == 1).all()):
        raise ValueError(f"{product_name}: found paired rows where selected_A + selected_B != 1")

    attr_cols = [col for col, _, _ in specs]
    left_records = left_df[["Response.ID", *attr_cols]].to_dict("records")
    right_records = right_df[["Response.ID", *attr_cols]].to_dict("records")

    task_counter: dict[str, int] = {}
    out_rows: list[dict[str, object]] = []
    for a_rec, b_rec, a_chosen in zip(left_records, right_records, sel_a.astype(int).tolist(), strict=True):
        rid = str(a_rec["Response.ID"])
        task_num = task_counter[rid] + 1 if rid in task_counter else 1
        task_counter[rid] = task_num
        code_a, desc_a = _encode_and_format_profile(a_rec, specs, product_prefix)
        code_b, desc_b = _encode_and_format_profile(b_rec, specs, product_prefix)
        out_rows.append(
            {
                "Response.ID": rid,
                "task": task_num,
                "code_a": code_a,
                "code_b": code_b,
                "desc_a": desc_a,
                "desc_b": desc_b,
                "choice_code": 1 if a_chosen == 1 else 2,
            }
        )
    return pd.DataFrame(out_rows)


def extract_all_tasks(comp_df: pd.DataFrame, laundry_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract validated `(comp_tasks_df, laundry_tasks_df)` paired choice tables."""
    comp_tasks = extract_product_tasks_df(
        comp_df,
        COMPUTER_ATTRIBUTE_SPECS,
        product_name="computers",
        product_prefix="c",
        expected_tasks=N_COMPUTERS_TASKS,
    )
    laundry_tasks = extract_product_tasks_df(
        laundry_df,
        LAUNDRY_ATTRIBUTE_SPECS,
        product_name="laundry_machines",
        product_prefix="l",
        expected_tasks=N_LAUNDRY_TASKS,
    )
    return comp_tasks, laundry_tasks


def _build_paired_variants(
    tasks_df: pd.DataFrame,
    stem: str,
    options: list[tuple[int, str]],
) -> tuple[list[dict], int]:
    """Group paired tasks by `(desc_a, desc_b)` and return `(top_variants, total_distinct_pairs)`."""
    opt_a, opt_b = options[0][1], options[1][1]
    groups: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for r in tasks_df.itertuples(index=False):
        key = (str(r.desc_a), str(r.desc_b))
        if key not in groups:
            groups[key] = []
        groups[key].append(int(r.choice_code))
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    variants: list[dict] = []
    for (desc_a, desc_b), choices in ranked[:MAX_PAIRED_CHOICE_VARIANTS]:
        cnt = Counter(choices)
        variants.append(
            {
                "n": len(choices),
                "stem": stem,
                "in_block_db_qid": "",
                "in_block_db_text": "",
                "qid_texts": {},
                "row_counts": [],
                "options": [
                    {"option": opt_a, "description": desc_a, "count": int(cnt[1])},
                    {"option": opt_b, "description": desc_b, "count": int(cnt[2])},
                ],
            }
        )
    return variants, len(groups)


def build_survey_audit_artifacts() -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_by_key, details_payload)` covering every item in the Dataverse archive."""
    comp_df, laundry_df = load_raw_dfs()
    resp_df = extract_respondent_df(comp_df, laundry_df)
    comp_tasks, laundry_tasks = extract_all_tasks(comp_df, laundry_df)
    mc_df = _read_rds_df(MANIPULATION_CHECKS_RDS)

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
        review_note: str,
        question_text: str,
        options: list[dict],
        tags: list[str],
        paired_variants: list[dict] | None = None,
        paired_variants_total: int | None = None,
        q_type_meta: str = "MC",
        selector_meta: str = "SAVR",
    ) -> None:
        idx = len(rows)
        row_key = f"{idx}:{qid}"
        paired_variants = paired_variants or []
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
                "context_source": "",
                "context_sep": "",
                "drop_options": "",
                "review_note": review_note,
                "QuestionText": question_text,
                "Context_preview": "",
                "Rows_preview": "",
                "Options_preview": (
                    f"[{len(options)}] " + " | ".join(str(o["option"]) for o in options) if options else ""
                ),
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
            "sample_texts": [],
            "in_block_db_text": "",
        }

    # 1. Computers conjoint (`computer_choice`)
    comp_variants, comp_n_distinct = _build_paired_variants(
        comp_tasks, COMPUTER_CHOICE_STEM, COMPUTER_CHOICE_OPTIONS
    )
    comp_first = comp_tasks.iloc[0]
    add_entry(
        qid="computer_choice",
        csv_cols=["selected", *[col for col, _, _ in COMPUTER_ATTRIBUTE_SPECS]],
        family="1_Choice_Multi",
        block="Computers Conjoint",
        n_answered=len(comp_tasks),
        include=True,
        kev_type="choice",
        review_note=(
            f"Included as choice: laptop computer paired conjoint outcome ({len(comp_tasks):,} answered tasks from "
            f"{N_COMPUTERS_RESPONDENTS:,} respondents; 2,232 completed all 6 tasks, 6 completed 4-5; "
            f"{comp_n_distinct:,} distinct pairs, top {len(comp_variants)} previewed). Exact Option A / Option B "
            "pairing verified across 100% of rows (`row i` vs `row i + 13,421`). Stem, option labels, attribute "
            "explanations, and participant-facing level text from Figure 1 (p. 114) and Table 1 (p. 113); canonical "
            "Table 1 attribute order used because randomized row order per respondent is not saved in `computers.RDS`."
        ),
        question_text=COMPUTER_CHOICE_STEM,
        options=[
            {
                "option": COMPUTER_CHOICE_OPTIONS[0][1],
                "description": str(comp_first["desc_a"]),
                "count": int((comp_tasks["choice_code"] == 1).sum()),
            },
            {
                "option": COMPUTER_CHOICE_OPTIONS[1][1],
                "description": str(comp_first["desc_b"]),
                "count": int((comp_tasks["choice_code"] == 2).sum()),
            },
        ],
        tags=[],
        paired_variants=comp_variants,
        paired_variants_total=comp_n_distinct,
    )

    # 2. Laundry machines conjoint (`laundry_choice`)
    laundry_variants, laundry_n_distinct = _build_paired_variants(
        laundry_tasks, LAUNDRY_CHOICE_STEM, LAUNDRY_CHOICE_OPTIONS
    )
    laundry_first = laundry_tasks.iloc[0]
    add_entry(
        qid="laundry_choice",
        csv_cols=["selected", *[col for col, _, _ in LAUNDRY_ATTRIBUTE_SPECS]],
        family="1_Choice_Multi",
        block="Laundry Machines Conjoint",
        n_answered=len(laundry_tasks),
        include=True,
        kev_type="choice",
        review_note=(
            f"Included as choice: laundry machine paired conjoint outcome ({len(laundry_tasks):,} answered tasks from "
            f"{N_LAUNDRY_RESPONDENTS:,} respondents; 2,226 completed all 6 tasks, 13 completed 1-5; "
            f"{laundry_n_distinct:,} distinct pairs, top {len(laundry_variants)} previewed). Exact Option A / Option B "
            "pairing verified across 100% of rows (`row i` vs `row i + 13,402`). Stem, option labels, attribute "
            "explanations, and participant-facing level text from Figure 1 (p. 114) and Table 2 (p. 113); canonical "
            "Table 2 attribute order used because randomized row order per respondent is not saved in "
            "`laundry_machines.RDS`."
        ),
        question_text=LAUNDRY_CHOICE_STEM,
        options=[
            {
                "option": LAUNDRY_CHOICE_OPTIONS[0][1],
                "description": str(laundry_first["desc_a"]),
                "count": int((laundry_tasks["choice_code"] == 1).sum()),
            },
            {
                "option": LAUNDRY_CHOICE_OPTIONS[1][1],
                "description": str(laundry_first["desc_b"]),
                "count": int((laundry_tasks["choice_code"] == 2).sum()),
            },
        ],
        tags=[],
        paired_variants=laundry_variants,
        paired_variants_total=laundry_n_distinct,
    )

    # 3. Respondent-level demographics (used in persona.py)
    demo_specs = [
        (
            "gender_dummy",
            "Respondent gender (binary split in RDS; exact Qualtrics wording not in archive).",
            "Excluded: mapped to persona attribute `SEX` in `persona.py` (`Female`, `Male`, NA -> NaN).",
        ),
        (
            "age_dummy",
            "Respondent age group (binary split in RDS; exact cutoff not documented in paper or supplement).",
            "Excluded: mapped to persona attribute `AGE` in `persona.py` (`Older`, `Younger`, NA -> NaN).",
        ),
        (
            "education_dummy",
            "Respondent education level (binary split in RDS; exact degree cutoff not documented).",
            "Excluded: mapped to persona attribute `EDULEVEL` in `persona.py` (`Below higher education`, `Higher education`, NA -> NaN).",
        ),
        (
            "income_dummy",
            "Respondent income range (binary split in RDS; exact dollar cutoff not documented).",
            "Excluded: mapped to persona attribute `US_INC` in `persona.py` (`Higher income`, `Lower income`, NA -> NaN).",
        ),
    ]
    for col, q_text, note in demo_specs:
        s = resp_df[col].dropna()
        vc = s.value_counts()
        add_entry(
            qid=col,
            csv_cols=[col],
            family="0_Demographics",
            block="Demographics",
            n_answered=len(s),
            include=False,
            kev_type="",
            review_note=note,
            question_text=q_text,
            options=[{"option": str(k), "count": int(v)} for k, v in vc.items()],
            tags=[],
        )

    # 4. Excluded respondent-level behavioral / cognitive columns
    trait_specs = [
        (
            "risk_aversion_dummy",
            "Derived binary risk-attitude indicator (no question wording or task description in paper or supplement).",
            "Excluded: undocumented behavioral lottery task with 22.3% missing values (`499` NA out of `2,242`), never documented or analyzed in the paper, supplement, or `script_final.R`; also excluded from persona.",
            ["missing-context"],
        ),
        (
            "analytical_thinking_dummy",
            "Binarized 3-item Cognitive Reflection Test (CRT) indicator (Supplement p. 5; raw item responses not in RDS).",
            "Excluded: derived binary indicator from a 3-item CRT math test with an undocumented score cutoff and no individual item responses in the RDS; also excluded from persona.",
            ["missing-context"],
        ),
        (
            "consistent_switch",
            "Derived 0/1 risk-task switching consistency indicator (undocumented in paper and supplement).",
            "Excluded: companion consistency flag for the undocumented behavioral risk task, unused in the paper or `script_final.R`; also excluded from persona.",
            ["missing-context"],
        ),
    ]
    for col, q_text, note, tags in trait_specs:
        s = resp_df[col].dropna()
        vc = s.value_counts()
        add_entry(
            qid=col,
            csv_cols=[col],
            family="1_Choice_Multi",
            block="Individual Characteristics",
            n_answered=len(s),
            include=False,
            kev_type="",
            review_note=note,
            question_text=q_text,
            options=[{"option": str(k), "count": int(v)} for k, v in vc.items()],
            tags=tags,
        )

    # 5. Excluded brand-level derived columns (vary by Brand within respondent)
    brand_fam_counts = pd.concat([comp_df["Brand_Familiarity"], laundry_df["Brand_Familiarity"]]).dropna().value_counts()
    add_entry(
        qid="Brand_Familiarity",
        csv_cols=["Brand_Familiarity"],
        family="1_Choice_Multi",
        block="Brand Covariates",
        n_answered=int(brand_fam_counts.sum()),
        include=False,
        kev_type="",
        review_note=(
            "Excluded: deterministic recode of the conjoint `Brand` attribute into More/Less familiar (counts shown are "
            "profile-row counts across both RDS files, not a respondent survey question)."
        ),
        question_text="Derived brand familiarity indicator (More familiar vs Less familiar brand arm).",
        options=[{"option": str(k), "count": int(v)} for k, v in brand_fam_counts.items()],
        tags=[],
    )

    trust_counts = pd.concat([comp_df["trust_dummy"], laundry_df["trust_dummy"]]).dropna().value_counts()
    add_entry(
        qid="trust_dummy",
        csv_cols=["trust_dummy"],
        family="1_Choice_Multi",
        block="Brand Covariates",
        n_answered=int(trust_counts.sum()),
        include=False,
        kev_type="",
        review_note=(
            "Excluded: derived binary split (`High` > 4 vs `Low` <= 4) of a 4-item brand reliability scale "
            "(Supplement pp. 5, 8) merged onto each displayed profile's `Brand` (varies by brand within respondent; "
            "the underlying 4-item 1-7 brand ratings are not in the Dataverse archive)."
        ),
        question_text="Derived per-brand trust indicator (High vs Low average across 4 reliability items).",
        options=[{"option": str(k), "count": int(v)} for k, v in trust_counts.items()],
        tags=["missing-context"],
    )

    # 6. Excluded unlinked comprehension / manipulation check table
    n_mc_nonnull = int(mc_df.notna().any(axis=1).sum())
    add_entry(
        qid="manipulation_checks",
        csv_cols=list(mc_df.columns),
        family="0_Slider",
        block="Comprehension Checks (Unlinked)",
        n_answered=n_mc_nonnull,
        include=False,
        kev_type="",
        review_note=(
            f"Excluded: `manipulation_checks.RDS` ({len(mc_df):,} rows, {n_mc_nonnull:,} with at least one rating) "
            "holds 1-7 slider ratings of how beneficial 12 legal contract terms are for consumers (Supplement pp. 3-4), "
            "but contains no `Response.ID` or demographic columns and therefore cannot be linked to respondent personas."
        ),
        question_text=(
            "Please indicate by moving the slider to which extent this contract term is beneficial (or not) for the "
            "consumers on a scale from 1-7 (1 = very unbeneficial for consumers; 7 = very beneficial for consumers)."
        ),
        options=[{"option": str(i), "count": 0} for i in range(1, 8)],
        tags=["missing-context"],
        q_type_meta="Slider",
        selector_meta="HSLIDER",
    )

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 2,
        "schema_kind": "question_audit",
        "study_contexts": {"__all__": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload


def write_kantorowicz2026_audit_files(out_dir: Path = KANTOROWICZ_DIR) -> tuple[Path, Path, Path]:
    """Generate `question_audit.csv`, `question_tags.json`, `.audit_details.json`; refuses to overwrite review work."""
    audit_df, tags_by_key, details = build_survey_audit_artifacts()
    csv_path = out_dir / "question_audit.csv"
    tags_path = out_dir / "question_tags.json"
    details_path = out_dir / ".audit_details.json"
    for p in (csv_path, tags_path):
        if p.exists():
            raise FileExistsError(f"{p} exists and may hold review work; move it away first to regenerate.")
    audit_df.to_csv(csv_path, index=False)
    tags_path.write_text(json.dumps(tags_by_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    details_path.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
    return csv_path, tags_path, details_path

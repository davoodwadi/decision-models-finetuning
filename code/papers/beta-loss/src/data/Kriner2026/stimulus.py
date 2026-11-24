"""Survey catalog and Audit UI indexers for Goldfarb, Kriner et al. (2026), sustainable aviation fuel survey.

Source files in this folder:
- `Goldfarb_etal_SAF_JCP_replication.dta`: Verasight survey (2025), 1,000 US respondents x 10 flight-profile rows.
- `1-s2.0-S0959652626019475-mmc1.pdf`: Supplemental Information; Appendix 2 = instrument for the SAF items only.

Question text sources, in this order:
1. Stata variable label, when shorter than Stata's 80-character limit (complete, as exported from Qualtrics).
2. Supplement Appendix 2, when the label is cut at 80 characters and the item is in the appendix.
3. Otherwise the cut 80-character label verbatim, tagged `truncated-text`.

Starting values (`include`, `kev_type`, `family`, tags) follow the Covid (Kreps et al. 2020) audit:
the conjoint choice starts `include=True`; every other row starts `include=False`.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import json
from pathlib import Path
import re

import pandas as pd

KRINER_DIR = Path(__file__).resolve().parent
RAW_DTA = KRINER_DIR / "Goldfarb_etal_SAF_JCP_replication.dta"

N_RESPONDENTS = 1_000
N_TASKS = 5
STATA_LABEL_LIMIT = 80

# --- Supplement Appendix 2 (pp. S12-S14), verbatim ---
SAF_DEFINITION_TEXT = (
    "New technology has significantly increased the production of sustainable aviation fuels made from "
    "renewable biomass and waste resources. These sustainable aviation fuels deliver the performance "
    "of petroleum-based jet fuel with a fraction of its carbon footprint. This enables airlines to "
    "significantly reduce greenhouse gas emissions."
)
SAF_FAMILIAR_STEM = "How informed are you about sustainable aviation fuels?"
CONJOINT_INTRO_TEXT = (
    "Please consider the following scenario. You have to take a domestic flight that lasts two hours. "
    "You have a choice between the two flight options below:"
)
CHOICE_STEM = "Which of these two flights would you choose?"
CHOICE_OPTIONS: list[tuple[int, str]] = [(1, "Flight 1"), (2, "Flight 2")]
SAF_SAFE_TEXT = (
    "Do you agree or disagree with the following statement:\n\n"
    "Sustainable aviation fuels are as safe to use in commercial flights as conventional jet fuels."
)
SCIKNOW_ROWS_SUPPLEMENT: dict[str, str] = {
    "sciknow1": "The center of the Earth is very hot.",
    "sciknow2": "The continents have been moving their location for millions of years and will continue to move.",
    "sciknow9": "Human beings, as we know them today, developed from earlier species of animals.",
}
CLIMATE_STEM = (
    "From what you know about global climate change or global warming, which one of the following "
    "statements comes closest to your opinion?"
)

# Raw Qualtrics variable (response source) -> supplement question text.
SUPPLEMENT_TEXT: dict[str, str] = {
    "saf_familiar_old": SAF_FAMILIAR_STEM,
    "saf_safe_old": SAF_SAFE_TEXT,
    "climate_change_old": CLIMATE_STEM,
    **SCIKNOW_ROWS_SUPPLEMENT,
}

# Conjoint: per-task outcome variables (1 = Flight 1, 2 = Flight 2) and per-profile display columns.
TASK_CHOICE_VARS = ["saf_choice1", "saf_choice2", "saf_conjoint3", "saf_conjoint4", "saf_conjoint5"]
PROFILE_TEXT_COLS = ["airline", "fuel_type", "baggage", "legroom", "price", "seat"]
ATTRIBUTE_TO_PROFILE_COL = {
    "Airline": "airline",
    "Fuel type": "fuel_type",
    "Included baggage": "baggage",
    "Legroom": "legroom",
    "Price": "price",
    "Seat type": "seat",
}
CONJOINT_DERIVED_COLS = [
    "choose_flight",
    *[f"choose_first_flight_{t}" for t in range(1, N_TASKS + 1)],
    *PROFILE_TEXT_COLS,
    "airline_n",
    "fuel_type_n",
    "baggage_n",
    "legroom_n",
    "price_n",
    "seat_n",
]

# Columns that are not survey questions (ids, metadata, author recodes listed under their source item).
NON_QUESTION_COLS = {"respondent", "choice_set", "case_id", "start_date", "end_date", "weight"}
DERIVED_COLS: dict[str, list[str]] = {
    "saf_familiar_old": ["saf_familiar"],
    "saf_safe_old": ["saf_safe"],
    "climate_change_old": ["climate_change"],
    "party": ["party3", "dem3", "gop3"],
    "party_lean": ["dem5", "gop5"],
    "approve": ["trump_approve"],
    "gender": ["female"],
    "education": ["college"],
    **{f"sciknow{k}": [f"sciknow{k}_correct", "sciknow_score"] for k in range(1, 10)},
}
RACE_COLS: list[tuple[str, str]] = [
    ("native_american", "American Indian"),
    ("asian", "Asian"),
    ("black", "Black or African American"),
    ("hispanic", "Hispanic"),
    ("white", "White"),
    ("race_6", "Other"),
]
RACE_STEM = "What best describes your race/ethnicity? Check all that apply."

# Randomized arms: each respondent saw one variable of each group (n per arm reported in review notes).
ARM_GROUPS: list[tuple[str, list[str]]] = [
    ("Electricity sources", ["electricity_control_", "electricity_treat_"]),
    ("Carbon source", ["carbon_coal", "recycled_coal_", "carbon_biomass", "recycled_biomass_",
                       "carbon_foodwaste", "recycled_foodwaste_", "carbon_sludge", "recycled_sludge_"]),
    ("Phosphorus recovery", ["phos_food", "phos_costs", "phos_sludge"]),
    ("Science spending", ["sci_spend_control", "sci_spend_treatment"]),
    ("Bioeconomy definition", ["bio_def_control", "bio_def_fossil"]),
    ("Campaign ad (Williams)", ["williams_control", "williams_job", "williams_climate", "williams_oil"]),
]

# Starting family for labelled items that are unordered choices; other labelled items start as ordinal.
CHOICE_FAMILY_VARS = {
    "sci_spend_initial", "sci_spend_control", "sci_spend_treatment", "williams_mc", "career_appointee",
    "urban_rural_best", "urban_rural_least", "climate_change_old", "approve",
    *[f"sciknow{k}" for k in range(1, 10)],
}
# First demographic variable in survey order; everything from here on is `0_Demographics`
# except the attitude items listed below.
FIRST_DEMOGRAPHIC_VAR = "age"
ATTITUDE_VARS_IN_DEMOGRAPHICS = {"approve", "sci_know_self"}
NO_SAMPLE_TEXT_VARS = {"zip"}

DONT_KNOW_RE = re.compile(r"don.?t know|unsure|not sure|no opinion", re.IGNORECASE)


def load_raw_df(raw_dta: Path = RAW_DTA) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[int, str]]]:
    """Load the `.dta` (`10,000` rows) with variable labels and per-variable value labels."""
    if not raw_dta.exists():
        raise FileNotFoundError(f"Kriner2026 Stata dataset not found: {raw_dta}")
    reader = pd.io.stata.StataReader(raw_dta)
    df = reader.read(convert_categoricals=False)
    var_labels = reader.variable_labels()
    label_sets = reader.value_labels()
    lbl_names = reader._lbllist  # value-label set name per column, in column order
    value_labels: dict[str, dict[int, str]] = {}
    for col, lbl in zip(df.columns, lbl_names, strict=True):
        if lbl and lbl in label_sets:
            value_labels[col] = {int(k): str(v) for k, v in label_sets[lbl].items()}
    if len(df) != N_RESPONDENTS * 2 * N_TASKS:
        raise ValueError(f"Expected {N_RESPONDENTS * 2 * N_TASKS} rows, got {len(df)}")
    if df["respondent"].nunique() != N_RESPONDENTS:
        raise ValueError(f"Expected {N_RESPONDENTS} respondents, got {df['respondent'].nunique()}")
    expected_sets = {10 * t + f for t in range(1, N_TASKS + 1) for f in (1, 2)}
    if set(df["choice_set"].astype(int).unique()) != expected_sets:
        raise ValueError(f"Unexpected choice_set values: {sorted(df['choice_set'].unique())}")
    df = df.copy()
    df["respondent"] = df["respondent"].astype(int)
    df["choice_set"] = df["choice_set"].astype(int)
    return df.sort_values(["respondent", "choice_set"]).reset_index(drop=True), var_labels, value_labels


def extract_respondent_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One row per respondent (`choice_set == 11`); survey answers repeat across the 10 profile rows."""
    resp = raw_df.loc[raw_df["choice_set"] == 11].reset_index(drop=True)
    if len(resp) != N_RESPONDENTS:
        raise ValueError(f"Expected {N_RESPONDENTS} respondent rows, got {len(resp)}")
    return resp


def build_persona_input_df(raw_df: pd.DataFrame, value_labels: dict[str, dict[int, str]]) -> pd.DataFrame:
    """One row per respondent with the raw persona columns plus `age_years` decoded from the `age` value labels.

    The `.dta` stores `age` as codes (1 = 18 ... 80 = 97), so `age_years` must come from the labels, never the codes.
    Pass the result to `persona.apply_persona_mapping(df, "kriner2026")`.
    """
    resp = extract_respondent_df(raw_df).copy()
    age_labels = value_labels["age"]
    codes = pd.to_numeric(resp["age"], errors="raise")
    if codes.isna().any():
        raise ValueError("Kriner2026: missing `age` codes")
    resp["age_years"] = [int(age_labels[int(c)]) for c in codes]
    return resp


def extract_choice_tasks_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (respondent, task) with the attribute rows exactly as displayed and the chosen flight."""
    resp = extract_respondent_df(raw_df)
    prof = raw_df.set_index(["respondent", "choice_set"])
    records: list[dict] = []
    for _, r in resp.iterrows():
        rid = int(r["respondent"])
        for t in range(1, N_TASKS + 1):
            attrs = [str(r[f"f_{t}_{k}"]) for k in range(1, 7)]
            if sorted(attrs) != sorted(ATTRIBUTE_TO_PROFILE_COL):
                raise ValueError(f"Respondent {rid} task {t}: unexpected attribute rows {attrs}")
            levels = {}
            for flight in (1, 2):
                shown = [str(r[f"f_{t}_{flight}_{k}"]) for k in range(1, 7)]
                prow = prof.loc[(rid, 10 * t + flight)]
                for attr, lvl in zip(attrs, shown, strict=True):
                    if str(prow[ATTRIBUTE_TO_PROFILE_COL[attr]]) != lvl:
                        raise ValueError(f"Respondent {rid} task {t} flight {flight}: display/profile mismatch on {attr}")
                levels[flight] = shown
            choice = int(r[TASK_CHOICE_VARS[t - 1]])
            chosen = [int(prof.loc[(rid, 10 * t + f), "choose_flight"]) for f in (1, 2)]
            if choice not in (1, 2) or chosen != [int(choice == 1), int(choice == 2)]:
                raise ValueError(f"Respondent {rid} task {t}: {TASK_CHOICE_VARS[t - 1]}={choice} vs choose_flight={chosen}")
            records.append(
                {
                    "respondent": rid,
                    "task": t,
                    "attributes": attrs,
                    "flight1": levels[1],
                    "flight2": levels[2],
                    "choice_code": choice,
                }
            )
    return pd.DataFrame(records)


def format_flight_description(attrs: list[str], levels: list[str]) -> str:
    """Attribute rows in the respondent's display order, e.g. `Price: $250; Airline: Delta; ...`."""
    return "; ".join(f"{a}: {lvl}" for a, lvl in zip(attrs, levels, strict=True))


def _count_codes(series: pd.Series, codes: list[int]) -> list[int]:
    vc = series.dropna().astype(int).value_counts()
    return [int(vc[c]) if c in vc.index else 0 for c in codes]


def _arm_of(var: str) -> tuple[str, list[str]] | None:
    for group, members in ARM_GROUPS:
        for m in members:
            if var == m or (m.endswith("_") and var.startswith(m) and var[len(m):].isdigit()):
                return group, members
    return None


def _display_order_col(var: str, columns: set[str]) -> str | None:
    m = re.match(r"^(.*)_(\d+)$", var)
    if m and f"{m.group(1)}_do_{m.group(2)}" in columns:
        return f"{m.group(1)}_do_{m.group(2)}"
    return None


def build_survey_audit_artifacts() -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_by_key, details_payload)` covering every survey question in the `.dta`."""
    raw_df, var_labels, value_labels = load_raw_df()
    resp_df = extract_respondent_df(raw_df)
    tasks_df = extract_choice_tasks_df(raw_df)
    all_cols = set(raw_df.columns)

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
            "sample_texts": sample_texts or [],
            "in_block_db_text": in_block_db_text,
        }
        if family == "0_DB":
            study_contexts.append(
                {
                    "qid": qid,
                    "block": block,
                    "is_question": False,
                    "n_variants": 1,
                    "preview": question_text[:140],
                    "full_text": question_text,
                }
            )

    def add_db(qid: str, block: str, text: str, note: str) -> None:
        add_entry(
            qid=qid, csv_cols=[], family="0_DB", block=block, n_answered=N_RESPONDENTS, include=False,
            kev_type="", context_source="", review_note=note, question_text=text, in_block_db_text="",
            options=[], tags=[], q_type_meta="DB", selector_meta="",
        )

    def add_conjoint() -> None:
        add_db(
            "saf_conjoint_intro", "SAF Conjoint", CONJOINT_INTRO_TEXT,
            "Prompt shown before the 5 flight choice sets (Supplement Appendix 2, p. S12).",
        )
        codes = [c for c, _ in CHOICE_OPTIONS]
        pooled = _count_codes(tasks_df["choice_code"], codes)
        groups: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
        for r in tasks_df.itertuples(index=False):
            key = (format_flight_description(r.attributes, r.flight1), format_flight_description(r.attributes, r.flight2))
            groups.setdefault(key, []).append(int(r.choice_code))
        ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        variants = []
        for (d1, d2), choices in ranked:
            cnt = Counter(choices)
            variants.append(
                {
                    "n": len(choices),
                    "stem": CHOICE_STEM,
                    "in_block_db_qid": "saf_conjoint_intro",
                    "in_block_db_text": CONJOINT_INTRO_TEXT,
                    "qid_texts": {},
                    "options": [
                        {"option": CHOICE_OPTIONS[0][1], "description": d1, "count": cnt[1]},
                        {"option": CHOICE_OPTIONS[1][1], "description": d2, "count": cnt[2]},
                    ],
                    "row_counts": [],
                }
            )
        first = tasks_df.iloc[0]
        add_entry(
            qid="flight_choice",
            csv_cols=[*TASK_CHOICE_VARS, *CONJOINT_DERIVED_COLS, *[c for c in raw_df.columns if c.startswith("f_")]],
            family="1_Choice_Multi",
            block="SAF Conjoint",
            n_answered=len(tasks_df),
            include=True,
            kev_type="choice",
            context_source="in_block_db",
            review_note=(
                f"Flight choice: {N_TASKS} tasks x {N_RESPONDENTS:,} respondents = {len(tasks_df):,} choices "
                f"({len(groups):,} distinct displayed pairs, all previewed). "
                "Prompt and question from Supplement Appendix 2; attribute names and levels are the on-screen "
                "text from the `f_*` columns (includes the on-screen spelling 'Sutainable aviation fuel produced "
                "from biomass'). Attribute row order is randomized per respondent and fixed across the 5 tasks. "
                "Shown after the SAF definition and familiarity question."
            ),
            question_text=CHOICE_STEM,
            in_block_db_text=CONJOINT_INTRO_TEXT,
            options=[
                {"option": CHOICE_OPTIONS[0][1], "description": format_flight_description(first.attributes, first.flight1), "count": pooled[0]},
                {"option": CHOICE_OPTIONS[1][1], "description": format_flight_description(first.attributes, first.flight2), "count": pooled[1]},
            ],
            tags=[],
            paired_variants=variants,
            paired_variants_total=len(groups),
        )

    def add_race() -> None:
        opts = [
            {"option": lbl, "count": int((pd.to_numeric(resp_df[col], errors="raise") == 1).sum())}
            for col, lbl in RACE_COLS
        ]
        add_entry(
            qid="race",
            csv_cols=[c for c, _ in RACE_COLS],
            family="0_Demographics",
            block="Demographics",
            n_answered=N_RESPONDENTS,
            include=False,
            kev_type="",
            context_source="",
            review_note="Multi-select checkboxes (one 0/1 column per category). Text source: Stata labels (complete).",
            question_text=RACE_STEM,
            in_block_db_text="",
            options=opts,
            tags=[],
            q_type_meta="MC_Multi",
            selector_meta="MAVR",
        )

    in_demographics = False
    conjoint_added = False
    race_added = False
    skip = (
        NON_QUESTION_COLS
        | set(CONJOINT_DERIVED_COLS)
        | {c for cols in DERIVED_COLS.values() for c in cols}
    )
    for col in raw_df.columns:
        if col == FIRST_DEMOGRAPHIC_VAR:
            in_demographics = True
        if col in TASK_CHOICE_VARS:
            if not conjoint_added:
                add_conjoint()
                conjoint_added = True
            continue
        if col in {c for c, _ in RACE_COLS}:
            if not race_added:
                add_race()
                race_added = True
            continue
        if (
            col in skip
            or col.startswith("f_")
            or col.startswith("fl_")
            or "_do_" in col
        ):
            continue

        label = var_labels[col]
        series = resp_df[col]
        n_answered = int(series.notna().sum()) if series.dtype != object else int((series.astype(str).str.strip() != "").sum())
        notes: list[str] = []
        tags: list[str] = []
        context_source = ""
        context_text = ""
        block = "Demographics" if in_demographics else re.sub(r"_?\d+$", "", col)

        # Question text
        if col in SUPPLEMENT_TEXT:
            question_text = SUPPLEMENT_TEXT[col]
            notes.append(f"Text source: Supplement Appendix 2 (Stata label cut at 80 characters: {label!r}).")
        elif len(label) >= STATA_LABEL_LIMIT:
            question_text = label
            tags.append("truncated-text")
            notes.append("Text source: Stata label, cut at 80 characters; full wording not available.")
        else:
            question_text = label
            notes.append("Text source: Stata label (complete).")

        if col == "saf_familiar_old":
            add_db(
                "saf_definition", "SAF Familiarity", SAF_DEFINITION_TEXT,
                "SAF definition shown with the familiarity question (Supplement Appendix 2, p. S12).",
            )
            context_source, context_text = "in_block_db", SAF_DEFINITION_TEXT
        if col.startswith("sciknow"):
            stem = var_labels["sciknow1"].split("\n")[0].strip()
            context_source, context_text = "in_block_db", stem
            if col not in SUPPLEMENT_TEXT and "truncated-text" not in tags:
                question_text = label
            notes.append(f"Statement shown under the stem {stem!r} (from the Stata label of sciknow1); order randomized.")
        if col in SUPPLEMENT_TEXT and col in ("saf_familiar_old", "saf_safe_old", "climate_change_old"):
            notes.append(
                "Options listed in Qualtrics code order from this raw variable; the supplement lists them in the "
                "reverse order of the authors' recoded variable."
            )

        arm = _arm_of(col)
        if arm is not None:
            notes.append(f"Randomized arm ({arm[0]}): shown to {n_answered:,} of {N_RESPONDENTS:,} respondents.")
        elif n_answered < N_RESPONDENTS:
            notes.append(f"Answered by {n_answered:,} of {N_RESPONDENTS:,} respondents.")
        do_col = _display_order_col(col, all_cols)
        battery = re.match(r"^(.*)_(\d+)$", col)
        siblings = (
            [c for c in raw_df.columns if re.fullmatch(rf"{re.escape(battery.group(1))}_\d+", c)]
            if battery
            else []
        )
        if len(siblings) > 1 and len({var_labels[c] for c in siblings}) == 1:
            notes.append(
                f"Battery sub-item ({len(siblings)} items share this Stata label): this sub-item's own text "
                "is missing from the data."
                + (f" Display order in `{do_col}`." if do_col else "")
            )
        if col.startswith("vs_"):
            notes.append("Verasight panel profile variable, not asked in this survey.")

        csv_cols = [col, *(DERIVED_COLS[col] if col in DERIVED_COLS else []), *([do_col] if do_col else [])]

        # Options and family
        sample_texts: list[str] = []
        q_type = "MC"
        selector = "SAVR"
        if col in value_labels:
            labels = value_labels[col]
            observed = sorted(int(v) for v in pd.to_numeric(series, errors="raise").dropna().unique())
            codes = sorted(set(labels) | set(observed))
            cnts = _count_codes(pd.to_numeric(series, errors="raise"), codes)
            unlabeled = [c for c in observed if c not in labels]
            if unlabeled:
                raise ValueError(f"{col}: observed codes without value labels: {unlabeled}")
            options = [{"option": labels[c], "count": n} for c, n in zip(codes, cnts, strict=True)]
            if in_demographics and col not in ATTITUDE_VARS_IN_DEMOGRAPHICS:
                family = "0_Demographics"
            elif col == "consent":
                family = "0_Consent_Check"
            elif col in CHOICE_FAMILY_VARS:
                family = "1_Choice_Multi"
            else:
                family = "2_Ordinal_Scale"
            if any(DONT_KNOW_RE.search(o["option"]) for o in options):
                tags.append("has-dont-know")
        else:
            options = []
            family = "0_Demographics" if in_demographics else "0_TE"
            q_type, selector = "TE", ""
            if col not in NO_SAMPLE_TEXT_VARS:
                vals = series.dropna().astype(str)
                sample_texts = vals[vals.str.strip() != ""].head(8).tolist()

        add_entry(
            qid=col,
            csv_cols=csv_cols,
            family=family,
            block=block,
            n_answered=n_answered,
            include=False,
            kev_type="",
            context_source=context_source,
            review_note=" ".join(notes),
            question_text=question_text,
            in_block_db_text=context_text,
            options=options,
            tags=tags,
            q_type_meta=q_type,
            selector_meta=selector,
            sample_texts=sample_texts,
        )

    if not (conjoint_added and race_added):
        raise ValueError("Conjoint or race entry was not built")

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 2,
        "schema_kind": "question_audit",
        "study_contexts": {"__all__": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload


def write_kriner2026_audit_files(out_dir: Path = KRINER_DIR) -> tuple[Path, Path, Path]:
    """Generate `question_audit.csv`, `question_tags.json`, and `.audit_details.json` in `out_dir`.

    Refuses to overwrite an existing `question_audit.csv` or `question_tags.json`, which hold review work.
    """
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
    for path in write_kriner2026_audit_files():
        print(f"Wrote {path}")

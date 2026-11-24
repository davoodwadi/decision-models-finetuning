"""Survey catalog, conjoint vignette stimulus builders, and Audit UI indexers for Magistro et al. (2025).

Data: `dataverse_files/clean_AJPS.csv` (6,056 YouGov respondents across the United States and Canada; decisions in
`DECISIONS.md`).

Wording sources:
- Conjoint definitions, scenario preamble, Table 1/2 attribute labels and levels, and outcome question:
  Magistro, Borwein, Alvarez, Bonikowski, and Loewen (2025), AJPS, pp. 8-10.
- Pre-/post-treatment policy items and response scales: `dataverse_files/Codebook_AJPS.pdf`, pp. 1-10.
- Post-conjoint manipulation check wording: Online Supporting Information (`ajps12959-sup-0001-appendix.pdf`), p. 14.

Starting `include` / `kev_type` values are initial classifications; the user verifies them in the Next.js Audit UI.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import pandas as pd

MAGISTRO_DIR = Path(__file__).resolve().parent
RAW_CSV = MAGISTRO_DIR / "dataverse_files" / "clean_AJPS.csv"

N_TOTAL_RESPONDENTS = 6_056
N_US_RESPONDENTS = 3_025
N_CANADA_RESPONDENTS = 3_031
N_TASKS_PER_RESPONDENT = 4

# --- Verbatim conjoint definitions (AJPS paper, p. 8) ---
AI_DEFINITION = (
    "Generative AI refers to artificial intelligence systems that can be used, for example, to write computer code, "
    "automate production tasks, streamline operations, or provide customer service via chatbots."
)
OFFSHORING_DEFINITION = (
    "Offshoring refers to the practice of relocating business activities, services, or jobs, ranging from production "
    "chain and customer service to software engineering, to another country."
)

# --- Verbatim conjoint scenario preamble (AJPS paper, p. 9; localized by country and shock) ---
SCENARIO_PREAMBLE_TEMPLATE = (
    "A {country_adj} company producing smartphones is {action_ing}. "
    "This change will decrease production costs, which could lead to lower product prices. "
    "However, while some jobs might be created, others might be lost. "
    "On the one hand, {shock_np} could reduce the need for some workers as jobs are {job_mech}, "
    "possibly leading to job losses. "
    "On the other hand, {shock_np} could lower production costs, making companies more productive and allowing them "
    "to charge lower prices. This could increase demand for their products, encouraging companies to expand "
    "production and hire more workers. These new jobs might require the same or different skills."
)

ARM_WORDING: dict[str, dict[str, str]] = {
    "AI": {
        "definition": AI_DEFINITION,
        "action_ing": "adopting new generative AI tools",
        "shock_np": "the new generative AI tools",
        "job_mech": "substituted by machines",
        "decision_inf": "adopt new generative AI tools",
        "policy_challenge": "AI and automation",
        "policy_regulate": "generative AI",
    },
    "Offshoring": {
        "definition": OFFSHORING_DEFINITION,
        "action_ing": "offshoring some of its activities abroad",
        "shock_np": "offshoring",
        "job_mech": "relocated abroad",
        "decision_inf": "offshore some of its activities abroad",
        "policy_challenge": "trade",
        "policy_regulate": "offshoring",
    },
}

COUNTRY_ADJ: dict[str, str] = {
    "US": "US",
    "Canada": "Canadian",
}

CONJOINT_AI_STEM = "Do you support or oppose the company's decision to adopt new generative AI tools?"
CONJOINT_OFFSHORING_STEM = "Do you support or oppose the company's decision to offshore some of its activities abroad?"

# Verbatim 5-point scale from Codebook_AJPS.pdf (p. 6) and AJPS paper (p. 9).
# Note: respondents also saw a 6th "Don't know" option, recoded to NA in clean_AJPS.csv.
SUPPORT_OPTIONS: list[tuple[int, str]] = [
    (1, "Strongly oppose"),
    (2, "Oppose"),
    (3, "Neither"),
    (4, "Support"),
    (5, "Strongly support"),
]

# Verbatim 5-point agreement scale for pre-/post-treatment policy items (Codebook_AJPS.pdf, pp. 2-5).
AGREE_OPTIONS: list[tuple[int, str]] = [
    (1, "Strongly disagree"),
    (2, "Disagree"),
    (3, "Neither agree nor disagree"),
    (4, "Agree"),
    (5, "Strongly agree"),
]

# --- Conjoint attributes (AJPS paper Tables 1-2, pp. 9-10) ---
# Each spec maps raw CSV value -> (code, before, after, change_abs, change_pct).
PRICE_LEVELS: OrderedDict[str, tuple[int, str, str, str, str]] = OrderedDict(
    [
        ("Unchanged at 600$", (0, "$600", "$600", "$0", "0%")),
        ("Decreases to 450$", (1, "$600", "$450", "-$150", "-25%")),
        ("Decreases to 300$", (2, "$600", "$300", "-$300", "-50%")),
    ]
)
CS_LEVELS: OrderedDict[str, tuple[int, str, str, str, str]] = OrderedDict(
    [
        ("Unchanged at 100", (0, "100", "100", "0", "0%")),
        ("Decrease to 75", (1, "100", "75", "-25", "-25%")),
        ("Increase to 125", (2, "100", "125", "+25", "+25%")),
    ]
)
FACTORY_LEVELS: OrderedDict[str, tuple[int, str, str, str, str]] = OrderedDict(
    [
        ("Unchanged at 400", (0, "400", "400", "0", "0%")),
        ("Decrease to 300", (1, "400", "300", "-100", "-25%")),
        ("Increase to 500", (2, "400", "500", "+100", "+25%")),
    ]
)
DS_LEVELS: OrderedDict[str, tuple[int, str, str, str, str]] = OrderedDict(
    [
        ("Unchanged at 200", (0, "200", "200", "0", "0%")),
        ("Decrease to 150", (1, "200", "150", "-50", "-25%")),
        ("Increase to 250", (2, "200", "250", "+50", "+25%")),
    ]
)

ATTRIBUTE_SPECS: list[tuple[str, str, OrderedDict[str, tuple[int, str, str, str, str]]]] = [
    ("price_after", "Price of smartphone", PRICE_LEVELS),
    ("cs_after", "Number of customer service jobs", CS_LEVELS),
    ("factory_after", "Number of factory jobs", FACTORY_LEVELS),
    ("ds_after", "Number of computer/data science jobs", DS_LEVELS),
]

# Pre-/post-treatment policy items (Codebook_AJPS.pdf, pp. 2-5).
POLICY_ITEMS: list[tuple[int, str, bool]] = [
    (1, "Expand social spending, such as unemployment benefits, to support workers facing potential job loss", False),
    (2, "Implement a basic income that covers essential living costs for everyone, regardless of their financial situation", False),
    (3, "Fund programs to re-skill workers for new jobs", False),
    (4, "Reduce the number of immigrants entering your country for work", False),
    (5, "Restrict international competition by increasing trade barriers on goods and services to your country", False),
    (6, "Regulate the use of {policy_regulate}", True),
]

POLICY_STEM_TEMPLATE = (
    "Think of the following scenario: Your government is crafting a policy plan to help with the challenges of "
    "{policy_challenge}. Do you agree that the government should implement the following policy: {policy_text}"
)


def format_conjoint_context(
    country: str,
    arm: str,
    price_raw: str,
    cs_raw: str,
    factory_raw: str,
    ds_raw: str,
) -> str:
    """Render the full participant-facing conjoint vignette (definition + preamble + scenario table)."""
    if country not in COUNTRY_ADJ:
        raise KeyError(f"Unexpected country {country!r}")
    if arm not in ARM_WORDING:
        raise KeyError(f"Unexpected arm {arm!r}")
    w = ARM_WORDING[arm]
    preamble = SCENARIO_PREAMBLE_TEMPLATE.format(
        country_adj=COUNTRY_ADJ[country],
        action_ing=w["action_ing"],
        shock_np=w["shock_np"],
        job_mech=w["job_mech"],
    )
    raw_vals = [price_raw, cs_raw, factory_raw, ds_raw]
    lines = ["Scenario (Before -> After | Change):"]
    for (_, attr_label, levels), raw_v in zip(ATTRIBUTE_SPECS, raw_vals, strict=True):
        if raw_v not in levels:
            raise KeyError(f"Unexpected value {raw_v!r} for {attr_label}")
        _, before, after, chg_abs, chg_pct = levels[raw_v]
        lines.append(f"- {attr_label}: {before} -> {after} ({chg_abs}, {chg_pct})")
    return f"{w['definition']}\n\n{preamble}\n\n" + "\n".join(lines)


def make_profile_code(country: str, price_raw: str, cs_raw: str, factory_raw: str, ds_raw: str) -> str:
    """Compact deterministic code for `(country, price, cs, factory, ds)`, e.g. `US_p2c0f2d1`."""
    c_prefix = "US" if country == "US" else "CA" if country == "Canada" else None
    if c_prefix is None:
        raise ValueError(f"Unexpected country {country!r}")
    p = PRICE_LEVELS[price_raw][0]
    c = CS_LEVELS[cs_raw][0]
    f = FACTORY_LEVELS[factory_raw][0]
    d = DS_LEVELS[ds_raw][0]
    return f"{c_prefix}_p{p}c{c}f{f}d{d}"


def load_raw_df(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    """Load `clean_AJPS.csv` (6,056 respondents across US and Canada) with fail-fast validation."""
    if not raw_csv.exists():
        raise FileNotFoundError(f"Magistro et al. (2025) dataset not found: {raw_csv}")
    df = pd.read_csv(raw_csv)
    if len(df) != N_TOTAL_RESPONDENTS:
        raise ValueError(f"Expected {N_TOTAL_RESPONDENTS} rows in {raw_csv.name}, got {len(df)}")
    if df["ID"].isna().any() or df["ID"].nunique() != N_TOTAL_RESPONDENTS:
        raise ValueError(f"Expected {N_TOTAL_RESPONDENTS} unique non-null IDs, got {df['ID'].nunique()}")
    countries = set(df["countr"].dropna().unique())
    if countries != {"US", "Canada"}:
        raise ValueError(f"Unexpected countr values: {countries}")
    arms = set(df["treat_offshoring"].dropna().unique())
    if arms != {"AI", "Offshoring"}:
        raise ValueError(f"Unexpected treat_offshoring values: {arms}")
    n_us = int((df["countr"] == "US").sum())
    n_ca = int((df["countr"] == "Canada").sum())
    if n_us != N_US_RESPONDENTS or n_ca != N_CANADA_RESPONDENTS:
        raise ValueError(f"Expected US={N_US_RESPONDENTS}, Canada={N_CANADA_RESPONDENTS}; got US={n_us}, Canada={n_ca}")
    return df


def extract_conjoint_tasks(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand `qtable_1..4` into answered task rows `(ai_tasks, offshoring_tasks)` with strict validation."""
    records: list[dict] = []
    valid_scores = {1, 2, 3, 4, 5}
    for row in raw_df.itertuples(index=False):
        rid = int(row.ID)
        country = str(row.countr)
        arm = str(row.treat_offshoring)
        for t in range(1, N_TASKS_PER_RESPONDENT + 1):
            y = getattr(row, f"qtable_{t}")
            if pd.isna(y):
                continue
            y_int = int(y)
            if y_int not in valid_scores or float(y) != float(y_int):
                raise ValueError(f"Unexpected qtable_{t} value {y!r} for ID={rid}")
            p_raw = str(getattr(row, f"price_after_{t}"))
            c_raw = str(getattr(row, f"cs_after_{t}"))
            f_raw = str(getattr(row, f"factory_after_{t}"))
            d_raw = str(getattr(row, f"ds_after_{t}"))
            code = make_profile_code(country, p_raw, c_raw, f_raw, d_raw)
            ctx = format_conjoint_context(country, arm, p_raw, c_raw, f_raw, d_raw)
            records.append(
                {
                    "ID": rid,
                    "countr": country,
                    "arm": arm,
                    "task": t,
                    "profile_code": code,
                    "context": ctx,
                    "price_after": p_raw,
                    "cs_after": c_raw,
                    "factory_after": f_raw,
                    "ds_after": d_raw,
                    "answer_code": y_int,
                }
            )
    tasks_df = pd.DataFrame(records)
    ai_tasks = tasks_df[tasks_df["arm"] == "AI"].reset_index(drop=True)
    off_tasks = tasks_df[tasks_df["arm"] == "Offshoring"].reset_index(drop=True)
    return ai_tasks, off_tasks


def _count_codes(series: pd.Series, codes: list[int]) -> list[int]:
    vc = series.dropna().astype(int).value_counts()
    return [int(vc[c]) if c in vc.index else 0 for c in codes]


def _build_conjoint_variants(tasks_df: pd.DataFrame, stem: str) -> tuple[list[dict], int]:
    """Build per-`(country, profile)` variant entries for the Audit UI (all 162 profiles fit in UI limit)."""
    opt_codes = [c for c, _ in SUPPORT_OPTIONS]
    opt_texts = [t for _, t in SUPPORT_OPTIONS]
    groups: list[tuple[int, str, str, list[int]]] = []
    for code, grp in tasks_df.groupby("profile_code", sort=False):
        ctx = str(grp["context"].iloc[0])
        cnts = _count_codes(grp["answer_code"], opt_codes)
        groups.append((len(grp), str(code), ctx, cnts))
    groups.sort(key=lambda x: (-x[0], x[1]))

    variants: list[dict] = []
    for n_obs, code, ctx, cnts in groups:
        variants.append(
            {
                "n": n_obs,
                "stem": stem,
                "in_block_db_qid": code,
                "in_block_db_text": ctx,
                "qid_texts": {},
                "row_counts": [],
                "options": [
                    {"option": txt, "count": cnt}
                    for txt, cnt in zip(opt_texts, cnts, strict=True)
                ],
            }
        )
    return variants, len(groups)


def build_survey_audit_artifacts(
    raw_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build `(audit_df, tags_by_key, details_payload)` covering every column in `clean_AJPS.csv`."""
    if raw_df is None:
        raw_df = load_raw_df()
    ai_tasks, off_tasks = extract_conjoint_tasks(raw_df)

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
        options: list[dict],
        tags: list[str],
        in_block_db_text: str = "",
        context_preview: str = "",
        paired_variants: list[dict] | None = None,
        paired_variants_total: int | None = None,
        q_type_meta: str = "MC",
        selector_meta: str = "SAVR",
    ) -> None:
        idx = len(rows)
        row_key = f"{idx}:{qid}"
        paired_variants = paired_variants or []
        qtext_variants: list[dict] = []
        db_variants: list[dict] = []
        if paired_variants:
            seen_stems: OrderedDict[str, tuple[str, int]] = OrderedDict()
            seen_dbs: OrderedDict[str, tuple[str, int]] = OrderedDict()
            for pv in paired_variants:
                st = pv["stem"]
                db_t = pv["in_block_db_text"]
                n_pv = int(pv["n"])
                lbl = pv["in_block_db_qid"] or "variant"
                if st:
                    if st not in seen_stems:
                        seen_stems[st] = (lbl, n_pv)
                    else:
                        prev_lbl, prev_n = seen_stems[st]
                        seen_stems[st] = (prev_lbl, prev_n + n_pv)
                if db_t:
                    if db_t not in seen_dbs:
                        seen_dbs[db_t] = (lbl, n_pv)
                    else:
                        prev_lbl, prev_n = seen_dbs[db_t]
                        seen_dbs[db_t] = (prev_lbl, prev_n + n_pv)
            if len(seen_stems) > 1:
                qtext_variants = [
                    {"label": lbl, "text": txt, "n": n}
                    for txt, (lbl, n) in list(seen_stems.items())[:15]
                ]
            if len(seen_dbs) > 1:
                db_variants = [
                    {"label": lbl, "text": txt, "n": n}
                    for txt, (lbl, n) in list(seen_dbs.items())[:15]
                ]

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
                "Context_preview": context_preview[:200],
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
            "qtext_variants": qtext_variants,
            "db_variants": db_variants,
            "opt_variant_sets": [],
            "paired_variants": paired_variants,
            "paired_variants_total": (
                paired_variants_total if paired_variants_total is not None else len(paired_variants)
            ),
            "sample_texts": [],
            "in_block_db_text": in_block_db_text,
        }

    # --- 1. Primary Conjoint Outcomes (AI arm and Offshoring arm) ---
    conjoint_cols = [
        "treat_offshoring",
        *[f"qtable_{t}" for t in range(1, 5)],
        *[f"price_after_{t}" for t in range(1, 5)],
        *[f"cs_after_{t}" for t in range(1, 5)],
        *[f"factory_after_{t}" for t in range(1, 5)],
        *[f"ds_after_{t}" for t in range(1, 5)],
    ]
    opt_codes = [c for c, _ in SUPPORT_OPTIONS]
    opt_texts = [t for _, t in SUPPORT_OPTIONS]

    for arm_qid, arm_name, tasks_df, stem in [
        ("conjoint_ai", "AI", ai_tasks, CONJOINT_AI_STEM),
        ("conjoint_offshoring", "Offshoring", off_tasks, CONJOINT_OFFSHORING_STEM),
    ]:
        variants, n_distinct = _build_conjoint_variants(tasks_df, stem)
        pooled_cnts = _count_codes(tasks_df["answer_code"], opt_codes)
        top_ctx = variants[0]["in_block_db_text"]
        n_resp_arm = int((raw_df["treat_offshoring"] == arm_name).sum())
        n_total_arm_tasks = n_resp_arm * N_TASKS_PER_RESPONDENT
        n_dk = n_total_arm_tasks - len(tasks_df)
        add_entry(
            qid=arm_qid,
            csv_cols=conjoint_cols,
            family="2_Ordinal_Scale",
            block=f"Conjoint ({arm_name})",
            n_answered=len(tasks_df),
            include=True,
            kev_type="score",
            context_source="in_block_db",
            review_note=(
                f"Included as score: {arm_name} conjoint outcome ({len(tasks_df):,} answered tasks out of "
                f"{n_total_arm_tasks:,} across {n_resp_arm:,} respondents; {n_dk:,} 'Don't know'/NA dropped; "
                f"all {n_distinct} distinct (country, profile) variants previewed). Definition, localized US/Canadian "
                "preamble, and Before -> After (Change) table follow Magistro et al. (2025) pp. 8-10 verbatim; "
                "canonical Table 1 attribute order used because per-respondent row order is not saved in clean_AJPS.csv."
            ),
            question_text=stem,
            context_preview=top_ctx,
            in_block_db_text=top_ctx,
            options=[
                {"option": txt, "count": cnt}
                for txt, cnt in zip(opt_texts, pooled_cnts, strict=True)
            ],
            paired_variants=variants,
            paired_variants_total=n_distinct,
            tags=[],
        )

    # --- 2. Pre-treatment and Post-treatment Policy Questions ---
    agree_codes = [c for c, _ in AGREE_OPTIONS]
    agree_texts = [t for _, t in AGREE_OPTIONS]
    for prefix, block_name, timing_note in [
        ("policy_ai_offshoregr", "Pre-treatment Policy Questions", "Pre-treatment policy item (before conjoint)"),
        ("tables_policygr", "Post-treatment Policy Questions", "Post-treatment repeat of pre-treatment policy item (after conjoint)"),
    ]:
        for item_idx, item_text, has_arm_placeholder in POLICY_ITEMS:
            col = f"{prefix}_{item_idx}"
            s_all = raw_df[col]
            pooled_cnts = _count_codes(s_all, agree_codes)
            p_variants: list[dict] = []
            for arm_name in ("AI", "Offshoring"):
                w = ARM_WORDING[arm_name]
                sub_s = raw_df.loc[raw_df["treat_offshoring"] == arm_name, col]
                cnts = _count_codes(sub_s, agree_codes)
                formatted_item = item_text.format(policy_regulate=w["policy_regulate"]) if has_arm_placeholder else item_text
                stem_arm = POLICY_STEM_TEMPLATE.format(
                    policy_challenge=w["policy_challenge"],
                    policy_text=formatted_item,
                )
                p_variants.append(
                    {
                        "n": int(sub_s.notna().sum()),
                        "stem": stem_arm,
                        "in_block_db_qid": arm_name,
                        "in_block_db_text": "",
                        "qid_texts": {},
                        "row_counts": [],
                        "options": [
                            {"option": txt, "count": cnt}
                            for txt, cnt in zip(agree_texts, cnts, strict=True)
                        ],
                    }
                )
            display_item = item_text.format(policy_regulate="[generative AI/offshoring]") if has_arm_placeholder else item_text
            display_stem = POLICY_STEM_TEMPLATE.format(
                policy_challenge="[AI and automation/trade]",
                policy_text=display_item,
            )
            add_entry(
                qid=col,
                csv_cols=[col],
                family="2_Ordinal_Scale",
                block=block_name,
                n_answered=int(s_all.notna().sum()),
                include=False,
                kev_type="",
                context_source="",
                review_note=(
                    f"Excluded by default: {timing_note} ({int(s_all.notna().sum()):,} answered, "
                    f"{int(s_all.isna().sum()):,} NA; wording varies by AI vs Offshoring arm)."
                ),
                question_text=display_stem,
                options=[
                    {"option": txt, "count": cnt}
                    for txt, cnt in zip(agree_texts, pooled_cnts, strict=True)
                ],
                paired_variants=p_variants,
                paired_variants_total=len(p_variants),
                tags=[],
            )

    # --- 3. Manipulation Check ---
    mc_s = raw_df["manipulation_check2"].dropna()
    mc_vc = mc_s.value_counts()
    add_entry(
        qid="manipulation_check2",
        csv_cols=["manipulation_check2"],
        family="0_Attention_Check",
        block="Manipulation Check",
        n_answered=len(mc_s),
        include=False,
        kev_type="",
        review_note=(
            "Excluded: post-conjoint factual manipulation check (Appendix p. 14; coded Pass/Fail in clean_AJPS.csv; "
            "raw 4-option selection not saved)."
        ),
        question_text="Which variables were included in the tables you just saw?",
        context_source="",
        options=[{"option": str(k), "count": int(v)} for k, v in mc_vc.items()],
        tags=[],
    )

    # --- 4. Demographics & Metadata Columns ---
    demo_specs = [
        ("countr", ["countr", "country"], "Respondent country (US vs Canada sample).", "Excluded: mapped to persona attribute `c_alphan` in `persona.py` (`United States`, `Canada`)."),
        ("region", ["region"], "US Census region (1=Northeast, 2=Midwest, 3=South, 4=West; US sample).", "Excluded: mapped to persona attribute `US_REG` for US respondents in `persona.py` (20 stale Canadian values ignored)."),
        ("GeoPC_Region1", ["GeoPC_Region1"], "Canadian province/territory (codes 1-12; Canada sample).", "Excluded: mapped to persona attribute `US_REG` for Canadian respondents in `persona.py`."),
        ("gender", ["gender"], "Respondent gender (Man, Woman, Other).", "Excluded: mapped to persona attribute `SEX` in `persona.py` (`Male`, `Female`, `Other`, 1 NA -> NaN)."),
        ("gender_bin", ["gender_bin"], "Binary gender recode (Woman, Man; Other coded as NA).", "Excluded: redundant binary recode of `gender` (drops 79 `Other` respondents)."),
        ("age_group", ["age_group"], "Respondent age bracket (18 to 29, 30 to 44, 45 to 64, 65 or older).", "Excluded: mapped to persona attribute `AGE` in `persona.py` (`18-29`, `30-44`, `45-64`, `65+`)."),
        ("education", ["education"], "Respondent education level (HS or less, Some college, College grad, Postgrad).", "Excluded: mapped to persona attribute `EDULEVEL` in `persona.py` (`High school or less`, `Some college`, `College graduate`, `Postgraduate`)."),
        ("education_bin", ["education_bin"], "Binary education recode (College vs No college).", "Excluded: redundant binary recode of `education`."),
        ("income", ["income"], "Respondent income level (Low, Middle, High; dollar cutoffs undocumented).", "Excluded: mapped to persona attribute `US_INC` in `persona.py` (`Low`, `Middle`, `High`, 785 NA -> NaN)."),
        ("party", ["party"], "Respondent political party (US: Democratic/Republican/Other; Canada: CPC/LPC/NDP/Other).", "Excluded: mapped to persona attribute `US_PRTY` in `persona.py` (`Democrat`, `Republican`, `Conservative Party of Canada`, `Liberal Party of Canada`, `New Democratic Party`, `Other`)."),
        ("party2", ["party2"], "Respondent political party with US leaners folded into Democratic/Republican.", "Excluded: redundant recode of `party` folding US leaners (consistent with `kriner2026` and `wu2025`, `party` without leaners is used)."),
    ]
    for col, cols_list, q_text, note in demo_specs:
        s = raw_df[col].dropna()
        vc = s.value_counts()
        add_entry(
            qid=col,
            csv_cols=cols_list,
            family="0_Demographics",
            block="Demographics",
            n_answered=len(s),
            include=False,
            kev_type="",
            context_source="",
            review_note=note,
            question_text=q_text,
            options=[{"option": str(k), "count": int(v)} for k, v in vc.items()],
            tags=[],
        )

    audit_df = pd.DataFrame(rows)
    details_payload = {
        "version": 2,
        "schema_kind": "question_audit",
        "study_contexts": {"__all__": study_contexts},
        "items": items_detail,
    }
    return audit_df, tags_by_key, details_payload


def write_magistro2025_audit_files(out_dir: Path = MAGISTRO_DIR) -> tuple[Path, Path, Path]:
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

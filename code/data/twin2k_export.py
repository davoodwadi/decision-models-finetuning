"""Unified Twin-2K-500 and Twin-2K-500-Mega-Study long-format parquet exporters.

Reads raw HuggingFace snapshots plus each dataset's audited `question_audit.csv`,
`question_overrides.json`, and `question_tags.json`, attaches standardized
respondent personas from `persona_loaders.load_raw_twin2k_personas` +
`persona_spec.yaml`, and exports:
  - `code/data/Twin-2K-500/twin2k_US_long.parquet`
  - `code/data/Twin-2K-500-Mega-Study/twin2k_megastudy_US_long.parquet`

Usage:
  .venv-cu13/bin/python -m data.twin2k_export              # export both datasets
  .venv-cu13/bin/python -m data.twin2k_export --dataset twin2k
  .venv-cu13/bin/python -m data.twin2k_export --dataset megastudy
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from data.persona_loaders import load_raw_twin2k_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

DATA_DIR = Path(__file__).resolve().parent

DEFAULT_SUBITEM_SEP = "\n\n"
TWIN2K_ALLOWED_OVERRIDE_FIELDS = {"stem", "subitems", "subitem_sep", "options"}
MEGA_ALLOWED_OVERRIDE_FIELDS = {"stem", "subitems", "options"}

MEGA_ALLOWED_TOP_KEYS = ("Elements", "EmbeddedData", "Metadata")
MEGA_ALLOWED_ELEMENT_TYPES = {"Block", "Branch", "Group", "Randomizer"}
MEGA_ALLOWED_BLOCK_KEYS = ("BlockID", "BlockName", "BlockType", "ElementType", "Questions")
MEGA_EXPECTED_PARQUET_COLS = ["PID", "survey_json_with_human_response", "survey_text"]
MEGA_DISCRETE_FAMILIES = ("1_Matrix_Bipolar", "2_Matrix_Likert", "3_MC_IdeaEval", "4_MC_Single")

BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
TERMINAL_PUNCT = frozenset(".?!:;…\"')")


def _decode_sep(raw_sep: str, *, default: str = "\n\n") -> str:
    if raw_sep == "":
        return default
    return raw_sep.replace("\\n", "\n").replace("\\t", "\t")


def _normalize_multiline_override(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def _norm_str(text: str) -> str:
    return " ".join(BR_RE.sub(" ", html.unescape(text)).split())


def _clean_mega_text(study: str, qid: str, text: str) -> str:
    norm = _norm_str(text)
    if study == "idea_evaluation" and re.match(r"^\d+_QID", qid):
        norm = re.sub(r"Idea #\d+ out of \d+:", "Idea:", norm)
    return norm


def _load_standardized_twin2k_personas(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Return `(persona_base_df_indexed_by_int_pid, canonical_persona_cols)` from persona_spec.yaml."""
    specs = load_persona_spec()
    raw_personas = load_raw_twin2k_personas(data_dir=data_dir)
    std_personas = standardize_dataframe(raw_personas, dataset="twin2k", specs=specs)
    persona_cols = [c for c in specs if c in std_personas.columns and std_personas[c].notna().any()]
    if not persona_cols:
        raise ValueError("Standardized Twin-2K-500 personas contain no populated spec columns")

    pids = std_personas["CASEID"].astype(int)
    if pids.duplicated().any():
        raise ValueError("Duplicate PIDs in standardized Twin-2K-500 personas")

    persona_base_df = std_personas[persona_cols].copy()
    persona_base_df.index = pids
    persona_base_df.index.name = "pid"
    persona_base_df = persona_base_df.sort_index()
    persona_base_df.insert(0, "CASEID", [f"twin_{pid}" for pid in persona_base_df.index])
    return persona_base_df, persona_cols


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must contain a JSON object, got {type(obj).__name__}")
    return obj


def _validate_tags_dict(tags: dict[str, Any], valid_keys: set[str], path: Path) -> None:
    for rk, tag_list in tags.items():
        if not isinstance(tag_list, list) or not all(isinstance(t, str) and t.strip() for t in tag_list):
            raise ValueError(f"{rk}: {path.name} entry must be a list of non-empty strings")
    unknown = sorted(set(tags.keys()) - valid_keys)
    if unknown:
        raise KeyError(f"Unknown row keys in {path.name}: {unknown}")


def _extract_selector(row: pd.Series) -> str:
    if row["QuestionType"] == "DB":
        if pd.notna(row["Settings"]):
            raise ValueError(f"{row['QuestionID']}: DB row has unexpected Settings: {row['Settings']}")
        return ""
    return row["Settings"]["Selector"]


def export_twin2k_parquet(
    *,
    data_dir: Path = DATA_DIR,
    write: bool = True,
) -> pd.DataFrame:
    """Export `Twin-2K-500/twin2k_US_long.parquet` with canonical `persona_spec.yaml` personas."""
    twin_dir = data_dir / "Twin-2K-500"
    hf_dir = twin_dir / "hf"
    csv_dir = hf_dir / "question_catalog_and_human_response_csv"

    catalog_path = csv_dir / "question_catalog.json"
    w13_path = csv_dir / "wave1_3_response_label.csv"
    w4_path = csv_dir / "wave4_response_label.csv"
    chunks_dir = hf_dir / "wave_split" / "chunks"

    for req in (catalog_path, w13_path, w4_path, chunks_dir):
        if not req.exists():
            raise FileNotFoundError(f"Required Twin-2K-500 raw artifact missing: {req}")

    catalog_df = pd.DataFrame(json.loads(catalog_path.read_text(encoding="utf-8")))
    catalog_df["Selector"] = catalog_df.apply(_extract_selector, axis=1)
    catalog_df["n_csv_cols"] = catalog_df["csv_columns"].map(len)

    w13_df = pd.read_csv(w13_path, dtype=str)
    w13_df["pid"] = w13_df["pid"].astype(int)
    w13_df = w13_df.set_index("pid")

    w4_df = pd.read_csv(w4_path, dtype=str)
    w4_df["pid"] = w4_df["pid"].astype(int)
    w4_df = w4_df.set_index("pid")

    if not w13_df.index.sort_values().equals(w4_df.index.sort_values()):
        raise ValueError("Waves 1-3 and wave 4 contain different pids")
    missing_in_w13 = sorted(set(w4_df.columns) - set(w13_df.columns))
    if missing_in_w13:
        raise ValueError(f"Wave 4 columns without a waves 1-3 counterpart: {missing_in_w13}")

    all_cat_cols = [c for cols in catalog_df["csv_columns"] for c in cols]
    if len(all_cat_cols) != len(set(all_cat_cols)):
        raise ValueError("Duplicate column names in catalog csv_columns")
    if set(all_cat_cols) != set(w13_df.columns):
        raise ValueError("Catalog csv_columns do not match w13_df columns")

    audit_path = twin_dir / "question_audit.csv"
    overrides_path = twin_dir / "question_overrides.json"
    tags_path = twin_dir / "question_tags.json"

    audit_raw_df = pd.read_csv(audit_path, keep_default_na=False)
    question_overrides = _load_json_dict(overrides_path)
    question_tags = _load_json_dict(tags_path)

    valid_override_keys: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    has_drop_opts = "drop_options" in audit_raw_df.columns
    for _, r in audit_raw_df.iterrows():
        idx_str = str(r["catalog_idx"])
        row_key = f"{idx_str}:{r['QuestionID']}"
        valid_override_keys.add(row_key)
        drop_opts = r["drop_options"] if has_drop_opts else ""
        inc = r["include"] == "True" if isinstance(r["include"], str) else bool(r["include"])
        if ".." in idx_str:
            start_idx, end_idx = map(int, idx_str.split(".."))
            for idx in range(start_idx, end_idx + 1):
                audit_rows.append(
                    {
                        "catalog_idx": idx,
                        "override_key": row_key,
                        "QuestionID": f"QID9_{idx - start_idx + 1}",
                        "family": r["family"],
                        "include": inc,
                        "kev_type": r["kev_type"],
                        "drop_options": drop_opts,
                    }
                )
        else:
            audit_rows.append(
                {
                    "catalog_idx": int(idx_str),
                    "override_key": row_key,
                    "QuestionID": r["QuestionID"],
                    "family": r["family"],
                    "include": inc,
                    "kev_type": r["kev_type"],
                    "drop_options": drop_opts,
                }
            )

    unknown_override_keys = sorted(set(question_overrides.keys()) - valid_override_keys)
    if unknown_override_keys:
        raise KeyError(f"Unknown row keys in {overrides_path.name}: {unknown_override_keys}")
    _validate_tags_dict(question_tags, valid_override_keys, tags_path)

    audit_expanded_df = pd.DataFrame(audit_rows).set_index("catalog_idx").sort_index()
    if not audit_expanded_df.index.equals(catalog_df.index):
        raise ValueError("Expanded question_audit.csv indices do not match catalog_df.index")
    if not (audit_expanded_df["QuestionID"] == catalog_df["QuestionID"]).all():
        raise ValueError("Expanded question_audit.csv QuestionIDs do not match catalog_df['QuestionID']")

    for col in ("override_key", "family", "include", "kev_type", "drop_options"):
        catalog_df[col] = audit_expanded_df[col]

    for rk, ov in question_overrides.items():
        if not isinstance(ov, dict) or not ov:
            raise ValueError(f"{rk}: override entry must be a non-empty dict")
        extra_fields = set(ov.keys()) - TWIN2K_ALLOWED_OVERRIDE_FIELDS
        if extra_fields:
            raise ValueError(f"{rk}: unknown override fields {sorted(extra_fields)}")
        matched_cat = catalog_df[catalog_df["override_key"] == rk]
        if matched_cat.empty:
            raise KeyError(f"{rk}: override key did not match any catalog_df row")
        if ".." in rk.split(":", 1)[0] and "stem" in ov:
            raise ValueError(f"{rk}: static stem override is not allowed on collapsed loop rows")
        cat0 = matched_cat.iloc[0]
        if "stem" in ov and not isinstance(ov["stem"], str):
            raise TypeError(f"{rk}: stem override must be str")
        if "subitems" in ov:
            if not isinstance(ov["subitems"], dict) or not ov["subitems"]:
                raise ValueError(f"{rk}: subitems override must be a non-empty dict")
            cat_rows = cat0["Rows"] if isinstance(cat0["Rows"], list) else []
            if not cat_rows:
                raise ValueError(f"{rk}: subitems override set on row without Matrix Rows")
            for sub_k, sub_v in ov["subitems"].items():
                if not isinstance(sub_k, str) or not sub_k.isdigit():
                    raise ValueError(f"{rk}: subitem key {sub_k!r} must be a 0-based integer string")
                sub_idx = int(sub_k)
                if not (0 <= sub_idx < len(cat_rows)):
                    raise IndexError(f"{rk}: subitem index {sub_idx} out of range (0..{len(cat_rows) - 1})")
                if not isinstance(sub_v, str) or not _normalize_multiline_override(sub_v):
                    raise ValueError(f"{rk}: subitem {sub_k!r} override must be a non-empty string")
        if "subitem_sep" in ov:
            if not isinstance(ov["subitem_sep"], str) or ov["subitem_sep"] == "":
                raise ValueError(f"{rk}: subitem_sep override must be a non-empty string")
            if cat0["family"] != "2_Matrix_Likert":
                raise ValueError(f"{rk}: subitem_sep override is only valid on 2_Matrix_Likert rows")
        if "options" in ov:
            if not isinstance(ov["options"], dict) or not ov["options"]:
                raise ValueError(f"{rk}: options override must be a non-empty dict")
            if cat0["family"] == "1_Matrix_Bipolar":
                raise ValueError(f"{rk}: override Bipolar pair options via subitems ('opt1:opt2') rather than options")
            raw_opt_list = (
                cat0["Columns"]
                if isinstance(cat0["Columns"], list)
                else (cat0["Options"] if isinstance(cat0["Options"], list) else [])
            )
            known_opts = {" ".join(str(x).split()) for x in raw_opt_list}
            for opt_k, opt_v in ov["options"].items():
                if opt_k not in known_opts:
                    raise KeyError(f"{rk}: option override key {opt_k!r} not in {sorted(known_opts)}")
                if not isinstance(opt_v, str) or not " ".join(opt_v.split()):
                    raise ValueError(f"{rk}: option override for {opt_k!r} must be a non-empty string")

    persona_base_df, persona_cols = _load_standardized_twin2k_personas(data_dir)
    if not persona_base_df.index.equals(w13_df.index.sort_values()):
        raise ValueError("Standardized persona PIDs do not match w13_df PIDs")

    def build_long_df(resp_df: pd.DataFrame, question_specs: list[dict[str, Any]]) -> pd.DataFrame:
        frames = []
        for spec in question_specs:
            col = spec["v"]
            s = resp_df[col].dropna()
            drop_raw = spec["drop_raw"] if "drop_raw" in spec else set()
            if drop_raw:
                s = s[~s.isin(drop_raw)]
            raw_to_code = spec["raw_to_code"]
            unexpected = set(s.unique()) - set(raw_to_code.keys())
            if unexpected:
                raise ValueError(f"{col}: unexpected response values {unexpected}")

            ans_codes = [raw_to_code[val] for val in s]
            code_to_text = dict(zip(spec["option_codes"], spec["option_texts"], strict=True))
            ans_texts = [code_to_text[c] for c in ans_codes]

            sub = persona_base_df.loc[s.index].copy()
            n = len(sub)
            sub["v"] = col
            sub["label"] = spec["label"]
            sub["question"] = spec["question"]
            sub["type"] = spec["type"]
            sub["reversed"] = False
            sub["option_codes"] = [list(spec["option_codes"])] * n
            sub["option_texts"] = [list(spec["option_texts"])] * n
            sub["answer_code"] = ans_codes
            sub["answer_text"] = ans_texts
            frames.append(sub.reset_index(drop=True))
        return pd.concat(frames, ignore_index=True)

    # Family 1: Matrix — Bipolar
    bipolar_entries = catalog_df[(catalog_df["family"] == "1_Matrix_Bipolar") & catalog_df["include"]]
    bipolar_specs: list[dict[str, Any]] = []
    for _, q in bipolar_entries.iterrows():
        ov = question_overrides[q["override_key"]] if q["override_key"] in question_overrides else None
        q_text = (
            _normalize_multiline_override(ov["stem"])
            if ov is not None and "stem" in ov
            else " ".join(q["QuestionText"].split())
        )
        q_type = q["kev_type"]
        if q_type != "choice":
            raise ValueError(f"{q['QuestionID']}: expected kev_type='choice', got {q_type!r}")
        rows = [
            _normalize_multiline_override(ov["subitems"][str(i)])
            if ov is not None and "subitems" in ov and str(i) in ov["subitems"]
            else r_str
            for i, r_str in enumerate(q["Rows"])
        ]
        for col, row_str in zip(q["csv_columns"], rows, strict=True):
            opt1, opt2 = row_str.split(":")
            bipolar_specs.append(
                {
                    "v": col,
                    "label": f"{q['BlockName'].strip()}: {row_str}",
                    "question": q_text,
                    "type": q_type,
                    "option_codes": [1, 2],
                    "option_texts": [opt1, opt2],
                    "raw_to_code": {"1": 1, "2": 2},
                }
            )
    bipolar_long_df = build_long_df(w13_df, bipolar_specs)

    # Family 2: Matrix — Likert
    likert_entries = catalog_df[(catalog_df["family"] == "2_Matrix_Likert") & catalog_df["include"]]
    likert_specs: list[dict[str, Any]] = []
    for _, q in likert_entries.iterrows():
        ov = question_overrides[q["override_key"]] if q["override_key"] in question_overrides else None
        q_text = (
            _normalize_multiline_override(ov["stem"])
            if ov is not None and "stem" in ov
            else " ".join(q["QuestionText"].split())
        )
        drop_set = {x.strip() for x in q["drop_options"].split("|") if x.strip()}
        raw_cols = list(q["Columns"])
        drop_raw = {raw_c for raw_c in raw_cols if " ".join(raw_c.split()) in drop_set}
        kept_raw_cols = [raw_c for raw_c in raw_cols if raw_c not in drop_raw]
        opt_ov = ov["options"] if ov is not None and "options" in ov else {}
        opts = [
            " ".join(opt_ov[" ".join(raw_c.split())].split()) if " ".join(raw_c.split()) in opt_ov else raw_c
            for raw_c in kept_raw_cols
        ]
        if len(opts) < 2 or len(set(opts)) != len(opts):
            raise ValueError(f"{q['QuestionID']}: invalid Likert options after override: {opts}")
        codes = list(range(1, len(opts) + 1))
        q_type = q["kev_type"]
        if q_type not in ("score", "choice"):
            raise ValueError(f"{q['QuestionID']}: unexpected kev_type {q_type!r}")
        raw_to_code = {raw_c: c for c, raw_c in zip(codes, kept_raw_cols, strict=True)}
        rows = [
            _normalize_multiline_override(ov["subitems"][str(i)])
            if ov is not None and "subitems" in ov and str(i) in ov["subitems"]
            else r_str
            for i, r_str in enumerate(q["Rows"])
        ]
        sub_sep = (
            _decode_sep(ov["subitem_sep"], default=DEFAULT_SUBITEM_SEP)
            if ov is not None and "subitem_sep" in ov
            else DEFAULT_SUBITEM_SEP
        )
        for col, row_str in zip(q["csv_columns"], rows, strict=True):
            joined_q = f"{q_text}{sub_sep}{row_str}" if q_text else row_str
            likert_specs.append(
                {
                    "v": col,
                    "label": f"{q['BlockName'].strip()}: {row_str}",
                    "question": joined_q,
                    "type": q_type,
                    "option_codes": codes,
                    "option_texts": opts,
                    "raw_to_code": raw_to_code,
                    "drop_raw": drop_raw,
                }
            )
    likert_long_df = build_long_df(w13_df, likert_specs)

    # Family 3: MC single-select — Product Preferences - Pricing
    pricing_entries = catalog_df[(catalog_df["family"] == "3_MC_Pricing") & catalog_df["include"]]
    if len(pricing_entries) != 40:
        raise ValueError(f"Expected 40 included pricing entries in question_audit.csv, got {len(pricing_entries)}")
    pricing_kev_types = set(pricing_entries["kev_type"].unique())
    if pricing_kev_types != {"choice"}:
        raise ValueError(f"Expected pricing kev_type='choice', got {pricing_kev_types}")
    pricing_ov_key = pricing_entries.iloc[0]["override_key"]
    pricing_opt_ov = (
        question_overrides[pricing_ov_key]["options"]
        if pricing_ov_key in question_overrides and "options" in question_overrides[pricing_ov_key]
        else {}
    )

    wave_split_df = pd.read_parquet(chunks_dir)
    wave_split_df["pid"] = wave_split_df["pid"].astype(int)
    wave_split_df = wave_split_df.set_index("pid").sort_index()
    if not wave_split_df.index.equals(w13_df.index.sort_values()):
        raise ValueError("wave_split_df and w13_df have different pids")

    for pid, j_str in wave_split_df["wave1_3_persona_json"].items():
        for b in json.loads(j_str):
            if b["BlockName"] == "Demographics":
                for q in b["Questions"]:
                    qid = q["QuestionID"]
                    ws_ans = q["Options"][int(q["Answers"]["SelectedByPosition"]) - 1]
                    csv_ans = w13_df.loc[pid, qid]
                    if ws_ans != csv_ans:
                        raise ValueError(f"Demographics mismatch for pid={pid}, {qid}: {ws_ans!r} != {csv_ans!r}")

    pricing_records: list[dict[str, Any]] = []
    for pid, j_str in wave_split_df["wave4_Q_wave1_3_A"].items():
        for b in json.loads(j_str):
            if b["BlockName"] == "Product Preferences - Pricing":
                for q in b["Questions"]:
                    if q["QuestionType"] == "MC":
                        slot_qid = q["QuestionID"]
                        q_text = " ".join(q["QuestionText"].split())
                        raw_opts = list(q["Options"])
                        ans_code = int(q["Answers"]["SelectedByPosition"])
                        raw_ans_text = raw_opts[ans_code - 1]
                        csv_ans = w13_df.loc[pid, slot_qid]
                        if raw_ans_text != csv_ans:
                            raise ValueError(
                                f"Pricing answer mismatch for pid={pid}, {slot_qid}: {raw_ans_text!r} != {csv_ans!r}"
                            )
                        opts = [
                            " ".join(pricing_opt_ov[" ".join(o.split())].split())
                            if " ".join(o.split()) in pricing_opt_ov
                            else o
                            for o in raw_opts
                        ]
                        if len(set(opts)) != len(opts):
                            raise ValueError(f"Pricing duplicate options after override: {opts}")
                        pricing_records.append(
                            {
                                "pid": pid,
                                "slot_qid": slot_qid,
                                "question": q_text,
                                "option_texts": opts,
                                "answer_code": ans_code,
                                "answer_text": opts[ans_code - 1],
                            }
                        )

    raw_pricing_df = pd.DataFrame(pricing_records)
    unique_qtexts = sorted(raw_pricing_df["question"].unique())
    qtext_to_v = {qt: f"QID9_{i:03d}" for i, qt in enumerate(unique_qtexts, 1)}
    raw_pricing_df["v"] = raw_pricing_df["question"].map(qtext_to_v)
    raw_pricing_df = raw_pricing_df.sort_values(["v", "pid"], ignore_index=True)
    raw_pricing_df["label"] = "Product Preferences - Pricing: " + raw_pricing_df["v"]
    raw_pricing_df["type"] = "choice"
    raw_pricing_df["reversed"] = False
    raw_pricing_df["option_codes"] = [[1, 2]] * len(raw_pricing_df)

    pricing_long_df = persona_base_df.loc[raw_pricing_df["pid"]].reset_index(drop=True)
    for c in [
        "v",
        "label",
        "question",
        "type",
        "reversed",
        "option_codes",
        "option_texts",
        "answer_code",
        "answer_text",
    ]:
        pricing_long_df[c] = raw_pricing_df[c].values

    if (pricing_long_df["CASEID"] != ("twin_" + raw_pricing_df["pid"].astype(str))).any():
        raise ValueError("Post-join CASEID misalignment in Pricing export")

    # Family 4: MC single-select
    mc_entries = catalog_df[(catalog_df["family"] == "4_MC_Single") & catalog_df["include"]]
    mc_specs: list[dict[str, Any]] = []
    for _, q in mc_entries.iterrows():
        if len(q["csv_columns"]) != 1:
            raise ValueError(f"{q['QuestionID']}: expected 1 csv_column, got {q['csv_columns']}")
        col = q["csv_columns"][0]
        block = q["BlockName"].strip()
        ov = question_overrides[q["override_key"]] if q["override_key"] in question_overrides else None
        q_text = (
            _normalize_multiline_override(ov["stem"])
            if ov is not None and "stem" in ov
            else " ".join(q["QuestionText"].split())
        )
        drop_set = {x.strip() for x in q["drop_options"].split("|") if x.strip()}
        raw_opts = list(q["Options"])
        drop_raw = {raw_o for raw_o in raw_opts if " ".join(raw_o.split()) in drop_set}
        kept_raw_opts = [raw_o for raw_o in raw_opts if raw_o not in drop_raw]
        opt_ov = ov["options"] if ov is not None and "options" in ov else {}
        opts = [
            " ".join(opt_ov[" ".join(raw_o.split())].split()) if " ".join(raw_o.split()) in opt_ov else raw_o
            for raw_o in kept_raw_opts
        ]
        if len(opts) < 2 or len(set(opts)) != len(opts):
            raise ValueError(f"{q['QuestionID']}: invalid MC options after override: {opts}")
        codes = list(range(1, len(opts) + 1))
        q_type = q["kev_type"]
        if q_type not in ("score", "choice"):
            raise ValueError(f"{q['QuestionID']}: unexpected kev_type {q_type!r}")
        raw_to_code = {raw_o: c for c, raw_o in zip(codes, kept_raw_opts, strict=True)}
        mc_specs.append(
            {
                "v": col,
                "label": f"{block}: {col}",
                "question": q_text,
                "type": q_type,
                "option_codes": codes,
                "option_texts": opts,
                "raw_to_code": raw_to_code,
                "drop_raw": drop_raw,
            }
        )
    mc_long_df = build_long_df(w13_df, mc_specs)

    multi_included = catalog_df[(catalog_df["family"] == "5_MC_Multi") & catalog_df["include"]]
    if len(multi_included) != 0:
        raise ValueError(f"Expected 0 included 5_MC_Multi entries, got {len(multi_included)}")

    long_df = pd.concat([bipolar_long_df, likert_long_df, pricing_long_df, mc_long_df], ignore_index=True)

    expected_cols = (
        ["CASEID"]
        + persona_cols
        + ["v", "label", "question", "type", "reversed", "option_codes", "option_texts", "answer_code", "answer_text"]
    )
    if list(long_df.columns) != expected_cols:
        raise ValueError(f"Twin-2K-500 column mismatch: {list(long_df.columns)} != {expected_cols}")

    if write:
        out_parquet = twin_dir / "twin2k_US_long.parquet"
        long_df.to_parquet(out_parquet, index=False)

    return long_df


def _iter_mega_blocks(elems: list[dict[str, Any]], depth: int = 0):
    for e in elems:
        et = e["ElementType"]
        if et not in MEGA_ALLOWED_ELEMENT_TYPES:
            raise ValueError(f"Unexpected ElementType: {et!r}")
        if et == "Block":
            if tuple(sorted(e.keys())) != MEGA_ALLOWED_BLOCK_KEYS:
                raise ValueError(f"Unexpected Block keys: {sorted(e.keys())}")
            yield e, depth
        else:
            yield from _iter_mega_blocks(e["Elements"], depth + 1)


def _mega_has_answer(qt: str, sel: str, ans: Any) -> bool:
    if qt == "DB" or not ans:
        return False
    if qt == "MC" and sel in ("SAVR", "SAHR", "DL"):
        return ans["SelectedByPosition"] is not None
    if qt == "MC" and sel == "MAVR":
        return bool(ans["SelectedByPosition"])
    if qt == "Matrix":
        return any(x is not None for x in ans["SelectedByPosition"])
    if qt in ("Slider", "CS"):
        return any(x is not None for x in ans["Values"])
    if qt == "TE":
        t = ans["Text"]
        if isinstance(t, list):
            return any(bool(str(x).strip()) for x in t if x is not None)
        return t is not None and bool(str(t).strip())
    raise ValueError(f"Unexpected QuestionType/Selector: {qt}:{sel}")


def _make_mega_base_slot(study: str, qid: str, rid: str | None = None) -> str:
    slug = study.replace("_", "-")
    m = re.match(r"^(\d+)_(QID.+)$", qid)
    if m:
        loop_idx, base_q = int(m.group(1)), m.group(2)
        loop_fmt = f"{loop_idx:03d}" if study == "idea_evaluation" else str(loop_idx)
        slot = f"{slug}-{base_q}_{loop_fmt}"
    else:
        slot = f"{slug}-{qid}"
    if rid is not None:
        slot = f"{slot}_{rid}"
    return slot


def _resolve_mega_context(
    ctx_spec: str,
    study: str,
    qid: str,
    in_block_db_txt: str | None,
    qid_to_text: dict[str, str],
) -> str:
    if ctx_spec == "":
        return ""
    if ctx_spec == "in_block_db":
        if not in_block_db_txt:
            raise ValueError(f"{study}:{qid} requires in_block_db context, but none found in block")
        return in_block_db_txt
    if ctx_spec.startswith("qid:"):
        ref_qids = ctx_spec.removeprefix("qid:").split("+")
        parts = []
        for rq in ref_qids:
            if rq not in qid_to_text or not qid_to_text[rq]:
                raise ValueError(f"{study}:{qid} missing referenced context {rq!r}")
            parts.append(qid_to_text[rq])
        return "\n\n".join(parts)
    if ctx_spec.startswith("prefix:"):
        pfx = ctx_spec.removeprefix("prefix:").strip()
        if not pfx:
            raise ValueError(f"{study}:{qid} empty prefix: context_source")
        return pfx
    raise ValueError(f"{study}:{qid} unsupported context_source={ctx_spec!r}")


def _join_context_and_question(ctx_txt: str, q_txt: str, sep: str) -> str:
    if ctx_txt and q_txt:
        return f"{ctx_txt}{sep}{q_txt}"
    if ctx_txt:
        return ctx_txt
    return q_txt


def _join_matrix_stem_and_row(
    stem: str,
    row_str: str,
    *,
    ctx_txt: str,
    q_txt: str,
    sep_spec: str,
) -> str:
    if not stem:
        return row_str
    if not q_txt and ctx_txt and sep_spec != "":
        return f"{ctx_txt}{_decode_sep(sep_spec)}{row_str}"
    if stem[-1] in TERMINAL_PUNCT:
        return f"{stem} {row_str}"
    return f"{stem}\n\n{row_str}"


def _apply_mega_option_overrides(
    study: str,
    qid: str,
    raw_list: list[str],
    drop_set: set[str],
    opt_overrides: dict[str, str] | None,
) -> tuple[list[str], list[str]]:
    if opt_overrides is not None:
        missing_keys = sorted(set(opt_overrides.keys()) - set(raw_list))
        if missing_keys:
            raise KeyError(f"{study}:{qid} option override keys {missing_keys} not found in raw options {raw_list}")
    kept_raw = [o for o in raw_list if o not in drop_set]
    if opt_overrides is None:
        kept_final = list(kept_raw)
    else:
        kept_final = [_norm_str(opt_overrides[o]) if o in opt_overrides else o for o in kept_raw]
    if len(kept_final) < 2 or len(set(kept_final)) != len(kept_final):
        raise ValueError(f"{study}:{qid} invalid kept options after override: {kept_final}")
    return kept_raw, kept_final


def export_twin2k_megastudy_parquet(
    *,
    data_dir: Path = DATA_DIR,
    write: bool = True,
) -> pd.DataFrame:
    """Export `Twin-2K-500-Mega-Study/twin2k_megastudy_US_long.parquet` with canonical `persona_spec.yaml` personas."""
    mega_dir = data_dir / "Twin-2K-500-Mega-Study"
    raw_data_dir = mega_dir / "hf" / "data"

    parquet_files = sorted(raw_data_dir.glob("*.parquet"))
    if len(parquet_files) != 19:
        raise ValueError(f"Expected 19 split parquet files in {raw_data_dir}, found {len(parquet_files)}")

    study_dfs: dict[str, pd.DataFrame] = {}
    catalog_entries: list[dict[str, Any]] = []

    for p in parquet_files:
        study = p.name.removesuffix("-00000-of-00001.parquet")
        df = pd.read_parquet(p)
        if list(df.columns) != MEGA_EXPECTED_PARQUET_COLS:
            raise ValueError(f"{study}: unexpected columns {list(df.columns)}")
        if df["PID"].isna().any() or df["PID"].duplicated().any():
            raise ValueError(f"{study}: missing or duplicate PIDs")
        study_dfs[study] = df[["PID", "survey_json_with_human_response"]].copy()

        slots: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for raw in df["survey_json_with_human_response"]:
            obj = json.loads(raw)
            if tuple(sorted(obj.keys())) != MEGA_ALLOWED_TOP_KEYS:
                raise ValueError(f"{study}: unexpected top-level JSON keys {sorted(obj.keys())}")
            seen_qids_in_resp: set[str] = set()
            for b, _ in _iter_mega_blocks(obj["Elements"]):
                bname = b["BlockName"]
                for q in b["Questions"]:
                    qid = q["QuestionID"]
                    if qid in seen_qids_in_resp:
                        raise ValueError(f"{study}: duplicate QuestionID {qid!r} within a single respondent")
                    seen_qids_in_resp.add(qid)
                    qt = q["QuestionType"]
                    sel = q["Settings"]["Selector"] if "Settings" in q else ""
                    if qid not in slots:
                        rows = [_norm_str(r) for r in q["Rows"]] if "Rows" in q else []
                        if qt == "Matrix":
                            opts = [_norm_str(c) for c in q["Columns"]]
                            n_sub = len(rows)
                        elif qt == "MC":
                            opts = [_norm_str(o) for o in q["Options"]]
                            n_sub = len(opts) if sel == "MAVR" else 1
                        elif qt == "DB":
                            opts, n_sub = [], 0
                        elif qt == "TE" and sel == "FORM":
                            opts, n_sub = [], len(rows)
                        elif qt in ("Slider", "CS"):
                            stmts = q["Statements"] if "Statements" in q else rows
                            opts, n_sub = [], max(1, len(stmts))
                        else:
                            opts, n_sub = [], 1
                        slots[qid] = {
                            "study": study,
                            "QuestionID": qid,
                            "BlockName": bname,
                            "QuestionType": qt,
                            "Selector": sel,
                            "n_subitems": n_sub,
                            "n_seen": 0,
                            "n_answered": 0,
                            "QuestionText": _clean_mega_text(study, qid, q["QuestionText"]),
                            "Rows": rows,
                            "Options_or_Columns": opts,
                        }
                    else:
                        if (slots[qid]["QuestionType"], slots[qid]["Selector"]) != (qt, sel):
                            raise ValueError(f"{study}:{qid} changed QuestionType/Selector across respondents")
                    slots[qid]["n_seen"] += 1
                    if _mega_has_answer(qt, sel, q["Answers"] if qt != "DB" else None):
                        slots[qid]["n_answered"] += 1

        slot_order = {k: i for i, k in enumerate(slots.keys())}

        def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int]:
            qid, _ = item
            if study == "idea_evaluation" and re.match(r"^\d+_QID", qid):
                num_str, base_q = qid.split("_", 1)
                arm_order = {"QID7": 1, "QID13": 2, "QID14": 3, "QID23": 4, "QID24": 5, "QID25": 6}[base_q]
                return (1, arm_order, int(num_str))
            return (0, 0, slot_order[qid])

        catalog_entries.extend(st for _, st in sorted(slots.items(), key=_sort_key))

    catalog_df = pd.DataFrame(catalog_entries)

    persona_base_df, persona_cols = _load_standardized_twin2k_personas(data_dir)

    # Verify 100% PID linkage against Twin-2K-500 standardized personas.
    for study, df in study_dfs.items():
        for pid_str in df["PID"]:
            if not pid_str.startswith("pid_"):
                raise ValueError(f"{study}: unexpected PID string {pid_str!r}")
            pid_int = int(pid_str.removeprefix("pid_"))
            if pid_int not in persona_base_df.index:
                raise KeyError(f"{study}: {pid_str!r} missing from Twin-2K-500 personas")

    audit_path = mega_dir / "question_audit.csv"
    overrides_path = mega_dir / "question_overrides.json"
    tags_path = mega_dir / "question_tags.json"

    audit_raw_df = pd.read_csv(audit_path, keep_default_na=False)
    question_overrides = _load_json_dict(overrides_path)
    question_tags = _load_json_dict(tags_path)

    valid_override_keys: set[str] = set()
    expanded_rows: list[dict[str, Any]] = []
    for _, r in audit_raw_df.iterrows():
        idx_str = str(r["catalog_idx"])
        row_key = f"{idx_str}:{r['study']}:{r['QuestionID']}"
        valid_override_keys.add(row_key)
        inc = r["include"] == "True" if isinstance(r["include"], str) else bool(r["include"])
        if ".." in idx_str:
            s_idx, e_idx = map(int, idx_str.split(".."))
            if not r["QuestionID"].startswith("1..200_"):
                raise ValueError(f"Unexpected range QuestionID pattern: {r['QuestionID']!r}")
            base_q = r["QuestionID"].split("_", 1)[1]
            if e_idx - s_idx + 1 != 200:
                raise ValueError(f"Expected 200 items in range {idx_str}")
            for offset, idx in enumerate(range(s_idx, e_idx + 1), 1):
                expanded_rows.append(
                    {
                        "catalog_idx": idx,
                        "override_key": row_key,
                        "study": r["study"],
                        "QuestionID": f"{offset}_{base_q}",
                        "family": r["family"],
                        "include": inc,
                        "kev_type": r["kev_type"],
                        "context_source": r["context_source"],
                        "context_sep": r["context_sep"],
                        "drop_options": r["drop_options"],
                    }
                )
        else:
            expanded_rows.append(
                {
                    "catalog_idx": int(idx_str),
                    "override_key": row_key,
                    "study": r["study"],
                    "QuestionID": r["QuestionID"],
                    "family": r["family"],
                    "include": inc,
                    "kev_type": r["kev_type"],
                    "context_source": r["context_source"],
                    "context_sep": r["context_sep"],
                    "drop_options": r["drop_options"],
                }
            )

    unknown_override_keys = sorted(set(question_overrides.keys()) - valid_override_keys)
    if unknown_override_keys:
        raise KeyError(f"Unknown row keys in {overrides_path.name}: {unknown_override_keys}")
    _validate_tags_dict(question_tags, valid_override_keys, tags_path)

    audit_exp_df = pd.DataFrame(expanded_rows).set_index("catalog_idx").sort_index()
    if not audit_exp_df.index.equals(catalog_df.index):
        raise ValueError("Expanded question_audit.csv indices do not match catalog_df.index")
    if not (audit_exp_df["study"] == catalog_df["study"]).all() or not (
        audit_exp_df["QuestionID"] == catalog_df["QuestionID"]
    ).all():
        raise ValueError("Expanded question_audit.csv (study, QuestionID) does not match catalog_df")

    for col in ("override_key", "family", "include", "kev_type", "context_source", "context_sep", "drop_options"):
        catalog_df[col] = audit_exp_df[col]

    for rk, ov in question_overrides.items():
        if not isinstance(ov, dict) or not ov:
            raise ValueError(f"{rk}: override entry must be a non-empty dict")
        extra_fields = set(ov.keys()) - MEGA_ALLOWED_OVERRIDE_FIELDS
        if extra_fields:
            raise ValueError(f"{rk}: unknown override fields {sorted(extra_fields)}")
        matched_cat = catalog_df[catalog_df["override_key"] == rk]
        if matched_cat.empty:
            raise KeyError(f"{rk}: override key did not match any catalog_df row")
        if ".." in rk.split(":", 1)[0] and "stem" in ov:
            raise ValueError(f"{rk}: static stem override is not allowed on collapsed loop rows")
        cat0 = matched_cat.iloc[0]
        if "stem" in ov and not isinstance(ov["stem"], str):
            raise TypeError(f"{rk}: stem override must be str, got {type(ov['stem']).__name__}")
        if "subitems" in ov:
            if not isinstance(ov["subitems"], dict) or not ov["subitems"]:
                raise ValueError(f"{rk}: subitems override must be a non-empty dict")
            n_rows = len(cat0["Rows"])
            if n_rows == 0:
                raise ValueError(f"{rk}: subitems override set on row without Matrix Rows")
            for sub_k, sub_v in ov["subitems"].items():
                if not isinstance(sub_k, str) or not sub_k.isdigit():
                    raise ValueError(f"{rk}: subitem key {sub_k!r} must be a 0-based integer string")
                sub_idx = int(sub_k)
                if not (0 <= sub_idx < n_rows):
                    raise IndexError(f"{rk}: subitem index {sub_idx} out of range (0..{n_rows - 1})")
                if not isinstance(sub_v, str) or not _normalize_multiline_override(sub_v):
                    raise ValueError(f"{rk}: subitem {sub_k!r} override must be a non-empty string")
        if "options" in ov:
            if not isinstance(ov["options"], dict) or not ov["options"]:
                raise ValueError(f"{rk}: options override must be a non-empty dict")
            known_opts = set(cat0["Options_or_Columns"])
            for opt_k, opt_v in ov["options"].items():
                if opt_k not in known_opts:
                    raise KeyError(f"{rk}: option override key {opt_k!r} not in {cat0['Options_or_Columns']}")
                if not isinstance(opt_v, str) or not _norm_str(opt_v):
                    raise ValueError(f"{rk}: option override for {opt_k!r} must be a non-empty string")

    for _, r in catalog_df.iterrows():
        if r["include"]:
            if r["kev_type"] not in ("choice", "score"):
                raise ValueError(f"{r['study']}:{r['QuestionID']} included with invalid kev_type={r['kev_type']!r}")
        else:
            if r["kev_type"] != "":
                raise ValueError(f"{r['study']}:{r['QuestionID']} excluded but has non-empty kev_type={r['kev_type']!r}")
        cs = r["context_source"]
        if cs != "" and cs != "in_block_db" and not cs.startswith("qid:") and not cs.startswith("prefix:"):
            raise ValueError(f"{r['study']}:{r['QuestionID']} invalid context_source={cs!r}")

    non_discrete_included = catalog_df[catalog_df["include"] & (~catalog_df["family"].isin(MEGA_DISCRETE_FAMILIES))]
    if not non_discrete_included.empty:
        raise ValueError(
            f"Unexpected included rows outside discrete families: {non_discrete_included[['study', 'QuestionID', 'family']]}"
        )

    included_audit_lookup = {(r["study"], r["QuestionID"]): r for _, r in catalog_df[catalog_df["include"]].iterrows()}
    stem_override_lookup = {
        (r["study"], r["QuestionID"]): _normalize_multiline_override(question_overrides[r["override_key"]]["stem"])
        for _, r in catalog_df.iterrows()
        if r["override_key"] in question_overrides and "stem" in question_overrides[r["override_key"]]
    }

    def extract_family_long_df(target_family: str) -> pd.DataFrame:
        raw_obs: list[dict[str, Any]] = []
        raw_qtext_variants: dict[tuple[str, str], set[str]] = {}
        target_studies = sorted(
            {s for (s, _), meta in included_audit_lookup.items() if meta["family"] == target_family}
        )
        for study in target_studies:
            df = study_dfs[study]
            for pid_str, raw_json in zip(df["PID"], df["survey_json_with_human_response"], strict=True):
                pid_int = int(pid_str.removeprefix("pid_"))
                obj = json.loads(raw_json)
                blocks = [b for b, _ in _iter_mega_blocks(obj["Elements"])]
                qid_to_text: dict[str, str] = {}
                for b in blocks:
                    for q in b["Questions"]:
                        q_id = q["QuestionID"]
                        raw_clean = _clean_mega_text(study, q_id, q["QuestionText"])
                        if (study, q_id) in stem_override_lookup:
                            seen_set = raw_qtext_variants.setdefault((study, q_id), set())
                            seen_set.add(raw_clean)
                            if len(seen_set) > 1:
                                raise ValueError(
                                    f"{study}:{q_id} has multiple raw QuestionText variants across respondents; "
                                    "static stem override is disallowed"
                                )
                            qid_to_text[q_id] = stem_override_lookup[(study, q_id)]
                        else:
                            qid_to_text[q_id] = raw_clean

                for b in blocks:
                    bname = b["BlockName"].strip()
                    in_block_db_txt: str | None = None
                    for q in b["Questions"]:
                        qid = q["QuestionID"]
                        qt = q["QuestionType"]
                        q_txt = qid_to_text[qid]
                        if qt == "DB":
                            in_block_db_txt = q_txt
                            continue

                        if (study, qid) not in included_audit_lookup:
                            continue
                        meta = included_audit_lookup[(study, qid)]
                        if meta["family"] != target_family:
                            continue

                        ans = q["Answers"]
                        if not ans:
                            continue

                        ov = (
                            question_overrides[meta["override_key"]]
                            if meta["override_key"] in question_overrides
                            else None
                        )
                        opt_ov = ov["options"] if ov is not None and "options" in ov else None
                        sub_ov = ov["subitems"] if ov is not None and "subitems" in ov else None

                        kev_type = meta["kev_type"]
                        drop_set = {x.strip() for x in meta["drop_options"].split("|") if x.strip()}
                        ctx_txt = _resolve_mega_context(
                            meta["context_source"], study, qid, in_block_db_txt, qid_to_text
                        )
                        sep_spec = meta["context_sep"]
                        stem = _join_context_and_question(ctx_txt, q_txt, _decode_sep(sep_spec))
                        if not stem:
                            raise ValueError(f"{study}:{qid} empty question stem")

                        if qt == "MC":
                            pos_raw = ans["SelectedByPosition"]
                            if pos_raw is None:
                                continue
                            pos = int(pos_raw)
                            raw_opts = [_norm_str(o) for o in q["Options"]]
                            sel_raw = _norm_str(ans["SelectedText"])
                            if raw_opts[pos - 1] != sel_raw:
                                raise ValueError(f"{study}:{qid} SelectedByPosition/SelectedText mismatch")
                            if sel_raw in drop_set:
                                continue
                            kept_raw_opts, kept_opts = _apply_mega_option_overrides(
                                study, qid, raw_opts, drop_set, opt_ov
                            )
                            ans_idx = kept_raw_opts.index(sel_raw)
                            raw_obs.append(
                                {
                                    "pid": pid_int,
                                    "base_slot": _make_mega_base_slot(study, qid),
                                    "label_prefix": f"{study} ({bname})",
                                    "label_suffix": qid,
                                    "question": stem,
                                    "type": kev_type,
                                    "option_texts": tuple(kept_opts),
                                    "answer_code": ans_idx + 1,
                                    "answer_text": kept_opts[ans_idx],
                                }
                            )
                        elif qt == "Matrix":
                            sel = q["Settings"]["Selector"]
                            raw_cols = [_norm_str(c) for c in q["Columns"]]
                            raw_rows = [_norm_str(r_str) for r_str in q["Rows"]]
                            if sub_ov is None:
                                rows = raw_rows
                            else:
                                if max(int(k) for k in sub_ov.keys()) >= len(raw_rows):
                                    raise IndexError(
                                        f"{study}:{qid} subitems override index exceeds respondent Rows length {len(raw_rows)}"
                                    )
                                rows = [
                                    _normalize_multiline_override(sub_ov[str(i)]) if str(i) in sub_ov else r_str
                                    for i, r_str in enumerate(raw_rows)
                                ]
                            rows_id = (
                                [str(x) for x in q["RowsID"]] if "RowsID" in q else [str(i + 1) for i in range(len(rows))]
                            )
                            pos_list = ans["SelectedByPosition"]
                            txt_list = ans["SelectedText"]
                            if not (len(rows) == len(rows_id) == len(pos_list) == len(txt_list)):
                                raise ValueError(f"{study}:{qid} Matrix length mismatch")
                            kept_raw_cols, kept_opts = _apply_mega_option_overrides(
                                study, qid, raw_cols, drop_set, opt_ov
                            )

                            if sel == "Bipolar":
                                if len(rows) != 1:
                                    raise ValueError(f"{study}:{qid} expected 1 row in Matrix:Bipolar, got {len(rows)}")
                                if pos_list[0] is None:
                                    continue
                                pos = int(pos_list[0])
                                sel_raw = _norm_str(txt_list[0])
                                if raw_cols[pos - 1] != sel_raw:
                                    raise ValueError(f"{study}:{qid} Bipolar SelectedText mismatch")
                                if sel_raw in drop_set:
                                    continue
                                ans_idx = kept_raw_cols.index(sel_raw)
                                raw_obs.append(
                                    {
                                        "pid": pid_int,
                                        "base_slot": _make_mega_base_slot(study, qid),
                                        "label_prefix": f"{study} ({bname})",
                                        "label_suffix": qid,
                                        "question": stem,
                                        "type": kev_type,
                                        "option_texts": tuple(kept_opts),
                                        "answer_code": ans_idx + 1,
                                        "answer_text": kept_opts[ans_idx],
                                    }
                                )
                            elif sel == "Likert":
                                for rid, row_str, p_val, t_val in zip(rows_id, rows, pos_list, txt_list, strict=True):
                                    if p_val is None:
                                        continue
                                    pos = int(p_val)
                                    sel_raw = _norm_str(t_val)
                                    if raw_cols[pos - 1] != sel_raw:
                                        raise ValueError(f"{study}:{qid}_{rid} Likert SelectedText mismatch")
                                    if sel_raw in drop_set:
                                        continue
                                    ans_idx = kept_raw_cols.index(sel_raw)
                                    raw_obs.append(
                                        {
                                            "pid": pid_int,
                                            "base_slot": _make_mega_base_slot(study, qid, rid),
                                            "label_prefix": f"{study} ({bname})",
                                            "label_suffix": row_str,
                                            "question": _join_matrix_stem_and_row(
                                                stem, row_str, ctx_txt=ctx_txt, q_txt=q_txt, sep_spec=sep_spec
                                            ),
                                            "type": kev_type,
                                            "option_texts": tuple(kept_opts),
                                            "answer_code": ans_idx + 1,
                                            "answer_text": kept_opts[ans_idx],
                                        }
                                    )
                            else:
                                raise ValueError(f"Unsupported Matrix selector: {sel!r}")
                        else:
                            raise ValueError(f"Unsupported QuestionType in included set: {qt!r}")

        obs_df = pd.DataFrame(raw_obs)
        v_map: dict[tuple[str, str, tuple[str, ...]], str] = {}
        for base_slot, grp in obs_df.groupby("base_slot", sort=False):
            variants = sorted(set(zip(grp["question"], grp["option_texts"], strict=True)))
            if len(variants) == 1:
                v_map[(base_slot, variants[0][0], variants[0][1])] = base_slot
            else:
                for v_idx, (q_str, opt_tup) in enumerate(variants, 1):
                    v_map[(base_slot, q_str, opt_tup)] = f"{base_slot}_v{v_idx:03d}"

        obs_df["v"] = [
            v_map[(bs, q_str, opt_tup)]
            for bs, q_str, opt_tup in zip(obs_df["base_slot"], obs_df["question"], obs_df["option_texts"], strict=True)
        ]
        obs_df = obs_df.sort_values(["v", "pid"], ignore_index=True)
        if obs_df.duplicated(subset=["v", "pid"]).any():
            raise ValueError(f"Duplicate (v, pid) in {target_family}")

        out_df = persona_base_df.loc[obs_df["pid"]].reset_index(drop=True)
        out_df["v"] = obs_df["v"]
        out_df["label"] = obs_df["label_prefix"] + ": " + obs_df["label_suffix"]
        out_df["question"] = obs_df["question"]
        out_df["type"] = obs_df["type"]
        out_df["reversed"] = False
        out_df["option_codes"] = [list(range(1, len(opts) + 1)) for opts in obs_df["option_texts"]]
        out_df["option_texts"] = [list(opts) for opts in obs_df["option_texts"]]
        out_df["answer_code"] = obs_df["answer_code"].astype("int64")
        out_df["answer_text"] = obs_df["answer_text"]
        return out_df

    family_frames = [extract_family_long_df(fam) for fam in MEGA_DISCRETE_FAMILIES]
    long_df = pd.concat(family_frames, ignore_index=True)

    expected_cols = (
        ["CASEID"]
        + persona_cols
        + ["v", "label", "question", "type", "reversed", "option_codes", "option_texts", "answer_code", "answer_text"]
    )
    if list(long_df.columns) != expected_cols:
        raise ValueError(f"Mega-Study column mismatch: {list(long_df.columns)} != {expected_cols}")

    if write:
        out_parquet = mega_dir / "twin2k_megastudy_US_long.parquet"
        long_df.to_parquet(out_parquet, index=False)

    return long_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Twin-2K-500 and Twin-2K-500-Mega-Study long parquets.")
    parser.add_argument(
        "--dataset",
        choices=("twin2k", "megastudy", "all"),
        default="all",
        help="Which dataset(s) to export (default: all).",
    )
    args = parser.parse_args()

    if args.dataset in ("twin2k", "all"):
        df = export_twin2k_parquet()
        print(
            f"twin2k_US_long.parquet: {len(df):,} rows x {df.shape[1]} cols | "
            f"{df['CASEID'].nunique():,} respondents | {df['v'].nunique():,} questions"
        )
    if args.dataset in ("megastudy", "all"):
        df = export_twin2k_megastudy_parquet()
        print(
            f"twin2k_megastudy_US_long.parquet: {len(df):,} rows x {df.shape[1]} cols | "
            f"{df['CASEID'].nunique():,} respondents | {df['v'].nunique():,} questions"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Precompute lightweight .audit_details.json caches for the Next.js Question Audit App.

Scans raw survey files (.parquet / question_catalog.json) once per dataset so the
Next.js UI can render full untruncated options, empirical response counts, matrix
row breakdowns, randomized variants, and context candidates with 0ms latency.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import html
import json
import os
from pathlib import Path
import re
import time

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "code" / "data"

BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
AUDIT_DETAILS_FILENAME = ".audit_details.json"
DETAILS_VERSION = 2
MAX_PAIRED_VARIANTS = 200


def norm_str(s: str) -> str:
    return WS_RE.sub(" ", html.unescape(BR_RE.sub(" ", s))).strip()


def discover_dataset_csvs(data_dir: Path) -> list[Path]:
    found: list[Path] = []
    mega = data_dir / "Twin-2K-500-Mega-Study" / "question_audit.csv"
    if mega.is_file():
        found.append(mega)
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir() or child.name == "audit-app" or child.name.startswith("."):
            continue
        for fname in ("question_audit.csv", "question_bank.csv"):
            p = child / fname
            if p.is_file() and p not in found:
                found.append(p)
        for sub in sorted(child.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            for fname in ("question_audit.csv", "question_bank.csv"):
                p = sub / fname
                if p.is_file() and p not in found:
                    found.append(p)
    return found


def make_row_key(schema_kind: str, has_study: bool, rec: dict[str, str]) -> str:
    if schema_kind == "question_audit":
        if has_study:
            return f"{rec['catalog_idx']}:{rec['study']}:{rec['QuestionID']}"
        return f"{rec['catalog_idx']}:{rec['QuestionID']}"
    return rec["v"]


def iter_blocks(elements: list[dict]):
    for el in elements:
        et = el["ElementType"]
        if et in ("Block", "Standard", "Default"):
            if "Questions" in el:
                yield el
        if "Elements" in el and el["Elements"]:
            yield from iter_blocks(el["Elements"])


def parse_mega_study_parquet(pq_path: Path) -> dict:
    df = pd.read_parquet(pq_path, columns=["survey_json_with_human_response"])
    docs = [json.loads(raw_json) for raw_json in df["survey_json_with_human_response"]]
    slots: OrderedDict[str, dict] = OrderedDict()
    db_catalog: OrderedDict[str, dict] = OrderedDict()

    # Pass 1: build study-wide context catalog and identify multi-variant DB blocks.
    for doc in docs:
        for el in iter_blocks(doc["Elements"]):
            bname = norm_str(el["BlockName"]) if "BlockName" in el and el["BlockName"] else ""
            for q in el["Questions"]:
                qid = q["QuestionID"]
                qt = q["QuestionType"]
                txt = norm_str(q["QuestionText"]) if "QuestionText" in q and q["QuestionText"] else ""
                if qt == "DB":
                    if qid not in db_catalog:
                        db_catalog[qid] = {
                            "qid": qid,
                            "block": bname,
                            "texts": Counter(),
                            "is_question": False,
                        }
                    db_catalog[qid]["texts"][txt] += 1
                elif qt in ("MC", "Matrix", "TE", "Slider", "CS") and txt:
                    if qid not in db_catalog:
                        db_catalog[qid] = {
                            "qid": qid,
                            "block": bname,
                            "texts": Counter(),
                            "is_question": True,
                        }
                    db_catalog[qid]["texts"][txt] += 1

    multi_db_qids = {
        qid
        for qid, meta in db_catalog.items()
        if not meta["is_question"] and len(meta["texts"]) > 1
    }

    # Pass 2: collect per-slot distributions and respondent-paired variants.
    for doc in docs:
        resp_multi_dbs: dict[str, str] = {}
        if multi_db_qids:
            for el in iter_blocks(doc["Elements"]):
                for q in el["Questions"]:
                    qid = q["QuestionID"]
                    if qid in multi_db_qids:
                        resp_multi_dbs[qid] = (
                            norm_str(q["QuestionText"])
                            if "QuestionText" in q and q["QuestionText"]
                            else ""
                        )
        qid_ctx_pairs = tuple(sorted(resp_multi_dbs.items()))

        for el in iter_blocks(doc["Elements"]):
            bname = norm_str(el["BlockName"]) if "BlockName" in el and el["BlockName"] else ""
            in_block_db: tuple[str, str] | None = None

            for q in el["Questions"]:
                qid = q["QuestionID"]
                qt = q["QuestionType"]
                sel = (
                    q["Settings"]["Selector"]
                    if "Settings" in q and q["Settings"] and "Selector" in q["Settings"] and q["Settings"]["Selector"]
                    else ""
                )
                txt = norm_str(q["QuestionText"]) if "QuestionText" in q and q["QuestionText"] else ""

                if qt == "DB":
                    in_block_db = (qid, txt)

                if qid not in slots:
                    rows = [norm_str(r) for r in q["Rows"]] if "Rows" in q and q["Rows"] else []
                    if qt == "Matrix":
                        opts = [norm_str(c) for c in q["Columns"]]
                    elif qt == "MC":
                        opts = [norm_str(o) for o in q["Options"]]
                    elif qt in ("Slider", "CS") and "Statements" in q and q["Statements"]:
                        rows = [norm_str(s) for s in q["Statements"]]
                        opts = []
                    else:
                        opts = []
                    slots[qid] = {
                        "QuestionType": qt,
                        "Selector": sel,
                        "BlockName": bname,
                        "rows": rows,
                        "options": opts,
                        "qtext_counts": Counter(),
                        "db_counts": Counter(),
                        "opt_variants": OrderedDict(),
                        "overall_counts": Counter(),
                        "row_counts": [Counter() for _ in rows],
                        "paired_buckets": OrderedDict(),
                        "sample_texts": [],
                    }

                st = slots[qid]
                st["qtext_counts"][txt] += 1
                if in_block_db is not None and qt != "DB":
                    st["db_counts"][in_block_db] += 1

                if qt == "MC":
                    cur_opts = tuple(norm_str(o) for o in q["Options"])
                    st["opt_variants"][cur_opts] = None
                    ib_qid, ib_txt = in_block_db if in_block_db is not None else ("", "")
                    vkey = (ib_qid, ib_txt, txt, cur_opts, qid_ctx_pairs)
                    if vkey not in st["paired_buckets"]:
                        st["paired_buckets"][vkey] = {
                            "n": 0,
                            "in_block_db_qid": ib_qid,
                            "in_block_db_text": ib_txt,
                            "stem": txt,
                            "opts": cur_opts,
                            "qid_ctx_pairs": qid_ctx_pairs,
                            "opt_counts": Counter(),
                            "row_counts": [],
                        }
                    pb = st["paired_buckets"][vkey]
                    pb["n"] += 1
                    ans = q["Answers"]
                    if ans:
                        if sel in ("SAVR", "SAHR", "DL") and ans["SelectedByPosition"] is not None:
                            nt = norm_str(ans["SelectedText"])
                            st["overall_counts"][nt] += 1
                            pb["opt_counts"][nt] += 1
                        elif sel == "MAVR" and ans["SelectedByPosition"]:
                            for stxt in ans["SelectedText"]:
                                nt = norm_str(stxt)
                                st["overall_counts"][nt] += 1
                                pb["opt_counts"][nt] += 1
                elif qt == "Matrix":
                    cur_opts = tuple(norm_str(c) for c in q["Columns"])
                    st["opt_variants"][cur_opts] = None
                    ib_qid, ib_txt = in_block_db if in_block_db is not None else ("", "")
                    vkey = (ib_qid, ib_txt, txt, cur_opts, qid_ctx_pairs)
                    if vkey not in st["paired_buckets"]:
                        st["paired_buckets"][vkey] = {
                            "n": 0,
                            "in_block_db_qid": ib_qid,
                            "in_block_db_text": ib_txt,
                            "stem": txt,
                            "opts": cur_opts,
                            "qid_ctx_pairs": qid_ctx_pairs,
                            "opt_counts": Counter(),
                            "row_counts": [Counter() for _ in st["rows"]],
                        }
                    pb = st["paired_buckets"][vkey]
                    pb["n"] += 1
                    ans = q["Answers"]
                    if ans:
                        for r_i, t_val in enumerate(ans["SelectedText"]):
                            if t_val is not None:
                                nt = norm_str(t_val)
                                st["overall_counts"][nt] += 1
                                pb["opt_counts"][nt] += 1
                                if r_i < len(st["row_counts"]):
                                    st["row_counts"][r_i][nt] += 1
                                    pb["row_counts"][r_i][nt] += 1
                elif qt == "TE":
                    ans = q["Answers"]
                    if ans and len(st["sample_texts"]) < 6:
                        t = ans["Text"]
                        val_str = (
                            " | ".join(str(x) for x in t if x)
                            if isinstance(t, list)
                            else (str(t).strip() if t else "")
                        )
                        if val_str:
                            st["sample_texts"].append(val_str[:220])
                elif qt in ("Slider", "CS"):
                    ans = q["Answers"]
                    if ans and len(st["sample_texts"]) < 6:
                        vals = [v for v in ans["Values"] if v is not None]
                        if vals:
                            st["sample_texts"].append(", ".join(str(v) for v in vals))

    context_candidates = []
    for qid, meta in db_catalog.items():
        top_txt = meta["texts"].most_common(1)[0][0]
        context_candidates.append({
            "qid": qid,
            "block": meta["block"],
            "is_question": bool(meta["is_question"]),
            "n_variants": len(meta["texts"]),
            "preview": top_txt[:180],
            "full_text": top_txt,
        })

    return {"slots": slots, "context_candidates": context_candidates}


def build_mega_study_details(csv_path: Path, df: pd.DataFrame) -> dict:
    dataset_dir = csv_path.parent
    study_cache: dict[str, dict] = {}
    study_contexts: dict[str, list[dict]] = {}
    items: dict[str, dict] = {}

    for _, row in df.iterrows():
        rec = {col: str(row[col]) for col in df.columns}
        key = make_row_key("question_audit", True, rec)
        study = rec["study"]
        qid = rec["QuestionID"]

        if study not in study_cache:
            pq_path = dataset_dir / "hf" / "data" / f"{study}-00000-of-00001.parquet"
            if not pq_path.is_file():
                raise FileNotFoundError(f"Expected Mega-Study split Parquet at {pq_path}")
            study_cache[study] = parse_mega_study_parquet(pq_path)
            study_contexts[study] = study_cache[study]["context_candidates"]

        slots = study_cache[study]["slots"]
        if qid.startswith("1..200_"):
            base_q = qid.split("_", 1)[1]
            loop_qids = [f"{k}_{base_q}" for k in range(1, 201) if f"{k}_{base_q}" in slots]
            if not loop_qids:
                raise KeyError(f"No loop QIDs found for {qid} in study {study}")
            rep = slots[loop_qids[0]]
            pooled_counts: Counter = Counter()
            q_variants = []
            for lq in loop_qids:
                pooled_counts.update(slots[lq]["overall_counts"])
                top_q, cnt = slots[lq]["qtext_counts"].most_common(1)[0]
                if len(q_variants) < 15:
                    q_variants.append({"label": lq, "text": top_q, "n": int(cnt)})
            opts = rep["options"]
            in_block_db_text = (
                rep["db_counts"].most_common(1)[0][0][1] if rep["db_counts"] else ""
            )
            items[key] = {
                "QuestionType": rep["QuestionType"],
                "Selector": rep["Selector"],
                "rows": [],
                "options": [{"option": o, "count": int(pooled_counts[o])} for o in opts],
                "row_breakdowns": [],
                "qtext_variants": q_variants,
                "db_variants": [],
                "opt_variant_sets": [list(opts)],
                "paired_variants": [],
                "paired_variants_total": 0,
                "sample_texts": [],
                "in_block_db_text": in_block_db_text,
            }
            continue

        if qid not in slots:
            raise KeyError(f"QuestionID {qid!r} missing from parsed slots in study {study!r}")

        st = slots[qid]
        all_opts: list[str] = []
        seen_opt_set: set[str] = set()
        for opt_seq in [st["options"], *st["opt_variants"].keys(), st["overall_counts"].keys()]:
            for o in opt_seq:
                if o not in seen_opt_set:
                    seen_opt_set.add(o)
                    all_opts.append(o)

        row_bds = []
        if st["QuestionType"] == "Matrix" and len(st["rows"]) >= 1:
            for r_txt, r_cnt in zip(st["rows"], st["row_counts"]):
                row_bds.append({
                    "row": r_txt,
                    "counts": [int(r_cnt[o]) for o in all_opts],
                })

        cs_raw = rec["context_source"].strip() if "context_source" in rec else ""
        ref_qids = (
            [rq.strip() for rq in cs_raw[4:].split("+") if rq.strip()]
            if cs_raw.startswith("qid:")
            else []
        )

        merged_buckets: OrderedDict[tuple, dict] = OrderedDict()
        for b in st["paired_buckets"].values():
            ctx_dict = dict(b["qid_ctx_pairs"])
            rel_pairs = tuple((rq, ctx_dict[rq]) for rq in ref_qids if rq in ctx_dict)
            mkey = (b["in_block_db_qid"], b["in_block_db_text"], b["stem"], b["opts"], rel_pairs)
            if mkey not in merged_buckets:
                merged_buckets[mkey] = {
                    "n": 0,
                    "in_block_db_qid": b["in_block_db_qid"],
                    "in_block_db_text": b["in_block_db_text"],
                    "stem": b["stem"],
                    "opts": b["opts"],
                    "rel_pairs": rel_pairs,
                    "opt_counts": Counter(),
                    "row_counts": [Counter() for _ in st["rows"]],
                }
            mb = merged_buckets[mkey]
            mb["n"] += b["n"]
            mb["opt_counts"].update(b["opt_counts"])
            for r_i, rc in enumerate(b["row_counts"]):
                mb["row_counts"][r_i].update(rc)

        paired_list = sorted(
            merged_buckets.values(),
            key=lambda b: (-b["n"], b["stem"], b["in_block_db_text"], b["opts"]),
        )
        paired_variants_total = len(paired_list)
        paired_variants_out = (
            [
                {
                    "n": int(b["n"]),
                    "stem": b["stem"],
                    "in_block_db_qid": b["in_block_db_qid"],
                    "in_block_db_text": b["in_block_db_text"],
                    "qid_texts": dict(b["rel_pairs"]),
                    "options": [{"option": o, "count": int(b["opt_counts"][o])} for o in b["opts"]],
                    "row_counts": (
                        [[int(rc[o]) for o in b["opts"]] for rc in b["row_counts"]]
                        if st["QuestionType"] == "Matrix"
                        else []
                    ),
                }
                for b in paired_list[:MAX_PAIRED_VARIANTS]
            ]
            if paired_variants_total > 1
            else []
        )

        matching_buckets = [b for b in paired_list if b["stem"] == rec["QuestionText"]]
        if matching_buckets:
            in_block_db_text = matching_buckets[0]["in_block_db_text"]
        elif paired_list:
            in_block_db_text = paired_list[0]["in_block_db_text"]
        else:
            in_block_db_text = st["db_counts"].most_common(1)[0][0][1] if st["db_counts"] else ""

        items[key] = {
            "QuestionType": st["QuestionType"],
            "Selector": st["Selector"],
            "rows": st["rows"],
            "options": [{"option": o, "count": int(st["overall_counts"][o])} for o in all_opts],
            "row_breakdowns": row_bds,
            "qtext_variants": [
                {"label": f"variant {i + 1}", "text": t, "n": int(c)}
                for i, (t, c) in enumerate(st["qtext_counts"].most_common(15))
            ],
            "db_variants": [
                {"label": db_qid, "text": db_txt, "n": int(c)}
                for (db_qid, db_txt), c in st["db_counts"].most_common(15)
            ],
            "opt_variant_sets": [list(ov) for ov in list(st["opt_variants"].keys())[:8]],
            "paired_variants": paired_variants_out,
            "paired_variants_total": paired_variants_total,
            "sample_texts": st["sample_texts"],
            "in_block_db_text": in_block_db_text,
        }

    return {
        "version": DETAILS_VERSION,
        "schema_kind": "question_audit",
        "study_contexts": study_contexts,
        "items": items,
    }


def build_twin2k_details(csv_path: Path, df: pd.DataFrame) -> dict:
    dataset_dir = csv_path.parent
    cat_path = dataset_dir / "hf" / "question_catalog_and_human_response_csv" / "question_catalog.json"
    w13_path = dataset_dir / "hf" / "question_catalog_and_human_response_csv" / "wave1_3_response_label.csv"
    if not (cat_path.is_file() and w13_path.is_file()):
        raise FileNotFoundError(f"Expected Twin-2K-500 raw files under {dataset_dir / 'hf'}")

    catalog = json.loads(cat_path.read_text(encoding="utf-8"))
    w13 = pd.read_csv(w13_path, dtype=str)

    qid_counts = Counter(e["QuestionID"] for e in catalog)
    dup_qids = {qid for qid, cnt in qid_counts.items() if cnt > 1}

    context_candidates: list[dict] = []
    in_block_by_idx: dict[int, tuple[str, str]] = {}
    cur_blk: str | None = None
    cur_db: tuple[str, str] | None = None

    for idx, entry in enumerate(catalog):
        qid = entry["QuestionID"]
        qt = entry["QuestionType"]
        bname = norm_str(entry["BlockName"]) if "BlockName" in entry and entry["BlockName"] else ""
        txt = norm_str(str(entry["QuestionText"])) if "QuestionText" in entry and entry["QuestionText"] else ""

        if bname != cur_blk:
            cur_blk = bname
            cur_db = None

        if qt == "DB":
            if txt.startswith("This completes this section"):
                cur_db = None
            else:
                cur_db = (qid, txt)
            if txt:
                context_candidates.append({
                    "qid": qid,
                    "block": bname,
                    "is_question": False,
                    "n_variants": 1,
                    "preview": txt[:180],
                    "full_text": txt,
                })
        else:
            if cur_db is not None:
                in_block_by_idx[idx] = cur_db
            if txt and not (196 <= idx <= 234):
                cand_id = f"{idx}:{qid}" if qid in dup_qids else qid
                context_candidates.append({
                    "qid": cand_id,
                    "block": bname,
                    "is_question": True,
                    "n_variants": 40 if idx == 195 else 1,
                    "preview": txt[:180],
                    "full_text": txt,
                })

    items: dict[str, dict] = {}

    for _, row in df.iterrows():
        rec = {col: str(row[col]) for col in df.columns}
        key = make_row_key("question_audit", False, rec)
        idx_spec = rec["catalog_idx"]
        if ".." in idx_spec:
            start_idx, end_idx = (int(x) for x in idx_spec.split("..", 1))
            loop_indices = list(range(start_idx, end_idx + 1))
        else:
            start_idx = int(idx_spec)
            loop_indices = [start_idx]

        entry = catalog[start_idx]
        qt = entry["QuestionType"]
        sel = entry["Settings"]["Selector"] if "Settings" in entry and entry["Settings"] and "Selector" in entry["Settings"] else ""
        rows = entry["Rows"] if "Rows" in entry and entry["Rows"] else (entry["Statements"] if "Statements" in entry and entry["Statements"] else [])
        opts = entry["Options"] if "Options" in entry and entry["Options"] else (entry["Columns"] if "Columns" in entry and entry["Columns"] else [])

        csv_cols: list[str] = []
        q_variants: list[dict] = []
        for l_idx in loop_indices:
            l_entry = catalog[l_idx]
            l_cols = l_entry["csv_columns"] if "csv_columns" in l_entry and l_entry["csv_columns"] else []
            csv_cols.extend(l_cols)
            if len(loop_indices) > 1 and len(q_variants) < 15:
                l_txt = norm_str(str(l_entry["QuestionText"])) if "QuestionText" in l_entry and l_entry["QuestionText"] else ""
                l_n = int(w13[l_cols[0]].notna().sum()) if l_cols and l_cols[0] in w13.columns else 0
                q_variants.append({
                    "label": l_entry["QuestionID"],
                    "text": l_txt,
                    "n": l_n,
                })

        pooled: Counter = Counter()
        for col in csv_cols:
            if col in w13.columns:
                pooled.update(w13[col].dropna().tolist())

        row_bds = []
        if qt == "Matrix" and sel == "Bipolar" and rows and ":" in rows[0]:
            opt1, opt2 = rows[0].split(":", 1)
            opt_counts = [
                {"option": f"1: {opt1}", "count": int(pooled["1"])},
                {"option": f"2: {opt2}", "count": int(pooled["2"])},
            ]
            if len(rows) == len(csv_cols):
                for r_txt, col in zip(rows, csv_cols):
                    col_cnt = Counter(w13[col].dropna().tolist()) if col in w13.columns else Counter()
                    row_bds.append({
                        "row": str(r_txt),
                        "counts": [int(col_cnt["1"]), int(col_cnt["2"])],
                    })
        else:
            opt_counts = [{"option": str(o), "count": int(pooled[str(o)])} for o in opts]
            if qt == "Matrix" and rows and len(rows) == len(csv_cols):
                for r_txt, col in zip(rows, csv_cols):
                    col_cnt = Counter(w13[col].dropna().tolist()) if col in w13.columns else Counter()
                    row_bds.append({
                        "row": str(r_txt),
                        "counts": [int(col_cnt[str(o)]) for o in opts],
                    })

        sample_texts: list[str] = []
        if qt in ("TE", "Slider") and csv_cols:
            present_cols = [c for c in csv_cols if c in w13.columns]
            if present_cols:
                sub_df = w13[present_cols].dropna(how="all")
                for _, srow in sub_df.head(20).iterrows():
                    vals = [norm_str(str(v)) for v in srow.tolist() if pd.notna(v) and str(v).strip()]
                    if vals:
                        sample_texts.append(" | ".join(vals)[:220])
                        if len(sample_texts) >= 6:
                            break

        ib = in_block_by_idx.get(start_idx)
        n_ans = int(w13[[c for c in csv_cols if c in w13.columns]].notna().any(axis=1).sum()) if any(c in w13.columns for c in csv_cols) else 0
        db_variants = (
            [{"label": ib[0], "text": ib[1], "n": n_ans}]
            if ib is not None
            else []
        )
        in_block_db_text = ib[1] if ib is not None else ""
        opt_variant_sets = [[o["option"] for o in opt_counts]] if opt_counts else []

        items[key] = {
            "QuestionType": qt,
            "Selector": sel,
            "rows": [str(r) for r in rows],
            "options": opt_counts,
            "row_breakdowns": row_bds,
            "qtext_variants": q_variants,
            "db_variants": db_variants,
            "opt_variant_sets": opt_variant_sets,
            "paired_variants": [],
            "paired_variants_total": 0,
            "sample_texts": sample_texts,
            "in_block_db_text": in_block_db_text,
        }

    return {
        "version": DETAILS_VERSION,
        "schema_kind": "question_audit",
        "study_contexts": {"__all__": context_candidates},
        "items": items,
    }


def build_question_bank_details(csv_path: Path, df: pd.DataFrame) -> dict:
    dataset_dir = csv_path.parent
    pqs = sorted(dataset_dir.glob("*_US_long.parquet"))
    long_counts: dict[str, list[tuple[str, int]]] = {}
    if pqs:
        ldf = pd.read_parquet(pqs[0], columns=["v", "option_texts", "answer_text"])
        for v_val, grp in ldf.groupby("v", sort=False):
            opts = list(grp["option_texts"].iloc[0])
            vc = grp["answer_text"].value_counts()
            long_counts[str(v_val)] = [(str(o), int(vc[o]) if o in vc.index else 0) for o in opts]

    items: dict[str, dict] = {}
    for _, row in df.iterrows():
        rec = {col: str(row[col]) for col in df.columns}
        v_id = rec["v"]
        if v_id in long_counts:
            opt_counts = [{"option": o, "count": c} for o, c in long_counts[v_id]]
        else:
            raw_opts = [o.strip() for o in rec["options"].split("|") if o.strip()]
            opt_counts = [{"option": o, "count": 0} for o in raw_opts]

        items[v_id] = {
            "QuestionType": "bank_item",
            "Selector": "",
            "rows": [],
            "options": opt_counts,
            "row_breakdowns": [],
            "qtext_variants": [],
            "db_variants": [],
            "opt_variant_sets": [],
            "paired_variants": [],
            "paired_variants_total": 0,
            "sample_texts": [],
            "in_block_db_text": "",
        }

    return {
        "version": DETAILS_VERSION,
        "schema_kind": "question_bank",
        "study_contexts": {},
        "items": items,
    }


def build_grummon2023_details(csv_path: Path) -> dict:
    import sys

    g23_dir = str(csv_path.parent.resolve())
    if g23_dir not in sys.path:
        sys.path.insert(0, g23_dir)
    import stimulus as g23_stimulus

    _, _, details_payload = g23_stimulus.build_survey_audit_artifacts()
    return details_payload


def index_csv(csv_path: Path) -> Path:
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    cols = set(df.columns)
    if {"catalog_idx", "QuestionID", "include", "kev_type"}.issubset(cols):
        if "study" in cols:
            payload = build_mega_study_details(csv_path, df)
        elif csv_path.parent.name == "grummon2023":
            payload = build_grummon2023_details(csv_path)
        else:
            payload = build_twin2k_details(csv_path, df)
    elif {"v", "type", "question", "options"}.issubset(cols):
        payload = build_question_bank_details(csv_path, df)
    else:
        raise ValueError(f"Unsupported CSV schema in {csv_path}: {list(df.columns)}")

    out_path = csv_path.parent / AUDIT_DETAILS_FILENAME
    tmp_path = csv_path.parent / f"{AUDIT_DETAILS_FILENAME}.tmp"
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    dt = time.perf_counter() - t0
    size_kb = out_path.stat().st_size / 1024.0
    print(f"Indexed {csv_path.parent.name}/{csv_path.name}: {len(payload['items'])} items -> {size_kb:.1f} KB ({dt:.2f}s)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build .audit_details.json caches for Next.js Audit UI")
    parser.add_argument("--dataset-dir", type=Path, help="Optional single dataset directory to index")
    args = parser.parse_args()

    if args.dataset_dir is not None:
        d = args.dataset_dir.resolve()
        targets = [d / f for f in ("question_audit.csv", "question_bank.csv") if (d / f).is_file()]
        if not targets:
            raise FileNotFoundError(f"No question_audit.csv or question_bank.csv found in {d}")
    else:
        targets = discover_dataset_csvs(DATA_DIR)

    for csv_path in targets:
        index_csv(csv_path)


if __name__ == "__main__":
    main()

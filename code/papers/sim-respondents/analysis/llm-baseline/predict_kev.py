"""Dedicated GPU inference script for released base KEV models on the 200-respondent `Vaccines` subsample.

Saves per-task choice probabilities (using each checkpoint's shipped temperature) to `llm-baseline/preds/` so
`01-gemini-vs-kev-covid-ols.ipynb` runs strictly on CPU without loading models.

Usage (run only when regenerating predictions on GPU):
    .venv-cu13/bin/python code/papers/sim-respondents/analysis/llm-baseline/predict_kev.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/home/dw/github/decision-models-finetuning")
CODE_DIR = REPO_ROOT / "code"
PAPER_DIR = CODE_DIR / "papers" / "sim-respondents"
SRC_DIR = PAPER_DIR / "src"
OUT_DIR = PAPER_DIR / "analysis" / "llm-baseline"
PREDS_DIR = OUT_DIR / "preds"
GEMINI_DIR = CODE_DIR / "proprietary-llms" / "gemini" / "data_sample1000"

for p in (str(CODE_DIR), str(CODE_DIR / "kev"), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import beta_ols.covid as ds_covid
from data.base import build_long_records, fused_eval_context, group_records_by_state, make_view, predict_probs_grouped
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint
from kev.model import encode as kev_encode, load_tokenizer as kev_load_tokenizer
from models.kev_pointer import KevPointerModel
from training import free_gpu

KEV_BASE_MODELS = [
    ("kev-0.8b", "KEV-0.8B", "jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f"),
    ("kev-4b",   "KEV-4B",   "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"),
    ("kev-9b",   "KEV-9B",   "jaredpalmer/kev-9b@db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"),
]


def load_verified_subsample() -> pd.DataFrame:
    """Load the exact 200-respondent (1,000-task, seed=42) `covid` subsample and verify against Gemini files."""
    full_df = ds_covid.load_choice_df().copy()
    full_df["respondent"] = full_df["respondent"].astype(int)
    full_df["pair_id"] = full_df.groupby("respondent").cumcount() + 1
    if len(full_df) != 9855 or full_df["respondent"].nunique() != 1971:
        raise ValueError(f"Unexpected covid choice_df shape: {len(full_df)} rows")

    all_resps = np.sort(full_df["respondent"].unique())
    sampled_resps = np.sort(np.random.default_rng(42).choice(all_resps, size=200, replace=False))
    sampled_set = {int(r) for r in sampled_resps}

    ref_file = GEMINI_DIR / "gemini-3.5-flash-lite_MINIMAL_persona.jsonl"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference Gemini file not found: {ref_file}")
    ref_resps = {int(json.loads(line)["respondent"]) for line in ref_file.read_text().splitlines()}
    if ref_resps != sampled_set:
        raise ValueError("Seed=42 respondent subsample does not match Gemini data_sample1000 respondent IDs")

    sub_df = full_df[full_df["respondent"].isin(sampled_set)].reset_index(drop=True)
    if len(sub_df) != 1000 or sub_df["respondent"].nunique() != 200:
        raise ValueError(f"Expected 1,000 tasks across 200 respondents, got {len(sub_df)}")
    return sub_df


def align_bundle_preds_to_df(bundle: dict, preds: list, df_tasks: pd.DataFrame) -> np.ndarray:
    """Align state-grouped bundle predictions back to `df_tasks` row order via (CASEID, qid, occurrence)."""
    by_key: dict[tuple[str, str, int], np.ndarray] = {}
    occ_seen: dict[tuple[str, str], int] = {}
    for rec, pred in zip(bundle["records"], preds, strict=True):
        k2 = (str(rec["CASEID"]), str(rec["qid"]))
        occ = occ_seen.get(k2, 0)
        occ_seen[k2] = occ + 1
        arr = np.asarray(pred, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Expected 3 option probabilities for {k2}, got shape {arr.shape}")
        by_key[(*k2, occ)] = arr

    aligned = []
    occ_df: dict[tuple[str, str], int] = {}
    for r in df_tasks.itertuples(index=False):
        k2 = (str(r.CASEID), str(r.qid))
        occ = occ_df.get(k2, 0)
        occ_df[k2] = occ + 1
        key = (*k2, occ)
        if key not in by_key:
            raise KeyError(f"Missing bundle prediction for {key}")
        aligned.append(by_key[key])
    return np.vstack(aligned)


def softmax_from_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Analytical rescaling softmax(log(p) / T) for fail-fast cross-checking against PointerHead temperature."""
    log_p = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run released base KEV-0.8B/4B/9B inference on the 200-respondent Vaccines subsample."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only check: print the exact KEV state/options input for a sample respondent and exit without GPU.",
    )
    parser.add_argument("--respondent", type=int, default=None, help="Respondent ID to inspect in --dry-run.")
    parser.add_argument("--pair-id", type=int, default=1, choices=(1, 2, 3, 4, 5), help="Pair ID (1-5) for --dry-run.")
    args = parser.parse_args()

    sub_df = load_verified_subsample()

    records_sub = build_long_records(
        sub_df,
        persona_cols=ds_covid.PERSONA_COLS,
        include_country=False,
        year=ds_covid.NAME,
    )
    bundle_sub = group_records_by_state(records_sub, view_fn=make_view, name="covid_sample200", valid=True)

    if args.dry_run:
        target_resp = args.respondent if args.respondent is not None else int(sub_df["respondent"].iloc[0])
        matches = sub_df[(sub_df["respondent"] == target_resp) & (sub_df["pair_id"] == args.pair_id)]
        if len(matches) != 1:
            raise ValueError(f"Respondent {target_resp} pair_id {args.pair_id} not found in 200-respondent subsample")
        row = matches.iloc[0]
        rec_idx = next(
            i
            for i, r in enumerate(bundle_sub["records"])
            if str(r["CASEID"]) == str(row["CASEID"]) and str(r["qid"]) == str(row["qid"])
        )
        view_idx, q_idx = bundle_sub["unpack_idx"][rec_idx][0]
        pv = bundle_sub["packed_views"][view_idx]
        q_key, q_obj = list(pv["questions"].items())[q_idx]
        single_req = SystemOneRequest.model_validate(
            {"state": pv["state"], "questions": {q_key: {k: v for k, v in q_obj.items() if k in ("type", "instructions", "criteria")}}}
        )
        kev_rec, _ = to_record(single_req)

        print(f"=== KEV DRY RUN (CPU-only) | respondent={target_resp} pair_id={args.pair_id} CASEID={row['CASEID']} qid={q_key} ===")
        print("\n--- KEV checkpoints & shipped calibrated temperatures ---")
        first_ckpt = None
        for _, display_name, repo_ref in KEV_BASE_MODELS:
            ckpt = Checkpoint(repo_ref)
            if first_ckpt is None:
                first_ckpt = ckpt
            ckpt_meta = ckpt.meta
            print(f"  {display_name:8s} | T_cal={float(ckpt_meta.temperature):.4f} | lora={ckpt_meta.lora} | head_dim={ckpt_meta.head_dim}")

        tok = kev_load_tokenizer(first_ckpt.path)
        enc = kev_encode(tok, kev_rec)
        decoded_exact = tok.decode(enc["ids"], skip_special_tokens=False)
        print(f"\n--- EXACT KEV token sequence (single question view, {len(enc['ids'])} tokens) ---")
        print(decoded_exact)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to run predict_kev.py")

    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    meta_out: dict[str, dict] = {
        "dataset": "covid",
        "n_respondents": 200,
        "n_tasks": len(sub_df),
        "subsample_seed": 42,
        "weights_dtype": "bf16",
        "models": {},
    }

    for slug, display_name, repo_ref in KEV_BASE_MODELS:
        ckpt_meta = Checkpoint(repo_ref).meta
        calibrated_temp = float(ckpt_meta.temperature)
        if not np.isfinite(calibrated_temp) or calibrated_temp <= 0.0:
            raise ValueError(f"{display_name}: invalid calibrated temperature {calibrated_temp}")

        print(f"Loading {display_name} ({repo_ref}) | shipped calibrated T={calibrated_temp:.4f}...")
        model = KevPointerModel.build(repo_ref, device="cuda", weights_dtype="bf16")
        model.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), fused_eval_context(
                model, eval_dtype=torch.bfloat16
            ):
                model.net.head.temperature = calibrated_temp
                preds_raw, _ = predict_probs_grouped(
                    model,
                    bundle_sub,
                    max_questions=512,
                    max_chars=120_000,
                    show_progress=False,
                    return_stats=True,
                )
        finally:
            del model
            gc.collect()
            free_gpu()

        p_kev = align_bundle_preds_to_df(bundle_sub, preds_raw, sub_df)

        out_df = pd.DataFrame(
            {
                "CASEID": sub_df["CASEID"].astype(str),
                "respondent": sub_df["respondent"].astype(int),
                "pair_id": sub_df["pair_id"].astype(int),
                "qid": sub_df["qid"].astype(str),
                "p_0": p_kev[:, 0],
                "p_1": p_kev[:, 1],
                "p_2": p_kev[:, 2],
            }
        )
        out_path = PREDS_DIR / f"{slug}_covid_sample200.parquet"
        out_df.to_parquet(out_path, index=False)
        meta_out["models"][slug] = {
            "display_name": display_name,
            "repo_ref": repo_ref,
            "calibrated_temperature": calibrated_temp,
            "lora_rank": int(ckpt_meta.lora),
            "head_dim": int(ckpt_meta.head_dim),
            "file": out_path.name,
        }
        print(
            f"  Saved {out_path.name} ({len(out_df)} rows) | "
            f"T={calibrated_temp:.4f} | opt_out={p_kev[:, 2].mean() * 100:.1f}%"
        )

    meta_path = PREDS_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2) + "\n")
    print(f"\nWrote metadata to {meta_path}")


if __name__ == "__main__":
    main()

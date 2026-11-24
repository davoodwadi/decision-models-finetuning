"""OpenAI-compatible API client for Qwen3.5 inference on the 200-respondent `Vaccines` subsample.

Talks to an already-running `transformers serve` (or vLLM / SGLang) server using Qwen3.5's official sampling
parameters from `qwen3.5-huggingface.md`:
  - Thinking mode (general tasks):
      temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
      chat_template_kwargs={"enable_thinking": True}
  - Non-thinking mode (general tasks):
      temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
      chat_template_kwargs={"enable_thinking": False}

Usage:
    # 1. Start server (example):
    transformers serve Qwen/Qwen3.5-0.8B --port 8000 --continuous-batching --dtype bfloat16

    # 2. Run this client:
    .venv-cu13/bin/python code/papers/sim-respondents/analysis/llm-baseline/predict_qwen.py \
        --model Qwen/Qwen3.5-9B --personas respondent random --thinking off on --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[5]
CODE_DIR = REPO_ROOT / "code"
PAPER_DIR = CODE_DIR / "papers" / "sim-respondents"
SRC_DIR = PAPER_DIR / "src"
OUT_DIR = PAPER_DIR / "analysis" / "llm-baseline"
PREDS_DIR = OUT_DIR / "preds"
GEMINI_DIR = CODE_DIR / "proprietary-llms" / "gemini" / "data_sample1000"
STATA_PATH = CODE_DIR / "data" / "covid" / "dataverse_files" / "Kreps_etal_vax_replication_data.dta"

for p in (str(CODE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.persona import DATASET_PERSONA_COLS, PERSONA_NAMES, apply_persona_mapping

FDA_WANG_TEXT_MAP = {
    "The vaccine has been approved and licensed by the US Food and Drug Administration.": (
        "Approved and licensed by the US Food and Drug Administration."
    ),
    (
        "The vaccine has received an emergency use authorization from the US Food and Drug Administration.  "
        "This allows the expedited use of promising drugs that the FDA has found it reasonable to believe "
        "may be effective in combatting the virus"
    ): (
        "Received an emergency use authorization from the US Food and Drug Administration.  "
        "This allows the expedited use of promising drugs that the FDA has found it reasonable "
        "to believe may be effective in combatting the virus."
    ),
}

RANDOM_SYSTEM_PROMPT = (
    "You should act like a random person deciding whether or not to receive a COVID-19 vaccination."
)

VALID_LETTERS = {"A", "B", "C"}

# Official Qwen3.5 sampling parameters (qwen3.5-huggingface.md, General Tasks)
OFFICIAL_SAMPLING = {
    "on": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "off": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
}


def format_wang_option(letter: str, row: dict) -> str:
    """Format a single vaccine option (A or B) using Wang (2026) Section 5.2.1 minimalist feature wording."""
    fda_raw = row["fda"]
    if fda_raw not in FDA_WANG_TEXT_MAP:
        raise ValueError(f"Unexpected fda text: {fda_raw!r}")
    fda_text = FDA_WANG_TEXT_MAP[fda_raw]
    return (
        f"{letter}: Efficacy: {row['efficacy']}. "
        f"Protective duration: {row['duration']}. "
        f"Major side effect: {row['majorside']}. "
        f"Minor side effect: {row['minorside']}. "
        f"Authorization: {fda_text} "
        f"Origin: {row['origin']}. "
        f"Endorsement: {row['endorsed']}."
    )


def build_options_block(row_a: dict, row_b: dict) -> str:
    opt_a = format_wang_option("A", row_a)
    opt_b = format_wang_option("B", row_b)
    return f"There are three options for you:\n{opt_a}\n{opt_b}\nC: You choose neither A nor B."


def build_verified_prompts() -> tuple[list[tuple[int, int]], dict[str, list[dict]]]:
    """Build exact Gemini Argon random & respondent prompts on the verified 200-respondent subsample."""
    if not STATA_PATH.exists():
        raise FileNotFoundError(f"Missing Kreps Stata file: {STATA_PATH}")

    raw_df = pd.read_stata(STATA_PATH, convert_categoricals=False)
    if len(raw_df) != 19710 or raw_df["respondent"].nunique() != 1971:
        raise ValueError(f"Unexpected Stata shape: {raw_df.shape}")
    raw_df["respondent"] = raw_df["respondent"].astype(int)
    raw_df["pair_id"] = raw_df["choice_set"] // 10
    raw_df["option_ab"] = raw_df["choice_set"] % 10
    if set(raw_df["pair_id"].unique()) != {1, 2, 3, 4, 5} or set(raw_df["option_ab"].unique()) != {1, 2}:
        raise ValueError("Unexpected pair_id or option_ab values in Kreps Stata file")
    eval_df = raw_df.sort_values(["respondent", "pair_id", "option_ab"]).reset_index(drop=True)

    resp_raw = eval_df[eval_df["choice_set"] == 11].sort_values("respondent").reset_index(drop=True)
    persona_cols = DATASET_PERSONA_COLS["covid"]
    resp_persona_df = apply_persona_mapping(resp_raw, "covid")[["respondent", *persona_cols]]
    resp_state_by_id: dict[int, str] = {}
    for row in resp_persona_df.to_dict("records"):
        lines = [
            f"{PERSONA_NAMES[col]}: {row[col]}"
            for col in persona_cols
            if row[col] is not None and pd.notna(row[col])
        ]
        resp_state_by_id[int(row["respondent"])] = "\n".join(lines)

    all_resps = np.sort(eval_df["respondent"].unique())
    sampled_resps = np.sort(np.random.default_rng(42).choice(all_resps, size=200, replace=False))
    sampled_set = {int(r) for r in sampled_resps}

    ref_file = GEMINI_DIR / "gemini-3.5-flash-lite_MINIMAL_persona.jsonl"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference Gemini file not found: {ref_file}")
    ref_resps = {int(json.loads(line)["respondent"]) for line in ref_file.read_text().splitlines()}
    if ref_resps != sampled_set:
        raise ValueError("Seed=42 subsample does not match Gemini data_sample1000 respondent IDs")

    df_a = eval_df[eval_df["option_ab"] == 1].set_index(["respondent", "pair_id"])
    df_b = eval_df[eval_df["option_ab"] == 2].set_index(["respondent", "pair_id"])

    task_keys = [(int(resp), pair_id) for resp in sampled_resps for pair_id in (1, 2, 3, 4, 5)]
    if len(task_keys) != 1000:
        raise ValueError(f"Expected 1,000 tasks, got {len(task_keys)}")

    prompts_by_persona: dict[str, list[dict]] = {"random": [], "respondent": []}
    for resp_id, pair_id in task_keys:
        key = (resp_id, pair_id)
        row_a = df_a.loc[key].to_dict()
        row_b = df_b.loc[key].to_dict()
        options_block = build_options_block(row_a, row_b)

        prompts_by_persona["random"].append(
            {
                "respondent": resp_id,
                "pair_id": pair_id,
                "system_prompt": RANDOM_SYSTEM_PROMPT,
                "user_prompt": (
                    f"{options_block}\n\n"
                    "Which option would you choose as a random person? Your response should be a single letter A, B, or C."
                ),
            }
        )
        persona_state = resp_state_by_id[resp_id]
        prompts_by_persona["respondent"].append(
            {
                "respondent": resp_id,
                "pair_id": pair_id,
                "system_prompt": (
                    "You should act like a person with the following characteristics deciding whether or not "
                    f"to receive a COVID-19 vaccination:\n{persona_state}"
                ),
                "user_prompt": (
                    f"{options_block}\n\n"
                    "Which option would you choose as this person? Your response should be a single letter A, B, or C."
                ),
            }
        )

    return task_keys, prompts_by_persona


def build_request_kwargs(
    item: dict, *, model_id: str, think_mode: str, max_tokens: int | None, backend: str = "vllm"
) -> dict:
    """Build `client.chat.completions.create(...)` kwargs using Qwen3.5's official sampling parameters."""
    params = OFFICIAL_SAMPLING[think_mode]
    extra_body: dict = {"chat_template_kwargs": {"enable_thinking": think_mode == "on"}}
    if backend == "vllm":
        extra_body.update(
            {
                "top_k": params["top_k"],
                "min_p": params["min_p"],
                "repetition_penalty": params["repetition_penalty"],
            }
        )
    kwargs: dict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["user_prompt"]},
        ],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "presence_penalty": params["presence_penalty"],
        "extra_body": extra_body,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def extract_reasoning_and_answer(message, *, enable_thinking: bool) -> tuple[str, str]:
    """Extract (reasoning_text, raw_response) whether the server splits reasoning_content or emits inline <think>."""
    content = (message.content or "").strip()
    reasoning_field = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if reasoning_field is not None:
        return str(reasoning_field).strip(), content

    if "</think>" in content:
        think_part, ans_part = content.split("</think>", 1)
        return think_part.removeprefix("<think>").strip(), ans_part.strip()
    if enable_thinking and content.startswith("<think>"):
        return content.removeprefix("<think>").strip(), ""
    return "", content


def load_existing_keys(jsonl_path: Path, expected_model: str, expected_thinking: str) -> set[tuple[int, int]]:
    """Read already-generated (respondent, pair_id) keys so interrupted runs resume cleanly."""
    seen: set[tuple[int, int]] = set()
    if not jsonl_path.exists():
        return seen
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["model"] != expected_model or rec["thinking_level"] != expected_thinking:
                raise ValueError(
                    f"{jsonl_path}:{line_num} metadata mismatch: "
                    f"({rec['model']!r}, {rec['thinking_level']!r}) != ({expected_model!r}, {expected_thinking!r})"
                )
            seen.add((int(rec["respondent"]), int(rec["pair_id"])))
    return seen


def slugify_model(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(" ", "_")


def collect_run(
    client: OpenAI,
    prompts: list[dict],
    out_path: Path,
    *,
    model_id: str,
    think_mode: str,
    persona_kind: str,
    max_tokens: int | None,
    backend: str,
    concurrency: int,
    overwrite: bool,
) -> dict:
    """Query the OpenAI-compatible server concurrently and persist each task progressively to JSONL."""
    enable_thinking = think_mode == "on"
    thinking_label = "THINKING_ON" if enable_thinking else "THINKING_OFF"

    if overwrite and out_path.exists():
        out_path.unlink()

    seen_keys = load_existing_keys(out_path, model_id, thinking_label)
    pending = [p for p in prompts if (p["respondent"], p["pair_id"]) not in seen_keys]

    if pending:
        write_lock = threading.Lock()
        n_valid_live = 0
        n_invalid_live = 0

        def _call_one(item: dict) -> dict:
            kwargs = build_request_kwargs(
                item, model_id=model_id, think_mode=think_mode, max_tokens=max_tokens, backend=backend
            )
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            reasoning_text, raw_response = extract_reasoning_and_answer(
                choice.message, enable_thinking=enable_thinking
            )
            usage = resp.usage
            prompt_tokens = int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else 0
            output_tokens = int(usage.completion_tokens) if usage and usage.completion_tokens is not None else 0
            total_tokens = int(usage.total_tokens) if usage and usage.total_tokens is not None else (prompt_tokens + output_tokens)
            return {
                "respondent": int(item["respondent"]),
                "pair_id": int(item["pair_id"]),
                "model": model_id,
                "thinking_level": thinking_label,
                "sampling": OFFICIAL_SAMPLING[think_mode],
                "raw_response": raw_response,
                "reasoning_text": reasoning_text,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        desc = f"{model_id.split('/')[-1]} [{think_mode}/{persona_kind}]"
        pbar = tqdm(total=len(prompts), initial=len(seen_keys), desc=desc, unit="task", dynamic_ncols=True)
        try:
            with out_path.open("a", encoding="utf-8") as fout:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(_call_one, item) for item in pending]
                    for fut in as_completed(futures):
                        rec = fut.result()
                        with write_lock:
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fout.flush()
                            if rec["raw_response"].strip() in VALID_LETTERS:
                                n_valid_live += 1
                            else:
                                n_invalid_live += 1
                            pbar.update(1)
                            pbar.set_postfix(valid=n_valid_live, inv=n_invalid_live)
        finally:
            pbar.close()

    all_rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(all_rows) != len(prompts):
        raise ValueError(f"{out_path.name}: expected {len(prompts)} rows, found {len(all_rows)}")
    valid_count = sum(1 for r in all_rows if r["raw_response"].strip() in VALID_LETTERS)
    invalid_count = len(all_rows) - valid_count
    opt_c = sum(1 for r in all_rows if r["raw_response"].strip() == "C")
    return {
        "n_total": len(all_rows),
        "n_valid": valid_count,
        "n_invalid": invalid_count,
        "opt_out_pct_of_valid": (opt_c / valid_count * 100.0) if valid_count > 0 else float("nan"),
        "mean_output_tokens": float(np.mean([r["output_tokens"] for r in all_rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Qwen3.5 predictions via OpenAI-compatible server (transformers serve / vLLM / SGLang)."
    )
    parser.add_argument("--model", required=True, help="Served model ID (e.g. Qwen/Qwen3.5-9B).")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible API base URL (default: $OPENAI_BASE_URL or http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key (default: $OPENAI_API_KEY or EMPTY).",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        choices=("respondent", "random"),
        default=["respondent", "random"],
        help="Prompt persona conditions to collect.",
    )
    parser.add_argument(
        "--thinking",
        nargs="+",
        choices=("off", "on"),
        default=["off", "on"],
        help="Thinking modes to collect using official Qwen3.5 sampling parameters.",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "transformers"),
        default="vllm",
        help="Serving backend: `vllm` (default, sends top_k/min_p/repetition_penalty) or `transformers`.",
    )
    parser.add_argument("--max-tokens-off", type=int, default=None, help="Optional max_tokens when thinking=off (omitted by default).")
    parser.add_argument("--max-tokens-on", type=int, default=None, help="Optional max_tokens when thinking=on (omitted by default).")
    parser.add_argument("--concurrency", type=int, default=64, help="Concurrent HTTP requests to the server.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request HTTP timeout in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing JSONL for selected runs before collecting.")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact API payload for a sample task and exit.")
    parser.add_argument("--respondent", type=int, default=None, help="Respondent ID to inspect in --dry-run.")
    parser.add_argument("--pair-id", type=int, default=1, choices=(1, 2, 3, 4, 5), help="Pair ID (1-5) for --dry-run.")
    args = parser.parse_args()

    task_keys, prompts_by_persona = build_verified_prompts()

    if args.dry_run:
        target_resp = args.respondent if args.respondent is not None else task_keys[0][0]
        print(f"=== QWEN API DRY RUN | model={args.model} | base_url={args.base_url} | respondent={target_resp} pair_id={args.pair_id} ===")
        for persona_kind in args.personas:
            item = next(
                p for p in prompts_by_persona[persona_kind] if p["respondent"] == target_resp and p["pair_id"] == args.pair_id
            )
            for think_mode in args.thinking:
                max_tok = args.max_tokens_on if think_mode == "on" else args.max_tokens_off
                kwargs = build_request_kwargs(
                    item, model_id=args.model, think_mode=think_mode, max_tokens=max_tok, backend=args.backend
                )
                print(f"\n--- persona={persona_kind} | thinking={think_mode} ---")
                print(json.dumps(kwargs, indent=2, ensure_ascii=False))
        return

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify_model(args.model)

    meta_path = PREDS_DIR / "qwen_meta.json"
    if meta_path.exists():
        meta_out = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta_out = {
            "dataset": "covid",
            "n_respondents": 200,
            "n_tasks": 1000,
            "subsample_seed": 42,
            "prompt_structure": "Wang (2026) / Gemini Argon system+user prompt via OpenAI Chat Completions API",
            "runs": {},
        }

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key, timeout=args.timeout)
    for think_mode in args.thinking:
        think_tag = "THINKING_ON" if think_mode == "on" else "THINKING_OFF"
        max_tok = args.max_tokens_on if think_mode == "on" else args.max_tokens_off
        for persona_kind in args.personas:
            out_name = f"{slug}_{think_tag}_{persona_kind}.jsonl"
            out_path = PREDS_DIR / out_name
            stats = collect_run(
                client,
                prompts_by_persona[persona_kind],
                out_path,
                model_id=args.model,
                think_mode=think_mode,
                persona_kind=persona_kind,
                max_tokens=max_tok,
                backend=args.backend,
                concurrency=args.concurrency,
                overwrite=args.overwrite,
            )
            run_key = f"{slug}__{think_tag}__{persona_kind}"
            meta_out["runs"][run_key] = {
                "model_id": args.model,
                "slug": slug,
                "thinking_level": think_tag,
                "persona": persona_kind,
                "sampling": OFFICIAL_SAMPLING[think_mode],
                "max_tokens": max_tok,
                "file": out_name,
                **stats,
            }
            print(
                f"  Saved {out_name} | valid={stats['n_valid']}/{stats['n_total']} "
                f"(invalid={stats['n_invalid']}) | opt_out={stats['opt_out_pct_of_valid']:.1f}% | "
                f"mean_tok={stats['mean_output_tokens']:.1f}"
            )

    meta_path.write_text(json.dumps(meta_out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote metadata to {meta_path}")


if __name__ == "__main__":
    main()

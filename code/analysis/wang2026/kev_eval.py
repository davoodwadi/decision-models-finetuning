"""Kev pointer-model dataset collection runner (default CPU, bfloat16) with progressive JSONL persistence."""

import gc
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "kev") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "kev"))

from data.persona import DATASET_PERSONA_COLS, PERSONA_NAMES, apply_persona_mapping  # noqa: E402
from kev.model import rows_of  # noqa: E402
from models.kev_pointer import Checkpoint, KevEncoder, KevPointerModel, LoadOptions  # noqa: E402

# Exact mapping from Stata `fda` strings to Wang (2026) Section 5.2.1 minimalist Authorization descriptions
FDA_WANG_TEXT_MAP: dict[str, str] = {
    "The vaccine has been approved and licensed by the US Food and Drug Administration.": (
        "Approved and licensed by the US Food and Drug Administration."
    ),
    "The vaccine has received an emergency use authorization from the US Food and Drug Administration.  This allows the expedited use of promising drugs that the FDA has found it reasonable to believe may be effective in combatting the virus": (
        "Received an emergency use authorization from the US Food and Drug Administration.  "
        "This allows the expedited use of promising drugs that the FDA has found it reasonable "
        "to believe may be effective in combatting the virus."
    ),
}

RANDOM_STATE_TEXT = ""
KEV_QUESTION_INSTRUCTIONS = (
    "Suppose you are deciding whether or not to receive a COVID-19 vaccination. "
    "There are three options for you. Which option would you choose?"
)
CHOICE_LETTERS = ("A", "B", "C")


def enable_qwen3_5_cpu_kernels() -> None:
    """Unwrap Hub/FLA Triton kernels on Qwen3.5 so CPU tensors use the pure-PyTorch reference implementation."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for fn_name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        fn = getattr(modeling_qwen3_5, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(modeling_qwen3_5, fn_name, fn.__wrapped__)


def format_wang_option_desc(row: pd.Series | dict[str, object]) -> str:
    """Format a vaccine option description (without the leading 'A: ' / 'B: ', which Kev's option_text adds)."""
    fda_raw = str(row["fda"])
    if fda_raw not in FDA_WANG_TEXT_MAP:
        raise ValueError(f"Unexpected fda text: {fda_raw!r}")
    fda_text = FDA_WANG_TEXT_MAP[fda_raw]
    return (
        f"Efficacy: {row['efficacy']}. "
        f"Protective duration: {row['duration']}. "
        f"Major side effect: {row['majorside']}. "
        f"Minor side effect: {row['minorside']}. "
        f"Authorization: {fda_text} "
        f"Origin: {row['origin']}. "
        f"Endorsement: {row['endorsed']}."
    )


def build_respondent_states(df_a: pd.DataFrame, include_country: bool = False) -> dict[int, str]:
    """Map each unique respondent in df_a to their Kev 'Key: Value' persona block via apply_persona_mapping(..., 'covid')."""
    resp_raw = df_a.groupby("respondent", as_index=False).first()
    resp_persona_df = apply_persona_mapping(resp_raw.copy(), "covid")
    cols = [c for c in DATASET_PERSONA_COLS["covid"] if include_country or c != "c_alphan"]

    resp_state_by_id: dict[int, str] = {}
    for row in resp_persona_df.to_dict("records"):
        lines = [f"{PERSONA_NAMES[c]}: {row[c]}" for c in cols if pd.notna(row[c])]
        resp_state_by_id[int(row["respondent"])] = "\n".join(lines)
    return resp_state_by_id


def build_kev_prompt_items(
    sub_df_a: pd.DataFrame,
    sub_df_b: pd.DataFrame,
    resp_state_by_id: dict[int, str],
    *,
    persona_mode: str,
) -> list[dict[str, object]]:
    """Build per-pair Kev view items for the sampled respondents under 'random' (empty state) or 'persona' mode."""
    if persona_mode not in {"random", "persona"}:
        raise ValueError(f"persona_mode must be 'random' or 'persona', got {persona_mode!r}")

    items: list[dict[str, object]] = []
    for row_a, row_b in zip(sub_df_a.to_dict("records"), sub_df_b.to_dict("records"), strict=True):
        resp_id = int(row_a["respondent"])
        pair_id = int(row_a["pair_id"])
        state_str = RANDOM_STATE_TEXT if persona_mode == "random" else resp_state_by_id[resp_id]

        y_a = float(row_a["vaxbin"])
        y_b = float(row_b["vaxbin"])
        y_c = 1.0 - y_a - y_b
        q_spec = {
            "type": "choice",
            "instructions": KEV_QUESTION_INSTRUCTIONS,
            "criteria": {
                "A": format_wang_option_desc(row_a),
                "B": format_wang_option_desc(row_b),
                "C": "You choose neither A nor B.",
            },
            "target": {"A": y_a, "B": y_b, "C": y_c},
        }
        items.append({
            "respondent": resp_id,
            "pair_id": pair_id,
            "persona_mode": persona_mode,
            "state": state_str,
            "question_spec": q_spec,
            "view": {"state": state_str, "questions": {f"pair_{pair_id}": q_spec}},
        })
    return items


def verify_matches_gemini_subsample(
    prompt_items: list[dict[str, object]],
    gemini_jsonl_path: str | Path = CODE_DIR / "proprietary-llms" / "gemini" / "data" / "gemini-3.5-flash-lite_MINIMAL_random.jsonl",
) -> None:
    """Fail fast if the (respondent, pair_id) pairs in prompt_items do not match the Gemini subsample JSONL."""
    g_path = Path(gemini_jsonl_path)
    if not g_path.exists():
        raise FileNotFoundError(f"Gemini reference JSONL not found at {g_path.resolve()}")
    gemini_keys: set[tuple[int, int]] = set()
    with g_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gemini_keys.add((int(rec["respondent"]), int(rec["pair_id"])))

    kev_keys = [(int(it["respondent"]), int(it["pair_id"])) for it in prompt_items]
    if len(kev_keys) != len(gemini_keys) or set(kev_keys) != gemini_keys:
        diff_kev = sorted(set(kev_keys) - gemini_keys)[:5]
        diff_gem = sorted(gemini_keys - set(kev_keys))[:5]
        raise ValueError(
            f"Subsample mismatch vs. {g_path.name}: kev has {len(kev_keys)} pairs, gemini has {len(gemini_keys)} pairs. "
            f"Only in kev: {diff_kev}, only in gemini: {diff_gem}"
        )


def preview_kev_item(item: dict[str, object], checkpoint: str | Path = "jaredpalmer/kev-9b") -> str:
    """Decode and format a single Kev prompt item using KevEncoder.preview (without loading backbone weights)."""
    enc = KevEncoder.for_base(str(checkpoint))
    header, text = enc.preview(item["view"])
    return f"{header}\n{text}"


def free_kev_memory() -> None:
    """Clear Python exception frames, run garbage collection, and empty CUDA cache if CUDA is initialized."""
    if hasattr(sys, "last_traceback") and sys.last_traceback is not None:
        traceback.clear_frames(sys.last_traceback)
        sys.last_traceback = None
    for attr in ("last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_kev_model(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    temperature: float | None = None,
) -> KevPointerModel:
    """Free memory/GPU cache and load a Kev checkpoint onto `device` (default 'cpu') using its checkpoint default temperature when temperature=None."""
    free_kev_memory()
    if str(device) == "cpu":
        enable_qwen3_5_cpu_kernels()
    ck = Checkpoint(str(checkpoint))
    tok, net = ck.load(device, LoadOptions(dtype=dtype, temperature=temperature))
    model = KevPointerModel(net, tok, ck.meta, "loaded")
    model.eval()
    return model


def load_kev_jsonl(
    output_path: str | Path,
    *,
    expected_model: str | None = None,
    expected_persona_mode: str | None = None,
    expected_temperature: float | None = None,
) -> dict[tuple[int, int], dict[str, object]]:
    """Load previously collected (respondent, pair_id) Kev records from JSONL and verify metadata."""
    path = Path(output_path)
    collected: dict[tuple[int, int], dict[str, object]] = {}
    if not path.exists():
        return collected
    required_keys = {
        "respondent",
        "pair_id",
        "model",
        "persona_mode",
        "device",
        "temperature",
        "meta_temperature",
        "logits",
        "probs",
        "choice",
        "state_tokens",
        "total_tokens",
    }
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = required_keys - rec.keys()
            if missing:
                raise ValueError(f"{path}:{line_num} missing required keys: {sorted(missing)}")
            if expected_model is not None and rec["model"] != expected_model:
                raise ValueError(f"{path}:{line_num} model mismatch: {rec['model']!r} != {expected_model!r}")
            if expected_persona_mode is not None and rec["persona_mode"] != expected_persona_mode:
                raise ValueError(
                    f"{path}:{line_num} persona_mode mismatch: {rec['persona_mode']!r} != {expected_persona_mode!r}"
                )
            if expected_temperature is not None and not np.isclose(float(rec["temperature"]), float(expected_temperature), atol=1e-4):
                raise ValueError(
                    f"{path}:{line_num} temperature mismatch: {rec['temperature']} != {expected_temperature}"
                )
            collected[(int(rec["respondent"]), int(rec["pair_id"]))] = rec
    return collected


@torch.no_grad()
def _score_packed_enc_with_cache(
    model: KevPointerModel,
    enc: dict[str, object],
    prefix_cache: dict[str, tuple[int, object, torch.Tensor | None]],
    temperature: float,
) -> list[tuple[list[float], list[float]]]:
    """Return per-question (raw_logits, probs_at_temperature) for a packed respondent view, reusing state prefixes."""
    state_ids_key = json.dumps(enc["ids"][: enc["seg"].count(0)])
    net = model.net
    if state_ids_key not in prefix_cache:
        # Keep at most 4 cached state prefixes in RAM
        if len(prefix_cache) >= 4:
            prefix_cache.clear()
        prefix_cache[state_ids_key] = net.prefix(enc)

    prefix_len, cache, _ = prefix_cache[state_ids_key]
    _, _, rows = rows_of(enc)
    hs = net._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=prefix_len)

    # Compute raw pointer logits (with head.temperature temporarily 1.0) and softmax probabilities at `temperature`
    orig_temp = net.head.temperature
    net.head.temperature = 1.0
    try:
        results: list[tuple[list[float], list[float]]] = []
        for h, r in zip(hs, rows, strict=True):
            opt_indices = torch.tensor(r["opts"], device=net.device)
            raw_z = net.head(h[r["decide"]], h[opt_indices]).float().cpu()
            probs = F.softmax(raw_z / float(temperature), dim=-1)
            results.append(([float(x) for x in raw_z.tolist()], [float(x) for x in probs.tolist()]))
        return results
    finally:
        net.head.temperature = orig_temp


def collect_kev_prompts(
    model: KevPointerModel,
    prompt_items: list[dict[str, object]],
    output_path: str | Path,
    *,
    model_name: str,
    n_personas: int | None = None,
    temperature: float | None = None,
) -> list[dict[str, object]]:
    """Incrementally run Kev inference (grouped by respondent) and progressively flush each choice set to JSONL.

    - Uses the checkpoint's default `model.meta.temperature` when `temperature=None` (e.g. 2.19 for Base Kev 9B, 1.0 for FT).
    - Groups the 5 choice sets of each respondent into one packed view so the respondent's state is encoded once.
    - Reuses the state KV/recurrent prefix whenever consecutive respondents share the same state (e.g. Random Persona).
    - Immediately flushes each respondent's 5 records to `output_path` so stopping mid-run loses 0 completed respondents.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_items:
        raise ValueError("prompt_items must not be empty")
    eff_temp = float(model.meta.temperature) if temperature is None else float(temperature)

    persona_mode = str(prompt_items[0]["persona_mode"])
    if n_personas is not None:
        seen_resp: list[int] = []
        seen_set: set[int] = set()
        for item in prompt_items:
            r_id = int(item["respondent"])
            if r_id not in seen_set:
                seen_set.add(r_id)
                seen_resp.append(r_id)
        target_resp = set(seen_resp[:n_personas])
        active_items = [item for item in prompt_items if int(item["respondent"]) in target_resp]
    else:
        active_items = list(prompt_items)

    collected = load_kev_jsonl(
        output_path,
        expected_model=model_name,
        expected_persona_mode=persona_mode,
        expected_temperature=eff_temp,
    )
    pending_items = [
        item for item in active_items if (int(item["respondent"]), int(item["pair_id"])) not in collected
    ]
    already_done = len(active_items) - len(pending_items)
    print(
        f"[{output_path.name}] model={model_name}, persona_mode={persona_mode}, temp={eff_temp:.2f}, device={model.net.device} | "
        f"Target: {len(active_items)} prompts ({already_done} already collected, {len(pending_items)} remaining)"
    )

    if pending_items:
        # Group pending items by respondent so all pending choice sets for a respondent run in one packed pass
        by_resp: dict[int, list[dict[str, object]]] = {}
        for item in pending_items:
            by_resp.setdefault(int(item["respondent"]), []).append(item)

        prefix_cache: dict[str, tuple[int, object, torch.Tensor | None]] = {}
        pbar = tqdm(total=len(pending_items), desc=output_path.stem, unit="pair")
        meta_temp = float(model.meta.temperature)
        device_str = str(model.net.device)

        with output_path.open("a", encoding="utf-8") as f_out:
            try:
                for resp_id, resp_items in by_resp.items():
                    state_str = str(resp_items[0]["state"])
                    packed_questions = {
                        f"pair_{int(it['pair_id'])}": it["question_spec"] for it in resp_items
                    }
                    packed_enc, _ = model.prepare({"state": state_str, "questions": packed_questions})
                    scored = _score_packed_enc_with_cache(model, packed_enc, prefix_cache, temperature=eff_temp)

                    for it, (raw_logits, probs) in zip(resp_items, scored, strict=True):
                        pair_id = int(it["pair_id"])
                        single_enc, _ = model.prepare(it["view"])
                        top_choice = CHOICE_LETTERS[int(np.argmax(probs))]
                        rec = {
                            "respondent": resp_id,
                            "pair_id": pair_id,
                            "model": model_name,
                            "persona_mode": persona_mode,
                            "device": device_str,
                            "temperature": eff_temp,
                            "meta_temperature": meta_temp,
                            "logits": raw_logits,
                            "probs": probs,
                            "choice": top_choice,
                            "state_tokens": int(single_enc["state_tokens"]),
                            "total_tokens": len(single_enc["ids"]),
                        }
                        collected[(resp_id, pair_id)] = rec
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    pbar.update(len(resp_items))
            finally:
                prefix_cache.clear()
                pbar.close()

    return [collected[(int(it["respondent"]), int(it["pair_id"]))] for it in active_items]


def extract_kev_prob_targets(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract soft probability arrays (p_A, p_B, p_C) from collected Kev JSONL records."""
    probs = np.array([rec["probs"] for rec in records], dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) probability array, got {probs.shape}")
    return probs[:, 0], probs[:, 1], probs[:, 2]


def extract_kev_hard_targets(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one-hot argmax choice arrays (y_A, y_B, y_C) from collected Kev JSONL records."""
    choices = np.array([str(rec["choice"]).strip() for rec in records])
    return (
        (choices == "A").astype(np.float64),
        (choices == "B").astype(np.float64),
        (choices == "C").astype(np.float64),
    )

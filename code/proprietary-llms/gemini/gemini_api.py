"""Async Gemini API dataset collection runner with progressive JSONL persistence, tenacity backoff, and strict letter parsing."""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def create_gemini_client() -> genai.Client:
    """Initialize Gemini client using Enterprise Agent Platform when configured, or Google AI Studio (GEMINI_API_KEY)."""
    if os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE"):
        return genai.Client()
    return genai.Client(vertexai=False, api_key=os.environ["GEMINI_API_KEY"])


def is_transient_api_error(exc: BaseException) -> bool:
    """Retry only on rate-limit (429) or transient server/network errors (5xx), failing fast on 4xx client errors."""
    if isinstance(exc, errors.APIError):
        return exc.code in {429, 500, 502, 503, 504}
    return isinstance(exc, (TimeoutError, ConnectionError))


def normalize_thinking_level(thinking_level: str | types.ThinkingLevel | None) -> str | None:
    """Convert a ThinkingLevel enum or string into a canonical uppercase string (or None for model default)."""
    if thinking_level is None:
        return None
    if isinstance(thinking_level, types.ThinkingLevel):
        return thinking_level.value
    return str(thinking_level).upper()


@retry(
    retry=retry_if_exception(is_transient_api_error),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
async def call_gemini_once(
    client: genai.Client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    thinking_level: str | types.ThinkingLevel | None = None,
) -> dict[str, object]:
    """Send one prompt pair to Gemini and return raw response text, reasoning text, and token counts."""
    thinking_kwargs: dict[str, object] = {"include_thoughts": True}
    if thinking_level is not None:
        thinking_kwargs["thinking_level"] = thinking_level
    resp = await client.aio.models.generate_content(
        model=model_id,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(**thinking_kwargs),
        ),
    )
    if resp.text is None or not resp.candidates or resp.candidates[0].content is None or resp.usage_metadata is None:
        raise ValueError(f"Gemini returned incomplete response or usage metadata: {resp!r}")

    parts = resp.candidates[0].content.parts or []
    reasoning_text = "".join(p.text for p in parts if p.thought and p.text)
    um = resp.usage_metadata
    return {
        "raw_response": resp.text,
        "reasoning_text": reasoning_text,
        "prompt_tokens": int(um.prompt_token_count),
        "output_tokens": int(um.candidates_token_count or 0),
        "thought_tokens": int(um.thoughts_token_count or 0),
        "total_tokens": int(um.total_token_count),
    }


def _item_key(item: dict[str, object]) -> tuple[int, int]:
    """Unique (respondent, pair_id) identifier for a choice-set prompt."""
    return int(item["respondent"]), int(item["pair_id"])


def load_collected_jsonl(
    output_path: Path,
    *,
    expected_model: str,
    expected_thinking_level: str | None,
) -> dict[tuple[int, int], dict[str, object]]:
    """Load previously collected (respondent, pair_id) records from JSONL and verify metadata and required fields."""
    collected: dict[tuple[int, int], dict[str, object]] = {}
    if not output_path.exists():
        return collected
    required_keys = {
        "respondent",
        "pair_id",
        "model",
        "thinking_level",
        "raw_response",
        "reasoning_text",
        "prompt_tokens",
        "output_tokens",
        "thought_tokens",
        "total_tokens",
    }
    with output_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = required_keys - rec.keys()
            if missing:
                raise ValueError(f"{output_path}:{line_num} missing required keys: {sorted(missing)}")
            rec_model = rec["model"]
            rec_thinking = rec["thinking_level"]
            if rec_model != expected_model or rec_thinking != expected_thinking_level:
                raise ValueError(
                    f"{output_path}:{line_num} metadata mismatch: file has (model={rec_model!r}, "
                    f"thinking_level={rec_thinking!r}), expected ({expected_model!r}, {expected_thinking_level!r})"
                )
            collected[(int(rec["respondent"]), int(rec["pair_id"]))] = rec
    return collected


async def query_gemini_prompts(
    prompt_items: list[dict[str, object]],
    output_path: str | Path,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    thinking_level: str | types.ThinkingLevel | None = None,
    n_personas: int | None = None,
    max_concurrency: int = 25,
) -> list[str]:
    """Incrementally collect Gemini responses and progressively flush each record (with tokens & reasoning_text) to JSONL.

    - If `n_personas` is set (e.g. 100), only the first `n_personas` unique respondents (5 * n_personas prompts) are targeted.
    - Any (respondent, pair_id) already present in `output_path` is skipped automatically.
    - Every completed API call is immediately appended and flushed to `output_path`, so stopping mid-run loses 0 completed calls.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm_thinking = normalize_thinking_level(thinking_level)

    if n_personas is not None:
        # Keep all 5 choice sets for the first n_personas unique respondents in prompt_items
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

    collected = load_collected_jsonl(
        output_path,
        expected_model=model,
        expected_thinking_level=norm_thinking,
    )
    pending_items = [item for item in active_items if _item_key(item) not in collected]
    already_done = len(active_items) - len(pending_items)
    print(
        f"[{output_path.name}] model={model}, thinking_level={norm_thinking} | "
        f"Target: {len(active_items)} prompts ({already_done} already collected, {len(pending_items)} remaining)"
    )

    if pending_items:
        client = create_gemini_client()
        sem = asyncio.Semaphore(max_concurrency)
        pbar = tqdm(total=len(pending_items), desc=output_path.stem, unit="req")

        # Open JSONL in append mode and flush every single record immediately upon completion
        with output_path.open("a", encoding="utf-8") as f_out:
            async def _worker(item: dict[str, object]) -> None:
                key = _item_key(item)
                async with sem:
                    result = await call_gemini_once(
                        client,
                        model,
                        str(item["system_prompt"]),
                        str(item["user_prompt"]),
                        thinking_level=norm_thinking,
                    )
                record = {
                    "respondent": key[0],
                    "pair_id": key[1],
                    "model": model,
                    "thinking_level": norm_thinking,
                    "raw_response": result["raw_response"],
                    "reasoning_text": result["reasoning_text"],
                    "prompt_tokens": result["prompt_tokens"],
                    "output_tokens": result["output_tokens"],
                    "thought_tokens": result["thought_tokens"],
                    "total_tokens": result["total_tokens"],
                }
                collected[key] = record
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                pbar.update(1)

            try:
                await asyncio.gather(*(_worker(item) for item in pending_items))
            finally:
                pbar.close()

    return [str(collected[_item_key(item)]["raw_response"]) for item in active_items]


def parse_letter_choices(raw_responses: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate that every stripped response is strictly 'A', 'B', or 'C' and return one-hot (y_A, y_B, y_C) arrays."""
    cleaned = [r.strip() for r in raw_responses]
    invalid = [(idx, raw_responses[idx]) for idx, c in enumerate(cleaned) if c not in {"A", "B", "C"}]
    if invalid:
        raise ValueError(f"{len(invalid)} responses were not a single letter A, B, or C; first 5: {invalid[:5]}")
    arr = np.array(cleaned)
    y_a = (arr == "A").astype(np.float64)
    y_b = (arr == "B").astype(np.float64)
    y_c = (arr == "C").astype(np.float64)
    return y_a, y_b, y_c

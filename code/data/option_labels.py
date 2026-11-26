"""Shared option-label formatting helpers for Stata-backed survey exporters (ISSP, WVS7, HINTS7)."""

from __future__ import annotations

import re


def _is_missing(text: object) -> bool:
    return not isinstance(text, str) or not text.strip()


def _has_alpha(text: object) -> bool:
    return isinstance(text, str) and bool(re.search(r"[A-Za-z]", text))


def _is_bare_code(text: object, code: int) -> bool:
    return isinstance(text, str) and bool(re.fullmatch(r"\d+", text.strip())) and int(text.strip()) == code


def format_option_texts(codes: list[int], texts: list[object]) -> list[str]:
    """Format Stata option labels cleanly, separating missing labels from explicit numeric labels.

    1. Any code missing a label (NaN/empty) -> render `{c}.` for missing, `{c}. {t}` for labelled
       (skipping the prefix if `t` already begins with `0*{c}` followed by space/dot/dash).
    2. All codes labelled, both endpoints have alphabetic text, and at least one interior point's
       label is the bare/zero-padded integer code `c` -> anchor endpoints and format bare interior
       points as `{c}.`.
    3. Everything else -> keep Stata labels verbatim.
    """
    if len(codes) != len(texts):
        raise ValueError(f"codes ({len(codes)}) and texts ({len(texts)}) length mismatch")
    any_missing = any(_is_missing(t) for t in texts)
    endpoint_anchored = (
        not any_missing
        and len(texts) >= 3
        and _has_alpha(texts[0])
        and _has_alpha(texts[-1])
        and any(_is_bare_code(t, c) for c, t in zip(codes[1:-1], texts[1:-1]))
    )
    if not (any_missing or endpoint_anchored):
        return [str(t) for t in texts]
    out: list[str] = []
    for c, t in zip(codes, texts):
        if _is_missing(t) or _is_bare_code(t, c):
            out.append(f"{c}.")
        else:
            s = str(t)
            if re.match(rf"0*{c}(?:\s|\.|[-–])", s):
                out.append(s)
            else:
                out.append(f"{c}. {s}")
    return out


def is_option_relabeled(codes: list[int], texts: list[object]) -> bool:
    """Return True when `format_option_texts` modifies any raw Stata option label."""
    if not codes:
        return False
    raw_str = [str(t) if not _is_missing(t) else "" for t in texts]
    return format_option_texts(codes, texts) != raw_str

#!/usr/bin/env python3
"""Compare each audited question with its dataset's codebook, for the audit app's Source tab.

Source side: the codebook text, re-extracted here with PyMuPDF. It deliberately does not reuse the parse that built
the dataset's catalog, so a parsing mistake there shows up as a difference instead of being repeated on both sides.
Final side: the question text and options the dataset's exporter would emit now (audit CSV, drop_options,
question_overrides.json), computed in memory, never read from the parquet.

Each dataset with a codebook gets a provider in PROVIDERS. Datasets without one return `configured: false`.
Prints one JSON object to stdout.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

SOURCE_CHECK_VERSION = 1
WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
PROMPT_START_RE = re.compile(
    r"^(do|does|did|are|is|were|was|have|has|had|would|could|should|will|can|"
    r"how|what|which|who|where|when|why|please|tell|indicate|choose|select|rate|rank|describe|compare)\b",
    re.I,
)
AGREE_STEM_RE = re.compile(
    r"\b(?:strongly\s+)?agree\s+(?:or|/)\s+(?:strongly\s+)?disagree\b|\bdisagree\s+(?:or|/)\s+agree\b",
    re.I,
)
AGREE_OPTION_RE = re.compile(r"\b(agree|disagree)\b", re.I)
REVIEW_WARNING_TAGS = frozenset(
    {
        "review-codebook-stem",
        "empty-stem",
        "empty-options",
        "backref",
        "missing-context",
        "short-text",
        "trunc-risk",
        "unclean-text",
        "unclean text",
        "html-table-flat",
        "img-ref",
        "prior-priming",
        "prior-task-dep",
        "rescuable-context",
    }
)


def _collapse(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def _norm_cmp(text: str) -> str:
    return NON_ALNUM_RE.sub(" ", text.lower()).strip()


def _load_tags(dataset_dir: Path) -> dict[str, list[str]]:
    tags_path = dataset_dir / "question_tags.json"
    if not tags_path.exists():
        return {}
    raw = json.loads(tags_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{tags_path}: expected a JSON object, got {type(raw).__name__}")
    out: dict[str, list[str]] = {}
    for key, val in raw.items():
        if not isinstance(key, str) or not isinstance(val, list) or not all(isinstance(t, str) for t in val):
            raise TypeError(f"{tags_path}: entry {key!r} must map to a list of strings")
        out[key] = [t.strip() for t in val if t.strip()]
    return out


def _stem_warnings(
    final_text: str,
    final_options: list[str],
    data_label: str,
    tags: list[str],
    has_stem_override: bool,
) -> list[str]:
    warnings: list[str] = []
    text = _collapse(final_text)
    if not text:
        warnings.append("Empty question stem")
    elif (
        data_label
        and _norm_cmp(text) == _norm_cmp(data_label)
        and "?" not in text
        and not PROMPT_START_RE.match(text)
    ):
        warnings.append("No question stem (matches variable label only)")
    if AGREE_STEM_RE.search(text) and final_options and not any(AGREE_OPTION_RE.search(o) for o in final_options):
        warnings.append("Stem asks agree/disagree, but answer options do not")
    for tag in tags:
        lower = tag.lower()
        if lower not in REVIEW_WARNING_TAGS:
            continue
        if lower == "review-codebook-stem" and has_stem_override:
            warnings.append(f"Tagged #{tag} (stem override set)")
        else:
            warnings.append(f"Tagged #{tag}")
    return warnings


def _norm_option(text: str) -> str:
    """Option text for matching only: straight apostrophes, collapsed spaces, exporter code prefixes removed
    ("1. None at all" -> "None at all", "2." -> "2")."""
    t = _collapse(text.replace("´", "'").replace("’", "'"))
    m = re.fullmatch(r"(\d+)\.(?:\s+(.*))?", t)
    if m:
        return m[2] if m[2] else m[1]
    return t


def word_diff(source: str, final: str) -> list[list[str]]:
    """Word-level diff as [op, text] pairs, op in eq / del / ins."""
    a = _collapse(source).split(" ")
    b = _collapse(final).split(" ")
    out: list[list[str]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(["eq", " ".join(a[i1:i2])])
        if tag in ("delete", "replace"):
            out.append(["del", " ".join(a[i1:i2])])
        if tag in ("insert", "replace"):
            out.append(["ins", " ".join(b[j1:j2])])
    return out


def compare_options(
    source_options: list[dict[str, object]],
    rendered: list[str],
    before_override: list[str],
    final: list[str],
    origins: list[str],
    unfilled: list[str],
) -> tuple[list[dict[str, object]], str]:
    """Align codebook categories (missing codes left out) with the exported options.

    Row status: same, overridden (text changed in question_overrides.json), filled (a codebook placeholder the
    exporter filled, e.g. [nationality] -> American), dropped (drop_options),
    final_only (exported but not in the codebook), source_only (in the codebook but not exported).
    """
    if not (len(before_override) == len(final) == len(origins)):
        raise ValueError(
            f"before_override / final / origins length mismatch: {before_override!r} {final!r} {origins!r}"
        )
    if len(rendered) != len(unfilled):
        raise ValueError(f"rendered / unfilled length mismatch: {rendered!r} {unfilled!r}")
    substantive = [str(o["text"]) for o in source_options if not o["missing_code"]]
    unmatched = {i: _norm_option(t) for i, t in enumerate(substantive)}

    def take_source(*candidates: str) -> str | None:
        for cand in candidates:
            key = _norm_option(cand)
            for i, norm in unmatched.items():
                if norm == key:
                    del unmatched[i]
                    return substantive[i]
        return None

    rows: list[dict[str, object]] = []
    kept = iter(zip(before_override, final, origins))
    kept_set = list(before_override)
    for text, raw_text in zip(rendered, unfilled):
        if text in kept_set:
            kept_set.remove(text)
            orig, fin, origin = next(kept)
            if orig != text:
                raise ValueError(f"option order changed between render and drop: {orig!r} vs {text!r}")
            # The codebook prints placeholders unfilled, so also try the Stata text before the exporter's fills.
            src = take_source(orig, fin, raw_text)
            if fin != orig:
                status = "overridden"
            elif origin == "stata_label_filled" and src is not None:
                status = "filled"
            elif src is None:
                status = "final_only"
            else:
                status = "same"
            rows.append(
                {"status": status, "source": src, "final": fin, "original": orig if fin != orig else None, "origin": origin}
            )
        else:
            rows.append(
                {"status": "dropped", "source": take_source(text, raw_text), "final": None, "original": text, "origin": None}
            )
    for i in sorted(unmatched):
        rows.append({"status": "source_only", "source": substantive[i], "final": None, "original": None, "origin": None})

    if not substantive or not rendered:
        return rows, "n/a"
    return rows, "match" if all(r["status"] == "same" for r in rows) else "differ"


# ---------------------------------------------------------------------------
# WVS Wave 7 US: codebook F00010562 (one "Qn- <wording>" block per variable, then a % table by sex and age)
# ---------------------------------------------------------------------------

WVS7_DIR_NAME = "WVS-Wave7"
WVS7_PDF = "F00010562-World_Values_Survey_Wave_7_2017-2020_United_States_v3.0.pdf"
WVS7_PAGE_FURNITURE = re.compile(
    r"^(World Values Survey Wave 7 \(2017-2020\)|United States: Distributions in %|v3\.0|Page \d+ of \d+)$"
)
WVS7_TABLE_HEADER = {"TOTAL", "Sex", "Age", "Male", "Female", "Up to 29", "30-49", "50 and more", "50 and", "more"}
WVS7_TABLE_COLUMNS = 6  # TOTAL, Male, Female, Up to 29, 30-49, 50 and more
WVS7_NUMBER_RE = re.compile(r"^(-|\d[\d,]*(\.\d+)?)$")
WVS7_MISSING_RE = re.compile(r"^(Don.t know|No answer|Not asked|Missing|Other missing|Not applicable|INAP)", re.I)


# One text line of the codebook: (0-based page index, text, bounding box (x0, y0, x1, y1)).
Line = tuple[int, str, tuple[float, float, float, float]]


def _wvs7_lines(pdf_path: Path, pages: list[int] | None = None) -> list[Line]:
    """Text lines in reading order (PyMuPDF), page header/footer lines removed. `pages` limits the pages read."""
    import pymupdf

    out: list[Line] = []
    with pymupdf.open(pdf_path) as doc:
        for index in range(len(doc)) if pages is None else pages:
            for block in doc[index].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"])
                    if WVS7_PAGE_FURNITURE.match(text.strip()):
                        continue
                    out.append((index, text, tuple(line["bbox"])))
    return out


def _wvs7_locate(v: str, lines: list[Line]) -> dict[str, object] | None:
    """Find one variable's block: wording lines and category rows (with their lines); None when absent."""
    start = next((i for i, (_, text, _) in enumerate(lines) if text.startswith(f"{v}- ")), None)
    if start is None:
        return None
    table = next((i for i in range(start + 1, len(lines)) if lines[i][1] in ("TOTAL", "Sex")), None)
    if table is None:
        raise ValueError(f"{v}: no table header after the question")
    end = next((i for i in range(table, len(lines)) if lines[i][1].startswith("(N)")), None)
    if end is None:
        raise ValueError(f"{v}: no (N) line closing the table")
    wording_lines = lines[start:table]
    wording = _collapse(" ".join([wording_lines[0][1][len(f"{v}- "):], *(t for _, t, _ in wording_lines[1:])]))

    raw = [ln for ln in lines[table:end] if ln[1].strip()]
    # Drop column headers (repeated when a table continues on the next page): runs of header words that start
    # with TOTAL or Sex and end with "50 and more" / "more". Single header-like lines between values are
    # categories (Q260 "Male" / "Female", Q262 "Up to 29").
    cells: list[Line] = []
    i = 0
    while i < len(raw):
        if raw[i][1].strip() in ("TOTAL", "Sex"):
            j = i
            while j < len(raw) and raw[j][1].strip() in WVS7_TABLE_HEADER:
                j += 1
                if raw[j - 1][1].strip() in ("50 and more", "more"):
                    break
            if raw[j - 1][1].strip() not in ("50 and more", "more"):
                raise ValueError(f"{v}: unterminated table header {[t for _, t, _ in raw[i:j]]!r}")
            i = j
            continue
        cells.append(raw[i])
        i += 1

    rows: list[dict[str, object]] = []
    label: list[Line] = []
    numbers = 0
    for cell in cells:
        text = cell[1].strip()
        if WVS7_NUMBER_RE.match(text) and label:
            numbers += 1
            if numbers == WVS7_TABLE_COLUMNS:
                label_text = " ".join(t.strip() for _, t, _ in label)
                rows.append(
                    {"text": label_text, "missing_code": bool(WVS7_MISSING_RE.match(label_text)), "lines": label}
                )
                label, numbers = [], 0
        elif numbers:
            joined = " ".join(t.strip() for _, t, _ in label)
            raise ValueError(f"{v}: table row {joined!r} has {numbers} values, expected {WVS7_TABLE_COLUMNS}")
        else:
            # Category label: a numeric label ("2" on 1-10 scales) is the first value-like line of a row.
            label.append(cell)
    if label:
        raise ValueError(f"{v}: table ends inside row {' '.join(t.strip() for _, t, _ in label)!r}")
    return {"page": wording_lines[0][0] + 1, "text": wording, "wording_lines": wording_lines, "rows": rows}


def _wvs7_block(v: str, lines: list[Line]) -> dict[str, object] | None:
    """Wording, 1-based page and response categories of one variable; None when the codebook has no block for it."""
    found = _wvs7_locate(v, lines)
    if found is None:
        return None
    options = [{"text": r["text"], "missing_code": r["missing_code"]} for r in found["rows"]]
    return {"page": found["page"], "text": found["text"], "options": options}


def wvs7_source(dataset_dir: Path) -> dict[str, object]:
    from data.wvs_export import wvs_final_questions

    lines = _wvs7_lines(dataset_dir / WVS7_PDF)
    final = wvs_final_questions()
    tags_by_key = _load_tags(dataset_dir)
    items: dict[str, dict[str, object]] = {}
    for row_key, f in final.items():
        error = None
        try:
            block = _wvs7_block(str(f["v"]), lines)
        except ValueError as exc:
            block, error = None, str(exc)
        items[row_key] = _item(f, "codebook", block, error, tags_by_key.get(row_key, []))
    return {
        "documents": {"codebook": {"label": WVS7_PDF, "path": WVS7_PDF}},
        "final_source": "Stata value labels, WVS_Wave_7_United_States_Stata_v5.1.dta",
        "items": items,
    }


def _item(
    final: dict[str, object],
    doc_id: str,
    block: dict[str, object] | None,
    error: str | None,
    tags: list[str],
) -> dict[str, object]:
    final_text = str(final["question"])
    final_options = [str(o) for o in final["options"]]
    warnings = _stem_warnings(
        final_text,
        final_options,
        str(final["data_label"]),
        tags,
        bool(final["has_stem_override"]),
    )
    base = {
        "v": final["v"],
        "included": final["included"],
        "is_persona": bool(final["is_persona"]),
        "type": final["type"],
        "export_problem": final["export_problem"],
        "doc": doc_id,
        "final_text": final_text,
        "final_options": final_options,
        "tags": tags,
        "warnings": warnings,
        "error": error,
    }
    if block is None:
        option_rows, options_status = compare_options(
            [],
            final["options_rendered"],
            final["options_before_override"],
            final["options"],
            final["options_origin"],
            final["options_unfilled"],
        )
        return {
            **base,
            "page": None,
            "source_text": None,
            "text_status": "not_in_source",
            "text_diff": [],
            "source_options": [],
            "option_rows": option_rows,
            "options_status": "n/a",
        }
    source_text = str(block["text"])
    option_rows, options_status = compare_options(
        block["options"],
        final["options_rendered"],
        final["options_before_override"],
        final["options"],
        final["options_origin"],
        final["options_unfilled"],
    )
    return {
        **base,
        "page": block["page"],
        "source_text": source_text,
        "text_status": "identical" if _collapse(source_text) == _collapse(final_text) else "changed",
        "text_diff": word_diff(source_text, final_text),
        "source_options": block["options"],
        "option_rows": option_rows,
        "options_status": options_status,
    }


# Highlight colours (RGB 0-1), matching the Source tab's chips.
HIGHLIGHT_COLORS: dict[str, tuple[float, float, float]] = {
    "wording": (1.0, 0.86, 0.2),
    "same": (0.45, 0.85, 0.5),
    "overridden": (0.72, 0.58, 1.0),
    "filled": (0.45, 0.75, 1.0),
    "dropped": (0.7, 0.7, 0.7),
    "source_only": (1.0, 0.5, 0.5),
    "missing_code": (0.8, 0.8, 0.8),
}


def wvs7_highlight(dataset_dir: Path, spec: dict[str, object]) -> bytes:
    """The codebook with one question's block highlighted: wording yellow, each category by its comparison status."""
    import pymupdf

    pdf_path = dataset_dir / WVS7_PDF
    v = str(spec["v"])
    page_index = int(spec["page"]) - 1
    statuses: dict[str, str] = spec["statuses"]
    with pymupdf.open(pdf_path) as doc:
        # The block starts on its page and may continue on the next one.
        pages = [i for i in (page_index, page_index + 1) if i < len(doc)]
        found = _wvs7_locate(v, _wvs7_lines(pdf_path, pages))
        if found is None:
            raise ValueError(f"{v}: no codebook block on PDF pages {[i + 1 for i in pages]}")
        marks: list[tuple[str, Line]] = [("wording", ln) for ln in found["wording_lines"]]
        for row in found["rows"]:
            status = "missing_code" if row["missing_code"] else statuses[row["text"]]
            marks.extend((status, ln) for ln in row["lines"])
        # Keep the Page objects alive while annotating (an annotation needs its page to exist).
        page_objects = {index: doc[index] for index in pages}
        for status, (index, _, bbox) in marks:
            annot = page_objects[index].add_highlight_annot(pymupdf.Rect(bbox))
            annot.set_colors(stroke=HIGHLIGHT_COLORS[status])
            annot.set_opacity(0.55)
            annot.update()
        return doc.tobytes()


# ---------------------------------------------------------------------------
# NCI HINTS 7 (2024): official Public Codebook PDF (one variable per page, metadata + frequency table)
# ---------------------------------------------------------------------------

HINTS_DIR_NAME = "HINTS"
HINTS_PDF = "HINTS 7 Public Codebook.pdf"
HINTS_ITEM_PREFIX_RE = re.compile(r'^"?([A-Q]\d+[A-Za-z]?)\.\s*')
HINTS_INT_CODE_RE = re.compile(r"^-?\d+$")
HINTS_NUM_CELL_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


def _hints7_pages_lines(pdf_path: Path, pages: list[int] | None = None) -> list[tuple[int, str | None, list[Line]]]:
    """Read lines + bounding boxes per page from HINTS 7 Public Codebook PDF."""
    import pymupdf

    out: list[tuple[int, str | None, list[Line]]] = []
    with pymupdf.open(pdf_path) as doc:
        for index in range(len(doc)) if pages is None else pages:
            lines: list[Line] = []
            for block in doc[index].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if text:
                        lines.append((index, text, tuple(line["bbox"])))
            vname = None
            for _, t, _ in lines:
                m = re.match(r"^Variable Name:\s*(\S+)$", t)
                if m:
                    vname = m.group(1)
                    break
            out.append((index, vname, lines))
    return out


def _hints7_group_by_var(page_entries: list[tuple[int, str | None, list[Line]]]) -> dict[str, list[Line]]:
    """Group contiguous codebook pages belonging to each variable."""
    by_var: dict[str, list[Line]] = {}
    cur_var: str | None = None
    for pno, vname, lines in page_entries:
        if vname is not None:
            cur_var = vname
            by_var[cur_var] = list(lines)
        elif cur_var is not None and pno > 20:
            by_var[cur_var].extend(lines)
    return by_var


def _hints7_locate(v: str, lines: list[Line]) -> dict[str, object]:
    """Parse one HINTS 7 codebook variable block into wording lines + category rows with bboxes."""
    i_lbl = next((i for i, (_, t, _) in enumerate(lines) if t.startswith("Variable Label:")), None)
    i_fmt = next((i for i, (_, t, _) in enumerate(lines) if t.startswith("Variable Format:")), None)
    i_back = next((i for i, (_, t, _) in enumerate(lines) if t == "Back to Table of Contents"), None)
    if None in (i_lbl, i_fmt, i_back):
        raise ValueError(f"{v}: missing Variable Label / Variable Format / Back to Table of Contents markers")

    wording_lines = lines[i_lbl:i_fmt]
    raw_wording = _collapse(
        " ".join([wording_lines[0][1][len("Variable Label:") :].strip(), *(t for _, t, _ in wording_lines[1:])])
    )
    wording = HINTS_ITEM_PREFIX_RE.sub("", raw_wording, count=1).strip()
    if wording.startswith('"') and wording.endswith('"'):
        wording = wording[1:-1].strip()
    wording = _collapse(wording)

    tbl = lines[i_back + 1 :]
    vl_x0 = next((bbox[0] for _, t, bbox in tbl if t == "Value Label"), None)
    unw_x0 = next((bbox[0] for _, t, bbox in tbl if t == "Unweighted"), None)
    est_idxs = [i for i, (_, t, _) in enumerate(tbl) if t == "(Estimated)"]
    if vl_x0 is None or unw_x0 is None or len(est_idxs) < 2:
        raise ValueError(f"{v}: could not find Value Label / Unweighted / (Estimated) table headers")

    body = tbl[est_idxs[1] + 1 :]
    rows: list[dict[str, object]] = []
    cur_code: int | None = None
    cur_label_lines: list[Line] = []
    cur_nums = 0
    for ln in body:
        _, t, bbox = ln
        if bbox[1] < 60 and HINTS_INT_CODE_RE.match(t):
            continue
        if bbox[0] < vl_x0 - 5 and HINTS_INT_CODE_RE.match(t):
            if cur_code is not None:
                rows.append(
                    {
                        "code": cur_code,
                        "text": _collapse(" ".join(x[1] for x in cur_label_lines)),
                        "missing_code": cur_code < 0,
                        "lines": list(cur_label_lines),
                    }
                )
            cur_code = int(t)
            cur_label_lines = []
            cur_nums = 0
        elif cur_code is not None:
            if bbox[0] >= unw_x0 - 10 and HINTS_NUM_CELL_RE.match(t):
                cur_nums += 1
            elif bbox[0] < unw_x0 - 10 and cur_nums == 0:
                cur_label_lines.append(ln)
    if cur_code is not None:
        rows.append(
            {
                "code": cur_code,
                "text": _collapse(" ".join(x[1] for x in cur_label_lines)),
                "missing_code": cur_code < 0,
                "lines": list(cur_label_lines),
            }
        )
    return {"page": wording_lines[0][0] + 1, "text": wording, "wording_lines": wording_lines, "rows": rows}


def hints7_source(dataset_dir: Path) -> dict[str, object]:
    from data.hints7_export import hints7_final_questions

    by_var = _hints7_group_by_var(_hints7_pages_lines(dataset_dir / HINTS_PDF))
    final = hints7_final_questions()
    tags_by_key = _load_tags(dataset_dir)
    items: dict[str, dict[str, object]] = {}
    for row_key, f in final.items():
        v = str(f["v"])
        error = None
        block = None
        if v in by_var:
            try:
                found = _hints7_locate(v, by_var[v])
                block = {
                    "page": found["page"],
                    "text": found["text"],
                    "options": [{"text": r["text"], "missing_code": r["missing_code"]} for r in found["rows"]],
                }
            except ValueError as exc:
                error = str(exc)
        items[row_key] = _item(f, "codebook", block, error, tags_by_key.get(row_key, []))
    return {
        "documents": {"codebook": {"label": HINTS_PDF, "path": HINTS_PDF}},
        "final_source": "Stata value labels, hints7_public.dta (HINTS7_STATA_20250731.zip)",
        "items": items,
    }


def hints7_highlight(dataset_dir: Path, spec: dict[str, object]) -> bytes:
    """Highlight one HINTS 7 variable's wording and category rows in HINTS 7 Public Codebook.pdf."""
    import pymupdf

    pdf_path = dataset_dir / HINTS_PDF
    v = str(spec["v"])
    page_index = int(spec["page"]) - 1
    statuses: dict[str, str] = spec["statuses"]
    with pymupdf.open(pdf_path) as doc:
        pages = [i for i in (page_index, page_index + 1) if i < len(doc)]
        by_var = _hints7_group_by_var(_hints7_pages_lines(pdf_path, pages))
        if v not in by_var:
            raise ValueError(f"{v}: no codebook block on PDF pages {[i + 1 for i in pages]}")
        found = _hints7_locate(v, by_var[v])
        marks: list[tuple[str, Line]] = [("wording", ln) for ln in found["wording_lines"]]
        for row in found["rows"]:
            status = "missing_code" if row["missing_code"] else statuses[row["text"]]
            marks.extend((status, ln) for ln in row["lines"])
        page_objects = {index: doc[index] for index in pages}
        for status, (index, _, bbox) in marks:
            annot = page_objects[index].add_highlight_annot(pymupdf.Rect(bbox))
            annot.set_colors(stroke=HIGHLIGHT_COLORS[status])
            annot.set_opacity(0.55)
            annot.update()
        return doc.tobytes(deflate=True)


# Federal Reserve SHED 2025: official Stata codebook PDF (dashed rules + Tabulation table)
SHED_DIR_NAME = "SHED"
SHED_PDF = "SHED_2025codebook.pdf"
SHED_DASH_RE = re.compile(r"^-{20,}$")
SHED_CODE_ROW_RE = re.compile(r"^(-?\d+)\s+(.*)$")
SHED_CODE_ONLY_RE = re.compile(r"^(-?\d+)$")


def _shed2025_pages_lines(pdf_path: Path, pages: list[int] | None = None) -> list[Line]:
    """Read lines + bounding boxes from SHED_2025codebook.pdf sorted into visual rows."""
    import pymupdf

    lines: list[Line] = []
    with pymupdf.open(pdf_path) as doc:
        indices = range(len(doc)) if pages is None else pages
        for pno in indices:
            page_lines: list[Line] = []
            for block in doc[pno].get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for ln in block["lines"]:
                    text = "".join(s["text"] for s in ln["spans"]).strip()
                    if text:
                        page_lines.append((pno, text, tuple(ln["bbox"])))
            page_lines.sort(key=lambda item: (item[2][1] + item[2][3]) / 2.0)
            visual_rows: list[list[object]] = []
            for item in page_lines:
                ymid = (item[2][1] + item[2][3]) / 2.0
                if visual_rows and abs(ymid - float(visual_rows[-1][0])) < 4.5:
                    visual_rows[-1][1].append(item)  # type: ignore[union-attr]
                else:
                    visual_rows.append([ymid, [item]])
            for _, grp in visual_rows:
                grp.sort(key=lambda item: item[2][0])  # type: ignore[union-attr]
                lines.extend(grp)  # type: ignore[arg-type]
    return lines


def _shed2025_group_by_var(lines: list[Line]) -> dict[str, list[Line]]:
    """Group codebook lines into per-variable blocks delimited by dashed rules."""
    by_var: dict[str, list[Line]] = {}
    cur_var: str | None = None
    cur_lines: list[Line] = []
    for i, ln in enumerate(lines):
        _, t, _ = ln
        if SHED_DASH_RE.match(t) and i + 1 < len(lines):
            _, nxt_t, nxt_bbox = lines[i + 1]
            if nxt_bbox[0] < 60 and not SHED_DASH_RE.match(nxt_t) and " " not in nxt_t:
                if cur_var is not None:
                    by_var[cur_var] = cur_lines
                cur_var = nxt_t
                cur_lines = [ln]
                continue
        if cur_var is not None:
            cur_lines.append(ln)
    if cur_var is not None:
        by_var[cur_var] = cur_lines
    return by_var


def _shed2025_locate(v: str, blk: list[Line]) -> dict[str, object]:
    """Parse one SHED 2025 codebook variable block into wording lines + tabulation rows with bboxes."""
    dash_idxs = [idx for idx, (_, t, _) in enumerate(blk) if SHED_DASH_RE.match(t)]
    if len(dash_idxs) < 2:
        raise ValueError(f"{v}: expected >= 2 header rule lines in SHED codebook, got {len(dash_idxs)}")
    header_lines = blk[dash_idxs[0] + 1 : dash_idxs[1]]
    wording_lines = [ln for ln in header_lines if not (ln[2][0] < 60 and ln[1] == v)]
    if not wording_lines:
        raise ValueError(f"{v}: empty wording header in SHED codebook")
    wording = _collapse(" ".join(t for _, t, _ in wording_lines))

    body = blk[dash_idxs[1] + 1 :]
    tab_idx: int | None = None
    for idx, (_, t, _) in enumerate(body):
        if t in ("Numeric Label", "Label"):
            tab_idx = idx
            break

    rows: list[dict[str, object]] = []
    if tab_idx is not None:
        cur_code: int | None = None
        cur_text_parts: list[str] = []
        cur_lines: list[Line] = []
        cur_row_y: float | None = None
        for ln in body[tab_idx + 1 :]:
            pno, t, bbox = ln
            if SHED_DASH_RE.match(t):
                break
            if t == "." or t.endswith(" ."):
                break
            # Skip frequency column (x0 < 235)
            if bbox[0] < 235:
                continue
            ymid = (bbox[1] + bbox[3]) / 2.0
            m = SHED_CODE_ROW_RE.match(t)
            m_only = SHED_CODE_ONLY_RE.match(t)
            if bbox[0] < 272 and (m or m_only):
                if cur_code is not None:
                    rows.append(
                        {
                            "code": cur_code,
                            "text": _collapse(" ".join(cur_text_parts)),
                            "missing_code": cur_code < 0,
                            "lines": list(cur_lines),
                        }
                    )
                if m:
                    cur_code = int(m.group(1))
                    cur_text_parts = [m.group(2).strip()]
                else:
                    cur_code = int(m_only.group(1))
                    cur_text_parts = []
                cur_lines = [ln]
                cur_row_y = ymid
            elif cur_code is not None and (bbox[0] >= 265 or (cur_row_y is not None and abs(ymid - cur_row_y) < 4.5)):
                cur_text_parts.append(t)
                cur_lines.append(ln)
        if cur_code is not None:
            rows.append(
                {
                    "code": cur_code,
                    "text": _collapse(" ".join(cur_text_parts)),
                    "missing_code": cur_code < 0,
                    "lines": list(cur_lines),
                }
            )
    return {"page": wording_lines[0][0] + 1, "text": wording, "wording_lines": wording_lines, "rows": rows}


def shed2025_source(dataset_dir: Path) -> dict[str, object]:
    from data.shed2025_export import shed2025_final_questions

    by_var = _shed2025_group_by_var(_shed2025_pages_lines(dataset_dir / SHED_PDF))
    final = shed2025_final_questions()
    tags_by_key = _load_tags(dataset_dir)
    items: dict[str, dict[str, object]] = {}
    for row_key, f in final.items():
        v = str(f["v"])
        error = None
        block = None
        if v in by_var:
            try:
                found = _shed2025_locate(v, by_var[v])
                block = {
                    "page": found["page"],
                    "text": found["text"],
                    "options": [{"text": r["text"], "missing_code": r["missing_code"]} for r in found["rows"]],
                }
            except ValueError as exc:
                error = str(exc)
        items[row_key] = _item(f, "codebook", block, error, tags_by_key.get(row_key, []))
    return {
        "documents": {"codebook": {"label": SHED_PDF, "path": SHED_PDF}},
        "final_source": "Stata value labels, public2025.dta (SHED_public_use_data_2025_(STATA).zip)",
        "items": items,
    }


def shed2025_highlight(dataset_dir: Path, spec: dict[str, object]) -> bytes:
    """Highlight one SHED 2025 variable's wording and tabulation rows in SHED_2025codebook.pdf."""
    import pymupdf

    pdf_path = dataset_dir / SHED_PDF
    v = str(spec["v"])
    page_index = int(spec["page"]) - 1
    statuses: dict[str, str] = spec["statuses"]
    with pymupdf.open(pdf_path) as doc:
        pages = [i for i in (page_index, page_index + 1) if 0 <= i < len(doc)]
        by_var = _shed2025_group_by_var(_shed2025_pages_lines(pdf_path, pages))
        if v not in by_var:
            raise ValueError(f"{v}: no codebook block on PDF pages {[i + 1 for i in pages]}")
        found = _shed2025_locate(v, by_var[v])
        marks: list[tuple[str, Line]] = [("wording", ln) for ln in found["wording_lines"]]
        for row in found["rows"]:
            status = "missing_code" if row["missing_code"] else statuses[row["text"]]
            marks.extend((status, ln) for ln in row["lines"])
        page_objects = {index: doc[index] for index in pages}
        for status, (index, _, bbox) in marks:
            annot = page_objects[index].add_highlight_annot(pymupdf.Rect(bbox))
            annot.set_colors(stroke=HIGHLIGHT_COLORS[status])
            annot.set_opacity(0.55)
            annot.update()
        return doc.tobytes(deflate=True)


PROVIDERS = {
    WVS7_DIR_NAME: wvs7_source,
    HINTS_DIR_NAME: hints7_source,
    SHED_DIR_NAME: shed2025_source,
}
HIGHLIGHTERS = {
    WVS7_DIR_NAME: wvs7_highlight,
    HINTS_DIR_NAME: hints7_highlight,
    SHED_DIR_NAME: shed2025_highlight,
}


def build(dataset_dir: Path) -> dict[str, object]:
    provider = PROVIDERS.get(dataset_dir.name)
    if provider is None:
        return {"version": SOURCE_CHECK_VERSION, "configured": False, "documents": {}, "final_source": None, "items": {}}
    return {"version": SOURCE_CHECK_VERSION, "configured": True, **provider(dataset_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebook vs exported question comparison for the Source tab.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--highlight-spec",
        help='JSON {"v", "page", "statuses": {codebook category: status}}: write the highlighted PDF to stdout.',
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if args.highlight_spec is not None:
        highlighter = HIGHLIGHTERS[dataset_dir.name]
        sys.stdout.buffer.write(highlighter(dataset_dir, json.loads(args.highlight_spec)))
        return
    json.dump(build(dataset_dir), sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()

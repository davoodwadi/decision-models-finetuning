"""Count words in manuscript.docx before the "Web Appendix" heading.

Marketing Science Frontiers limit: 6,000 words for all pages of the paper,
including references, tables, graphs, and appendices; online appendices do
not count. So everything (title, abstract, body, tables, references,
footnotes) up to the "Web Appendix" heading is counted.

Usage: python count_word.py [path/to/manuscript.docx] [--limit 6000]
"""

import argparse
import re
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

STOP_HEADING = "web appendix"
WORD_RE = re.compile(r"\S+")


def n_words(text):
    return len(WORD_RE.findall(text))


def iter_blocks(document):
    """Yield paragraphs and tables in document order."""
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


def table_text(table):
    # Merged cells repeat across the grid; count each underlying cell once.
    seen, parts = set(), []
    for row in table.rows:
        for cell in row.cells:
            if id(cell._tc) not in seen:
                seen.add(id(cell._tc))
                parts.append(cell.text)
    return " ".join(parts)


def footnote_ids(paragraph):
    ns = paragraph._p.nsmap.get("w")
    return [ref.get(f"{{{ns}}}id") for ref in paragraph._p.iter(f"{{{ns}}}footnoteReference")]


def footnote_texts(document):
    """Map footnote id -> text, read from word/footnotes.xml if present."""
    part = next((r.target_part for r in document.part.rels.values()
                 if r.reltype.endswith("/footnotes")), None)
    if part is None:
        return {}
    from lxml import etree
    root = etree.fromstring(part.blob)
    w = root.nsmap["w"]
    return {fn.get(f"{{{w}}}id"): "".join(fn.itertext()) for fn in root.iter(f"{{{w}}}footnote")}


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=here / "manuscript.docx")
    ap.add_argument("--limit", type=int, default=6000)
    args = ap.parse_args()

    document = docx.Document(args.path)
    footnotes = footnote_texts(document)

    sections = {}  # Heading 1 title -> word count
    current = "Front matter (title, abstract, keywords)"
    found_stop = False

    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            if block.style.name == "Heading 1":
                if block.text.strip().lower() == STOP_HEADING:
                    found_stop = True
                    break
                current = block.text.strip()
            words = n_words(block.text)
            words += sum(n_words(footnotes[i]) for i in footnote_ids(block))
        else:
            words = n_words(table_text(block))
        sections[current] = sections.get(current, 0) + words

    if not found_stop:
        raise ValueError(f'No Heading 1 titled "Web Appendix" found in {args.path}')

    total = sum(sections.values())
    width = max(len(s) for s in sections)
    for name, words in sections.items():
        print(f"{name:<{width}}  {words:>6,}")
    print("-" * (width + 8))
    print(f"{'Total before Web Appendix':<{width}}  {total:>6,}")
    print(f"{'Limit':<{width}}  {args.limit:>6,}")
    diff = total - args.limit
    label = "Over limit by" if diff > 0 else "Under limit by"
    print(f"{label:<{width}}  {abs(diff):>6,}")


if __name__ == "__main__":
    main()

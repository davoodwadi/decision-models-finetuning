"""Post-process a Quarto-rendered docx for Marketing Science / INFORMS submission:

1. Bake explicit top/bottom/header cell borders into APATable tables, instead
   of relying on Word table-style conditional formatting (w:tblStylePr
   type="firstRow"), which LibreOffice and the journal's docx-to-PDF pipeline
   render inconsistently.
2. Left-justify the first (stub) column of every APATable, per APA/INFORMS style.
3. Set table and figure captions in INFORMS style: "Table 1. Caption",
   the label bold with a trailing period, text roman, flush left, kept with float.
4. Ensure a centered page-number footer and strip author metadata from core.xml
   for double-anonymous review.

The input is left untouched; the result is written alongside it with a "_"
appended to the stem, e.g. manuscript.docx -> manuscript_.docx.

Usage: python3 fix-docx.py [manuscript.docx]
"""
import re
import sys
import pathlib
import zipfile
from lxml import etree

W_URI = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = "{%s}" % W_URI
NSMAP = {"w": W_URI}

BORDER_ATTRS = {
    f"{W}val": "single",
    f"{W}sz": "12",
    f"{W}space": "0",
    f"{W}color": "000000",
}

TC_PR_PRE = {f"{W}cnfStyle", f"{W}tcW", f"{W}gridSpan", f"{W}hMerge", f"{W}vMerge"}
BORDER_SIDE_ORDER = ["top", "start", "left", "bottom", "end", "right", "insideH", "insideV"]


def ensure_tc_pr(tc):
    tcPr = tc.find(f"{W}tcPr")
    if tcPr is None:
        tcPr = etree.Element(f"{W}tcPr")
        tc.insert(0, tcPr)
    return tcPr


def ensure_tc_borders(tcPr):
    tcBorders = tcPr.find(f"{W}tcBorders")
    if tcBorders is None:
        tcBorders = etree.Element(f"{W}tcBorders")
        idx = 0
        for i, child in enumerate(tcPr):
            if child.tag in TC_PR_PRE:
                idx = i + 1
            else:
                break
        tcPr.insert(idx, tcBorders)
    return tcBorders


def set_border_side(tcBorders, side):
    tag = f"{W}{side}"
    el = tcBorders.find(tag)
    if el is None:
        el = etree.Element(tag)
        idx = BORDER_SIDE_ORDER.index(side)
        insert_at = len(tcBorders)
        for i, child in enumerate(tcBorders):
            child_side = child.tag[len(W):]
            if BORDER_SIDE_ORDER.index(child_side) > idx:
                insert_at = i
                break
        tcBorders.insert(insert_at, el)
    for k, v in BORDER_ATTRS.items():
        el.set(k, v)


def add_row_border(row, side):
    for tc in row.findall(f"{W}tc"):
        tcPr = ensure_tc_pr(tc)
        tcBorders = ensure_tc_borders(tcPr)
        set_border_side(tcBorders, side)


# Elements that, per CT_PPr's sequence, must follow w:jc.
P_PR_POST_JC = [
    "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl",
    "divId", "cnfStyle", "rPr", "sectPr", "pPrChange",
]


def set_paragraph_jc(p, value):
    """Force a paragraph's alignment.

    Pandoc plus the apa-table filter can leave several w:jc children in one
    w:pPr; the schema allows only one and renderers honour the last, so drop
    every existing one and write a single replacement.
    """
    for pPr in p.findall(f"{W}pPr")[1:]:
        p.remove(pPr)
    pPr = p.find(f"{W}pPr")
    if pPr is None:
        pPr = etree.Element(f"{W}pPr")
        p.insert(0, pPr)
    for jc in pPr.findall(f"{W}jc"):
        pPr.remove(jc)
    jc = etree.Element(f"{W}jc")
    jc.set(f"{W}val", value)
    insert_at = len(pPr)
    for i, child in enumerate(pPr):
        if child.tag[len(W):] in P_PR_POST_JC:
            insert_at = i
            break
    pPr.insert(insert_at, jc)


def left_justify_first_column(tbl):
    """Left-align the stub column. Cells spanning several grid columns are
    spanner headings, not stub entries, so they keep their own alignment."""
    val_attr = f"{W}val"
    for row in tbl.findall(f"{W}tr"):
        cells = row.findall(f"{W}tc")
        if not cells:
            continue
        tc = cells[0]
        gridSpan = tc.find(f"{W}tcPr/{W}gridSpan")
        if gridSpan is not None and val_attr in gridSpan.attrib and gridSpan.attrib[val_attr] != "1":
            continue
        for p in tc.findall(f"{W}p"):
            set_paragraph_jc(p, "left")


# --------------------------------------------------------------------------
# Float captions, INFORMS / Marketing Science style:
#
#     Table 1. Summary of the respondent-level training corpus
#
# with "Table 1." bold as a unit and the caption text roman, flush left with no
# indent.
# --------------------------------------------------------------------------

# Gap between the bold label (with period) and the caption text.
CAPTION_SEP = " "
CAPTION_SPACE_BEFORE = "240"  # 12pt above, separating it from the body text
CAPTION_SPACE_AFTER = "120"   # 6pt below, holding it against its float
CAPTION_STYLE = "ImageCaption"
CAPTION_STYLES = {"ImageCaption", "TableCaption", "Caption"}

# Longer labels first: "Table W" and "Figure W" must win over "Table" and "Figure".
CAPTION_PREFIX_RE = re.compile(
    r"^[\s ]*(Table W|Figure W|Online Resource|Table|Figure|Fig\.)"
    r"([\s ]*)(\d+(?:\.\d+)?)[\s .:]*",
    re.S,
)


def run_text(r):
    return "".join(t.text or "" for t in r.findall(f"{W}t"))


def set_run_text(r, text):
    ts = r.findall(f"{W}t")
    for extra in ts[1:]:
        r.remove(extra)
    if ts:
        t = ts[0]
    else:
        t = etree.SubElement(r, f"{W}t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text


def is_bold(r):
    return r.find(f"{W}rPr/{W}b") is not None


# Run properties are emitted in CT_RPr order; these follow w:i / w:iCs.
R_PR_POST_I = ["caps", "smallCaps", "strike", "dstrike", "outline", "shadow",
               "emboss", "imprint", "noProof", "snapToGrid", "vanish",
               "webHidden", "color", "spacing", "w", "kern", "position", "sz",
               "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
               "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
               "specVanish", "oMath"]


def set_run_roman(r):
    """Turn italics off explicitly; ImageCaption is an italic style, but
    Springer sets caption text roman."""
    rPr = r.find(f"{W}rPr")
    if rPr is None:
        rPr = etree.Element(f"{W}rPr")
        r.insert(0, rPr)
    for tag in (f"{W}i", f"{W}iCs"):
        el = rPr.find(tag)
        if el is None:
            el = etree.Element(tag)
            insert_at = len(rPr)
            for i, child in enumerate(rPr):
                if child.tag[len(W):] in R_PR_POST_I:
                    insert_at = i
                    break
            rPr.insert(insert_at, el)
        el.set(f"{W}val", "false")


def reset_caption_pPr(p):
    """Replace the caption's paragraph properties with a single, valid w:pPr."""
    style = CAPTION_STYLE
    val_attr = f"{W}val"
    for pPr in p.findall(f"{W}pPr"):
        pStyle = pPr.find(f"{W}pStyle")
        if pStyle is not None and val_attr in pStyle.attrib and pStyle.attrib[val_attr]:
            style = pStyle.attrib[val_attr]
        p.remove(pPr)

    pPr = etree.Element(f"{W}pPr")
    etree.SubElement(pPr, f"{W}pStyle").set(f"{W}val", style)
    etree.SubElement(pPr, f"{W}keepNext")  # never strand a caption from its float
    spacing = etree.SubElement(pPr, f"{W}spacing")
    spacing.set(f"{W}before", CAPTION_SPACE_BEFORE)
    spacing.set(f"{W}after", CAPTION_SPACE_AFTER)
    ind = etree.SubElement(pPr, f"{W}ind")  # flush left, no first-line indent
    ind.set(f"{W}left", "0")
    ind.set(f"{W}firstLine", "0")
    etree.SubElement(pPr, f"{W}jc").set(f"{W}val", "left")
    p.insert(0, pPr)


def is_caption(p):
    val_attr = f"{W}val"
    for pPr in p.findall(f"{W}pPr"):
        pStyle = pPr.find(f"{W}pStyle")
        if pStyle is not None and val_attr in pStyle.attrib and pStyle.attrib[val_attr] in CAPTION_STYLES:
            return True
    return False


def make_label_run(text):
    r = etree.Element(f"{W}r")
    rPr = etree.SubElement(r, f"{W}rPr")
    etree.SubElement(rPr, f"{W}b")
    etree.SubElement(rPr, f"{W}bCs")
    set_run_text(r, text)
    set_run_roman(r)
    return r


def drop_leading_chars(p, runs, count):
    """Consume `count` characters from the head of the run sequence, deleting
    runs that empty out, and return the runs that survive."""
    kept = []
    for r in runs:
        text = run_text(r)
        if count >= len(text):
            count -= len(text)
            p.remove(r)
        else:
            if count:
                set_run_text(r, text[count:])
                count = 0
            kept.append(r)
    return kept


def fix_captions(tree):
    fixed = 0
    for p in tree.iter(f"{W}p"):
        if not is_caption(p):
            continue
        runs = p.findall(f"{W}r")
        if not runs:
            continue
        m = CAPTION_PREFIX_RE.match("".join(run_text(r) for r in runs))
        if m is None:
            continue
        label, gap, number = m.group(1), m.group(2), m.group(3)
        # "Table 1" is spaced, the appendix floats' "Table W13" is not; keep
        # whichever the source used, but make the space non-breaking.
        sep = " " if gap else ""

        anchor = list(p).index(runs[0])
        kept = drop_leading_chars(p, runs, m.end())
        p.insert(anchor, make_label_run(f"{label}{sep}{number}."))
        if kept:
            set_run_text(kept[0], CAPTION_SEP + run_text(kept[0]))
        for r in kept:
            if r.find(f"{W}rPr/{W}i") is None:  # keep deliberate emphasis
                set_run_roman(r)
        reset_caption_pPr(p)
        fixed += 1
    return fixed


# --------------------------------------------------------------------------
# Equation numbers, Marketing Science style: "numbers should be given in
# parentheses, flush with the right margin of the page."
#
# Quarto writes "(1)" inside the display equation after a \quad\quad gap, so
# Word centres equation and number together. Each numbered equation paragraph
# becomes a borderless one-row, two-cell table: the display equation centred in
# the wide left cell and "(n)" flush right in a narrow right cell. The equation
# keeps display size (a tab-stop layout would make it inline). There is no
# matching empty cell on the left: the longest equations (4, 5) are ~6" wide,
# so symmetric side cells would clip them; the equation sits ~0.2" left of
# true centre instead.
# The paragraph's cross-reference bookmark moves onto the number.
# --------------------------------------------------------------------------

M_URI = "http://schemas.openxmlformats.org/officeDocument/2006/math"
M = "{%s}" % M_URI
EQ_BOOKMARK_PREFIX = "eq-"
EQ_NUMBER_CELL = 576  # twips (0.4") for the number cell


def m_run_text(r):
    return "".join(t.text or "" for t in r.findall(f"{M}t"))


def text_block_width(tree):
    """Usable text width in twips, from the (single) body section."""
    widths = set()
    for sectPr in tree.iter(f"{W}sectPr"):
        pgSz = sectPr.find(f"{W}pgSz")
        pgMar = sectPr.find(f"{W}pgMar")
        if pgSz is None or pgMar is None:
            raise ValueError("Section without w:pgSz/w:pgMar; cannot size equation rows.")
        widths.add(int(pgSz.attrib[f"{W}w"]) - int(pgMar.attrib[f"{W}left"])
                   - int(pgMar.attrib[f"{W}right"]))
    if len(widths) != 1:
        raise ValueError(f"Expected one text-block width across sections, got {sorted(widths)}.")
    return widths.pop()


def split_equation_number(omath):
    """Remove the trailing gap and "(n)" runs from an m:oMath; return "n"."""
    tail = list(omath)[-3:]
    if len(tail) != 3 or any(c.tag != f"{M}r" for c in tail):
        raise ValueError("Numbered equation does not end in m:r runs '(' n ')'.")
    opening, number, closing = (m_run_text(c) for c in tail)
    if opening != "(" or closing != ")" or not number.isdigit():
        raise ValueError(f"Expected equation to end in '(n)', got {opening!r}{number!r}{closing!r}.")
    for c in tail:
        omath.remove(c)
    # Drop the whitespace gap Quarto puts before the number.
    while len(omath) and omath[-1].tag == f"{M}r" and not m_run_text(omath[-1]).strip():
        omath.remove(omath[-1])
    if not len(omath):
        raise ValueError("Numbered equation has no math left after removing its number.")
    return number


def equation_cell(width, paragraph):
    tc = etree.Element(f"{W}tc")
    tcPr = etree.SubElement(tc, f"{W}tcPr")
    tcW = etree.SubElement(tcPr, f"{W}tcW")
    tcW.set(f"{W}w", str(width))
    tcW.set(f"{W}type", "dxa")
    etree.SubElement(tcPr, f"{W}vAlign").set(f"{W}val", "center")
    tc.append(paragraph)
    return tc


def equation_paragraph(jc, style):
    p = etree.Element(f"{W}p")
    pPr = etree.SubElement(p, f"{W}pPr")
    etree.SubElement(pPr, f"{W}pStyle").set(f"{W}val", style)
    spacing = etree.SubElement(pPr, f"{W}spacing")
    spacing.set(f"{W}before", "0")
    spacing.set(f"{W}after", "0")
    ind = etree.SubElement(pPr, f"{W}ind")
    ind.set(f"{W}left", "0")
    ind.set(f"{W}right", "0")
    ind.set(f"{W}firstLine", "0")
    etree.SubElement(pPr, f"{W}jc").set(f"{W}val", jc)
    return p


def equation_table(text_width):
    tbl = etree.Element(f"{W}tbl")
    tblPr = etree.SubElement(tbl, f"{W}tblPr")
    tblW = etree.SubElement(tblPr, f"{W}tblW")
    tblW.set(f"{W}w", str(text_width))
    tblW.set(f"{W}type", "dxa")
    etree.SubElement(tblPr, f"{W}jc").set(f"{W}val", "center")
    borders = etree.SubElement(tblPr, f"{W}tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        etree.SubElement(borders, f"{W}{side}").set(f"{W}val", "nil")
    etree.SubElement(tblPr, f"{W}tblLayout").set(f"{W}type", "fixed")
    mar = etree.SubElement(tblPr, f"{W}tblCellMar")
    for side in ("left", "right"):
        el = etree.SubElement(mar, f"{W}{side}")
        el.set(f"{W}w", "0")
        el.set(f"{W}type", "dxa")
    grid = etree.SubElement(tbl, f"{W}tblGrid")
    middle = text_width - EQ_NUMBER_CELL
    for w in (middle, EQ_NUMBER_CELL):
        etree.SubElement(grid, f"{W}gridCol").set(f"{W}w", str(w))
    return tbl, middle


def fix_equation_numbers(tree):
    text_width = text_block_width(tree)
    expected = {
        b.attrib[f"{W}name"] for b in tree.iter(f"{W}bookmarkStart")
        if b.attrib[f"{W}name"].startswith(EQ_BOOKMARK_PREFIX)
    }
    fixed = set()
    for p in list(tree.iter(f"{W}p")):
        starts = [b for b in p.findall(f"{W}bookmarkStart")
                  if b.attrib[f"{W}name"].startswith(EQ_BOOKMARK_PREFIX)]
        if not starts:
            continue
        if len(starts) != 1:
            raise ValueError(f"Paragraph holds several equation bookmarks: "
                             f"{[b.attrib[f'{W}name'] for b in starts]}.")
        start = starts[0]
        name = start.attrib[f"{W}name"]
        bid = start.attrib[f"{W}id"]
        ends = [e for e in p.findall(f"{W}bookmarkEnd") if e.attrib[f"{W}id"] == bid]
        paras = p.findall(f"{M}oMathPara")
        if len(ends) != 1 or len(paras) != 1:
            raise ValueError(f"Equation {name}: expected one m:oMathPara and one bookmarkEnd.")
        omaths = paras[0].findall(f"{M}oMath")
        if len(omaths) != 1:
            raise ValueError(f"Equation {name}: expected one m:oMath, got {len(omaths)}.")
        number = split_equation_number(omaths[0])

        pStyle = p.find(f"{W}pPr/{W}pStyle")
        style = pStyle.attrib[f"{W}val"] if pStyle is not None else "BodyText"

        tbl, middle = equation_table(text_width)
        tr = etree.SubElement(tbl, f"{W}tr")
        etree.SubElement(etree.SubElement(tr, f"{W}trPr"), f"{W}cantSplit")

        eq_p = equation_paragraph("center", style)
        eq_p.append(paras[0])  # moves the m:oMathPara out of the old paragraph
        tr.append(equation_cell(middle, eq_p))

        num_p = equation_paragraph("right", style)
        num_p.append(start)
        r = etree.SubElement(num_p, f"{W}r")
        set_run_text(r, f"({number})")
        num_p.append(ends[0])
        tr.append(equation_cell(EQ_NUMBER_CELL, num_p))

        if len(p) and any(c.tag != f"{W}pPr" for c in p):
            raise ValueError(f"Equation {name}: unexpected content left in its paragraph.")
        p.getparent().replace(p, tbl)
        fixed.add(name)
    if fixed != expected:
        raise ValueError(f"Equation bookmarks not converted: {sorted(expected - fixed)}.")
    return len(fixed)


# --------------------------------------------------------------------------
# Table column widths
#
# Pandoc sizes docx columns from the dash counts in the markdown separator row,
# i.e. from character counts, so long headers ("Observations", "Respondents")
# and numbers ("1,121,477") get broken mid-word. Re-size every APATable from
# real Times New Roman metrics: each column gets at least its widest
# unbreakable word (split at spaces and after hyphens, where Word may break),
# and the remaining width goes to columns in proportion to how much more they
# would need to sit on one line. A table that cannot fit without breaking a
# word raises instead of silently overflowing.
# --------------------------------------------------------------------------

TABLE_FONT_FILES = {
    False: "/usr/share/fonts/TTF/Times.TTF",
    True: "/usr/share/fonts/TTF/Timesbd.TTF",
}
TABLE_FONT_PT = 11          # Compact style, ms_template.docx
TABLE_CELL_MARGIN = 57      # APATable left/right cell margin, twips
TABLE_WIDTH_SLACK = 40      # twips per column, guards against rounding/kerning
WRAPPER_CELL_MARGIN = 108   # "Table" style cell margin of Quarto's float wrapper
_table_fonts = {}


def table_font(bold):
    if bold not in _table_fonts:
        from PIL import ImageFont
        # Size in tenths of a point keeps sub-point precision in getlength().
        _table_fonts[bold] = ImageFont.truetype(TABLE_FONT_FILES[bold], TABLE_FONT_PT * 10)
    return _table_fonts[bold]


def text_twips(text, bold):
    # getlength() is in pixels at size TABLE_FONT_PT*10; 1 pt = 20 twips.
    return table_font(bold).getlength(text) / 10 * 20


def cell_runs(tc):
    """(text, bold) per paragraph of a cell."""
    out = []
    for p in tc.findall(f"{W}p"):
        runs = p.findall(f".//{W}r")
        text = "".join(run_text(r) for r in runs)
        bold = bool(runs) and all(r.find(f"{W}rPr/{W}b") is not None for r in runs)
        out.append((text, bold))
    return out


def unbreakable_pieces(text):
    return [t for t in re.split(r"\s+|(?<=-)", text) if t]


def fit_table_columns(tbl, available):
    grid = tbl.find(f"{W}tblGrid")
    cols = grid.findall(f"{W}gridCol")
    n = len(cols)
    min_w = [0.0] * n
    one_line = [0.0] * n
    for row in tbl.findall(f"{W}tr"):
        cells = row.findall(f"{W}tc")
        if len(cells) != n:
            raise ValueError(f"APATable row has {len(cells)} cells for {n} grid columns.")
        for k, tc in enumerate(cells):
            if tc.find(f"{W}tcPr/{W}gridSpan") is not None:
                raise ValueError("APATable with merged cells: column fitting not implemented.")
            for text, bold in cell_runs(tc):
                for piece in unbreakable_pieces(text):
                    min_w[k] = max(min_w[k], text_twips(piece, bold))
                one_line[k] = max(one_line[k], text_twips(text.strip(), bold))
    pad = 2 * TABLE_CELL_MARGIN + TABLE_WIDTH_SLACK
    min_w = [w + pad for w in min_w]
    one_line = [w + pad for w in one_line]
    if sum(min_w) > available:
        raise ValueError(f"APATable needs {sum(min_w):.0f} twips without breaking words; "
                         f"only {available} available.")
    if sum(one_line) <= available:
        # Everything fits on one line: share the spare width in proportion.
        widths = [w * available / sum(one_line) for w in one_line]
    else:
        spare = available - sum(min_w)
        want = [o - m for o, m in zip(one_line, min_w)]
        widths = [m + spare * w / sum(want) for m, w in zip(min_w, want)]
    widths = [int(w) for w in widths]
    widths[-1] += available - sum(widths)
    for col, w in zip(cols, widths):
        col.set(f"{W}w", str(w))
    for row in tbl.findall(f"{W}tr"):
        for tc, w in zip(row.findall(f"{W}tc"), widths):
            tcPr = ensure_tc_pr(tc)
            tcW = tcPr.find(f"{W}tcW")
            if tcW is None:
                tcW = etree.Element(f"{W}tcW")
                insert_at = 1 if tcPr.find(f"{W}cnfStyle") is not None else 0
                tcPr.insert(insert_at, tcW)
            tcW.set(f"{W}w", str(w))
            tcW.set(f"{W}type", "dxa")


def table_available_width(tbl, text_width):
    """Width the table fills: the text block, or Quarto's wrapper cell inside it."""
    parent = tbl.getparent()
    if parent.tag == f"{W}body":
        return text_width
    if parent.tag != f"{W}tc":
        raise ValueError(f"APATable inside unexpected element {parent.tag}.")
    wrapper = parent.getparent().getparent()
    if len(wrapper.find(f"{W}tblGrid").findall(f"{W}gridCol")) != 1:
        raise ValueError("APATable wrapper is not a one-cell float table.")
    return text_width - 2 * WRAPPER_CELL_MARGIN


FALSE_VALUES = {"false", "off", "0"}


def is_header_row(row):
    tblHeader = row.find(f"{W}trPr/{W}tblHeader")
    if tblHeader is None:
        return False
    val_attr = f"{W}val"
    if val_attr not in tblHeader.attrib:
        return True
    return tblHeader.attrib[val_attr].lower() not in FALSE_VALUES


# --------------------------------------------------------------------------
# Page numbers
#
# Quarto/pandoc emits no header or footer part, and the reference doc has none
# either, so the output has no page numbers. Add a footer part holding a PAGE
# field and point every section at it.
# --------------------------------------------------------------------------

R_URI = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
R = "{%s}" % R_URI
PKG_REL_URI = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_URI = "http://schemas.openxmlformats.org/package/2006/content-types"

FOOTER_PART = "word/footer-pagenum.xml"
FOOTER_REL_TYPE = f"{R_URI}/footer"
FOOTER_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"
)

# Where the number sits: "center" (Springer/JBE house style) or "right" (APA 7).
PAGE_NUMBER_ALIGN = "center"
# Distance from the page edge to the footer, in twips (720 = 0.5").
FOOTER_MARGIN = "720"

FOOTER_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W_URI}">
  <w:p>
    <w:pPr><w:pStyle w:val="Footer"/><w:jc w:val="{PAGE_NUMBER_ALIGN}"/></w:pPr>
    <w:r><w:fldChar w:fldCharType="begin"/></w:r>
    <w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>
    <w:r><w:fldChar w:fldCharType="separate"/></w:r>
    <w:r><w:t>1</w:t></w:r>
    <w:r><w:fldChar w:fldCharType="end"/></w:r>
  </w:p>
</w:ftr>
""".encode("utf-8")

# Per CT_SectPr, header/footer references come before everything else.
SECT_PR_REF_TAGS = {f"{W}headerReference", f"{W}footerReference"}


def serialize(tree):
    return etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)


def next_rel_id(rels_tree):
    used = set()
    for rel in rels_tree:
        rid = rel.attrib["Id"]
        if rid.startswith("rId") and rid[3:].isdigit():
            used.add(int(rid[3:]))
    return "rId%d" % (max(used, default=0) + 1)


def add_footer_relationship(rels_bytes):
    tree = etree.fromstring(rels_bytes)
    for rel in tree:
        if rel.attrib["Target"] == FOOTER_PART.split("/", 1)[1]:
            return rels_bytes, rel.attrib["Id"]
    rid = next_rel_id(tree)
    etree.SubElement(
        tree,
        f"{{{PKG_REL_URI}}}Relationship",
        Id=rid,
        Type=FOOTER_REL_TYPE,
        Target=FOOTER_PART.split("/", 1)[1],
    )
    return serialize(tree), rid


def add_footer_content_type(ct_bytes):
    tree = etree.fromstring(ct_bytes)
    part = "/" + FOOTER_PART
    for child in tree:
        if "PartName" in child.attrib and child.attrib["PartName"] == part:
            return ct_bytes
    etree.SubElement(
        tree,
        f"{{{CT_URI}}}Override",
        PartName=part,
        ContentType=FOOTER_CONTENT_TYPE,
    )
    return serialize(tree)


DC_URI = "http://purl.org/dc/elements/1.1/"
CP_URI = "http://schemas.openxmlformats.org/package/core-properties"
CP_URI_2006 = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"


def strip_author_metadata(core_bytes):
    tree = etree.fromstring(core_bytes)
    for tag in (
        f"{{{DC_URI}}}creator",
        f"{{{CP_URI}}}lastModifiedBy",
        f"{{{CP_URI_2006}}}lastModifiedBy",
    ):
        for el in tree.findall(tag):
            el.text = ""
    return serialize(tree)


def reference_footer(tree, rid):
    """Point every section at the footer part and give it room to breathe."""
    sections = 0
    type_attr = f"{W}type"
    footer_attr = f"{W}footer"
    for sectPr in tree.iter(f"{W}sectPr"):
        existing = [
            el for el in sectPr.findall(f"{W}footerReference")
            if type_attr not in el.attrib or el.attrib[type_attr] == "default"
        ]
        if existing:
            for el in existing:
                el.set(f"{R}id", rid)
        else:
            ref = etree.Element(f"{W}footerReference")
            ref.set(type_attr, "default")
            ref.set(f"{R}id", rid)
            insert_at = 0
            for i, child in enumerate(sectPr):
                if child.tag in SECT_PR_REF_TAGS:
                    insert_at = i + 1
                else:
                    break
            sectPr.insert(insert_at, ref)
        pgMar = sectPr.find(f"{W}pgMar")
        if pgMar is not None and (footer_attr not in pgMar.attrib or pgMar.attrib[footer_attr] in ("0", "")):
            pgMar.set(footer_attr, FOOTER_MARGIN)
        sections += 1
    return sections


def fix_document_xml(xml_bytes, footer_rid):
    tree = etree.fromstring(xml_bytes)
    # lxml needs the r: prefix declared on the root to serialize w:footerReference.
    etree.register_namespace("r", R_URI)
    tables = tree.findall(f".//{W}tbl")
    fixed = 0
    val_attr = f"{W}val"
    for tbl in tables:
        tblStyle = tbl.find(f"{W}tblPr/{W}tblStyle")
        if tblStyle is None or val_attr not in tblStyle.attrib or tblStyle.attrib[val_attr] != "APATable":
            continue
        rows = tbl.findall(f"{W}tr")
        if not rows:
            continue
        header_rows = [r for r in rows if is_header_row(r)] or [rows[0]]
        add_row_border(header_rows[0], "top")
        add_row_border(header_rows[-1], "bottom")
        add_row_border(rows[-1], "bottom")
        left_justify_first_column(tbl)
        fit_table_columns(tbl, table_available_width(tbl, text_block_width(tree)))
        fixed += 1
    captions = fix_captions(tree)
    equations = fix_equation_numbers(tree)
    sections = reference_footer(tree, footer_rid)
    return serialize(tree), fixed, captions, equations, sections


def main():
    docx_path = sys.argv[1] if len(sys.argv) > 1 else "manuscript.docx"
    # Leave the Quarto output untouched; write a sibling with a "_" suffix.
    src = pathlib.Path(docx_path)
    out_path = src.with_name(src.stem + "_" + src.suffix)

    rels_name = "word/_rels/document.xml.rels"
    ct_name = "[Content_Types].xml"
    core_name = "docProps/core.xml"

    with zipfile.ZipFile(docx_path, "r") as zin:
        names = set(zin.namelist())
        new_rels, footer_rid = add_footer_relationship(zin.read(rels_name))
        new_ct = add_footer_content_type(zin.read(ct_name))
        new_document, fixed, captions, equations, sections = fix_document_xml(
            zin.read("word/document.xml"), footer_rid
        )
        replacements = {
            "word/document.xml": new_document,
            rels_name: new_rels,
            ct_name: new_ct,
            FOOTER_PART: FOOTER_XML,
        }
        if core_name in names:
            replacements[core_name] = strip_author_metadata(zin.read(core_name))

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            written = set()
            for item in zin.infolist():
                if item.filename in replacements:
                    zout.writestr(item, replacements[item.filename])
                else:
                    zout.writestr(item, zin.read(item.filename))
                written.add(item.filename)
            for name, data in replacements.items():
                if name not in written:
                    zout.writestr(name, data)

    print(f"Baked borders, left-aligned stub column and fitted column widths in {fixed} "
          f"APATable table(s).")
    print(f"Reformatted {captions} float caption(s) to INFORMS style.")
    print(f"Set {equations} equation number(s) flush right.")
    print(f"Added a {PAGE_NUMBER_ALIGN}ed page-number footer to {sections} section(s).")
    print(f"Wrote {out_path}.")


if __name__ == "__main__":
    main()

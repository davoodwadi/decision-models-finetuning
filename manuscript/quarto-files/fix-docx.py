"""Post-process a Quarto-rendered docx for submission:

1. Bake explicit top/bottom/header cell borders into APATable tables, instead
   of relying on Word table-style conditional formatting (w:tblStylePr
   type="firstRow"), which LibreOffice and the journal's docx-to-PDF pipeline
   render inconsistently.
2. Left-justify the first (stub) column of every APATable, per APA style.
3. Set table and figure captions in Springer/JBE style: "Table 1  Caption",
   the label bold as a unit, text roman, flush left, kept with its float.
4. Add a page-number footer, which neither pandoc nor the reference doc emits.

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
    for row in tbl.findall(f"{W}tr"):
        cells = row.findall(f"{W}tc")
        if not cells:
            continue
        tc = cells[0]
        gridSpan = tc.find(f"{W}tcPr/{W}gridSpan")
        if gridSpan is not None and gridSpan.get(f"{W}val", "1") != "1":
            continue
        for p in tc.findall(f"{W}p"):
            set_paragraph_jc(p, "left")


# --------------------------------------------------------------------------
# Float captions, Springer/JBE house style:
#
#     Table 1  Alternatives used in Study 1
#
# with "Table 1" bold as a unit and the caption text roman, flush left with no
# indent.
#
# Two shapes reach us. Ordinary floats arrive pre-split by Quarto's
# `tbl-title: "**Table**"`, which bolds only the literal word and leaves the
# number in a following, unbolded run. The custom appendix floats (apptbl,
# appfig) arrive as one plain run, "Table W1 Full logistic regression ...",
# with nothing bold at all. Keying off the ImageCaption paragraph style catches
# both, and rebuilding the label run from scratch treats them identically.
# --------------------------------------------------------------------------

# Gap between the bold label and the caption text.
CAPTION_SEP = "  "
CAPTION_SPACE_BEFORE = "240"  # 12pt above, separating it from the body text
CAPTION_SPACE_AFTER = "120"   # 6pt below, holding it against its float
CAPTION_STYLE = "ImageCaption"

# Longer labels first: "Table W" must win over "Table".
CAPTION_PREFIX_RE = re.compile(
    r"^[\s ]*(Table W|Figure W|Online Resource|Table|Fig\.)"
    r"([\s ]*)(\d+(?:\.\d+)?)[\s :]*",
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
    """Replace the caption's paragraph properties with a single, valid w:pPr.

    Pandoc leaves two sibling w:pPr here; only the first is honoured, which is
    why the ImageCaption style and its left alignment are currently dropped and
    the caption renders centred. Children are emitted in CT_PPr order.
    """
    style = CAPTION_STYLE
    for pPr in p.findall(f"{W}pPr"):
        pStyle = pPr.find(f"{W}pStyle")
        if pStyle is not None and pStyle.get(f"{W}val"):
            style = pStyle.get(f"{W}val")
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
    for pPr in p.findall(f"{W}pPr"):
        pStyle = pPr.find(f"{W}pStyle")
        if pStyle is not None and pStyle.get(f"{W}val") == CAPTION_STYLE:
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
        p.insert(anchor, make_label_run(f"{label}{sep}{number}"))
        if kept:
            set_run_text(kept[0], CAPTION_SEP + run_text(kept[0]))
        for r in kept:
            if r.find(f"{W}rPr/{W}i") is None:  # keep deliberate emphasis
                set_run_roman(r)
        reset_caption_pPr(p)
        fixed += 1
    return fixed


FALSE_VALUES = {"false", "off", "0"}


def is_header_row(row):
    tblHeader = row.find(f"{W}trPr/{W}tblHeader")
    if tblHeader is None:
        return False
    val = tblHeader.get(f"{W}val")
    return val is None or val.lower() not in FALSE_VALUES


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
    <w:pPr><w:jc w:val="{PAGE_NUMBER_ALIGN}"/></w:pPr>
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
        rid = rel.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            used.add(int(rid[3:]))
    return "rId%d" % (max(used, default=0) + 1)


def add_footer_relationship(rels_bytes):
    tree = etree.fromstring(rels_bytes)
    for rel in tree:
        if rel.get("Target") == FOOTER_PART.split("/", 1)[1]:
            return rels_bytes, rel.get("Id")
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
        if child.get("PartName") == part:
            return ct_bytes
    etree.SubElement(
        tree,
        f"{{{CT_URI}}}Override",
        PartName=part,
        ContentType=FOOTER_CONTENT_TYPE,
    )
    return serialize(tree)


def reference_footer(tree, rid):
    """Point every section at the footer part and give it room to breathe."""
    sections = 0
    for sectPr in tree.iter(f"{W}sectPr"):
        existing = [
            el for el in sectPr.findall(f"{W}footerReference")
            if el.get(f"{W}type") in (None, "default")
        ]
        if existing:
            for el in existing:
                el.set(f"{R}id", rid)
        else:
            ref = etree.Element(f"{W}footerReference")
            ref.set(f"{W}type", "default")
            ref.set(f"{R}id", rid)
            insert_at = 0
            for i, child in enumerate(sectPr):
                if child.tag in SECT_PR_REF_TAGS:
                    insert_at = i + 1
                else:
                    break
            sectPr.insert(insert_at, ref)
        pgMar = sectPr.find(f"{W}pgMar")
        if pgMar is not None and pgMar.get(f"{W}footer", "0") in ("0", ""):
            pgMar.set(f"{W}footer", FOOTER_MARGIN)
        sections += 1
    return sections


def fix_document_xml(xml_bytes, footer_rid):
    tree = etree.fromstring(xml_bytes)
    # lxml needs the r: prefix declared on the root to serialize w:footerReference.
    etree.register_namespace("r", R_URI)
    tables = tree.findall(f".//{W}tbl")
    fixed = 0
    for tbl in tables:
        tblStyle = tbl.find(f"{W}tblPr/{W}tblStyle")
        if tblStyle is None or tblStyle.get(f"{W}val") != "APATable":
            continue
        rows = tbl.findall(f"{W}tr")
        if not rows:
            continue
        header_rows = [r for r in rows if is_header_row(r)] or [rows[0]]
        add_row_border(header_rows[0], "top")
        add_row_border(header_rows[-1], "bottom")
        add_row_border(rows[-1], "bottom")
        left_justify_first_column(tbl)
        fixed += 1
    captions = fix_captions(tree)
    sections = reference_footer(tree, footer_rid)
    return serialize(tree), fixed, captions, sections


def main():
    docx_path = sys.argv[1] if len(sys.argv) > 1 else "manuscript.docx"
    # Leave the Quarto output untouched; write a sibling with a "_" suffix.
    src = pathlib.Path(docx_path)
    out_path = src.with_name(src.stem + "_" + src.suffix)

    rels_name = "word/_rels/document.xml.rels"
    ct_name = "[Content_Types].xml"

    with zipfile.ZipFile(docx_path, "r") as zin:
        new_rels, footer_rid = add_footer_relationship(zin.read(rels_name))
        new_ct = add_footer_content_type(zin.read(ct_name))
        new_document, fixed, captions, sections = fix_document_xml(
            zin.read("word/document.xml"), footer_rid
        )
        replacements = {
            "word/document.xml": new_document,
            rels_name: new_rels,
            ct_name: new_ct,
            FOOTER_PART: FOOTER_XML,
        }

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            written = set()
            for item in zin.infolist():
                data = replacements.get(item.filename)
                zout.writestr(item, zin.read(item.filename) if data is None else data)
                written.add(item.filename)
            for name, data in replacements.items():
                if name not in written:
                    zout.writestr(name, data)

    print(f"Baked explicit borders and left-aligned stub column in {fixed} "
          f"APATable table(s).")
    print(f"Reformatted {captions} float caption(s) to Springer/JBE style.")
    print(f"Added a {PAGE_NUMBER_ALIGN}ed page-number footer to {sections} section(s).")
    print(f"Wrote {out_path}.")


if __name__ == "__main__":
    main()

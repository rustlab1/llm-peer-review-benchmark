"""Convert a PMC JATS XML article into plain markdown for the review experiment.

Keeps: title, abstract (structured sections as 'Label: text'), body sections, figure legends and tables
(tables as markdown pipe tables with their footnotes). Drops: author list, affiliations, key points box,
reference list, citation markers, acknowledgements, funding, conflicts, author contributions and other back matter.
The output is deterministic for a given XML file, so the same script can rebuild the text on another machine.

Usage: python3 jats2md.py input.xml output.md
"""
import re
import sys
import xml.etree.ElementTree as ET

CITE = "\u0001"          # placeholder for a removed citation marker
MML = "{http://www.w3.org/1998/Math/MathML}"
XLINK = "{http://www.w3.org/1999/xlink}"


def local(tag):
    """Tag name without namespace."""
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def is_cite_only(el):
    """True if an element (e.g. <sup>) holds nothing but citation xrefs and punctuation."""
    txt = []
    has_xref = False
    for node in el.iter():
        if local(node.tag) == "xref" and node.get("ref-type") in ("bibr", "ref"):
            has_xref = True
    if not has_xref:
        return False
    # text outside the bibr xrefs must be punctuation only
    def outside(e):
        out = [e.text or ""]
        for c in e:
            if not (local(c.tag) == "xref" and c.get("ref-type") in ("bibr", "ref")):
                out.append(outside(c))
            out.append(c.tail or "")
        return "".join(out)
    return re.fullmatch(r"[\s,;\-–—\[\]\(\)]*", outside(el)) is not None


def inline(el):
    """Flatten an element's mixed content to text, handling citations, sub/sup, math and line breaks."""
    t = local(el.tag)
    if t == "xref" and el.get("ref-type") in ("bibr", "ref"):
        return CITE
    if t in ("sup",) and is_cite_only(el):
        return CITE
    if t in ("fn", "table-wrap", "fig", "boxed-text", "supplementary-material", "graphic", "alternatives") and t != "alternatives":
        return ""
    if t == "alternatives":
        # prefer a text-like alternative
        for c in el:
            if local(c.tag) in ("tex-math",):
                continue
            return inline(c)
        return ""
    if t == "math":
        return "".join(el.itertext()).strip()
    if t == "break":
        return " "
    parts = [el.text or ""]
    for c in el:
        parts.append(inline(c))
        parts.append(c.tail or "")
    s = "".join(parts)
    if t == "sup" and s.strip():
        s = "^" + s.strip()
    elif t == "sub" and s.strip():
        s = "_" + s.strip()
    return s


def clean_text(s):
    """Remove citation placeholders with their brackets/punctuation and normalise whitespace."""
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    # bracketed or parenthesised groups that contain only citations
    s = re.sub(r"\s*\[\s*(?:" + CITE + r"[\s,;\-–]*)+\]", "", s)
    s = re.sub(r"\s*\(\s*(?:" + CITE + r"[\s,;\-–]*)+\)", "", s)
    # runs of citation placeholders joined by commas or dashes
    s = re.sub(r"\s*" + CITE + r"(?:[\s,;\-–]*" + CITE + r")*", "", s)
    s = s.replace(CITE, "")
    s = re.sub(r"[ \t\r\n]+", " ", s)
    s = re.sub(r" ([,.;:)])", r"\1", s)
    s = re.sub(r"\( ", "(", s)
    return s.strip()


def para(el):
    return clean_text(inline(el))


def table_to_md(tw):
    """Render a <table-wrap> as label/caption, a pipe table and its footnotes."""
    out = []
    label = clean_text(inline(tw.find("./label"))) if tw.find("./label") is not None else ""
    cap = tw.find("./caption")
    cap_txt = ""
    if cap is not None:
        cap_txt = " ".join(para(c) for c in cap if local(c.tag) in ("title", "p"))
    head = (label + ". " if label and not label.endswith(".") else (label + " " if label else "")) + cap_txt
    out.append("**" + head.strip() + "**")
    out.append("")
    for table in tw.iter():
        if local(table.tag) != "table":
            continue
        grid = []          # list of rows; each row a list of cell strings
        pending = {}       # (row_index, col_index) -> text for rowspans
        rows = [r for r in table.iter() if local(r.tag) == "tr"]
        nhead = sum(1 for r in table.iter() if local(r.tag) == "thead" for _ in r.iter() if local(_.tag) == "tr")
        for ri, tr in enumerate(rows):
            row = []
            ci = 0
            cells = [c for c in tr if local(c.tag) in ("td", "th")]
            k = 0
            while k < len(cells) or (ri, ci) in pending:
                if (ri, ci) in pending:
                    row.append(pending.pop((ri, ci)))
                    ci += 1
                    continue
                c = cells[k]
                k += 1
                txt = clean_text(inline(c)).replace("|", "/")
                cs = int(c.get("colspan", "1") or 1)
                rs = int(c.get("rowspan", "1") or 1)
                for j in range(cs):
                    row.append(txt if j == 0 else "")
                    for r2 in range(1, rs):
                        pending[(ri + r2, ci + j)] = ""
                ci += cs
            grid.append(row)
        if not grid:
            continue
        width = max(len(r) for r in grid)
        grid = [r + [""] * (width - len(r)) for r in grid]
        hdr_n = max(1, nhead)
        header = grid[0]
        if hdr_n > 1:
            # merge multi-row headers column-wise
            header = []
            for j in range(width):
                bits = [grid[i][j] for i in range(hdr_n) if grid[i][j]]
                header.append(" / ".join(bits))
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * width)
        for r in grid[hdr_n:]:
            out.append("| " + " | ".join(r) + " |")
        out.append("")
    foot = tw.find("./table-wrap-foot")
    if foot is not None:
        notes = []
        for node in foot.iter():
            if local(node.tag) in ("p",):
                t = para(node)
                if t:
                    notes.append(t)
        if not notes:
            t = para(foot)
            if t:
                notes.append(t)
        if notes:
            out.append("Note: " + " ".join(notes))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def fig_legend(fig):
    label = clean_text(inline(fig.find("./label"))) if fig.find("./label") is not None else "Figure"
    cap = fig.find("./caption")
    txt = ""
    if cap is not None:
        txt = " ".join(para(c) for c in cap if local(c.tag) in ("title", "p"))
    return f"{label}. {txt}".replace("..", ".").strip()


class Doc:
    def __init__(self):
        self.lines = []
        self.tables = []
        self.figs = []

    def add(self, s=""):
        self.lines.append(s)


SKIP_SEC = re.compile(r"^(acknowledg|funding|author contribution|conflict|competing|data (sharing|availability)|"
                      r"code availability|declaration|disclosure|supplement|additional information|reporting summary|"
                      r"open access|online content|consortia|group information|role of the funder|article information|"
                      r"references?$|ethics declarations$)", re.I)


def walk_sec(sec, doc, level):
    title_el = sec.find("./title")
    title = para(title_el) if title_el is not None else ""
    if title and SKIP_SEC.search(title):
        return
    if title:
        doc.add("#" * min(level, 4) + " " + title)
        doc.add("")
    for c in sec:
        t = local(c.tag)
        if t == "sec":
            walk_sec(c, doc, level + 1)
        elif t == "p":
            # floats embedded in paragraphs
            for f in c.iter():
                if local(f.tag) == "table-wrap":
                    doc.tables.append(f)
                elif local(f.tag) == "fig":
                    doc.figs.append(f)
            txt = para(c)
            if txt:
                doc.add(txt)
                doc.add("")
        elif t in ("list",):
            for it in c.iter():
                if local(it.tag) == "list-item":
                    txt = clean_text(" ".join(para(p) for p in it if local(p.tag) == "p") or inline(it))
                    if txt:
                        doc.add("- " + txt)
            doc.add("")
        elif t == "table-wrap":
            doc.tables.append(c)
        elif t == "fig":
            doc.figs.append(c)
        elif t in ("disp-quote", "boxed-text"):
            # boxed text in the body (rare) kept as paragraphs
            for p in c.iter():
                if local(p.tag) == "p":
                    txt = para(p)
                    if txt:
                        doc.add(txt)
                        doc.add("")
        elif t == "table-wrap-group":
            for tw in c.iter():
                if local(tw.tag) == "table-wrap":
                    doc.tables.append(tw)


def convert(xml_path):
    root = ET.parse(xml_path).getroot()
    art = root if local(root.tag) == "article" else root.find(".//{*}article")
    if art is None:
        art = root.find(".//article")
    meta = art.find(".//front/article-meta")
    doc = Doc()
    title = para(meta.find("./title-group/article-title"))
    doc.add("# " + title)
    doc.add("")
    # main abstract: first <abstract> without abstract-type, else the first one that is not key points/teaser
    abstracts = meta.findall("./abstract")
    main = None
    for a in abstracts:
        if not a.get("abstract-type"):
            main = a
            break
    if main is None:
        for a in abstracts:
            if a.get("abstract-type") not in ("key-points", "teaser", "graphical", "toc", "summary", "author-summary", "precis"):
                main = a
                break
    if main is not None:
        doc.add("## Abstract")
        doc.add("")
        secs = main.findall("./sec")
        if secs:
            for s in secs:
                lab = para(s.find("./title")) if s.find("./title") is not None else ""
                body = " ".join(para(p) for p in s.findall("./p"))
                if lab and re.search(r"trial registration|clinical trial registration|registration", lab, re.I):
                    doc.add(f"{lab.rstrip(':')}: identifier withheld for review.")
                else:
                    doc.add((lab.rstrip(":") + ": " if lab else "") + body)
                doc.add("")
        else:
            for p in main.findall("./p"):
                doc.add(para(p))
                doc.add("")
    body = art.find("./body")
    if body is not None:
        for c in body:
            t = local(c.tag)
            if t == "sec":
                walk_sec(c, doc, 2)
            elif t == "p":
                for f in c.iter():
                    if local(f.tag) == "table-wrap":
                        doc.tables.append(f)
                    elif local(f.tag) == "fig":
                        doc.figs.append(f)
                txt = para(c)
                if txt:
                    doc.add(txt)
                    doc.add("")
            elif t == "table-wrap":
                doc.tables.append(c)
            elif t == "fig":
                doc.figs.append(c)
    fg = art.find("./floats-group")
    if fg is not None:
        for f in fg.iter():
            if local(f.tag) == "table-wrap":
                doc.tables.append(f)
            elif local(f.tag) == "fig":
                doc.figs.append(f)
    # de-duplicate floats while keeping order
    seen = set()
    figs = []
    for f in doc.figs:
        if id(f) not in seen:
            seen.add(id(f))
            figs.append(f)
    tables = []
    for t in doc.tables:
        if id(t) not in seen:
            seen.add(id(t))
            tables.append(t)
    if figs:
        doc.add("## Figure legends")
        doc.add("")
        for f in figs:
            doc.add(fig_legend(f))
            doc.add("")
    if tables:
        doc.add("## Tables")
        doc.add("")
        for t in tables:
            doc.add(table_to_md(t))
    text = "\n".join(doc.lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    # registry identifiers are always withheld
    text = re.sub(r"\b(NCT\d{8}|ISRCTN\d{6,}|ChiCTR[\w-]+|ACTRN\d+|jRCT\w+|UMIN\d+|CTRI/[\d/]+|DRKS\d+|EudraCT[\s:]*\d{4}-\d{6}-\d{2}|\d{4}-\d{6}-\d{2}|KCT\d+|IRCT\w+|PACTR\d+|NL\d{4,}|TCTR\d+|CRD\d+|EUCT[\s:]*[\d-]+)\b",
                  "[identifier withheld for review]", text)
    return text


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    open(dst, "w", encoding="utf-8").write(convert(src))

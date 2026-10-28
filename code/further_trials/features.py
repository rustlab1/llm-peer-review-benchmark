"""Summarize structural features of each candidate PMC XML to judge eligibility."""
import json, re, glob
import xml.etree.ElementTree as ET
cand = {c["pmc"]: c for c in json.load(open("cand/esummary.json"))}
rows = []
for f in sorted(glob.glob("raw/PMC*.xml")):
    pmc = f.split("/")[-1][:-4]
    try:
        root = ET.parse(f).getroot()
    except Exception as e:
        rows.append({"pmc": pmc, "err": str(e)}); continue
    art = root.find(".//article")
    if art is None: art = root
    atype = art.get("article-type")
    body = art.find(".//body")
    btxt = " ".join(body.itertext()) if body is not None else ""
    abst = art.find(".//abstract")
    atxt = " ".join(abst.itertext()) if abst is not None else ""
    lic = art.find(".//license")
    lictype = (lic.get("license-type") if lic is not None else None) or ""
    lichref = ""
    if lic is not None:
        for k, v in lic.attrib.items():
            if "href" in k: lichref = v
        lt = " ".join(lic.itertext())
        m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)", lt + lichref)
        if m: lichref = m.group(1)
    dates = {}
    for d in art.findall(".//article-meta/pub-date"):
        t = d.get("pub-type") or d.get("date-type") or "?"
        y, mo, da = (d.findtext("year"), d.findtext("month"), d.findtext("day"))
        dates[t] = "-".join(x for x in (y, mo, da) if x)
    low = btxt.lower()
    feat = {
        "pmc": pmc, "journal": cand.get(pmc, {}).get("journal"), "title": cand.get(pmc, {}).get("title"),
        "pmid": cand.get(pmc, {}).get("pmid"), "doi": cand.get(pmc, {}).get("doi"),
        "atype": atype, "license": lictype or lichref, "dates": dates,
        "body_words": len(btxt.split()), "n_tables": len(art.findall(".//body//table-wrap")) + len(art.findall(".//floats-group//table-wrap")),
        "n_sec": len(body.findall("./sec")) if body is not None else 0,
        "kw_samplesize": bool(re.search(r"sample size|power calculation|statistical power|% power|powered", low)),
        "kw_ethics": bool(re.search(r"institutional review board|ethics committee|ethical approval|ethics board|irb", low)),
        "kw_itt": bool(re.search(r"intention[- ]to[- ]treat|intent[- ]to[- ]treat|full analysis set", low)),
        "kw_random": bool(re.search(r"randomi[sz]ation|randomly assigned|randomi[sz]ed", low)),
        "kw_screened": bool(re.search(r"screened|assessed for eligibility", low)),
        "kw_limit": bool(re.search(r"limitation", low)),
        "abs_words": len(atxt.split()),
    }
    rows.append(feat)
json.dump(rows, open("cand/features.json", "w"), indent=1)
print(len(rows))

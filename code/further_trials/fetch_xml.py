"""Fetch PMC JATS XML for candidate articles into raw/<PMCID>.xml (efetch db=pmc)."""
import json, os, sys, time, urllib.parse, urllib.request
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
def fetch(pmc):
    path = f"raw/{pmc}.xml"
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    q = urllib.parse.urlencode({"db": "pmc", "id": pmc.replace("PMC", ""), "retmode": "xml"})
    for i in range(5):
        try:
            data = urllib.request.urlopen(E + q, timeout=90).read()
            open(path, "wb").write(data)
            return
        except Exception as e:
            time.sleep(3 + 3 * i)
    print("FAILED", pmc, file=sys.stderr)
if __name__ == "__main__":
    ids = sys.argv[1:] or [c["pmc"] for c in json.load(open("cand/esummary.json")) if c["pmc"]]
    for k, p in enumerate(ids):
        fetch(p); time.sleep(0.35)
    print("done", len(ids))

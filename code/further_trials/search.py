"""Search PubMed for candidate RCTs: open-access full text in PMC, leading clinical journals,
entered/published from 2026-07-15 onward."""
import json, time, urllib.parse, urllib.request
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
JOURNALS = ["JAMA", "JAMA Netw Open", "JAMA Intern Med", "JAMA Neurol", "JAMA Cardiol", "JAMA Oncol",
            "JAMA Psychiatry", "JAMA Pediatr", "JAMA Surg", "JAMA Dermatol", "JAMA Ophthalmol", "JAMA Otolaryngol Head Neck Surg",
            "BMJ", "Lancet", "Lancet Neurol", "Lancet Oncol", "Lancet Infect Dis", "Lancet Respir Med", "Lancet Psychiatry",
            "Lancet Diabetes Endocrinol", "Lancet Gastroenterol Hepatol", "Lancet Glob Health", "Lancet Public Health",
            "Lancet Child Adolesc Health", "Lancet HIV", "Lancet Rheumatol", "Lancet Haematol", "Lancet Healthy Longev",
            "EClinicalMedicine", "Lancet Reg Health Eur", "Lancet Reg Health Am",
            "N Engl J Med", "NEJM Evid", "Nat Med", "PLoS Med", "Ann Intern Med", "Circulation", "Eur Heart J",
            "J Clin Oncol", "Arthritis Rheumatol", "Ann Rheum Dis", "Gastroenterology", "Gut", "Diabetes Care",
            "Am J Respir Crit Care Med", "Intensive Care Med", "Stroke", "Neurology", "Br J Psychiatry", "Am J Psychiatry",
            "Clin Infect Dis", "Kidney Int", "J Am Soc Nephrol", "Hepatology", "J Hepatol", "Eur Respir J", "Thorax",
            "Brain", "Alzheimers Dement", "BMC Med", "CMAJ", "Ann Surg", "Br J Surg", "Anesthesiology", "Br J Anaesth",
            "Crit Care Med", "Pediatrics", "Obstet Gynecol", "Am J Obstet Gynecol", "Hum Reprod", "J Am Coll Cardiol",
            "Blood", "J Allergy Clin Immunol", "Am J Clin Nutr", "Addiction", "Psychol Med", "World Psychiatry"]
jq = " OR ".join(f'"{j}"[ta]' for j in JOURNALS)
term = (f'("Randomized Controlled Trial"[pt]) AND ("pubmed pmc"[sb]) AND ({jq}) '
        f'AND ("2026/07/15"[edat] : "2026/10/26"[edat])')
def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            time.sleep(2 + i * 2)
    raise RuntimeError(url)
q = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": 2000, "retmode": "json"})
r = json.loads(get(E + "esearch.fcgi?" + q))
ids = r["esearchresult"]["idlist"]
print("hits:", r["esearchresult"]["count"], len(ids))
out = []
for i in range(0, len(ids), 200):
    q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids[i:i+200]), "retmode": "json"})
    s = json.loads(get(E + "esummary.fcgi?" + q))["result"]
    for pid in s["uids"]:
        d = s[pid]
        art = {x["idtype"]: x["value"] for x in d.get("articleids", [])}
        out.append({"pmid": pid, "title": d.get("title"), "journal": d.get("source"), "pubdate": d.get("pubdate"),
                    "epubdate": d.get("epubdate"), "pmc": art.get("pmc") or art.get("pmcid"), "doi": art.get("doi"),
                    "pubtype": d.get("pubtype")})
    time.sleep(0.4)
json.dump(out, open("cand/esummary.json", "w"), indent=1)
from collections import Counter
print(Counter(o["journal"] for o in out).most_common(60))

import json, re, sys
from collections import Counter
sys.path.insert(0, '.')
from constitutional_bioguard.config import DATA_PROCESSED, DATA_EXTERNAL
from constitutional_bioguard.dual_mode import DualModeGuard, has_bio_context

def load(n):
    for d in (DATA_PROCESSED, DATA_EXTERNAL):
        p = d / (n + ".jsonl")
        if p.exists(): return [json.loads(l) for l in open(p) if l.strip()]
    return []
def qf(r):
    for k in ("query","prompt","text","instruction","behavior","goal"):
        if k in r and r[k]: return str(r[k])
    return ""

CYBER=r"\b(malware|ransomware|payload|exploit|vulnerabilit|software|firmware|device|network|hack|cyber|phish|encrypt|ddos|botnet|keylog|trojan|backdoor|server|antivirus|computer virus|data breach)\b"
CHEM=r"\b(mercury|mdma|safrole|sarin|tabun|nerve agent|explosive|tnt|rdx|chlorine gas|methamphetamine|fentanyl|cocaine|thermite|ammonium nitrate|dimethylmercury|nitroglycerin)\b"
VIOL=r"\b(jaw|assault|punch|stab|gunshot|knife|shoot|hurt someone|attack someone|kill (someone|him|her|them)|beat up|strangl|self.harm)\b"
SEQ=r"<sequence>|<name>|[ACGT]{20,}|[ACDEFGHIKLMNPQRSTVWY]{25,}"
BIO=(r"\b(anthrax|ebola|smallpox|variola|botulin|ricin|abrin|plague|yersinia|francisella|tularem|"
     r"burkholderia|glanders|melioidosis|coxiella|rickettsia|clostridium|marburg|nipah|hendra|lassa|"
     r"machupo|junin|equine encephalitis|1918 influenza|h5n1|h7n9|hpai|sars-cov|mers|fmdv|rinderpest|"
     r"swine fever|pathogen|biolog|bioweapon|bioterror|gain.of.function|select agent|virulen|"
     r"transmissib|contagio|pandemic|gene drive|aerosoliz|\bspore|\btoxin|gene synthesis|synthetic biolog|"
     r"culture.*(bacteri|virus|pathogen)|weaponiz.*(virus|pathogen|bacteri|agent)|"
     r"synthesi.*(virus|pathogen|gene|genome|nucleic)|release.*(pathogen|virus|agent))\b")

def genuine_bio(t):
    tl=t.lower()
    if re.search(SEQ,t) or re.search(CYBER,tl) or re.search(CHEM,tl) or re.search(VIOL,tl): return False
    return bool(re.search(BIO,tl))

# pdual training positive sources -> in-dist
POS_TRAIN = {"saladbench_cbrn","alert_cbrn","advbench_bio","harmbench_bio","jailbreakbench_bio","clearharm_bio"}
PROMPT_SRC = ["harmbench_bio","clearharm_bio","alert_cbrn_strict","saladbench_cbrn_strict",
              "harmbench_full","saladbench_cbrn","alert_cbrn"]

pos, seen = [], set()
for n in PROMPT_SRC:
    base = re.sub(r"_strict$|_full$","",n)
    indist = base in POS_TRAIN or n in POS_TRAIN
    for r in load(n):
        q = qf(r).strip()
        if not q or q.lower() in seen: continue
        if genuine_bio(q):
            seen.add(q.lower())
            pos.append({"query": q, "response": "", "label": 1,
                        "source": n, "split": "in-dist" if indist else "OOD-cand"})

# benign-bio negatives (clean, from prior validation)
neg, nseen = [], set()
for n in ["false_reject_test","or_bench_health"]:
    for r in load(n):
        if n=="or_bench_health" and int(r.get("label",0))!=0: continue
        q = qf(r).strip()
        if q and has_bio_context(q) and q.lower() not in nseen and q.lower() not in seen:
            nseen.add(q.lower())
            neg.append({"query": q, "response": "", "label": 0, "source": n, "split": "benign"})

print("CLEAN BIO BENCHMARK")
print("  positives (genuine bio-harm prompts): %d" % len(pos))
print("    by source:", dict(Counter(p["source"] for p in pos)))
print("    by split:", dict(Counter(p["split"] for p in pos)))
print("  negatives (benign-bio):", len(neg), dict(Counter(n2["source"] for n2 in neg)))

# save
allrows = pos + neg
out = DATA_PROCESSED / "bio_clean_eval.jsonl"
with open(out,"w") as f:
    for r in allrows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print("  saved:", out, "(%d rows)" % len(allrows))

# AUDIT: print all positives (capped) so they are hand-verifiable
print("\n  -- ALL POSITIVES (audit) --")
for p in pos: print("   [%s] %s" % (p["source"][:10], p["query"][:108]))

# dual-mode self-eval
import numpy as np
g = DualModeGuard()
vp = g.classify_batch([p["query"] for p in pos])
vn = g.classify_batch([n2["query"] for n2 in neg])
rec = np.mean([x.prompt_flag for x in vp]); fp = np.mean([x.prompt_flag for x in vn])
print("\n  dual-mode PROMPT recall(pos)=%.3f  over-flag(neg)=%.3f" % (rec, fp))

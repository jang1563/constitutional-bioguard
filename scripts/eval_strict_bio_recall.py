import json, re, sys
sys.path.insert(0, '.')
from collections import Counter
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

CYBER = r"\b(malware|ransomware|payload|exploit|vulnerabilit|software|firmware|device|network|hack|cyber|phish|encrypt|ddos|botnet|keylog|trojan|backdoor|server|antivirus|computer virus|data breach|steal.*data)\b"
CHEM  = r"\b(mercury|mdma|safrole|sarin|tabun|nerve agent|explosive|tnt|rdx|nitrate|chlorine gas|methamphetamine|fentanyl|cocaine|thermite|ammonium nitrate|dimethylmercury)\b"
VIOL  = r"\b(jaw|assault|punch|stab|gunshot|knife|shoot|hurt someone|attack someone|kill (someone|him|her|them)|beat up|strangl|suicide|self.harm)\b"
SEQ   = r"<sequence>|<name>|[ACGT]{20,}|[ACDEFGHIKLMNPQRSTVWY]{25,}"
BIO   = r"\b(anthrax|ebola|smallpox|variola|botulin|ricin|plague|yersinia|tularem|burkholderia|nipah|marburg|h5n1|h7n9|1918 influenza|pathogen|biolog|bioweapon|bioterror|gain.of.function|select agent|virulen|transmissib|contagio|pandemic|gene drive|aerosoliz|spore|toxin|pcr|synthesi.*(virus|gene|pathogen|dna)|culture.*(bacteri|virus|pathogen)|weaponiz.*(virus|pathogen|agent))\b"

def is_genuine_bio(t):
    tl = t.lower()
    if re.search(SEQ, t): return False
    if re.search(CYBER, tl) or re.search(CHEM, tl) or re.search(VIOL, tl): return False
    return bool(re.search(BIO, tl))

SETS = ["harmbench_bio","advbench_bio","clearharm_bio","alert_cbrn_strict",
        "saladbench_cbrn_strict","scisafeeval_bio","simple_safety_bio"]
pool, seen = [], set()
for n in SETS:
    for r in load(n):
        q = qf(r)
        if is_genuine_bio(q) and q.strip().lower() not in seen:
            seen.add(q.strip().lower()); pool.append((q, n))
print("Strict genuine-bio pool: n=%d (deduped across %d sets)" % (len(pool), len(SETS)))
print("  by source:", dict(Counter(s for _, s in pool)))

guard = DualModeGuard()
qs = [q for q, _ in pool]
v = guard.classify_batch(qs)
import numpy as np
flags = np.array([int(x.prompt_flag) for x in v])
lex = np.mean([x.prompt_source in ("lex","both") for x in v])
learn = np.mean([x.prompt_source in ("learned","both") for x in v])
print("\n  PROMPT-AXIS recall on genuine-bio: %.3f  (lex=%.3f learned=%.3f)" % (flags.mean(), lex, learn))
print("\n  -- MISSES (genuine bio, flag=False) --")
miss = [(qs[i], pool[i][1]) for i in range(len(qs)) if not flags[i]]
for q, s in miss[:12]:
    print("   [%s] %s" % (s, q[:115]))
print("  ... total misses: %d/%d" % (len(miss), len(qs)))

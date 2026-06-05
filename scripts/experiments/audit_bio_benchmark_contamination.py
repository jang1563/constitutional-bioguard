import json, re
from constitutional_bioguard.config import DATA_PROCESSED, DATA_EXTERNAL

def load(n):
    for d in (DATA_PROCESSED, DATA_EXTERNAL):
        p = d / (n + ".jsonl")
        if p.exists(): return [json.loads(l) for l in open(p) if l.strip()]
    return []
def qf(r):
    for k in ("query","prompt","text","instruction","behavior","goal"):
        if k in r and r[k]: return str(r[k])
    return ""

CYBER = r"\b(malware|ransomware|payload|exploit|vulnerabilit|software|firmware|device|network|hack|cyber|phish|encrypt|ddos|botnet|keylog|trojan|backdoor|database|server|antivirus|computer virus)\b"
CHEM  = r"\b(mercury|mdma|safrole|sarin|tabun|vx |nerve agent|explosive|tnt|rdx|nitrate|chemical weapon|chlorine|methamphetamine|fentanyl|cocaine|thermite|ammonium)\b"
VIOL  = r"\b(jaw|assault|punch|stab|gun|knife|shoot|bomb(?!ardment)|hurt someone|attack someone|kill (someone|him|her|them)|beat up|strangl)\b"
BIO   = r"\b(anthrax|ebola|smallpox|variola|botulin|ricin|plague|yersinia|tularem|pathogen|bioweapon|biological weapon|gain.of.function|select agent|virus strain|bacteri|toxin|spore|pandemic|gene drive|crispr|aerosol|culture .*(bacteri|virus)|influenza|h5n1|nipah|marburg|sars|enhance (transmiss|virulen))\b"
SEQ   = r"<sequence>|<name>|[ACGT]{20,}|[ACDEFGHIKLMNPQRSTVWY]{25,}"

def bucket(t):
    tl = t.lower()
    if re.search(SEQ, t): return "sequence/placeholder"
    if re.search(CYBER, tl): return "cyber"
    if re.search(CHEM, tl): return "chem"
    if re.search(VIOL, tl): return "violence"
    if re.search(BIO, tl): return "genuine_bio"
    return "other/ambiguous"

from collections import Counter
sets = ["scisafeeval_bio","simple_safety_bio","harmbench_bio","advbench_bio",
        "clearharm_bio","alert_cbrn_strict","saladbench_cbrn_strict","real_response_bio"]
print("%-24s %6s  %s" % ("set","n","bucket fractions (genuine_bio first)"))
for n in sets:
    rows = load(n)
    if not rows: continue
    b = Counter(bucket(qf(r)) for r in rows)
    tot = sum(b.values())
    gb = b.get("genuine_bio",0)/tot
    order = ["genuine_bio","sequence/placeholder","cyber","chem","violence","other/ambiguous"]
    frac = "  ".join("%s=%.2f" % (k, b.get(k,0)/tot) for k in order if b.get(k,0))
    print("%-24s %6d  bio=%.2f | %s" % (n, tot, gb, frac))

#!/usr/bin/env python
"""Phase G quality audit: genuine-bio retention, length, benign-drift, diversity."""
import json
import re
import sys
from collections import Counter

paras = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("Total paraphrases:", len(paras))

CYBER = r"\b(malware|ransomware|payload|exploit|software|device|network|hack|cyber)\b"
CHEM = r"\b(mercury|mdma|sarin|nerve agent|explosive|tnt|fentanyl)\b"
BIO = (r"\b(anthrax|ebola|smallpox|variola|botulin|ricin|plague|yersinia|tularem|"
       r"marburg|nipah|lassa|coxiella|clostridium|pathogen|biolog|bioweapon|virulen|"
       r"transmissib|pandemic|aerosoliz|spore|toxin|weaponiz|q fever|bacill)\b")


def genuine(t):
    tl = t.lower()
    if re.search(CYBER, tl) or re.search(CHEM, tl):
        return False
    return bool(re.search(BIO, tl))


gb = sum(genuine(p["query"]) for p in paras)
print("genuine_bio pass: %d/%d (%.1f%%)" % (gb, len(paras), 100 * gb / len(paras)))

lens = [len(p["query"]) for p in paras]
print("length: min=%d median=%d max=%d" % (min(lens), sorted(lens)[len(lens) // 2], max(lens)))

# benign-drift: did paraphrase keep an operational/harm verb, or soften to pure education?
HARM = re.compile(r"\b(make|create|produc|synthesi|weaponi|obtain|acquir|deploy|releas|"
                  r"spread|dispers|cultivat|grow|kill|poison|how to|how much|how do|how can|"
                  r"steps|method|approach|process)\b", re.I)
soft = sum(0 if HARM.search(p["query"]) else 1 for p in paras)
print("no-harm-verb (possible benign drift): %d/%d (%.1f%%)" % (soft, len(paras), 100 * soft / len(paras)))

print("\nby source:", dict(Counter(p["source"] for p in paras)))

print("\n-- 12 random samples (audit) --")
import random
random.seed(1)
for p in random.sample(paras, min(12, len(paras))):
    g = "OK " if genuine(p["query"]) else "DROP"
    h = "H" if HARM.search(p["query"]) else "soft"
    print("  [%s|%s] %s" % (g, h, p["query"][:110]))

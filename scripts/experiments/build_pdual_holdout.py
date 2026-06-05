#!/usr/bin/env python
"""Phase R Part 2: source-held-out pdual data for TRUE OOD recall.

Mirrors build_pdual_data.py EXACTLY but excludes one positive source from
training (default: saladbench_cbrn -- the most diverse / natural-phrasing bio
source) and emits its genuine-bio items as a held-out OOD eval set. Retraining
pdual on this and measuring recall on the held-out source removes the in-dist
home-field advantage from the Phase R head-to-head: it answers "does the prompt
head GENERALIZE to bio phrasings it never trained on, or did it memorize?".

No new content generated -- pure reuse, just a different train/test partition.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED

random.seed(42)
BIO_KW = [k.lower() for k in
          json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]

POS_FILES = ["saladbench_cbrn.jsonl", "alert_cbrn.jsonl", "advbench_bio.jsonl",
             "harmbench_bio.jsonl", "jailbreakbench_bio.jsonl", "clearharm_bio.jsonl"]
ALLBIO_NEG_FILES = ["or_bench_health.jsonl", "lab_bench.jsonl",
                    "pubmed_qa_pqa_labeled.jsonl", "med_qa_test.jsonl"]
KW_BIO_NEG_FILES = ["false_reject_train.jsonl", "false_reject_test.jsonl",
                    "or_bench_hard_1k.jsonl"]
NEG_NONBIO_HARM_FILES = ["harmbench_full.jsonl", "advbench_full.jsonl",
                         "saladbench_cbrn.jsonl", "alert_cbrn.jsonl",
                         "wmdp_cyber.jsonl", "wmdp_chem.jsonl"]
NEG_EASY_FILES = ["or_bench_hard_1k.jsonl", "false_reject_train.jsonl"]

# genuine-bio filter (decontaminate cyber/chem/violence/sequence) for the OOD set
CYBER = r"\b(malware|ransomware|payload|exploit|vulnerabilit|software|firmware|device|network|hack|cyber|phish|encrypt|server|computer virus)\b"
CHEM = r"\b(mercury|mdma|safrole|sarin|nerve agent|explosive|tnt|chlorine gas|methamphetamine|fentanyl|dimethylmercury)\b"
VIOL = r"\b(jaw|assault|punch|stab|gunshot|knife|shoot|hurt someone|attack someone|beat up|strangl)\b"
SEQ = r"<sequence>|<name>|[ACGT]{20,}"
BIO = (r"\b(anthrax|ebola|smallpox|variola|botulin|ricin|abrin|plague|yersinia|francisella|tularem|"
       r"burkholderia|marburg|nipah|lassa|coxiella|clostridium|pathogen|biolog|bioweapon|bioterror|"
       r"gain.of.function|select agent|virulen|transmissib|contagio|pandemic|aerosoliz|spore|toxin|"
       r"weaponiz|gene synthesis)\b")


def genuine_bio(t):
    tl = t.lower()
    if re.search(SEQ, t) or re.search(CYBER, tl) or re.search(CHEM, tl) or re.search(VIOL, tl):
        return False
    return bool(re.search(BIO, tl))


def qh(t):
    return hashlib.sha1((t or "").strip().lower().encode()).hexdigest()[:16]


def is_bio(t):
    return any(k in str(t or "").lower() for k in BIO_KW)


def prompt_of(r):
    for k in ("prompt", "query", "instruction", "question", "behavior", "goal", "text"):
        if k in r and r[k]:
            return str(r[k])
    return ""


def load(fn):
    p = DATA_EXTERNAL / fn
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="saladbench_cbrn.jsonl",
                    help="positive source to hold out of training")
    args = ap.parse_args()
    holdout = args.holdout
    pos_files = [f for f in POS_FILES if f != holdout]
    assert holdout in POS_FILES, f"{holdout} not in POS_FILES"
    print(f"HOLDOUT: {holdout}  -> training positives from {pos_files}")

    seen, items, src = set(), [], Counter()

    def add(prompt, label, source):
        h = qh(prompt)
        if not prompt or len(prompt.strip()) < 8 or h in seen:
            return False
        seen.add(h)
        items.append({"query": prompt.strip(), "response": "", "label": label, "source": source})
        src[source] += 1
        return True

    for fn in pos_files:
        for r in load(fn):
            pr = prompt_of(r)
            if is_bio(pr):
                add(pr, 1, fn.replace(".jsonl", ""))
    for fn in ALLBIO_NEG_FILES:
        for r in load(fn):
            add(prompt_of(r), 0, "negbio_" + fn.split(".")[0][:8])
    for fn in KW_BIO_NEG_FILES:
        for r in load(fn):
            pr = prompt_of(r)
            if is_bio(pr):
                add(pr, 0, "negbiokw_" + fn.split(".")[0][:6])

    nonbio_harm, nb_seen = [], set()
    for fn in NEG_NONBIO_HARM_FILES:
        for r in load(fn):
            pr = prompt_of(r)
            if pr and not is_bio(pr) and len(pr.strip()) >= 8:
                h = qh(pr)
                if h in seen or h in nb_seen:
                    continue
                nb_seen.add(h)
                nonbio_harm.append(pr)
    random.shuffle(nonbio_harm)
    cut = int(len(nonbio_harm) * 0.8)
    for pr in nonbio_harm[:cut]:
        add(pr, 0, "neg_nonbio_harm")

    easy = []
    for fn in NEG_EASY_FILES:
        for r in load(fn):
            pr = prompt_of(r)
            if pr and not is_bio(pr) and qh(pr) not in seen:
                easy.append(pr)
    random.shuffle(easy)
    for pr in easy[:500]:
        add(pr, 0, "neg_easy_nonbio")

    random.shuffle(items)
    pos = sum(it["label"] for it in items)
    n_val = int(len(items) * 0.15)
    val, train = items[:n_val], items[n_val:]
    for name, data in (("pdual_train_ho", train), ("pdual_val_ho", val)):
        with open(DATA_PROCESSED / f"{name}.jsonl", "w") as f:
            for it in data:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # OOD eval set = genuine-bio items from the HELD-OUT source (never trained)
    ood = []
    oseen = set()
    for r in load(holdout):
        pr = prompt_of(r).strip()
        if genuine_bio(pr) and pr.lower() not in oseen:
            oseen.add(pr.lower())
            ood.append({"query": pr, "response": "", "label": 1,
                        "source": holdout.replace(".jsonl", "") + "_OOD"})
    with open(DATA_PROCESSED / "pdual_ood_ho.jsonl", "w") as f:
        for it in ood:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"train={len(train)} val={len(val)} pos={pos} ({round(100*pos/len(items),1)}%)")
    print(f"OOD held-out genuine-bio eval (never trained): n={len(ood)}")


if __name__ == "__main__":
    main()

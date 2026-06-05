#!/usr/bin/env python
"""Phase G: build pdual training data with paraphrase augmentation.

Two variants for controlled experiment:
  G1a (additive): pdual_ho base (202 original pos) + paraphrases
  G1b (clean+aug): genuine-bio only (~50 pos) + paraphrases

Both hold out saladbench_cbrn from positives (same as pdual_ho). The ONLY
difference from pdual_ho is the paraphrase addition (G1a) or positive
decontamination + paraphrase (G1b). Same negatives, same val split seed.

OOD test = same saladbench 41 genuine-bio items (pdual_ood_ho.jsonl).
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

POS_FILES_NO_SALAD = ["alert_cbrn.jsonl", "advbench_bio.jsonl",
                      "harmbench_bio.jsonl", "jailbreakbench_bio.jsonl",
                      "clearharm_bio.jsonl"]
ALLBIO_NEG_FILES = ["or_bench_health.jsonl", "lab_bench.jsonl",
                    "pubmed_qa_pqa_labeled.jsonl", "med_qa_test.jsonl"]
KW_BIO_NEG_FILES = ["false_reject_train.jsonl", "false_reject_test.jsonl",
                    "or_bench_hard_1k.jsonl"]
NEG_NONBIO_HARM_FILES = ["harmbench_full.jsonl", "advbench_full.jsonl",
                         "saladbench_cbrn.jsonl", "alert_cbrn.jsonl",
                         "wmdp_cyber.jsonl", "wmdp_chem.jsonl"]
NEG_EASY_FILES = ["or_bench_hard_1k.jsonl", "false_reject_train.jsonl"]

# genuine-bio filter for G1b
CYBER = r"\b(malware|ransomware|payload|exploit|vulnerabilit|software|firmware|device|network|hack|cyber|phish|encrypt|server|computer virus)\b"
CHEM = r"\b(mercury|mdma|safrole|sarin|nerve agent|explosive|tnt|chlorine gas|methamphetamine|fentanyl|dimethylmercury)\b"
VIOL = r"\b(jaw|assault|punch|stab|gunshot|knife|shoot|hurt someone|attack someone|beat up|strangl)\b"
SEQ = r"<sequence>|<name>|[ACGT]{20,}"
BIO_RE = (r"\b(anthrax|ebola|smallpox|variola|botulin|ricin|abrin|plague|yersinia|francisella|tularem|"
          r"burkholderia|marburg|nipah|lassa|coxiella|clostridium|pathogen|biolog|bioweapon|bioterror|"
          r"gain.of.function|select agent|virulen|transmissib|contagio|pandemic|aerosoliz|spore|toxin|"
          r"weaponiz|gene synthesis)\b")


def genuine_bio(t):
    tl = t.lower()
    if re.search(SEQ, t) or re.search(CYBER, tl) or re.search(CHEM, tl) or re.search(VIOL, tl):
        return False
    return bool(re.search(BIO_RE, tl))


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
    ap.add_argument("--paraphrases", required=True,
                    help="Path to pdual_paraphrases.jsonl")
    ap.add_argument("--variant", default="g1a", choices=["g1a", "g1b"])
    args = ap.parse_args()

    # load paraphrases
    paras = [json.loads(l) for l in open(args.paraphrases) if l.strip()]
    print("Loaded %d paraphrases" % len(paras))

    seen, items, src = set(), [], Counter()

    def add(prompt, label, source):
        h = qh(prompt)
        if not prompt or len(prompt.strip()) < 8 or h in seen:
            return False
        seen.add(h)
        items.append({"query": prompt.strip(), "response": "", "label": label,
                      "source": source})
        src[source] += 1
        return True

    # --- POSITIVES ---
    if args.variant == "g1a":
        # G1a: original pdual_ho positives (is_bio filter, same as build_pdual_holdout)
        for fn in POS_FILES_NO_SALAD:
            for r in load(fn):
                pr = prompt_of(r)
                if is_bio(pr):
                    add(pr, 1, fn.replace(".jsonl", ""))
    else:
        # G1b: genuine-bio only (decontaminated)
        for fn in POS_FILES_NO_SALAD:
            for r in load(fn):
                pr = prompt_of(r)
                if genuine_bio(pr):
                    add(pr, 1, "genuine_" + fn.replace(".jsonl", ""))

    # add paraphrases (all variants)
    for p in paras:
        add(p["query"], 1, p.get("source", "paraphrase"))

    n_pos_before_neg = sum(it["label"] for it in items)

    # --- NEGATIVES (identical to build_pdual_holdout / build_pdual_data) ---
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

    # --- SPLIT + SAVE ---
    random.shuffle(items)
    pos = sum(it["label"] for it in items)
    n_val = int(len(items) * 0.15)
    val, train = items[:n_val], items[n_val:]

    tag = args.variant
    for name, data in ((f"pdual_train_{tag}", train), (f"pdual_val_{tag}", val)):
        with open(DATA_PROCESSED / f"{name}.jsonl", "w") as f:
            for it in data:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # class weights
    tp = sum(it["label"] for it in train)
    tn = len(train) - tp
    w = {"0": round(len(train) / (2 * tn), 4) if tn else 1.0, "1": 1.0}
    json.dump(w, open(DATA_PROCESSED / f"pdual_class_weights_{tag}.json", "w"))

    print("Variant: %s" % tag)
    print("  total=%d pos=%d neg=%d (%.1f%% pos)" % (len(items), pos, len(items)-pos, 100*pos/len(items)))
    print("  paraphrases included: %d" % len(paras))
    print("  train=%d val=%d  class_weights=%s" % (len(train), len(val), w))
    print("  by_source: %s" % dict(src.most_common(15)))


if __name__ == "__main__":
    main()

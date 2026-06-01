#!/usr/bin/env python
"""P2/P2b build: prompt-harm bio dataset for the standalone prompt-head.

P2b adds NON-BIO harmful prompts as label-0 negatives so the head learns to flag
bio-harm but NOT non-bio harm (directly optimizing the bio-selectivity ratio S).
The non-bio-harm pool is split 80/20: 80% trains, 20% is HELD OUT as the
selectivity denominator (no leakage). WMDP-bio is the dual-use boundary (held out,
never trained). Format query=prompt, response="" (reuse the pair pipeline).

Writes data/processed/pdual_{train,val}.jsonl + pdual_selectivity.jsonl +
pdual_class_weights.json. Content-blind.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED

random.seed(42)
BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]

POS_FILES = ["saladbench_cbrn.jsonl", "alert_cbrn.jsonl", "advbench_bio.jsonl",
             "harmbench_bio.jsonl", "jailbreakbench_bio.jsonl", "clearharm_bio.jsonl"]
NEG_BIO_FILES = ["or_bench_health.jsonl", "lab_bench.jsonl", "false_reject_train.jsonl",
                 "false_reject_test.jsonl", "med_qa_test.jsonl", "pubmed_qa_pqa_labeled.jsonl",
                 "or_bench_hard_1k.jsonl"]
# NON-BIO harmful prompts -> the selectivity-teaching negatives (label 0 for bio head)
NEG_NONBIO_HARM_FILES = ["harmbench_full.jsonl", "advbench_full.jsonl", "saladbench_cbrn.jsonl",
                         "alert_cbrn.jsonl", "wmdp_cyber.jsonl", "wmdp_chem.jsonl"]
NEG_EASY_FILES = ["or_bench_hard_1k.jsonl", "false_reject_train.jsonl"]


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
    seen = set()
    items = []
    src = Counter()

    def add(prompt, label, source):
        h = qh(prompt)
        if not prompt or len(prompt.strip()) < 8 or h in seen:
            return False
        seen.add(h)
        items.append({"query": prompt.strip(), "response": "", "label": label, "source": source})
        src[source] += 1
        return True

    # positives (bio-harm prompts)
    for fn in POS_FILES:
        for r in load(fn):
            pr = prompt_of(r)
            if is_bio(pr):
                add(pr, 1, fn.replace(".jsonl", ""))

    # benign-bio hard negatives
    for fn in NEG_BIO_FILES:
        for r in load(fn):
            pr = prompt_of(r)
            if is_bio(pr):
                add(pr, 0, "neg_bio_" + fn.split(".")[0][:10])

    # NON-BIO harmful -> collect, dedup, split 80 train / 20 held-out selectivity
    nonbio_harm = []
    nb_seen = set()
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
    train_nbh, held_nbh = nonbio_harm[:cut], nonbio_harm[cut:]
    for pr in train_nbh:
        add(pr, 0, "neg_nonbio_harm")

    # easy non-bio benign negatives (smaller now)
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
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, data in (("pdual_train", train), ("pdual_val", val)):
        with open(DATA_PROCESSED / f"{name}.jsonl", "w") as f:
            for it in data:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tp = sum(it["label"] for it in train); tn = len(train) - tp
    w = {"0": round(len(train) / (2 * tn), 4) if tn else 1.0, "1": 1.0}
    json.dump(w, open(DATA_PROCESSED / "pdual_class_weights.json", "w"))

    # selectivity set: WMDP-bio boundary + HELD-OUT non-bio-harm (disjoint from train)
    sel = []
    for r in load("wmdp_bio.jsonl"):
        pr = prompt_of(r)
        if is_bio(pr):
            sel.append({"query": pr, "response": "", "tier": "dual_use_boundary", "source": "wmdp_bio"})
    for pr in held_nbh:
        sel.append({"query": pr, "response": "", "tier": "nonbio_harm", "source": "held_out"})
    with open(DATA_PROCESSED / "pdual_selectivity.jsonl", "w") as f:
        for it in sel:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"pdual P2b: total={len(items)} pos={pos} neg={len(items)-pos} ({round(100*pos/len(items),1)}% pos)")
    print(f"  train={len(train)} val={len(val)}  class_weights={w}")
    print(f"  by_source: {dict(src.most_common())}")
    print(f"  non-bio-harm: {len(nonbio_harm)} ({len(train_nbh)} train-neg / {len(held_nbh)} held-out selectivity)")
    print(f"  selectivity set: {dict(Counter(s['tier'] for s in sel))}")


if __name__ == "__main__":
    main()

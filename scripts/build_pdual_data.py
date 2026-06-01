#!/usr/bin/env python
"""P2 build: prompt-harm bio dataset for the standalone prompt-head baseline.

Assembles bio-harm PROMPT positives + benign-bio hard-negatives + non-bio easy-
negatives from cache (reuse-only). Format query=prompt, response="" so it flows
through the existing pair-encoding train pipeline unchanged (a prompt classifier =
a response head trained on (prompt, empty); infer the same way). WMDP-bio is held
out as the DUAL-USE boundary for the selectivity eval, never trained on.

Writes data/processed/pdual_{train,val}.jsonl + pdual_selectivity.jsonl (WMDP
boundary + non-bio-harmful denominator) + pdual_class_weights.json. Content-blind.
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

# positive (bio-harm prompt) sources: apply bio filter to the prompt
POS_FILES = ["saladbench_cbrn.jsonl", "alert_cbrn.jsonl", "advbench_bio.jsonl",
             "harmbench_bio.jsonl", "jailbreakbench_bio.jsonl",
             "scisafeeval_bio.jsonl", "clearharm_bio.jsonl"]
# benign-bio hard negatives
NEG_BIO_FILES = ["or_bench_health.jsonl", "lab_bench.jsonl", "false_reject_train.jsonl",
                 "false_reject_test.jsonl", "med_qa_test.jsonl", "pubmed_qa_pqa_labeled.jsonl",
                 "or_bench_hard_1k.jsonl"]
# non-bio easy negatives (sample from these, NON-bio rows only)
NEG_EASY_FILES = ["or_bench_hard_1k.jsonl", "false_reject_train.jsonl"]
SCISAFE_CAP = 350   # cap templated SciSafeEval so it does not dominate


def qh(t):
    return hashlib.sha1((t or "").strip().lower().encode()).hexdigest()[:16]


def is_bio(t):
    t = str(t or "").lower()
    return any(k in t for k in BIO_KW)


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
            return
        seen.add(h)
        items.append({"query": prompt.strip(), "response": "", "label": label, "source": source})
        src[source] += 1

    # positives
    for fn in POS_FILES:
        rows = load(fn)
        if "scisafeeval" in fn:
            rows = rows[:SCISAFE_CAP]
        for r in rows:
            pr = prompt_of(r)
            if is_bio(pr):
                add(pr, 1, fn.replace(".jsonl", ""))

    # benign-bio hard negatives
    for fn in NEG_BIO_FILES:
        for r in load(fn):
            pr = prompt_of(r)
            if is_bio(pr):
                add(pr, 0, "neg_" + fn.replace(".jsonl", ""))

    # non-bio easy negatives (sample)
    easy = []
    for fn in NEG_EASY_FILES:
        for r in load(fn):
            pr = prompt_of(r)
            if pr and not is_bio(pr) and qh(pr) not in seen:
                easy.append(pr)
    random.shuffle(easy)
    for pr in easy[:1200]:
        add(pr, 0, "neg_easy_nonbio")

    random.shuffle(items)
    pos = sum(it["label"] for it in items)
    neg = len(items) - pos
    # 85/15 split, stratified-ish by shuffle
    n_val = int(len(items) * 0.15)
    val, train = items[:n_val], items[n_val:]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, data in (("pdual_train", train), ("pdual_val", val)):
        with open(DATA_PROCESSED / f"{name}.jsonl", "w") as f:
            for it in data:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # class weights (auto SAFE)
    tp = sum(it["label"] for it in train); tn = len(train) - tp
    w = {"0": round(len(train) / (2 * tn), 4) if tn else 1.0, "1": 1.0}
    json.dump(w, open(DATA_PROCESSED / "pdual_class_weights.json", "w"))

    # selectivity eval set: WMDP-bio (dual-use boundary, expect mostly NOT flagged)
    # + non-bio harmful (the denominator: should NOT flag, that is other heads' job)
    sel = []
    for r in load("wmdp_bio.jsonl"):
        pr = prompt_of(r)
        if is_bio(pr):
            sel.append({"query": pr, "response": "", "tier": "dual_use_boundary", "source": "wmdp_bio"})
    for fn in ("harmbench_full.jsonl", "advbench_full.jsonl"):
        for r in load(fn):
            pr = prompt_of(r)
            if pr and not is_bio(pr):
                sel.append({"query": pr, "response": "", "tier": "nonbio_harm", "source": fn.replace(".jsonl", "")})
    with open(DATA_PROCESSED / "pdual_selectivity.jsonl", "w") as f:
        for it in sel:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"pdual: total={len(items)} pos={pos} neg={neg} ({round(100*pos/len(items),1)}% pos)")
    print(f"  train={len(train)} val={len(val)}  class_weights={w}")
    print(f"  by_source: {dict(src.most_common())}")
    print(f"  selectivity set: {len(sel)} ({dict(Counter(s['tier'] for s in sel))})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""R3 data harvest — ASSESS the available pool of NEW bio-harmful responses.

Before building a larger eval (62 -> 140/390 positives), measure how many
genuinely-harmful, bio, REAL-response items each cached source can contribute that
are NOT already in v8 training or the #106 eval. Also detect stub-template
responses (a source whose "responses" are one repeated COMPLIANCE_TEMPLATE is
useless for response-harm and must be excluded, like the gate benchmarks).

Content-blind: counts only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED

BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]


def qh(q):
    return hashlib.sha1((q or "").strip().lower().encode()).hexdigest()[:16]


def rh(r):
    return hashlib.sha1((r or "").strip().encode()).hexdigest()[:16]


def is_bio(q, r):
    t = (str(q or "") + " " + str(r or "")).lower()
    return any(k in t for k in BIO_KW)


def harmful(r):
    lab = r.get("label")
    if isinstance(lab, (int, float)):
        return int(lab) == 1
    if r.get("is_safe") is False:
        return True
    rhl = r.get("response_harm_label")
    if isinstance(rhl, str):
        return rhl.lower() == "harmful"
    return False


def field(r, *ks):
    for k in ks:
        if k in r and r[k] not in (None, ""):
            return r[k]
    return ""


def main():
    # training + #106 exclusion sets
    train_q = set()
    train_qr = set()
    for tp in ("v8c_train", "v8c_val", "v8b_train", "v8b_val", "v8_train", "v8_val"):
        p = DATA_PROCESSED / f"{tp}.jsonl"
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                train_q.add(qh(r.get("query", "")))
                train_qr.add((qh(r.get("query", "")), rh(r.get("response", ""))))
    eval_qr = set()
    ep = DATA_EXTERNAL / "real_response_bio.jsonl"
    if ep.exists():
        for line in open(ep):
            r = json.loads(line)
            eval_qr.add((qh(r.get("query", "")), rh(r.get("response", ""))))

    print(f"exclusion: train_q={len(train_q)} train_qr={len(train_qr)} eval_qr={len(eval_qr)}\n")
    print(f"{'source':30s} {'tot':>5s} {'bio':>5s} {'bioharm':>7s} {'NEW':>5s} {'uniqResp%':>9s}")

    cands = ["harmbench_bio.jsonl", "harmbench_full.jsonl", "beavertails_subset.jsonl",
             "advbench_bio.jsonl", "advbench_full.jsonl", "wildguard_test.jsonl"]
    new_total = set()
    for fn in cands:
        p = DATA_EXTERNAL / fn
        if not p.exists():
            print(f"{fn:30s} MISSING"); continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        bio = [r for r in rows if is_bio(field(r, "query", "prompt"), field(r, "response"))]
        bioharm = [r for r in bio if harmful(r)]
        new = []
        for r in bioharm:
            q = field(r, "query", "prompt"); resp = field(r, "response")
            k = (qh(q), rh(resp))
            if qh(q) in train_q or k in train_qr or k in eval_qr:
                continue
            new.append(r); new_total.add(k)
        uniq = len(set(rh(field(r, "response")) for r in bioharm))
        uniqpct = round(100 * uniq / len(bioharm), 1) if bioharm else 0.0
        print(f"{fn:30s} {len(rows):5d} {len(bio):5d} {len(bioharm):7d} {len(new):5d} {uniqpct:8.1f}%")

    print(f"\nTOTAL distinct NEW bio-harmful positives available: {len(new_total)}")
    print(f"(current #106 has 62; target 140 min / 390 publication-grade)")


if __name__ == "__main__":
    main()

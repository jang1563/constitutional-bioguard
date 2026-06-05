#!/usr/bin/env python
# Broad density debiasing: add DIVERSE dense-but-safe bio responses (multiple distributions) as hard
# negatives, decontaminated vs the eval sets, to test if over-refusal closes BROADLY (not just on
# the FORTRESS distribution). Prioritizes LONG responses (the density-bias trigger).
import json
import hashlib
from collections import Counter
from datasets import load_dataset

KW = [k.lower() for k in json.load(open("configs/bio_keywords_adv.json")).get("keywords", [])]


def is_bio(q, r):
    t = (str(q) + " " + str(r)).lower()
    return any(k in t for k in KW)


def hh(q, r):
    return hashlib.md5((str(q).strip().lower() + "|" + str(r).strip().lower()).encode()).hexdigest()


# eval exclusion: (query,response) hashes from the held-out eval sets
block = set()
for fn in ["data/external/real_response_bio_large.jsonl", "data/external/fortress_safe_heldout.jsonl"]:
    for l in open(fn):
        if l.strip():
            r = json.loads(l)
            block.add(hh(r.get("query") or r.get("prompt"), r.get("response")))
print("eval-exclusion (query,response) hashes:", len(block))

MINLEN = 300  # target the density tail: long safe responses
pool = []


def add(q, resp, src):
    if resp and len(resp) >= MINLEN and is_bio(q, resp) and hh(q, resp) not in block:
        pool.append({"query": q, "response": resp, "label": 0, "prompt_harm": None,
                     "response_harm": "unharmful", "bio": True, "source": src})


# FORTRESS safe train-half (already split)
for l in open("data/processed/v8bh_train.jsonl"):
    r = json.loads(l)
    if r.get("source") == "fortress_safe_resp":
        add(r["query"], r["response"], "fortress_safe")
# BeaverTails 330k_TRAIN safe bio (disjoint from 330k_test eval)
for r in load_dataset("PKU-Alignment/BeaverTails", split="330k_train"):
    if r.get("is_safe") is True:
        add(r.get("prompt", ""), r.get("response", ""), "beavertails_train_safe")
# SafeRLHF TRAIN safe bio
for r in load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train"):
    q = r.get("prompt", "")
    for i in (0, 1):
        if r.get(f"is_response_{i}_safe"):
            add(q, r.get(f"response_{i}", ""), "saferlhf_train_safe")

# dedup
seen, ded = set(), []
for r in pool:
    k = hh(r["query"], r["response"])
    if k not in seen:
        seen.add(k)
        ded.append(r)
# cap per source so no single distribution dominates
import random
random.seed(42)
random.shuffle(ded)
cap = {"beavertails_train_safe": 700, "saferlhf_train_safe": 700, "fortress_safe": 200}
kept, cnt = [], Counter()
for r in ded:
    if cnt[r["source"]] < cap.get(r["source"], 300):
        kept.append(r)
        cnt[r["source"]] += 1

orig = [json.loads(l) for l in open("data/processed/v8b_train.jsonl") if l.strip()]
merged = orig + kept
with open("data/processed/v8b2_train.jsonl", "w") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("data/processed/v8b_val.jsonl") as fi, open("data/processed/v8b2_val.jsonl", "w") as fo:
    fo.write(fi.read())
print("dense-safe negatives added:", dict(cnt), "total", len(kept))
print("v8b2_train:", len(merged), "(orig", len(orig), "+ dense-safe", len(kept), ")")

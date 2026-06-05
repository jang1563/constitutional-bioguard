#!/usr/bin/env python
# Proper debiasing split: train on HALF of FORTRESS safe responses, hold out the other half for eval.
import json
import hashlib

orig = [json.loads(l) for l in open("data/processed/v8b_train.jsonl") if l.strip()]
fort = [json.loads(l) for l in open("data/external/fortress_safe_responses.jsonl") if l.strip()]

# deterministic split by query+response hash parity
def h(r):
    return int(hashlib.md5((r["query"] + r["response"]).encode()).hexdigest(), 16)

train_half = [r for r in fort if h(r) % 2 == 0]
eval_half = [r for r in fort if h(r) % 2 == 1]

aug = [{"query": r["query"], "response": r["response"], "label": 0,
        "prompt_harm": None, "response_harm": "unharmful", "bio": True,
        "source": "fortress_safe_resp"} for r in train_half]
merged = orig + aug
with open("data/processed/v8bh_train.jsonl", "w") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
# val unchanged
with open("data/processed/v8b_val.jsonl") as fi, open("data/processed/v8bh_val.jsonl", "w") as fo:
    fo.write(fi.read())
# held-out eval half
with open("data/external/fortress_safe_heldout.jsonl", "w") as f:
    for r in eval_half:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"v8bh_train: {len(merged)} (orig {len(orig)} + fortress-train-half {len(aug)})")
print(f"fortress_safe_heldout (eval): {len(eval_half)} (bio {sum(1 for r in eval_half if r['bio'])})")

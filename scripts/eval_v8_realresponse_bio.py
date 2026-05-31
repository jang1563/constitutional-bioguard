#!/usr/bin/env python
"""#106 — the real-response BIO benchmark + v8/v4 measurement.

The gate benchmarks (SaladBench/ALERT/SimpleSafety) carry a constant
COMPLIANCE_TEMPLATE stub as the "response", so they test PROMPT harm, not
response harm. A genuine response-harm classifier (v8) correctly flags ~0 on
them. To measure v8's TRUE bio recall we need REAL bio responses.

This assembles the best available HELD-OUT real-response bio set without
generating any new content:
  - wildguard_test bio subset (real GPT-4/Claude/Llama responses, response_harm
    labels; leakage-clean -- v8 trained on WildGuardMix TRAIN, this is TEST)
  - BeaverTails subset bio (real responses, response-harm `label`)
Both filtered to bio via the same keyword set used elsewhere.

Honest about size: the bio slice of public real-response data is small (the
crux). Report n_pos and treat recall as directional, not a tight estimate.

Content-blind: prints counts/metrics only. Out: data/external/real_response_bio.jsonl
+ results/metrics/v8_realresponse_bio_{model}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from constitutional_bioguard.config import (
    CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED, METRICS_DIR, MODELS_DIR)
import hashlib


def _qh(q):
    return hashlib.sha1((q or "").strip().lower().encode()).hexdigest()[:16]


def _rh(r):
    return hashlib.sha1((r or "").strip().encode()).hexdigest()[:16]

MODEL_DIRS = {"v8b": MODELS_DIR / "deberta_bioguard_v8b",
              "v8": MODELS_DIR / "deberta_bioguard_v8",
              "v4": MODELS_DIR / "deberta_bioguard_v4_response_diverse"}

BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]


def is_bio(q, r):
    t = (str(q or "") + " " + str(r or "")).lower()
    return any(k in t for k in BIO_KW)


def build_benchmark():
    # Leakage guard: anything whose QUERY the model trained on is excluded.
    train_q = set(); train_qr = set()
    for tp in (DATA_PROCESSED / "v8b_train.jsonl", DATA_PROCESSED / "v8b_val.jsonl",
               DATA_PROCESSED / "v8_train.jsonl", DATA_PROCESSED / "v8_val.jsonl"):
        if tp.exists():
            for line in open(tp):
                r = json.loads(line)
                train_q.add(_qh(r.get("query", "")))
                train_qr.add((_qh(r.get("query", "")), _rh(r.get("response", ""))))

    items = []
    # wildguard_test bio (real responses, response_harm label; held-out from training)
    for r in (json.loads(l) for l in open(DATA_EXTERNAL / "wildguard_test.jsonl") if l.strip()):
        rh = r.get("response_harm_label")
        if not (isinstance(rh, str) and rh.lower() in ("harmful", "unharmful")):
            continue
        q = r.get("prompt", r.get("query", "")); resp = r.get("response", "")
        if not is_bio(q, resp) or _qh(q) in train_q:
            continue
        items.append({"query": q, "response": resp,
                      "label": 1 if rh.lower() == "harmful" else 0, "source": "wildguard_test_bio"})
    # BeaverTails 330k_TEST bio (real responses; disjoint from v8b's 330k_train,
    # plus explicit train-query exclusion for airtight no-leakage)
    try:
        from datasets import load_dataset
        bds = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    except Exception as e:
        print(f"WARN: BeaverTails 330k_test load failed ({e})"); bds = []
    for r in bds:
        q = r.get("prompt", r.get("query", "")); resp = r.get("response", "")
        if not is_bio(q, resp):
            continue
        if _qh(q) in train_q or (_qh(q), _rh(resp)) in train_qr:
            continue
        harmful = (r.get("is_safe") is False) or (int(r.get("label", 0)) == 1)
        items.append({"query": q, "response": resp, "label": 1 if harmful else 0,
                      "source": "beavertails330k_test_bio"})
    # dedup by (query, response)
    seen = set(); out = []
    for it in items:
        k = (_qh(it["query"]), _rh(it["response"]))
        if k in seen:
            continue
        seen.add(k); out.append(it)
    with open(DATA_EXTERNAL / "real_response_bio.jsonl", "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return out


def predict(model_dir, queries, responses):
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer, predict_batch)
    m, t = load_model_and_tokenizer(model_dir)
    preds = predict_batch(model=m, tokenizer=t, queries=queries, responses=responses, normalize=True)
    import gc
    import torch
    del m, t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [int(p[0]) for p in preds]


def main():
    items = build_benchmark()
    from collections import Counter
    src = Counter(it["source"] for it in items)
    pos = sum(it["label"] for it in items)
    print(f"real-response bio benchmark: n={len(items)} pos={pos} neg={len(items) - pos} by_source={dict(src)}")
    if pos < 5:
        print("WARNING: <5 positives -- recall will be extremely noisy")
    q = [it["query"] for it in items]; resp = [it["response"] for it in items]
    lab = np.array([it["label"] for it in items])
    srcs = np.array([it["source"] for it in items])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for model in ("v8b", "v8", "v4"):
        md = MODEL_DIRS[model]
        if not md.exists():
            print(f"{model}: missing {md}")
            continue
        pred = np.array(predict(md, q, resp))
        npos = int((lab == 1).sum()); nneg = int((lab == 0).sum())
        out = {"n": len(lab), "n_pos": npos, "n_neg": nneg,
               "recall": round(float(pred[lab == 1].mean()), 4) if npos else None,
               "fpr": round(float(pred[lab == 0].mean()), 4) if nneg else None,
               "precision": round(float(precision_score(lab, pred, zero_division=0)), 4),
               "f1": round(float(f1_score(lab, pred, zero_division=0)), 4)}
        json.dump({"overall": out}, open(METRICS_DIR / f"v8_realresponse_bio_{model}.json", "w"), indent=2)
        print(f"  {model}: recall={out['recall']} fpr={out['fpr']} precision={out['precision']} f1={out['f1']} (n_pos={npos})")
        # per-source (leakage / home-field check): wildguard_test_bio is OUT-of-training
        # distribution for v8b; beavertails_bio is IN-training distribution.
        for s in sorted(set(srcs.tolist())):
            mpos = (lab == 1) & (srcs == s)
            mneg = (lab == 0) & (srcs == s)
            r = round(float(pred[mpos].mean()), 3) if mpos.any() else None
            fp = round(float(pred[mneg].mean()), 3) if mneg.any() else None
            print(f"      [{s}] recall={r} (n_pos={int(mpos.sum())})  fpr={fp} (n_neg={int(mneg.sum())})")


if __name__ == "__main__":
    main()

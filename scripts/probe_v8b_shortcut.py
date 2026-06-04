#!/usr/bin/env python
"""Goodhart / shortcut probe for v8b — bio-HARM content vs bio-KEYWORD style?

The decisive test (per JK): does v8b flag because the response is genuinely
harmful, or because it contains bio keywords (a style/keyword shortcut, the v1/v3
failure mode)? We mask every bio keyword in BOTH query and response with "[X]"
(removing bio IDENTITY but preserving harm STRUCTURE) and re-evaluate on the
real-response bio benchmark (#106).

Reading:
  - recall on POSITIVES drops sharply under ablation  -> positives flagged by keyword (shortcut)
  - FPR on bio-BENIGN drops sharply under ablation     -> over-refusal is keyword-driven (style shortcut)
  - recall holds under ablation                        -> content/structure based (genuine)

Run v8b vs v8 vs v4 to contrast keyword-dependence.
Content-blind: prints counts/metrics only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, MODELS_DIR

MODEL_DIRS = {"v8b": MODELS_DIR / "deberta_bioguard_v8b",
              "v8": MODELS_DIR / "deberta_bioguard_v8",
              "v4": MODELS_DIR / "deberta_bioguard_v4_response_diverse"}

BIO_KW = sorted(
    (k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])),
    key=len, reverse=True)  # longest first so multi-word keywords mask before sub-words
_PAT = re.compile("|".join(re.escape(k) for k in BIO_KW), re.IGNORECASE) if BIO_KW else None


def ablate(text):
    if not _PAT:
        return text
    return _PAT.sub("[X]", text or "")


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
    return np.array([int(p[0]) for p in preds])


def main():
    rows = [json.loads(l) for l in open(DATA_EXTERNAL / "real_response_bio.jsonl") if l.strip()]
    lab = np.array([int(r["label"]) for r in rows])
    q = [r.get("query", "") for r in rows]
    resp = [r.get("response", "") for r in rows]
    q_abl = [ablate(x) for x in q]
    resp_abl = [ablate(x) for x in resp]
    npos = int((lab == 1).sum()); nneg = int((lab == 0).sum())
    # sanity: how many items actually contained a bio keyword that got masked?
    changed = sum(1 for a, b in zip(q + resp, q_abl + resp_abl) if a != b)
    print(f"#106: n={len(rows)} pos={npos} neg={nneg}; items changed by ablation: {changed}/{2*len(rows)} fields")

    for model in ("v8b", "v8", "v4"):
        md = MODEL_DIRS[model]
        if not md.exists():
            print(f"{model}: missing"); continue
        p0 = predict(md, q, resp)
        p1 = predict(md, q_abl, resp_abl)

        def rec(p):
            return round(float(p[lab == 1].mean()), 3) if npos else None

        def fpr(p):
            return round(float(p[lab == 0].mean()), 3) if nneg else None

        print(f"\n=== {model} ===")
        print(f"  recall : orig {rec(p0)} -> ablated {rec(p1)}   (drop = keyword-driven positives)")
        print(f"  bio-FPR: orig {fpr(p0)} -> ablated {fpr(p1)}   (drop = keyword-driven over-refusal)")
        # how many predictions flipped when only bio identity was masked
        flip = int((p0 != p1).sum())
        print(f"  predictions changed by bio-keyword masking: {flip}/{len(rows)} ({round(100*flip/len(rows))}%)")


if __name__ == "__main__":
    main()

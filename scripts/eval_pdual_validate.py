#!/usr/bin/env python
"""Pre-P3 validation of the prompt-head: honest OOD recall + selectivity + shortcut.

The in-dist recall (0.983 on saladbench/alert) is a same-distribution shortcut
(P2 AUROC 1.0). This measures the HONEST signals on data the prompt head never
trained on:
  - OOD bio-harm recall: wildguard_test prompts with prompt_harm_label==harmful AND
    bio (a different distribution than saladbench/alert) -> does it generalize?
  - OOD selectivity: wildguard_test NON-bio harmful prompts -> flag-rate (should be
    LOW = passes non-bio harm by design); benign-bio prompts -> FPR.
  - Lexical (bio-keyword) ablation: mask every bio keyword in the in-dist positives
    -> flip-rate to safe (HIGH = the head relied on bio keywords = shortcut).
Note: the credibility gate is reframed for a bio-SPECIALIST (OOD bio recall +
selectivity + no-shortcut), not "match a general guard on general harm" (which a
bio-specific head should NOT, by design). Content-blind. Input query=prompt, response="".
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED, MODELS_DIR

MODEL = MODELS_DIR / "deberta_pdual_v3"
BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]
_BIO_RE = re.compile("|".join(re.escape(k) for k in sorted(BIO_KW, key=len, reverse=True)), re.IGNORECASE)


def is_bio(t):
    return any(k in str(t or "").lower() for k in BIO_KW)


def mask_bio(t):
    return _BIO_RE.sub("[X]", str(t or ""))


def predict(qs):
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer, predict_batch)
    m, t = load_model_and_tokenizer(MODEL)
    preds = predict_batch(model=m, tokenizer=t, queries=qs, responses=[""] * len(qs), normalize=True)
    return np.array([int(p[0]) for p in preds])


def main():
    # OOD: wildguard_test prompts (never in prompt-head training)
    wg = [json.loads(l) for l in open(DATA_EXTERNAL / "wildguard_test.jsonl") if l.strip()]
    def phl(r):
        v = r.get("prompt_harm_label")
        return v.lower() if isinstance(v, str) else None
    prompts = [(r.get("prompt", r.get("query", "")), phl(r)) for r in wg]
    bio_harm = [p for p, l in prompts if l == "harmful" and is_bio(p)]
    nonbio_harm = [p for p, l in prompts if l == "harmful" and not is_bio(p)]
    bio_benign = [p for p, l in prompts if l == "unharmful" and is_bio(p)]

    print(f"OOD wildguard_test prompts: bio_harm={len(bio_harm)} nonbio_harm={len(nonbio_harm)} bio_benign={len(bio_benign)}")
    if bio_harm:
        r = float(predict(bio_harm).mean())
        print(f"  OOD bio-harm RECALL = {r:.3f}  (vs in-dist 0.983; honest generalization)")
    if nonbio_harm:
        fr = float(predict(nonbio_harm).mean())
        print(f"  OOD non-bio-harm flag-rate = {fr:.3f}  (LOW = passes non-bio harm = selectivity holds OOD)")
    if bio_benign:
        fp = float(predict(bio_benign).mean())
        print(f"  OOD benign-bio FPR = {fp:.3f}  (over-refusal on benign bio prompts)")

    # Lexical (bio-keyword) ablation on the in-dist positives
    val = [json.loads(l) for l in open(DATA_PROCESSED / "pdual_val.jsonl") if l.strip()]
    pos = [r["query"] for r in val if int(r["label"]) == 1]
    if pos:
        base = predict(pos)
        flagged = [pos[i] for i in range(len(pos)) if base[i] == 1]
        masked = predict([mask_bio(p) for p in flagged])
        flip = float((masked == 0).mean()) if len(flagged) else None
        print(f"\nlexical ablation (mask bio keywords) on {len(flagged)} flagged positives:")
        print(f"  flip-to-safe rate = {flip:.3f}  (HIGH = relied on bio keywords = shortcut; LOW = robust)")


if __name__ == "__main__":
    main()

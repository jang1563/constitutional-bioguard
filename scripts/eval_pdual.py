#!/usr/bin/env python
"""P2 eval: the standalone prompt-harm head baseline + bio-selectivity ratio.

Reports (input format query=prompt, response="" matching training):
  - in-distribution: prompt-harm recall (positives) + benign-bio FPR (over-refusal)
    on pdual_val
  - selectivity set: flag-rate on WMDP-bio (dual-use boundary; LOW = selective,
    does not over-flag legitimate research) and on non-bio harmful prompts (LOW =
    bio-SPECIFIC, not a generic harm detector)
  - **bio-selectivity ratio S = recall(bio-harm) / flag-rate(non-bio-harm)**, the
    defined metric: S >> 1 means the head is bio-specific.
Content-blind: rates/counts only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from constitutional_bioguard.config import DATA_PROCESSED, METRICS_DIR, MODELS_DIR

MODEL = MODELS_DIR / "deberta_pdual_v1"


def predict(qs):
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer, predict_batch)
    m, t = load_model_and_tokenizer(MODEL)
    preds = predict_batch(model=m, tokenizer=t, queries=qs, responses=[""] * len(qs), normalize=True)
    return np.array([int(p[0]) for p in preds])


def main():
    val = [json.loads(l) for l in open(DATA_PROCESSED / "pdual_val.jsonl") if l.strip()]
    sel = [json.loads(l) for l in open(DATA_PROCESSED / "pdual_selectivity.jsonl") if l.strip()]

    # in-distribution val
    vq = [r["query"] for r in val]
    vy = np.array([int(r["label"]) for r in val])
    vp = predict(vq)
    npos = int((vy == 1).sum()); nneg = int((vy == 0).sum())
    recall = float(vp[vy == 1].mean()) if npos else None
    fpr = float(vp[vy == 0].mean()) if nneg else None
    print(f"in-dist (pdual_val): n={len(val)} pos={npos} neg={nneg}")
    print(f"  prompt-harm recall={recall:.3f}  benign-bio FPR(over-refusal)={fpr:.3f}")

    # selectivity set
    sq = [r["query"] for r in sel]
    sp = predict(sq)
    tiers = np.array([r["tier"] for r in sel])
    rates = {}
    for tier in sorted(set(tiers.tolist())):
        mask = tiers == tier
        rates[tier] = float(sp[mask].mean())
        print(f"  [{tier}] flag-rate={rates[tier]:.3f} (n={int(mask.sum())})")

    nonbio = rates.get("nonbio_harm", None)
    S = (recall / nonbio) if (recall and nonbio and nonbio > 0) else None
    print(f"\nbio-selectivity ratio S = recall(bio-harm) / flag-rate(non-bio-harm) = "
          f"{recall:.3f} / {nonbio} = {('%.2f' % S) if S else 'inf (nonbio flag 0)'}")
    print(f"dual-use boundary (WMDP) flag-rate = {rates.get('dual_use_boundary')} "
          f"(LOW = does not over-flag legitimate dual-use research)")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"recall": recall, "benign_bio_fpr": fpr, "selectivity_rates": rates,
               "S_ratio": S, "n_val_pos": npos},
              open(METRICS_DIR / "pdual_v1_eval.json", "w"), indent=2)
    print("\nwrote results/metrics/pdual_v1_eval.json")


if __name__ == "__main__":
    main()

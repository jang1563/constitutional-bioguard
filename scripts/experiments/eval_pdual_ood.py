#!/usr/bin/env python
"""Phase R Part 2: in-dist vs OOD recall on the held-out source.

Runs BOTH the original pdual_v3 (saladbench WAS in training -> in-dist) and the
holdout retrain (saladbench held out -> OOD) on the same held-out genuine-bio
saladbench set. If OOD recall ~= in-dist recall, the prompt head GENERALIZES
(the Phase R head-to-head recall edge is real). If OOD << in-dist, it memorized
and the edge was home-field. Also re-checks benign-bio over-refusal is stable.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR
from constitutional_bioguard.dual_mode import DualModeGuard


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def evalrec(model_path, prompts):
    g = DualModeGuard(prompt_model_path=model_path)
    v = g.classify_batch(prompts)
    hybrid = float(np.mean([x.prompt_flag for x in v]))
    learned = float(np.mean([x.prompt_source in ("learned", "both") for x in v]))
    lex = float(np.mean([x.prompt_source in ("lex", "both") for x in v]))
    return hybrid, learned, lex


def main():
    ood = [r["query"] for r in load(DATA_PROCESSED / "pdual_ood_ho.jsonl")]
    # benign-bio negatives from the clean bench, to confirm over-refusal stable
    clean = load(DATA_PROCESSED / "bio_clean_eval.jsonl")
    neg = [r["query"] for r in clean if int(r["label"]) == 0]

    orig = MODELS_DIR / "deberta_pdual_v3"
    holdout = MODELS_DIR / "deberta_pdual_ho"

    print(f"Held-out saladbench genuine-bio OOD set: n={len(ood)}")
    print(f"Benign-bio negatives: n={len(neg)}\n")

    print("=== RECALL on held-out saladbench (the OOD test) ===")
    print(f"  {'model':28} {'hybrid':>7} {'learned':>8} {'lex':>6}")
    for name, path in (("pdual_v3 (saladbench IN-dist)", orig),
                       ("pdual_ho (saladbench OOD)", holdout)):
        if not Path(path).exists():
            print(f"  {name:28}  MODEL MISSING ({path})")
            continue
        h, lr, lx = evalrec(path, ood)
        print(f"  {name:28} {h:7.3f} {lr:8.3f} {lx:6.3f}")

    print("\n=== benign-bio over-refusal (should stay ~0.01) ===")
    for name, path in (("pdual_v3", orig), ("pdual_ho", holdout)):
        if not Path(path).exists():
            continue
        h, _, _ = evalrec(path, neg)
        print(f"  {name:12} over-refusal={h:.3f}")


if __name__ == "__main__":
    main()

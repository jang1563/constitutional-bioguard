#!/usr/bin/env python
"""Paired significance tests: v7.C vs response-harm baselines on wildguard_native.

Both v7.C (eval_v7b) and the baselines (eval_baselines_response_harm) use the
IDENTICAL loader -> item rows align 1:1. We verify that (gold arrays must match
element-for-element) before testing.

Two tests per baseline:
  1. McNemar (accuracy / error homogeneity) -- exact binomial on discordant pairs.
  2. Paired bootstrap on the F1 difference -- the F1-appropriate test (McNemar
     tests accuracy, not F1). 10k resamples, fixed seed.

Content-blind: reads only integer label/pred arrays; prints only statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def load_v7c(fp):
    d = json.load(open(fp))
    preds = d["predictions"]
    labels = np.array([int(p["label"]) for p in preds])
    yhat = np.array([int(p["pred"]) for p in preds])
    return labels, yhat


def load_baseline(fp):
    d = json.load(open(fp))
    return np.array([int(x) for x in d["labels"]]), np.array([int(x) for x in d["preds"]])


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar on per-item correctness booleans."""
    n10 = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    n01 = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = n10 + n01
    # exact two-sided binomial p (H0: p=0.5)
    try:
        from scipy.stats import binomtest
        p = binomtest(min(n10, n01), n, 0.5, alternative="two-sided").pvalue if n else 1.0
    except Exception:
        from math import comb
        k = min(n10, n01)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
    return n10, n01, p


def paired_bootstrap_f1(gold, yhat_a, yhat_b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(gold))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        g = gold[s]
        if g.sum() == 0 or g.sum() == len(g):
            diffs[i] = 0.0
            continue
        diffs[i] = (f1_score(g, yhat_a[s], zero_division=0)
                    - f1_score(g, yhat_b[s], zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi), float((diffs > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7c", required=True)
    ap.add_argument("--baselines", nargs="+", required=True,
                    help="name=path pairs, e.g. lg3=results/metrics/v7c_baseline_rh_llama_guard_3_8b_wildguard_native.json")
    args = ap.parse_args()

    gold, v7c = load_v7c(args.v7c)
    print(f"v7.C: n={len(gold)} n_pos={int(gold.sum())} "
          f"F1={f1_score(gold, v7c, zero_division=0):.4f} "
          f"acc={(gold==v7c).mean():.4f}")

    for spec in args.baselines:
        name, path = spec.split("=", 1)
        bgold, bpred = load_baseline(path)
        # ALIGNMENT CHECK: gold arrays must be identical
        if len(bgold) != len(gold) or not np.array_equal(bgold, gold):
            print(f"\n[{name}] ALIGNMENT FAILED: gold arrays differ -> test invalid")
            n_match = int(np.sum(bgold[:len(gold)] == gold[:len(bgold)])) if len(bgold)==len(gold) else -1
            print(f"  len v7c={len(gold)} base={len(bgold)} elementwise_gold_match={n_match}")
            continue
        bf1 = f1_score(gold, bpred, zero_division=0)
        print(f"\n[{name}] F1={bf1:.4f} acc={(gold==bpred).mean():.4f}  (gold aligned OK)")

        ca = (v7c == gold)
        cb = (bpred == gold)
        n10, n01, p_mc = mcnemar_exact(ca, cb)
        print(f"  McNemar (accuracy): v7c-only-right={n10} base-only-right={n01} "
              f"exact p={p_mc:.4f}")

        mean, lo, hi, pgt = paired_bootstrap_f1(gold, v7c, bpred)
        sig = "0 NOT in CI -> significant" if (lo > 0 or hi < 0) else "0 in CI -> indistinguishable"
        print(f"  Bootstrap dF1 (v7c - {name}): {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(v7c>base)={pgt:.3f}  -> {sig}")


if __name__ == "__main__":
    main()

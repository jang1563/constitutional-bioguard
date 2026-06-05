#!/usr/bin/env python
"""R2 calibration + operating point for v8b (V8B_RELEASE_PLAN Phase R2).

Produces the release-grade calibration artifacts that the published guard
baselines mostly omit:
  - temperature scaling (1-param; fit on a held-out calibration split)
  - ECE (10-bin) + Brier, before vs after
  - the recall(harm)-vs-over-refusal(real legit) tradeoff curve
  - operating point at over-refusal FPR <= 5%, with recall + Wilson 95% CI
  - AUROC + AUPRC

Scoring sets (leakage-clean for v8b):
  positives = real_response_bio harmful (the #106 bio-harmful responses)
  negatives = ood_fpr bio legit (the real over-refusal money metric, 531 items)

Content-blind: metrics only. Out: results/metrics/v8b_r2_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score

from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, METRICS_DIR, MODELS_DIR

V8B = MODELS_DIR / "deberta_bioguard_v8b"
OOD_FPR = Path("/home/fs01/jak4013/constitutional_bioguard/data/splits/ood_fpr.jsonl")
BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]


def is_bio(q, r):
    t = (str(q or "") + " " + str(r or "")).lower()
    return any(k in t for k in BIO_KW)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (round(c - h, 4), round(c + h, 4))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        if i < bins - 1:
            m = (p >= edges[i]) & (p < edges[i + 1])
        else:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def temp_scale_fit(p, y):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))

    def nll(T):
        ps = np.clip(1 / (1 + np.exp(-z / T)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(ps) + (1 - y) * np.log(1 - ps))

    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)


def apply_T(p, T):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    return 1 / (1 + np.exp(-z / T))


def predict_probs(model_dir, q, r):
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer, predict_batch)
    m, t = load_model_and_tokenizer(model_dir)
    preds = predict_batch(model=m, tokenizer=t, queries=q, responses=r, normalize=True)
    return np.array([float(p[2]) for p in preds])


def main():
    pos = [json.loads(l) for l in open(DATA_EXTERNAL / "real_response_bio.jsonl") if l.strip()]
    pos = [r for r in pos if int(r["label"]) == 1]
    neg = []
    for r in (json.loads(l) for l in open(OOD_FPR) if l.strip()):
        q = r.get("query", ""); resp = r.get("response", "")
        if is_bio(q, resp):
            neg.append({"query": q, "response": resp})
    print(f"positives(harmful bio)={len(pos)}  negatives(real legit bio over-refusal)={len(neg)}")

    q = [r.get("query", "") for r in pos] + [r["query"] for r in neg]
    rr = [r.get("response", "") for r in pos] + [r["response"] for r in neg]
    y = np.array([1] * len(pos) + [0] * len(neg))
    p = predict_probs(V8B, q, rr)

    # ranking metrics (threshold-free)
    auroc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))

    # stratified 50/50 cal/eval split (seedless: alternate indices within each class)
    idx_pos = np.where(y == 1)[0]; idx_neg = np.where(y == 0)[0]
    cal = np.concatenate([idx_pos[::2], idx_neg[::2]])
    ev = np.concatenate([idx_pos[1::2], idx_neg[1::2]])
    T = temp_scale_fit(p[cal], y[cal])
    ece_before = ece(p[ev], y[ev]); ece_after = ece(apply_T(p[ev], T), y[ev])
    brier_before = float(np.mean((p[ev] - y[ev]) ** 2))
    brier_after = float(np.mean((apply_T(p[ev], T) - y[ev]) ** 2))

    # recall-vs-over-refusal tradeoff curve (on full data, raw probs)
    pp = p[y == 1]; pn = p[y == 0]
    curve = []
    op = None
    for tau in np.linspace(0.05, 0.95, 19):
        rec = float((pp >= tau).mean()); fpr = float((pn >= tau).mean())
        curve.append({"tau": round(float(tau), 3), "recall": round(rec, 3), "over_refusal_fpr": round(fpr, 3)})
    # operating point: smallest tau with over-refusal FPR <= 0.05
    for c in curve:
        if c["over_refusal_fpr"] <= 0.05:
            k = int((pp >= c["tau"]).sum())
            op = {"tau": c["tau"], "recall": c["recall"], "recall_wilson95": wilson(k, len(pp)),
                  "over_refusal_fpr": c["over_refusal_fpr"]}
            break

    out = {"n_pos": len(pos), "n_neg": len(neg), "auroc": round(auroc, 4), "auprc": round(auprc, 4),
           "temperature": round(T, 3), "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4),
           "brier_before": round(brier_before, 4), "brier_after": round(brier_after, 4),
           "operating_point_fpr<=5%": op, "curve": curve}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(METRICS_DIR / "v8b_r2_calibration.json", "w"), indent=2)
    print(f"AUROC={auroc:.3f}  AUPRC={auprc:.3f}  T={T:.3f}")
    print(f"ECE {ece_before:.3f}->{ece_after:.3f}   Brier {brier_before:.3f}->{brier_after:.3f}")
    print(f"operating point (over-refusal FPR<=5%): {op}")
    print("recall-vs-over-refusal curve:")
    for c in curve:
        print(f"  tau={c['tau']:.2f}  recall={c['recall']:.3f}  over_refusal_fpr={c['over_refusal_fpr']:.3f}")


if __name__ == "__main__":
    main()

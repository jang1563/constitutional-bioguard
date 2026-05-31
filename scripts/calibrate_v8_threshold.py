#!/usr/bin/env python
"""Threshold calibration for the v8 koaug3 classifier — can a single operating
point pass BOTH OOD gates (FPR<=0.10 AND FNR<=0.05)?

The official eval uses argmax (threshold 0.5 on P(harmful)). koaug3 sits at
FPR 0.0475 / FNR 0.076 there. FPR has headroom to the 0.10 gate, so lowering the
threshold may pull FNR under 0.05 while FPR stays legal. This script measures
that honestly:

  - Scores ood_fpr legit (label 0) and ood_fnr harmful (label 1) with P(harmful),
    using the SAME text format + class index as training/eval.
  - FPR(t) = mean[p>=t] over ood_fpr legit ; FNR(t) = mean[p<t] over ood_fnr harmful.
  - Reports the full tradeoff curve and the feasible t-window where both gates hold.
  - Avoids eval-on-eval overfit: 50/50 calib/test split (seed) of each OOD set,
    picks t* on calib, reports held-out FPR/FNR at t* on test.

Read-only inference. No training, no session data used for fitting.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_text(r: dict) -> str:
    q = (r.get("query", "") or "").strip()
    resp = (r.get("response", "") or "").strip()
    return f"{q} [SEP] {resp[:256]}" if resp else q


def score(model, tok, recs, device, batch_size=64):
    import torch
    texts = [build_text(r) for r in recs]
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], max_length=512, truncation=True,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
        probs.extend(p)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ood-fpr", default="data/splits/ood_fpr.jsonl")
    ap.add_argument("--ood-fnr", default="data/splits/ood_fnr.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-gate", type=float, default=0.10)
    ap.add_argument("--fnr-gate", type=float, default=0.05)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    def load(path, keep):
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        return [r for r in recs if r.get("binary_label") == keep]

    legit = load(args.ood_fpr, "legitimate")   # label 0 -> FPR side
    harm = load(args.ood_fnr, "negative")       # label 1 -> FNR side
    p_legit = np.array(score(model, tok, legit, device))
    p_harm = np.array(score(model, tok, harm, device))

    def fpr_at(t, p=p_legit): return float(np.mean(p >= t)) if len(p) else 0.0
    def fnr_at(t, p=p_harm): return float(np.mean(p < t)) if len(p) else 0.0

    # sanity: reproduce official argmax (t=0.5)
    base = {"threshold": 0.5, "fpr": fpr_at(0.5), "fnr": fnr_at(0.5),
            "n_legit": len(legit), "n_harm": len(harm)}

    # full-set tradeoff curve + feasible window
    grid = np.round(np.linspace(0.01, 0.99, 99), 3)
    curve = [{"t": float(t), "fpr": round(fpr_at(t), 4), "fnr": round(fnr_at(t), 4)} for t in grid]
    # FPR is non-increasing, FNR non-decreasing in t
    t_lo = next((c["t"] for c in curve if c["fpr"] <= args.fpr_gate), None)  # min t with FPR ok
    t_hi = None
    for c in curve:  # max t with FNR ok
        if c["fnr"] <= args.fnr_gate:
            t_hi = c["t"]
    feasible = (t_lo is not None and t_hi is not None and t_lo <= t_hi)
    window = {"t_lo_fpr_ok": t_lo, "t_hi_fnr_ok": t_hi, "feasible": feasible,
              "fpr_at_t_hi": round(fpr_at(t_hi), 4) if t_hi is not None else None,
              "fnr_at_t_lo": round(fnr_at(t_lo), 4) if t_lo is not None else None}

    # held-out calib/test: 50/50 deterministic split of each set
    rng = np.random.default_rng(args.seed)
    def split(p):
        idx = rng.permutation(len(p)); h = len(p) // 2
        return p[idx[:h]], p[idx[h:]]
    pl_cal, pl_test = split(p_legit)
    ph_cal, ph_test = split(p_harm)
    # pick t* on calib: prefer the window where both gates hold; else min FPR s.t. FNR<=gate
    cal_curve = [(float(t), float(np.mean(pl_cal >= t)), float(np.mean(ph_cal < t))) for t in grid]
    cal_feasible = [t for (t, f, n) in cal_curve if f <= args.fpr_gate and n <= args.fnr_gate]
    if cal_feasible:
        # center of the feasible window (max margin)
        t_star = round(float(np.median(cal_feasible)), 3)
        sel = "feasible-window-median"
    else:
        # no joint solution on calib: take max t with FNR<=gate (prioritise FNR gate)
        ok = [t for (t, f, n) in cal_curve if n <= args.fnr_gate]
        t_star = round(float(max(ok)), 3) if ok else 0.5
        sel = "fnr-gate-priority (no joint window on calib)"
    heldout = {
        "t_star": t_star, "selection": sel,
        "calib": {"fpr": round(float(np.mean(pl_cal >= t_star)), 4),
                  "fnr": round(float(np.mean(ph_cal < t_star)), 4),
                  "n_legit": len(pl_cal), "n_harm": len(ph_cal)},
        "test": {"fpr": round(float(np.mean(pl_test >= t_star)), 4),
                 "fnr": round(float(np.mean(ph_test < t_star)), 4),
                 "n_legit": len(pl_test), "n_harm": len(ph_test)},
    }
    heldout["test"]["both_gates_pass"] = (heldout["test"]["fpr"] <= args.fpr_gate
                                          and heldout["test"]["fnr"] <= args.fnr_gate)

    out = {"model": Path(args.model_dir).parent.name, "gates": {"fpr": args.fpr_gate, "fnr": args.fnr_gate},
           "argmax_baseline": base, "feasible_window_fullset": window,
           "heldout_calibration": heldout, "curve": curve}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"argmax_baseline": base, "feasible_window_fullset": window,
                      "heldout_calibration": heldout}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

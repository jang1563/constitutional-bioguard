#!/usr/bin/env python
"""B.4 Confusion matrices + B.7 misclassification subgroup analysis.

Confusion matrices per benchmark per model (where preds available).
Subgroup breakdown for v3 on BioThreat-Eval (if predictions stored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from constitutional_bioguard.config import METRICS_DIR

OUT_PATH = METRICS_DIR / "audit_confusion_misclass.json"


def confusion(y_true, y_pred):
    y, p = np.array(y_true), np.array(y_pred)
    return {
        "tp": int(((y == 1) & (p == 1)).sum()),
        "fp": int(((y == 0) & (p == 1)).sum()),
        "tn": int(((y == 0) & (p == 0)).sum()),
        "fn": int(((y == 1) & (p == 0)).sum()),
    }


def load_preds(model, bench, phase):
    candidates = []
    if model == "v3":
        candidates.extend([
            METRICS_DIR / f"phase2_v3_{bench}.json",
            METRICS_DIR / f"v3_compare_v3_{bench}.json",
        ])
    else:
        candidates.extend([
            METRICS_DIR / f"baseline_{model}_{bench}.json",
            METRICS_DIR / f"phase2_{model}_{bench}.json",
        ])
    for fp in candidates:
        if not fp.exists():
            continue
        d = json.load(fp.open())
        preds = d.get("predictions", [])
        if preds and isinstance(preds[0], dict):
            return [(p.get("label", 0), p.get("pred", 0), p.get("prob", 0.0),
                     p.get("category", "") or p.get("subcategory", "") or
                     p.get("semantic_category", "") or p.get("primary_category", "") or
                     p.get("type", "")) for p in preds]
        if preds:
            return [(p[0], p[1], p[2], "") for p in preds]
    return []


def main():
    models = ["v3", "wildguard_7b", "llama_guard_3_8b"]
    benches_phase1 = ["biothreat", "harmbench_bio_ho", "advbench_bio_ho",
                      "wildguard_test_ho", "lab_bench_ho"]
    benches_phase2 = ["harmbench_full", "advbench_full", "xstest",
                      "beavertails", "wildguard_native"]

    report = {}
    print("=" * 100)
    print("CONFUSION MATRICES per (model, benchmark)")
    print("=" * 100)
    print(f"{'Model':<22}{'Benchmark':<25}{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}{'n':>8}")
    print("-" * 100)
    for model in models:
        report[model] = {}
        all_benches = [(b, 1) for b in benches_phase1] + [(b, 2) for b in benches_phase2]
        for bench, phase in all_benches:
            preds = load_preds(model, bench, phase)
            if not preds:
                continue
            y = [p[0] for p in preds]
            pl = [p[1] for p in preds]
            cm = confusion(y, pl)
            cm["n"] = len(preds)
            key = bench if phase == 1 else f"phase2/{bench}"
            report[model][key] = cm
            print(f"{model:<22}{key:<25}{cm['tp']:>6}{cm['fp']:>6}{cm['tn']:>6}{cm['fn']:>6}{cm['n']:>8}")

    # B.7 v3 misclassification subgroup analysis (v3 on benchmarks with diverse categories)
    print("\n" + "=" * 100)
    print("B.7 v3 MISCLASSIFICATION subgroup analysis")
    print("=" * 100)
    subgroups = {}
    for bench, phase in [("beavertails", 2), ("xstest", 2), ("wildguard_native", 2)]:
        preds = load_preds("v3", bench, phase)
        if not preds:
            continue
        print(f"\n--- {bench} ---")
        by_cat = {}
        for label, pl, prob, cat in preds:
            if not cat:
                cat = "_uncat"
            by_cat.setdefault(cat, []).append((label, pl, prob))
        # Sort by FP+FN count
        sorted_cats = []
        for cat, items in by_cat.items():
            cm = confusion([i[0] for i in items], [i[1] for i in items])
            cm["n"] = len(items)
            cm["fpr"] = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) > 0 else 0
            cm["fnr"] = cm["fn"] / (cm["fn"] + cm["tp"]) if (cm["fn"] + cm["tp"]) > 0 else 0
            sorted_cats.append((cat, cm))
        sorted_cats.sort(key=lambda x: -(x[1]["fp"] + x[1]["fn"]))
        for cat, cm in sorted_cats[:8]:
            print(f"  {cat[:40]:<42}  n={cm['n']:>4}  FPR={cm['fpr']*100:>5.1f}%  FNR={cm['fnr']*100:>5.1f}%  (TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']})")
        subgroups[bench] = {cat: cm for cat, cm in sorted_cats}

    report["v3_subgroups"] = subgroups

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

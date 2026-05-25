#!/usr/bin/env python
"""B.2 Bootstrap CIs on Phase 1 + Phase 2 results.

1000-resample bootstrap on F1, AUROC, AUPRC for v3 + WildGuard + LLaMA-Guard 3
across BioThreat-Eval, HarmBench bio held-out, AdvBench bio held-out,
WildGuard_native, BeaverTails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score, roc_auc_score,
)

from constitutional_bioguard.config import METRICS_DIR

N_BOOTSTRAP = 1000
RNG = np.random.RandomState(42)


def load_preds(model: str, bench: str, phase: int) -> list[tuple[int, int, float]]:
    """Load (label, pred, prob) triples from multiple possible file formats."""
    candidates = []
    if model == "v3":
        # v3 Phase 1 uses v3_compare_v3_*.json (or compare_v3_*.json)
        candidates.extend([
            METRICS_DIR / f"v3_compare_v3_{bench}.json",
            METRICS_DIR / f"compare_v3_{bench}.json",
            METRICS_DIR / f"phase2_v3_{bench}.json",
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
        # Try several schemas
        if "predictions" in d and d["predictions"]:
            preds = d["predictions"]
            if isinstance(preds[0], dict):
                return [(p.get("label", 0), p.get("pred", 0), p.get("prob", 0.0)) for p in preds]
            return [(p[0], p[1], p[2]) for p in preds]
        # v3 biothreat schema: by_strategy/threat_level_4/variant_a/...
        if "by_strategy" in d:
            tl4 = d["by_strategy"].get("threat_level_4", {})
            # We don't have per-item predictions here; skip
            continue
        # External benchmark schema: overall + by_category, no predictions
        if "overall" in d:
            continue
    return []


def bootstrap_metrics(preds: list[tuple], n_boot: int = N_BOOTSTRAP) -> dict:
    if not preds:
        return {}
    arr = np.array(preds)
    y = arr[:, 0].astype(int)
    p = arr[:, 1].astype(int)
    probs = arr[:, 2].astype(float)
    n = len(arr)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    f1s, aurocs, auprcs, recalls, fprs = [], [], [], [], []
    has_mixed = n_pos > 0 and n_neg > 0

    for _ in range(n_boot):
        idx = RNG.choice(n, size=n, replace=True)
        yi, pi, pri = y[idx], p[idx], probs[idx]
        if pi.sum() > 0:
            f1s.append(f1_score(yi, pi, zero_division=0))
        else:
            f1s.append(0.0)
        recalls.append(recall_score(yi, pi, zero_division=0))
        if (yi == 0).sum() > 0:
            fprs.append(pi[yi == 0].mean())
        if has_mixed and len(set(yi)) > 1:
            try:
                aurocs.append(roc_auc_score(yi, pri))
                auprcs.append(average_precision_score(yi, pri))
            except Exception:
                pass

    def ci(arr_or_list):
        if not arr_or_list:
            return None
        a = np.array(arr_or_list)
        return {
            "mean": round(float(a.mean()), 4),
            "lo_2_5": round(float(np.percentile(a, 2.5)), 4),
            "hi_97_5": round(float(np.percentile(a, 97.5)), 4),
        }

    return {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "f1": ci(f1s),
        "recall": ci(recalls),
        "fpr": ci(fprs),
        "auroc": ci(aurocs),
        "auprc": ci(auprcs),
    }


def main():
    benchmarks = [
        ("biothreat", 1),
        ("harmbench_bio_ho", 1),
        ("advbench_bio_ho", 1),
        ("wildguard_test_ho", 1),
        ("lab_bench_ho", 1),
        # Phase 2:
        ("biothreat", 2),  # phase2 prefix doesn't exist for biothreat — skip
    ]
    models = ["v3", "wildguard_7b", "llama_guard_3_8b"]

    # Phase 2 benchmark names for v3 (have predictions saved with prob)
    phase2_benches = ["harmbench_full", "advbench_full", "xstest",
                       "beavertails", "wildguard_native"]
    phase1_benches = ["biothreat", "harmbench_bio_ho", "advbench_bio_ho",
                      "wildguard_test_ho", "lab_bench_ho"]

    report = {}
    for model in models:
        report[model] = {}
        # Try phase 1 benchmark names
        for bench in phase1_benches:
            preds = load_preds(model, bench, phase=1)
            if not preds:
                continue
            print(f"  bootstrapping {model} / {bench} (phase 1)...")
            report[model][bench] = bootstrap_metrics(preds)
        # Try phase 2 benchmark names
        for bench in phase2_benches:
            preds = load_preds(model, bench, phase=2)
            if not preds:
                continue
            print(f"  bootstrapping {model} / {bench} (phase 2)...")
            report[model][f"phase2/{bench}"] = bootstrap_metrics(preds)

    out_path = METRICS_DIR / "audit_bootstrap_cis.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 90)
    print("BOOTSTRAP 95% CIs (1000 resamples)")
    print("=" * 90)
    for model in models:
        print(f"\n--- {model} ---")
        for bench, stats in report.get(model, {}).items():
            if not stats:
                continue
            line = f"  {bench:25s} n={stats['n']:5d}  "
            for metric in ["f1", "auroc", "auprc", "recall", "fpr"]:
                ci = stats.get(metric)
                if ci:
                    line += f"{metric}={ci['mean']:.3f}[{ci['lo_2_5']:.3f}-{ci['hi_97_5']:.3f}] "
            print(line)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

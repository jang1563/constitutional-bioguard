#!/usr/bin/env python
"""Rebuild v3_eval_summary.json from individual v3_compare_*.json files.

Solves the race condition where the BioThreat patch overwrote the
main pipeline's summary (patch loaded the summary before the main pipeline
finished writing it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import METRICS_DIR

MODELS = ["a_full", "v2", "v3"]

# Bio adversarial held-out benchmarks (label=1, so "FAR" = recall)
BIO_ADV_HO = ["harmbench_bio_held_out", "advbench_bio_held_out"]

# Cross-domain benchmarks (label=0)
CROSS_DOMAIN = [
    "wmdp_chem",
    "wmdp_cyber",
    "lab_bench",
    "pubmed_qa_pqa_labeled",
    "med_qa_test",
    "wildguard_test",
]

# WMDP-Bio uses mixed labels
WMDP_BIO = "wmdp_bio"


def main():
    summary = {}

    for tag in MODELS:
        # ── BioThreat-Eval ──
        bt_path = METRICS_DIR / f"v3_compare_{tag}_biothreat.json"
        if bt_path.exists():
            d = json.load(bt_path.open())
            try:
                m = d["by_strategy"]["threat_level_4"]["variant_a"][
                    "metrics_at_0.5"
                ]
                summary[f"biothreat_{tag}"] = {
                    "f1": m["f1"],
                    "auroc": m["auroc"],
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "fpr": m["fpr"],
                }
            except KeyError:
                pass

        # ── WMDP-Bio ──
        wb_path = METRICS_DIR / f"v3_compare_{tag}_wmdp_bio.json"
        if wb_path.exists():
            d = json.load(wb_path.open())
            try:
                m = d["metrics_at_0.5"]
                summary[f"wmdp_bio_{tag}"] = {
                    "f1": m["f1"],
                    "auroc": m["auroc"],
                    "auprc": m.get("au_prc"),
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "fpr": m["fpr"],
                }
            except KeyError:
                pass

        # ── Bio adversarial held-out ──
        for bench in BIO_ADV_HO:
            p = METRICS_DIR / f"v3_compare_{tag}_{bench}.json"
            if p.exists():
                d = json.load(p.open())
                o = d.get("overall", {})
                key = f"{bench.replace('_held_out', '_ho')}_{tag}"
                summary[key] = {
                    "far": o.get("false_alarm_rate"),
                    "ci_lower": o.get("wilson_95_lower"),
                    "ci_upper": o.get("wilson_95_upper"),
                    "n": o.get("n"),
                    "n_flagged": o.get("n_flagged_unsafe"),
                    "mean_prob": o.get("mean_prob_unsafe"),
                }

        # ── Cross-domain ──
        for bench in CROSS_DOMAIN:
            p = METRICS_DIR / f"v3_compare_{tag}_{bench}.json"
            if p.exists():
                d = json.load(p.open())
                o = d.get("overall", {})
                summary[f"{bench}_{tag}"] = {
                    "far": o.get("false_alarm_rate"),
                    "ci_lower": o.get("wilson_95_lower"),
                    "ci_upper": o.get("wilson_95_upper"),
                    "n": o.get("n"),
                    "n_flagged": o.get("n_flagged_unsafe"),
                    "mean_prob": o.get("mean_prob_unsafe"),
                }

    # Save
    summary_path = METRICS_DIR / "v3_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Rebuilt summary: {summary_path}")
    print(f"  {len(summary)} entries")
    for k in sorted(summary.keys()):
        v = summary[k]
        if "far" in v:
            print(f"    {k}: FAR={v['far']:.4f} (n={v['n']})")
        else:
            print(f"    {k}: F1={v.get('f1', 'N/A')} AUROC={v.get('auroc', 'N/A')} recall={v.get('recall', 'N/A')}")


if __name__ == "__main__":
    main()

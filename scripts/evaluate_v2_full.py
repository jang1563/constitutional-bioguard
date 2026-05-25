#!/usr/bin/env python
"""Full v2 evaluation across all 8 external benchmarks.

Runs deberta_bioguard_v2_augmented on:
  - WMDP-Bio (5092 items, MCQ-derived labels)
  - BioThreat-Eval (558 items, expert labels)
  - WMDP-Chem held out (408 items)
  - WMDP-Cyber held out (1787 items)
  - LAB-Bench held out (976 items, stratified by subtask)
  - PubMedQA held out (1000 items)
  - MedQA held out (1273 items)
  - WildGuardMix held out (1109 items, stratified by adversarial)

Output: results/metrics/v2_eval_{benchmark}.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import (
    DATA_EXTERNAL,
    METRICS_DIR,
    MODELS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from constitutional_bioguard.evaluation.corrective_experiments import (
        run_biothreat_ood_evaluation,
        run_external_benchmark_evaluation,
        run_ood_evaluation,
    )

    V2_DIR = MODELS_DIR / "deberta_bioguard_v2_augmented"
    SPLITS = DATA_EXTERNAL / "v2_splits"

    if not V2_DIR.exists():
        logger.error("v2 model not found at %s", V2_DIR)
        sys.exit(1)

    logger.info("=== v2 Full Evaluation ===")
    logger.info("Model: %s", V2_DIR)
    summary = {}

    # 1. WMDP-Bio (uses cached jsonl directly)
    logger.info("\n[1/8] WMDP-Bio")
    try:
        out = METRICS_DIR / "v2_eval_wmdp_bio.json"
        r = run_ood_evaluation(model_dir=V2_DIR, output_file=out)
        m = r["metrics_at_0.5"]
        summary["wmdp_bio"] = {
            "auroc": m["auroc"], "auprc": m["au_prc"],
            "f1": m["f1"], "fpr": m["fpr"],
        }
    except Exception as e:
        logger.error("WMDP-Bio failed: %s", e)
        summary["wmdp_bio"] = {"error": str(e)}

    # 2. BioThreat-Eval
    logger.info("\n[2/8] BioThreat-Eval")
    try:
        out = METRICS_DIR / "v2_eval_biothreat.json"
        r = run_biothreat_ood_evaluation(model_dir_a=V2_DIR, output_file=out)
        tl4 = r["by_strategy"]["threat_level_4"]["variant_a"]["metrics_at_0.5"]
        summary["biothreat_eval"] = {
            "auroc": tl4["auroc"], "auprc": tl4["au_prc"],
            "f1": tl4["f1"], "fpr": tl4["fpr"], "recall": tl4["recall"],
        }
    except Exception as e:
        logger.error("BioThreat failed: %s", e)
        summary["biothreat_eval"] = {"error": str(e)}

    # 3-8. External benchmarks via unified evaluator
    eval_specs = [
        ("wmdp_chem_held_out", None),
        ("wmdp_cyber_held_out", None),
        ("lab_bench_held_out", "subtask"),
        ("pubmed_qa_pqa_labeled_held_out", None),
        ("med_qa_test_held_out", None),
        ("wildguard_test_held_out", "adversarial"),
    ]
    for i, (name, strat) in enumerate(eval_specs, start=3):
        cache_path = SPLITS / f"{name}.jsonl"
        if not cache_path.exists():
            logger.warning("[%d/8] %s: cache missing", i, name)
            summary[name] = {"error": "missing"}
            continue
        logger.info("\n[%d/8] %s (strat=%s)", i, name, strat)
        try:
            out_path = METRICS_DIR / f"v2_eval_{name}.json"
            r = run_external_benchmark_evaluation(
                cache_path=cache_path,
                model_dir=V2_DIR,
                benchmark_name=name,
                output_file=out_path,
                stratification_field=strat,
            )
            o = r["overall"]
            summary[name] = {
                "far": o["false_alarm_rate"],
                "ci_lower": o["wilson_95_lower"],
                "ci_upper": o["wilson_95_upper"],
                "n": o["n"],
                "n_flagged": o["n_flagged_unsafe"],
            }
        except Exception as e:
            logger.error("%s failed: %s", name, e)
            summary[name] = {"error": str(e)}

    # Save combined summary
    summary_path = METRICS_DIR / "v2_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nSummary saved: %s", summary_path)

    # Print readable table
    print("\n" + "=" * 70)
    print(f"{'Benchmark':<35s} {'Metric':<10s} {'Value':<10s}")
    print("=" * 70)
    for bench, m in summary.items():
        if "error" in m:
            print(f"{bench:<35s} ERROR: {m['error']}")
            continue
        if "auroc" in m:
            print(f"{bench:<35s} {'F1':<10s} {m['f1']:.4f}")
            print(f"{bench:<35s} {'AUROC':<10s} {m['auroc']:.4f}")
            print(f"{bench:<35s} {'AUPRC':<10s} {m['auprc']:.4f}")
            if "recall" in m:
                print(f"{bench:<35s} {'Recall':<10s} {m['recall']:.4f}")
            print(f"{bench:<35s} {'FPR':<10s} {m['fpr']:.4f}")
        if "far" in m:
            print(f"{bench:<35s} {'FAR':<10s} {m['far']:.4f}")
            print(f"{bench:<35s} {'n':<10s} {m['n_flagged']}/{m['n']}")
        print("-" * 70)


if __name__ == "__main__":
    main()

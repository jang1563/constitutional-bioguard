#!/usr/bin/env python
"""Apples-to-apples comparison: A_full vs v2 on identical held-out splits.

Runs both models on the same held-out subsets to ensure fair v1 vs v2 comparison.
Saves to results/metrics/compare_{model}_{benchmark}.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR

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

    A_FULL = MODELS_DIR / "deberta_bioguard_v1_A_full"
    V2 = MODELS_DIR / "deberta_bioguard_v2_augmented"
    SPLITS = DATA_EXTERNAL / "v2_splits"

    # External benchmark held-out files
    eval_specs = [
        ("wmdp_chem_held_out", None),
        ("wmdp_cyber_held_out", None),
        ("lab_bench_held_out", "subtask"),
        ("pubmed_qa_pqa_labeled_held_out", None),
        ("med_qa_test_held_out", None),
        ("wildguard_test_held_out", "adversarial"),
    ]

    models = [("a_full", A_FULL), ("v2", V2)]
    summary = {}

    # 1. WMDP-Bio (full set, eval only)
    for tag, m_dir in models:
        logger.info(f"\n[WMDP-Bio | {tag}]")
        try:
            out = METRICS_DIR / f"compare_{tag}_wmdp_bio.json"
            r = run_ood_evaluation(model_dir=m_dir, output_file=out)
            mm = r["metrics_at_0.5"]
            summary[f"wmdp_bio_{tag}"] = {
                "f1": mm["f1"], "auroc": mm["auroc"], "auprc": mm["au_prc"],
                "fpr": mm["fpr"], "recall": mm["recall"], "precision": mm["precision"],
            }
        except Exception as e:
            logger.error(f"WMDP-Bio {tag} failed: {e}")
            summary[f"wmdp_bio_{tag}"] = {"error": str(e)}

    # 2. BioThreat-Eval (full 558)
    for tag, m_dir in models:
        logger.info(f"\n[BioThreat | {tag}]")
        try:
            out = METRICS_DIR / f"compare_{tag}_biothreat.json"
            r = run_biothreat_ood_evaluation(model_dir_a=m_dir, output_file=out)
            tl4 = r["by_strategy"]["threat_level_4"]["variant_a"]["metrics_at_0.5"]
            summary[f"biothreat_{tag}"] = {
                "f1": tl4["f1"], "auroc": tl4["auroc"], "auprc": tl4["au_prc"],
                "fpr": tl4["fpr"], "recall": tl4["recall"], "precision": tl4["precision"],
            }
        except Exception as e:
            logger.error(f"BioThreat {tag} failed: {e}")
            summary[f"biothreat_{tag}"] = {"error": str(e)}

    # 3-8. External benchmarks (held-out splits)
    for name, strat in eval_specs:
        cache_path = SPLITS / f"{name}.jsonl"
        if not cache_path.exists():
            logger.warning(f"Skip {name}: cache missing")
            continue
        for tag, m_dir in models:
            logger.info(f"\n[{name} | {tag}]")
            try:
                out = METRICS_DIR / f"compare_{tag}_{name}.json"
                r = run_external_benchmark_evaluation(
                    cache_path=cache_path,
                    model_dir=m_dir,
                    benchmark_name=f"{tag}_{name}",
                    output_file=out,
                    stratification_field=strat,
                )
                o = r["overall"]
                summary[f"{name}_{tag}"] = {
                    "far": o["false_alarm_rate"],
                    "ci_lower": o["wilson_95_lower"],
                    "ci_upper": o["wilson_95_upper"],
                    "n": o["n"],
                    "n_flagged": o["n_flagged_unsafe"],
                    "mean_prob": o["mean_prob_unsafe"],
                }
            except Exception as e:
                logger.error(f"{name} {tag} failed: {e}")
                summary[f"{name}_{tag}"] = {"error": str(e)}

    # Save and print
    summary_path = METRICS_DIR / "compare_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("APPLES-TO-APPLES: A_full vs v2 on IDENTICAL held-out splits")
    print("=" * 90)

    # Extract benchmark roots
    benches = set()
    for k in summary:
        if k.endswith("_a_full"):
            benches.add(k[:-len("_a_full")])
        elif k.endswith("_v2"):
            benches.add(k[:-len("_v2")])

    for b in sorted(benches):
        a = summary.get(f"{b}_a_full", {})
        v = summary.get(f"{b}_v2", {})
        print(f"\n{b}:")
        if "f1" in a and "f1" in v:
            print(f"  F1:       A={a['f1']:.4f}  v2={v['f1']:.4f}  Δ={v['f1']-a['f1']:+.4f}")
            print(f"  AUROC:    A={a['auroc']:.4f}  v2={v['auroc']:.4f}  Δ={v['auroc']-a['auroc']:+.4f}")
            print(f"  Recall:   A={a['recall']:.4f}  v2={v['recall']:.4f}  Δ={v['recall']-a['recall']:+.4f}")
            print(f"  Precision:A={a['precision']:.4f}  v2={v['precision']:.4f}  Δ={v['precision']-a['precision']:+.4f}")
            print(f"  FPR:      A={a['fpr']:.4f}  v2={v['fpr']:.4f}  Δ={v['fpr']-a['fpr']:+.4f}")
        if "far" in a and "far" in v:
            print(f"  FAR:      A={a['far']:.4f}  v2={v['far']:.4f}  Δ={v['far']-a['far']:+.4f}")
            print(f"  n_flagged: A={a['n_flagged']}/{a['n']}  v2={v['n_flagged']}/{v['n']}")
            print(f"  mean_prob: A={a['mean_prob']:.4f}  v2={v['mean_prob']:.4f}")


if __name__ == "__main__":
    main()

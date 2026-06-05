#!/usr/bin/env python
"""Full v3 evaluation: all three models (A_full, v2, v3) on identical benchmarks.

Runs the complete benchmark suite and produces a comparison table.
This is the definitive evaluation for the v3 balanced approach.

Benchmarks:
  1. BioThreat-Eval (558 items, expert labels) — bio recall
  2. HarmBench bio held-out (if available) — bio recall
  3. WMDP-Bio (5092 items) — bio detection
  4. WMDP-Chem held-out — domain specificity
  5. WMDP-Cyber held-out — cross-domain FAR
  6. LAB-Bench held-out — bio-legitimate FAR
  7. WildGuardMix held-out — adversarial FAR
  8. PubMedQA held-out — medical-legitimate FAR
  9. MedQA held-out — medical-legitimate FAR

Output: results/metrics/v3_eval_{benchmark}.json + comparison tables
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

    # Model directories
    A_FULL = MODELS_DIR / "deberta_bioguard_v1_A_full"
    V2 = MODELS_DIR / "deberta_bioguard_v2_augmented"
    V3 = MODELS_DIR / "deberta_bioguard_v3_balanced"

    models = [("a_full", A_FULL), ("v2", V2), ("v3", V3)]

    # Check model availability
    for tag, m_dir in models:
        if not m_dir.exists():
            logger.warning("Model not found: %s (%s)", tag, m_dir)

    # Use v3 splits for held-out data (v3 has different split sizes than v2)
    # For eval-only datasets (wmdp_chem, pubmed_qa, med_qa), both use full set
    V3_SPLITS = DATA_EXTERNAL / "v3_splits"
    V2_SPLITS = DATA_EXTERNAL / "v2_splits"

    # Prefer v3 splits; fall back to v2 splits for datasets unchanged between versions
    def get_held_out(name: str) -> Path:
        v3_path = V3_SPLITS / f"{name}_held_out.jsonl"
        if v3_path.exists():
            return v3_path
        v2_path = V2_SPLITS / f"{name}_held_out.jsonl"
        if v2_path.exists():
            return v2_path
        return v3_path  # will trigger "missing" warning

    summary = {}

    # ── 1. BioThreat-Eval ──
    for tag, m_dir in models:
        if not m_dir.exists():
            continue
        logger.info(f"\n[BioThreat-Eval | {tag}]")
        try:
            out = METRICS_DIR / f"v3_compare_{tag}_biothreat.json"
            r = run_biothreat_ood_evaluation(model_dir_a=m_dir, output_file=out)
            tl4 = r["by_strategy"]["threat_level_4"]["variant_a"]["metrics_at_0.5"]
            summary[f"biothreat_{tag}"] = {
                "f1": tl4["f1"], "auroc": tl4["auroc"],
                "recall": tl4["recall"], "precision": tl4["precision"],
                "fpr": tl4["fpr"],
            }
        except Exception as e:
            logger.error(f"BioThreat {tag}: {e}")
            summary[f"biothreat_{tag}"] = {"error": str(e)}

    # ── 2. WMDP-Bio ──
    for tag, m_dir in models:
        if not m_dir.exists():
            continue
        logger.info(f"\n[WMDP-Bio | {tag}]")
        try:
            out = METRICS_DIR / f"v3_compare_{tag}_wmdp_bio.json"
            r = run_ood_evaluation(model_dir=m_dir, output_file=out)
            mm = r["metrics_at_0.5"]
            summary[f"wmdp_bio_{tag}"] = {
                "f1": mm["f1"], "auroc": mm["auroc"],
                "recall": mm["recall"], "precision": mm["precision"],
                "fpr": mm["fpr"],
            }
        except Exception as e:
            logger.error(f"WMDP-Bio {tag}: {e}")
            summary[f"wmdp_bio_{tag}"] = {"error": str(e)}

    # ── 3. Bio adversarial held-outs ──
    bio_adv_held_outs = ["harmbench_bio", "advbench_bio"]
    for bench_name in bio_adv_held_outs:
        cache_path = V3_SPLITS / f"{bench_name}_held_out.jsonl"
        if not cache_path.exists():
            logger.warning(f"Skip {bench_name} held-out: not found")
            continue
        for tag, m_dir in models:
            if not m_dir.exists():
                continue
            logger.info(f"\n[{bench_name} held-out | {tag}]")
            try:
                out = METRICS_DIR / f"v3_compare_{tag}_{bench_name}_held_out.json"
                r = run_external_benchmark_evaluation(
                    cache_path=cache_path,
                    model_dir=m_dir,
                    benchmark_name=f"{tag}_{bench_name}_ho",
                    output_file=out,
                    stratification_field=None,
                )
                o = r["overall"]
                summary[f"{bench_name}_ho_{tag}"] = {
                    "far": o["false_alarm_rate"],
                    "n": o["n"],
                    "n_flagged": o["n_flagged_unsafe"],
                    "mean_prob": o["mean_prob_unsafe"],
                }
            except Exception as e:
                logger.error(f"{bench_name} {tag}: {e}")
                summary[f"{bench_name}_ho_{tag}"] = {"error": str(e)}

    # ── 4-9. Cross-domain benchmarks ──
    cross_domain_specs = [
        ("wmdp_chem", None),
        ("wmdp_cyber", None),
        ("lab_bench", "subtask"),
        ("pubmed_qa_pqa_labeled", None),
        ("med_qa_test", None),
        ("wildguard_test", "adversarial"),
    ]
    for bench_name, strat in cross_domain_specs:
        cache_path = get_held_out(bench_name)
        if not cache_path.exists():
            logger.warning(f"Skip {bench_name}: held-out missing at {cache_path}")
            continue
        for tag, m_dir in models:
            if not m_dir.exists():
                continue
            logger.info(f"\n[{bench_name} | {tag}]")
            try:
                out = METRICS_DIR / f"v3_compare_{tag}_{bench_name}.json"
                r = run_external_benchmark_evaluation(
                    cache_path=cache_path,
                    model_dir=m_dir,
                    benchmark_name=f"{tag}_{bench_name}",
                    output_file=out,
                    stratification_field=strat,
                )
                o = r["overall"]
                summary[f"{bench_name}_{tag}"] = {
                    "far": o["false_alarm_rate"],
                    "ci_lower": o["wilson_95_lower"],
                    "ci_upper": o["wilson_95_upper"],
                    "n": o["n"],
                    "n_flagged": o["n_flagged_unsafe"],
                    "mean_prob": o["mean_prob_unsafe"],
                }
            except Exception as e:
                logger.error(f"{bench_name} {tag}: {e}")
                summary[f"{bench_name}_{tag}"] = {"error": str(e)}

    # ── Save summary ──
    summary_path = METRICS_DIR / "v3_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Print comparison table ──
    print("\n" + "=" * 95)
    print("THREE-WAY COMPARISON: A_full vs v2 vs v3")
    print("=" * 95)

    # Group by benchmark
    benchmarks = set()
    for k in summary:
        for tag_suffix in ("_a_full", "_v2", "_v3"):
            if k.endswith(tag_suffix):
                benchmarks.add(k[:-len(tag_suffix)])
                break

    for bench in sorted(benchmarks):
        a = summary.get(f"{bench}_a_full", {})
        v2 = summary.get(f"{bench}_v2", {})
        v3 = summary.get(f"{bench}_v3", {})
        print(f"\n{bench}:")

        if "f1" in (a or v2 or v3):
            for metric in ["f1", "auroc", "recall", "precision", "fpr"]:
                vals = []
                for tag, d in [("A", a), ("v2", v2), ("v3", v3)]:
                    if metric in d:
                        vals.append(f"{tag}={d[metric]:.4f}")
                    else:
                        vals.append(f"{tag}=N/A")
                print(f"  {metric:12s} {' | '.join(vals)}")

        if "far" in (a or v2 or v3):
            for metric in ["far", "n_flagged", "mean_prob"]:
                vals = []
                for tag, d in [("A", a), ("v2", v2), ("v3", v3)]:
                    if metric in d:
                        if metric == "n_flagged":
                            vals.append(f"{tag}={d[metric]}/{d.get('n', '?')}")
                        else:
                            vals.append(f"{tag}={d[metric]:.4f}")
                    else:
                        vals.append(f"{tag}=N/A")
                print(f"  {metric:12s} {' | '.join(vals)}")

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()

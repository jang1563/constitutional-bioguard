#!/usr/bin/env python
"""CLI driver for corrective experiments (Section 6 of technical report).

Usage:
    # Run all experiments
    python scripts/run_corrective.py --model-a /path/to/A_full

    # Run single experiment
    python scripts/run_corrective.py --experiment 6.1 --model-a /path/to/A_full

    # Quick test with limited WMDP-Bio data
    python scripts/run_corrective.py --experiment 6.1 --wmdp-limit 10

    # Cache WMDP-Bio data for air-gapped compute nodes
    python scripts/run_corrective.py --cache-wmdp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrective experiments for Constitutional BioGuard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Experiments:\n"
            "  6.1  OOD evaluation on WMDP-Bio\n"
            "  6.2  Bootstrap CIs for WS-2 kappa\n"
            "  6.3  Pre- vs post-preprocessing adversarial comparison\n"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["6.1", "6.2", "6.3", "all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--model-a",
        type=Path,
        default=None,
        help="Path to A_full model directory",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        default=None,
        help="Path to B_bowhard model directory (for experiment 6.2)",
    )
    parser.add_argument(
        "--wmdp-limit",
        type=int,
        default=None,
        help="Limit WMDP-Bio questions for quick testing",
    )
    parser.add_argument(
        "--cache-wmdp",
        action="store_true",
        help="Download and cache WMDP-Bio data locally, then exit",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help="Experiment IDs to skip when running all (e.g., --skip 6.2)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    from constitutional_bioguard.evaluation.corrective_experiments import (
        cache_wmdp_bio,
        run_adversarial_comparison,
        run_all_corrective,
        run_bootstrap_kappa,
        run_ood_evaluation,
    )

    # Handle --cache-wmdp
    if args.cache_wmdp:
        path = cache_wmdp_bio(limit=args.wmdp_limit)
        print(f"WMDP-Bio cached to: {path}")
        return

    # Run selected experiment(s)
    if args.experiment == "all":
        results = run_all_corrective(
            model_dir_a=args.model_a,
            model_dir_b=args.model_b,
            wmdp_limit=args.wmdp_limit,
            skip=args.skip,
        )
        _print_summary(results)

    elif args.experiment == "6.1":
        results = run_ood_evaluation(
            model_dir=args.model_a, limit=args.wmdp_limit,
        )
        _print_ood_summary(results)

    elif args.experiment == "6.2":
        results = run_bootstrap_kappa(
            model_dir_a=args.model_a, model_dir_b=args.model_b,
        )
        _print_bootstrap_summary(results)

    elif args.experiment == "6.3":
        results = run_adversarial_comparison(model_dir=args.model_a)
        _print_adversarial_summary(results)


def _print_summary(results: dict):
    """Print formatted summary of all experiments."""
    print("\n" + "=" * 60)
    print("CORRECTIVE EXPERIMENTS SUMMARY")
    print("=" * 60)

    if "6.1" in results and "error" not in results["6.1"]:
        r = results["6.1"]
        m = r["metrics_at_0.5"]
        print(f"\n6.1 OOD (WMDP-Bio): AU-PRC={m['au_prc']:.4f}  "
              f"AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}  FPR={m['fpr']:.4f}")
    elif "6.1" in results:
        print(f"\n6.1 OOD: FAILED - {results['6.1'].get('error', 'unknown')}")

    if "6.2" in results and "error" not in results["6.2"]:
        r = results["6.2"]
        k = r["variant_a"]["kappa"]
        print(f"\n6.2 Bootstrap CI (A_full): kappa={k['point_estimate']:.4f} "
              f"[{k['ci_lower']:.4f}, {k['ci_upper']:.4f}]")
        if "variant_b" in r and "kappa" in r["variant_b"]:
            kb = r["variant_b"]["kappa"]
            print(f"    Bootstrap CI (B_bowhard): kappa={kb['point_estimate']:.4f} "
                  f"[{kb['ci_lower']:.4f}, {kb['ci_upper']:.4f}]")
        if "delta_analysis" in r:
            d = r["delta_analysis"]
            print(f"    Delta: {d['delta_kappa_A_minus_B']:.4f} "
                  f"CIs overlap: {d['cis_overlap']}")
    elif "6.2" in results:
        print(f"\n6.2 Bootstrap: FAILED - {results['6.2'].get('error', 'unknown')}")

    if "6.3" in results and "error" not in results["6.3"]:
        s = results["6.3"]["summary"]
        print(f"\n6.3 Adversarial: pre-VDR={s['pre_preprocessing_vdr_per_1k']}/1k  "
              f"post-VDR={s['post_preprocessing_vdr_per_1k']}/1k  "
              f"blocked={s['attacks_blocked_by_preprocessing']}")
    elif "6.3" in results:
        print(f"\n6.3 Adversarial: FAILED - {results['6.3'].get('error', 'unknown')}")

    print()


def _print_ood_summary(r: dict):
    m = r["metrics_at_0.5"]
    ws1 = r["metrics_at_0.65_ws1"]
    print(f"\n6.1 OOD Evaluation (WMDP-Bio, {r['n_questions']} questions)")
    print(f"  At threshold 0.50: AU-PRC={m['au_prc']:.4f} AUROC={m['auroc']:.4f} "
          f"F1={m['f1']:.4f} FPR={m['fpr']:.4f}")
    print(f"  At threshold 0.65: recall={ws1['recall']:.4f} FPR={ws1['fpr']:.4f}")
    esc = r["escalation_at_ws1_point"]
    print(f"  Escalation@1%: {esc['escalation_rate_base_0.01']:.4f}")
    wmdp = r["wmdp_specific"]
    print(f"  Correct answers flagged: {wmdp['correct_answers_flagged_unsafe']}/"
          f"{r['n_questions']} ({wmdp['correct_answers_flagged_pct']}%)")


def _print_bootstrap_summary(r: dict):
    k = r["variant_a"]["kappa"]
    print(f"\n6.2 Bootstrap CIs (n={r['n_iterations']} iterations)")
    print(f"  A_full kappa: {k['point_estimate']:.4f} [{k['ci_lower']:.4f}, {k['ci_upper']:.4f}]")
    if "variant_b" in r and "kappa" in r["variant_b"]:
        kb = r["variant_b"]["kappa"]
        print(f"  B_bowhard kappa: {kb['point_estimate']:.4f} [{kb['ci_lower']:.4f}, {kb['ci_upper']:.4f}]")


def _print_adversarial_summary(r: dict):
    s = r["summary"]
    print(f"\n6.3 Adversarial Comparison ({s['total_attacks']} attacks)")
    print(f"  Pre-preprocessing  VDR: {s['pre_preprocessing_vdr_per_1k']}/1k "
          f"({s['pre_preprocessing_total_flipped']} flipped)")
    print(f"  Post-preprocessing VDR: {s['post_preprocessing_vdr_per_1k']}/1k "
          f"({s['post_preprocessing_total_flipped']} flipped)")
    print(f"  Attacks blocked by preprocessing: {s['attacks_blocked_by_preprocessing']}")
    print(f"  Attacks classifier handles alone: {s['attacks_classifier_handles_independently']}")
    print("\n  Per-category:")
    for cat, info in r["by_category"].items():
        print(f"    {cat:15s}: pre={info['pre_asr']:.4f} post={info['post_asr']:.4f} "
              f"delta={info['delta']:.4f}")


if __name__ == "__main__":
    main()

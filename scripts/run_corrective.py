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
            "  6.7  OOD evaluation on BioThreat-Eval (expert labels)\n"
            "  6.8  WildGuardMix adversarial cross-domain OOD\n"
            "  6.8b Stratified diagnosis of WildGuardMix false alarms\n"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["6.1", "6.2", "6.3", "6.7", "6.8", "6.8b", "all"],
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
        run_biothreat_ood_evaluation,
        run_bootstrap_kappa,
        run_ood_evaluation,
        run_wildguard_adversarial_ood,
        run_wildguard_stratified_diagnosis,
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

    elif args.experiment == "6.7":
        results = run_biothreat_ood_evaluation(
            model_dir_a=args.model_a, model_dir_b=args.model_b,
        )
        _print_biothreat_summary(results)

    elif args.experiment == "6.8":
        results = run_wildguard_adversarial_ood(
            model_dir_a=args.model_a, model_dir_b=args.model_b,
        )
        _print_wildguard_summary(results)

    elif args.experiment == "6.8b":
        results = run_wildguard_stratified_diagnosis(
            model_dir_a=args.model_a,
        )
        _print_stratified_summary(results)


def _print_stratified_summary(r: dict):
    print(f"\n6.8b Stratified diagnosis ({r['n_items']} items)")
    print(f"  Length correlation (Spearman rho): "
          f"{r['length_correlation']['spearman_rho_length_vs_prob']:.4f} "
          f"(p={r['length_correlation']['spearman_p_value']:.4g})")
    print(f"    -> {r['length_correlation']['interpretation']}")
    pp = r["probability_percentiles"]
    print(f"  Prob medians: overall={pp['overall']['p50']:.3f} "
          f"adv={pp['adversarial']['p50']:.3f} "
          f"vani={pp['vanilla']['p50']:.3f}")
    print("\n  Length x Adversarial FAR:")
    for s in r["length_x_adversarial_x_far"]:
        print(f"    Q{s['length_quartile']} {'adv' if s['adversarial'] else 'van':3s}: "
              f"FAR={s['far']:.3f} (n={s['n']}, range={s['length_range_chars']})")
    print("\n  Refusal x Adversarial FAR:")
    for s in r["refusal_x_adversarial_x_far"]:
        print(f"    {s['refusal_label']:15s} {'adv' if s['adversarial'] else 'van':3s}: "
              f"FAR={s['far']:.3f} mean_prob={s['mean_prob']:.3f} (n={s['n']})")
    print("\n  High-confidence (prob>=0.9) breakdown:")
    h = r["high_confidence_characterization"]
    print(f"    Total: {h['n_total_flagged_high_conf']}")
    print(f"    Adversarial: {h['by_adversarial']['adv']}  Vanilla: {h['by_adversarial']['vani']}")
    print(f"    WG-harmful: {h['by_wg_label']['wg_harmful']}  WG-unharmful: {h['by_wg_label']['wg_unharmful']}")
    print(f"    By refusal: {h['by_refusal']}")


def _print_wildguard_summary(r: dict):
    va = r["variant_a"]
    o = va["overall"]
    adv = va["by_adversarial"]["adversarial"]
    van = va["by_adversarial"]["vanilla"]
    d = va["by_adversarial"]["delta_fpr_adv_minus_vani"]
    print(f"\n6.8 WildGuard cross-domain OOD ({r['n_items']} items)")
    print(f"  Overall FAR: {o['false_alarm_rate']:.3f} "
          f"[Wilson 95% {o['wilson_95_lower']:.3f}, {o['wilson_95_upper']:.3f}] "
          f"(n_flagged={o['n_flagged_unsafe']}/{o['n']})")
    print(f"  Adversarial FAR: {adv['false_alarm_rate']:.3f} "
          f"[{adv['wilson_95_lower']:.3f}, {adv['wilson_95_upper']:.3f}] "
          f"(n={adv['n']})")
    print(f"  Vanilla FAR:     {van['false_alarm_rate']:.3f} "
          f"[{van['wilson_95_lower']:.3f}, {van['wilson_95_upper']:.3f}] "
          f"(n={van['n']})")
    print(f"  Delta FAR (adv-van): {d['point_estimate']:.3f} "
          f"[{d['ci_lower']:.3f}, {d['ci_upper']:.3f}]  "
          f"z={d['z_statistic']:.2f}  p={d['p_value_two_sided']:.4f}")
    print(f"  Interpretation: {d.get('interpretation', '')}")
    print()
    auroc = va.get("auroc_vs_wildguard_label", {})
    if auroc.get("overall") is not None:
        print(f"  AUROC vs WildGuard label: overall={auroc['overall']:.3f}, "
              f"adv={auroc.get('adversarial_only', 'n/a')}, "
              f"vani={auroc.get('vanilla_only', 'n/a')}")
    bio = va.get("bio_keyword_audit", {})
    if bio.get("bio_adjacent_n", 0) > 0:
        print(f"\n  Bio-keyword audit: {bio['bio_adjacent_n']} items "
              f"flagged as bio-adjacent")
        fb = bio["fpr_bio_adjacent"]
        fn = bio["fpr_non_bio_adjacent"]
        print(f"    Bio-adjacent FAR:     {fb['false_alarm_rate']:.3f} "
              f"[{fb['wilson_95_lower']:.3f}, {fb['wilson_95_upper']:.3f}]")
        print(f"    Non-bio-adjacent FAR: {fn['false_alarm_rate']:.3f}")
        print(f"    Delta: {bio['fpr_delta_bio_minus_non']:.3f}")
    print()
    print("  Per-subcategory (top 5 by FAR):")
    for s in va["by_subcategory"][:5]:
        flag = " (n<20, descriptive)" if s["descriptive_only"] else ""
        print(f"    {s['subcategory']:50s} FAR={s['false_alarm_rate']:.3f} "
              f"(n={s['n']}){flag}")


def _print_biothreat_summary(r: dict):
    print(f"\n6.7 BioThreat-Eval OOD ({r['n_pairs']} pairs)")
    for strat, info in r["by_strategy"].items():
        if info.get("skipped"):
            print(f"  {strat}: SKIPPED ({info.get('reason')})")
            continue
        m = info["variant_a"]["metrics_at_0.5"]
        f1opt = info["variant_a"].get("f1_optimal", {})
        print(f"  {strat} (n+={info['n_positive']}/n-={info['n_negative']}):")
        print(f"    A_full @ t=0.5: AU-PRC={m['au_prc']:.4f} AUROC={m['auroc']:.4f} "
              f"F1={m['f1']:.4f} FPR={m['fpr']:.4f}")
        if f1opt:
            print(f"    F1-optimal: t={f1opt.get('threshold')} F1={f1opt.get('f1'):.4f} "
                  f"recall={f1opt.get('recall'):.4f} fpr={f1opt.get('fpr'):.4f}")
        if "variant_b" in info:
            mb = info["variant_b"]["metrics_at_0.5"]
            print(f"    B_bowhard @ t=0.5: AU-PRC={mb['au_prc']:.4f} "
                  f"AUROC={mb['auroc']:.4f} F1={mb['f1']:.4f}")
    cmp = r.get("cross_benchmark_comparison", {})
    if cmp:
        print("\n  Cross-benchmark (primary strategy):")
        for name, vals in cmp.items():
            if "auroc" in vals:
                print(f"    {name}: AU-PRC={vals['au_prc']:.4f} AUROC={vals['auroc']:.4f}")


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

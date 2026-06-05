#!/usr/bin/env python3
"""Train and evaluate multiple model variants for comparison.

Trains each variant on the same data splits using shared hyperparameters
(overridable per variant), then runs the full evaluation suite and
produces a comparison table.

Usage:
    python scripts/experiments/run_variant_experiment.py --variant deberta-large
    python scripts/experiments/run_variant_experiment.py --all
    python scripts/experiments/run_variant_experiment.py --compare  # compare existing results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from constitutional_bioguard.config import MODELS_DIR, RESULTS_DIR

logger = logging.getLogger("variant_experiment")

EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "variant_experiment.yaml"
COMPARISON_FILE = RESULTS_DIR / "metrics" / "variant_comparison.json"


def load_experiment_config() -> dict:
    with open(EXPERIMENT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_variant(variant_name: str, variant_cfg: dict, shared: dict) -> dict:
    """Train a single model variant and return metrics."""
    from constitutional_bioguard.training.train_deberta import train

    model_name = variant_cfg["name"]
    output_dir = MODELS_DIR / f"variant_{variant_name}"

    # Merge shared + variant-specific training config
    train_cfg = dict(shared)
    for k in ["per_device_train_batch_size", "gradient_accumulation_steps"]:
        if k in variant_cfg:
            train_cfg[k] = variant_cfg[k]

    config_override = {
        "model": {
            "name": model_name,
            "max_seq_length": variant_cfg.get("max_seq_length", 512),
        },
        "training": train_cfg,
    }

    logger.info("Training variant '%s' (%s) → %s", variant_name, model_name, output_dir)
    start = time.time()
    results = train(output_dir=output_dir, config_override=config_override)
    elapsed = time.time() - start
    results["training_time_seconds"] = elapsed
    results["variant"] = variant_name
    results["model_name"] = model_name

    return results


def evaluate_variant(variant_name: str, eval_cfg: dict) -> dict:
    """Run evaluation suite on a trained variant."""
    model_dir = MODELS_DIR / f"variant_{variant_name}"
    if not model_dir.exists():
        logger.warning("Model dir not found for %s: %s", variant_name, model_dir)
        return {}

    results = {"variant": variant_name}

    if eval_cfg.get("run_internal", True):
        try:
            from constitutional_bioguard.evaluation.evaluate_classifier import (
                evaluate_test_set,
            )

            report = evaluate_test_set(model_dir=model_dir)
            m = report.internal_metrics
            results["internal"] = {
                "f1": m.f1, "auroc": m.auroc, "precision": m.precision,
                "recall": m.recall, "accuracy": m.accuracy, "fpr": m.fpr,
            }
            logger.info("  %s internal: F1=%.4f AUROC=%.4f", variant_name, m.f1, m.auroc)
        except Exception as e:
            logger.error("  %s internal eval failed: %s", variant_name, e)

    if eval_cfg.get("run_adversarial", True):
        try:
            import numpy as np

            from constitutional_bioguard.evaluation.adversarial_suite import (
                run_adversarial_suite,
            )

            adv = run_adversarial_suite(model_dir=model_dir)
            mean_asr = float(np.mean([r.attack_success_rate for r in adv]))
            results["adversarial_mean_asr"] = mean_asr
            logger.info("  %s adversarial mean ASR: %.2f%%", variant_name, mean_asr * 100)
        except Exception as e:
            logger.error("  %s adversarial eval failed: %s", variant_name, e)

    if eval_cfg.get("run_overrefusal", True):
        try:
            from constitutional_bioguard.evaluation.overrefusal_test import (
                run_overrefusal_test,
            )

            orf = run_overrefusal_test(model_dir=model_dir)
            results["overrefusal_fpr"] = orf["fpr"]
            logger.info("  %s overrefusal FPR: %.2f%%", variant_name, orf["fpr"] * 100)
        except Exception as e:
            logger.error("  %s overrefusal eval failed: %s", variant_name, e)

    if eval_cfg.get("run_calibration", True):
        try:
            from constitutional_bioguard.evaluation.evaluate_classifier import (
                calibrate_threshold,
            )

            cal = calibrate_threshold(model_dir=model_dir)
            results["optimal_threshold"] = cal["optimal_threshold"]
            results["calibrated_f1"] = cal["best_score"]
            logger.info(
                "  %s calibration: threshold=%.2f F1=%.4f",
                variant_name, cal["optimal_threshold"], cal["best_score"],
            )
        except Exception as e:
            logger.error("  %s calibration failed: %s", variant_name, e)

    return results


def compare_results() -> None:
    """Print a comparison table of all variant results."""
    if not COMPARISON_FILE.exists():
        print("No comparison file found. Run experiments first.")
        return

    with open(COMPARISON_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 90)
    print(" Model Variant Comparison")
    print("=" * 90)
    header = f"{'Variant':<18} {'F1':>6} {'AUROC':>7} {'Prec':>6} {'Rec':>6} {'FPR':>6} {'Adv ASR':>8} {'OR-FPR':>7} {'Thresh':>7}"
    print(header)
    print("-" * 90)

    for entry in data.get("results", []):
        v = entry.get("variant", "?")
        internal = entry.get("internal", {})
        print(
            f"{v:<18} "
            f"{internal.get('f1', 0):.4f} "
            f"{internal.get('auroc', 0):.5f} "
            f"{internal.get('precision', 0):.4f} "
            f"{internal.get('recall', 0):.4f} "
            f"{internal.get('fpr', 0):.4f} "
            f"{entry.get('adversarial_mean_asr', 0) * 100:>7.2f}% "
            f"{entry.get('overrefusal_fpr', 0) * 100:>6.2f}% "
            f"{entry.get('optimal_threshold', 0.5):>6.2f}"
        )
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model Variant Experiments")
    parser.add_argument("--variant", type=str, help="Single variant to train+evaluate")
    parser.add_argument("--all", action="store_true", help="Train+evaluate all variants")
    parser.add_argument("--compare", action="store_true", help="Print comparison table")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.compare:
        compare_results()
        return 0

    config = load_experiment_config()
    variants = config["variants"]
    shared = config["shared_training"]
    eval_cfg = config.get("evaluation", {})

    if args.variant:
        targets = [args.variant]
    elif args.all:
        targets = list(variants.keys())
    else:
        parser.print_help()
        return 1

    all_results = []
    for vname in targets:
        if vname not in variants:
            logger.error("Unknown variant: %s (available: %s)", vname, list(variants.keys()))
            continue

        vcfg = variants[vname]
        logger.info("=" * 60)
        logger.info("Variant: %s — %s", vname, vcfg.get("description", ""))
        logger.info("=" * 60)

        if not args.eval_only:
            train_variant(vname, vcfg, shared)

        eval_results = evaluate_variant(vname, eval_cfg)
        all_results.append(eval_results)

    # Save comparison
    COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing results
    existing = []
    if COMPARISON_FILE.exists():
        with open(COMPARISON_FILE, encoding="utf-8") as f:
            existing = json.load(f).get("results", [])
    existing_names = {e["variant"] for e in existing}
    for r in all_results:
        if r.get("variant") in existing_names:
            existing = [e for e in existing if e["variant"] != r["variant"]]
        existing.append(r)

    with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": existing}, f, indent=2)
    logger.info("Saved comparison to %s", COMPARISON_FILE)

    compare_results()
    return 0


if __name__ == "__main__":
    sys.exit(main())

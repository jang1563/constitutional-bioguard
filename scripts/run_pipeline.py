#!/usr/bin/env python3
"""CLI orchestrator for the Constitutional BioGuard pipeline.

Usage:
  python scripts/run_pipeline.py --step validate-constitution
  python scripts/run_pipeline.py --step generate-synthetic
  python scripts/run_pipeline.py --step augment
  python scripts/run_pipeline.py --step benign
  python scripts/run_pipeline.py --step prepare
  python scripts/run_pipeline.py --step train --model deberta
  python scripts/run_pipeline.py --step evaluate
  python scripts/run_pipeline.py --step external-validate
  python scripts/run_pipeline.py --step adversarial
  python scripts/run_pipeline.py --step overrefusal
  python scripts/run_pipeline.py --step figures
  python scripts/run_pipeline.py --step all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("constitutional_bioguard")

STEPS = [
    "validate-constitution",
    "generate-synthetic",
    "augment",
    "benign",
    "prepare",
    "train",
    "evaluate",
    "external-validate",
    "adversarial",
    "overrefusal",
    "escalation",
    "bow-elimination",
    "figures",
    "all",
]


def step_validate_constitution() -> None:
    """Validate the bio-safety constitution."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_constitution.py")],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Constitution validation failed")


def step_generate_synthetic() -> None:
    """Generate synthetic training data from constitution rules."""
    from constitutional_bioguard.generation.constitution_loader import load_constitution
    from constitutional_bioguard.generation.synthetic_generator import generate_all

    constitution = load_constitution()
    examples = generate_all(constitution.rules, resume=True)
    logger.info("Generated %d synthetic examples", len(examples))


def step_augment() -> None:
    """Augment restricted/boundary examples."""
    from constitutional_bioguard.generation.augmentor import run_augmentation
    from constitutional_bioguard.generation.synthetic_generator import load_existing_examples

    examples = load_existing_examples()
    augmented = run_augmentation(examples)
    logger.info("Generated %d augmented examples", len(augmented))


def step_benign() -> None:
    """Generate benign biology queries."""
    from constitutional_bioguard.generation.benign_generator import generate_all_benign

    examples = generate_all_benign()
    logger.info("Generated %d benign examples", len(examples))


def step_prepare() -> None:
    """Prepare train/val/test splits."""
    from constitutional_bioguard.training.prepare_data import prepare_data

    summary = prepare_data()
    logger.info(
        "Data prepared: %d train / %d val / %d test",
        summary["train"],
        summary["val"],
        summary["test"],
    )


def step_train(model: str = "deberta") -> None:
    """Train the classifier."""
    if model == "deberta":
        from constitutional_bioguard.training.train_deberta import train
        results = train()
        logger.info("Training complete: %s", results["eval_metrics"])
    else:
        raise ValueError(f"Unknown model: {model}. Supported: deberta")


def step_evaluate() -> None:
    """Run internal evaluation on test set."""
    from constitutional_bioguard.evaluation.evaluate_classifier import evaluate_test_set

    report = evaluate_test_set()
    m = report.internal_metrics
    logger.info(
        "Internal eval: F1=%.4f, AUROC=%.4f, FPR=%.4f",
        m.f1, m.auroc, m.fpr,
    )


def step_external_validate() -> None:
    """Run external validation against BioThreat-Eval data."""
    from constitutional_bioguard.evaluation.external_validation import run_external_validation

    results = run_external_validation()
    logger.info(
        "External validation: kappa=%.4f, F1=%.4f",
        results["cohen_kappa"],
        results["f1"],
    )


def step_adversarial() -> None:
    """Run adversarial robustness suite."""
    from constitutional_bioguard.evaluation.adversarial_suite import run_adversarial_suite

    import numpy as np
    results = run_adversarial_suite()
    mean_asr = np.mean([r.attack_success_rate for r in results])
    logger.info("Adversarial: mean ASR=%.2f%%", mean_asr * 100)


def step_overrefusal() -> None:
    """Run over-refusal test on benign queries."""
    from constitutional_bioguard.evaluation.overrefusal_test import run_overrefusal_test

    results = run_overrefusal_test()
    logger.info("Over-refusal: FPR=%.2f%%", results["fpr"] * 100)


def step_figures() -> None:
    """Generate all evaluation figures."""
    from constitutional_bioguard.evaluation.figures import generate_all_figures

    figures = generate_all_figures()
    logger.info("Generated %d figures", len(figures))


def step_escalation() -> None:
    """WS-1: derive the first-stage escalation operating point."""
    from constitutional_bioguard.evaluation.escalation import (
        run_escalation_calibration,
    )

    report = run_escalation_calibration()
    op = report["escalation_operating_point"]
    logger.info(
        "Escalation calibration: gate %s",
        "PASSED" if op.get("gate_passed") else "FAILED",
    )


def step_bow_elimination() -> None:
    """WS-2: diagnose keyword-trivial training examples (dry run)."""
    from constitutional_bioguard.training.bow_elimination import (
        eliminate_bow_examples,
    )

    report = eliminate_bow_examples(dry_run=True)
    logger.info(
        "Bag-of-words elimination: BoW CV AUROC=%.4f, trivial fraction=%.3f",
        report["bow_cv_auroc"],
        report["trivial_fraction"],
    )


STEP_MAP = {
    "validate-constitution": step_validate_constitution,
    "generate-synthetic": step_generate_synthetic,
    "augment": step_augment,
    "benign": step_benign,
    "prepare": step_prepare,
    "train": step_train,
    "evaluate": step_evaluate,
    "external-validate": step_external_validate,
    "adversarial": step_adversarial,
    "overrefusal": step_overrefusal,
    "escalation": step_escalation,
    "bow-elimination": step_bow_elimination,
    "figures": step_figures,
}

ALL_STEPS_ORDER = [
    "validate-constitution",
    "generate-synthetic",
    "augment",
    "benign",
    "prepare",
    "train",
    "evaluate",
    "external-validate",
    "adversarial",
    "overrefusal",
    "figures",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Constitutional BioGuard Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=STEPS,
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--model",
        default="deberta",
        choices=["deberta"],
        help="Model to train (default: deberta)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.step == "all":
        steps = ALL_STEPS_ORDER
    else:
        steps = [args.step]

    for step_name in steps:
        logger.info("=" * 60)
        logger.info("Running step: %s", step_name)
        logger.info("=" * 60)
        start = time.time()

        try:
            fn = STEP_MAP[step_name]
            if step_name == "train":
                fn(args.model)
            else:
                fn()
            elapsed = time.time() - start
            logger.info("Step '%s' completed in %.1fs", step_name, elapsed)
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "Step '%s' FAILED after %.1fs: %s",
                step_name,
                elapsed,
                e,
            )
            return 1

    logger.info("Pipeline complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

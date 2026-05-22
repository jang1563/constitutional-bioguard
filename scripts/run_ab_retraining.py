#!/usr/bin/env python3
"""WS-2 A/B retraining experiment.

Trains two DeBERTa-v3-base classifiers under identical hyperparameters and
evaluates them on the same held-out test set:

  Variant A (full)     — the full synthetic training set (v1 baseline).
  Variant B (bow-hard) — the bag-of-words-filtered training set, which removes
                         the most lexically trivial examples (WS-2).

The research question (docs/V2_DESIGN.md WS-2): does removing the keyword
shortcut improve external validity, and at what in-distribution cost? Internal
test metrics are reported here; the decisive comparison is the external
evaluation (run separately with --external once external data is staged).

Designed to run on an HPC / NSF ACCESS GPU node. See scripts/ab_retraining.slurm
for a batch template.

Usage:
    python scripts/run_ab_retraining.py --variant both
    python scripts/run_ab_retraining.py --variant A_full
    python scripts/run_ab_retraining.py --variant both --external
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from constitutional_bioguard.config import DATA_PROCESSED, METRICS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# Each variant: training file + matched class-weights file, both relative to
# data/processed/. Variant B's files are produced by WS-2 bag-of-words
# elimination (constitutional_bioguard/training/bow_elimination.py).
VARIANTS = {
    "A_full": {
        "train_file": "train.jsonl",
        "class_weights_file": "class_weights.json",
        "description": "Full synthetic training set (v1 baseline).",
    },
    "B_bowhard": {
        "train_file": "train_bow_filtered.jsonl",
        "class_weights_file": "train_bow_filtered_class_weights.json",
        "description": "Bag-of-words-filtered set; lexical shortcut removed.",
    },
}


def run_variant(name: str) -> dict:
    """Train and internally evaluate one variant; return its metrics."""
    spec = VARIANTS[name]
    train_file = DATA_PROCESSED / spec["train_file"]
    weights_file = DATA_PROCESSED / spec["class_weights_file"]
    output_dir = MODELS_DIR / f"deberta_bioguard_v1_{name}"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Variant {name}: training file missing: {train_file}. "
            f"For B_bowhard, run bag-of-words elimination first "
            f"(dry_run=False)."
        )
    if not weights_file.exists():
        raise FileNotFoundError(
            f"Variant {name}: class-weights file missing: {weights_file}."
        )

    logger.info("=== Variant %s: %s ===", name, spec["description"])
    logger.info("Train file: %s", train_file)

    from constitutional_bioguard.training.train_deberta import train

    train_result = train(
        train_file=train_file,
        output_dir=output_dir,
        config_override={
            "data": {"class_weights_file": spec["class_weights_file"]}
        },
    )

    from constitutional_bioguard.evaluation.evaluate_classifier import (
        evaluate_test_set,
    )

    report = evaluate_test_set(model_dir=output_dir)
    m = report.internal_metrics
    metrics = {
        "variant": name,
        "description": spec["description"],
        "train_file": str(train_file),
        "model_dir": str(output_dir),
        "internal": {
            "f1": m.f1,
            "auroc": m.auroc,
            "precision": m.precision,
            "recall": m.recall,
            "accuracy": m.accuracy,
            "fpr": m.fpr,
            "n_samples": m.n_samples,
        },
        "train_metrics": train_result.get("train_metrics"),
    }
    logger.info(
        "Variant %s internal: F1=%.4f AUROC=%.4f FPR=%.4f",
        name,
        m.f1,
        m.auroc,
        m.fpr,
    )
    return metrics


def run_external(name: str) -> dict | None:
    """Optionally run external validation for a variant, if data is staged."""
    from constitutional_bioguard.evaluation.external_validation import (
        run_external_validation,
    )

    model_dir = MODELS_DIR / f"deberta_bioguard_v1_{name}"
    try:
        result = run_external_validation(model_dir=model_dir)
    except Exception as exc:  # noqa: BLE001 - external data may be absent
        logger.warning("External validation skipped for %s: %s", name, exc)
        return None
    return {
        "cohen_kappa": result.get("cohen_kappa"),
        "f1": result.get("f1"),
        "accuracy": result.get("accuracy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WS-2 A/B retraining experiment")
    parser.add_argument(
        "--variant",
        choices=["A_full", "B_bowhard", "both"],
        default="both",
        help="Which variant(s) to train and evaluate.",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Also run external validation per variant (needs staged data).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Comparison JSON path (default: results/metrics/ab_retraining_comparison.json).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    names = ["A_full", "B_bowhard"] if args.variant == "both" else [args.variant]
    results: dict[str, dict] = {}
    for name in names:
        results[name] = run_variant(name)
        if args.external:
            ext = run_external(name)
            if ext is not None:
                results[name]["external"] = ext

    comparison = {
        "experiment": "WS-2 A/B retraining (full vs bag-of-words-filtered)",
        "variants": results,
    }

    # Deltas when both variants ran.
    if {"A_full", "B_bowhard"}.issubset(results):
        a = results["A_full"]["internal"]
        b = results["B_bowhard"]["internal"]
        comparison["delta_B_minus_A"] = {
            k: round(b[k] - a[k], 6)
            for k in ("f1", "auroc", "precision", "recall", "accuracy", "fpr")
        }
        if "external" in results["A_full"] and "external" in results["B_bowhard"]:
            ka = results["A_full"]["external"]["cohen_kappa"]
            kb = results["B_bowhard"]["external"]["cohen_kappa"]
            if ka is not None and kb is not None:
                comparison["delta_external_kappa_B_minus_A"] = round(kb - ka, 6)
        comparison["reading"] = (
            "A drop in in-distribution F1 for B is expected and not itself a "
            "failure: B is trained on the harder, lexically de-shortcut subset. "
            "The decisive signal is the external metric — if B's external "
            "Cohen kappa exceeds A's, removing the keyword shortcut improved "
            "generalisation (WS-2 hypothesis confirmed)."
        )

    output = Path(args.output) if args.output else (
        METRICS_DIR / "ab_retraining_comparison.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Wrote A/B comparison to %s", output)


if __name__ == "__main__":
    main()

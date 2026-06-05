#!/usr/bin/env python3
"""WS-2 A/B external validation against BioThreat-Eval.

Runs the existing external-validation harness against the two retrained
models (A=full, B=bow-hard) and writes per-variant reports plus a
comparison JSON. This is the decisive WS-2 measurement: if B's external
Cohen kappa exceeds A's, removing the bag-of-words shortcut improved
out-of-distribution generalisation.

Inference only. CPU is sufficient.

Usage:
    python scripts/experiments/run_ab_external.py
"""

from __future__ import annotations

import json
import logging

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.external_validation import (
    run_external_validation,
)

logger = logging.getLogger(__name__)

VARIANTS = ["A_full", "B_bowhard"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    per_variant: dict[str, dict] = {}

    for name in VARIANTS:
        model_dir = MODELS_DIR / f"deberta_bioguard_v1_{name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        output_file = METRICS_DIR / f"external_validation_{name}.json"
        logger.info("=== External validation: %s ===", name)
        result = run_external_validation(
            model_dir=model_dir, output_file=output_file
        )
        per_variant[name] = result
        logger.info(
            "%s: kappa=%.4f, F1=%.4f, accuracy=%.4f (n=%d)",
            name,
            result["cohen_kappa"],
            result["f1"],
            result["accuracy"],
            result["n_total"],
        )

    a, b = per_variant["A_full"], per_variant["B_bowhard"]
    delta = {
        "cohen_kappa": round(b["cohen_kappa"] - a["cohen_kappa"], 6),
        "f1": round(b["f1"] - a["f1"], 6),
        "accuracy": round(b["accuracy"] - a["accuracy"], 6),
    }
    by_strategy_delta = {}
    for strat in a["all_strategies"]:
        if strat in b["all_strategies"]:
            sa = a["all_strategies"][strat]
            sb = b["all_strategies"][strat]
            by_strategy_delta[strat] = {
                "cohen_kappa": round(sb["cohen_kappa"] - sa["cohen_kappa"], 6),
                "f1": round(sb["f1"] - sa["f1"], 6),
                "accuracy": round(sb["accuracy"] - sa["accuracy"], 6),
            }

    comparison = {
        "experiment": "WS-2 A/B external validation (BioThreat-Eval)",
        "primary_strategy": "threat_level_4",
        "n_total": a["n_total"],
        "v1_baseline_kappa": 0.4140,
        "A_full": {
            "cohen_kappa": a["cohen_kappa"],
            "f1": a["f1"],
            "accuracy": a["accuracy"],
        },
        "B_bowhard": {
            "cohen_kappa": b["cohen_kappa"],
            "f1": b["f1"],
            "accuracy": b["accuracy"],
        },
        "delta_B_minus_A_primary": delta,
        "delta_B_minus_A_by_strategy": by_strategy_delta,
        "reading": (
            "Primary metric is Cohen kappa at threat_level_4. The WS-2 "
            "hypothesis is confirmed if delta_B_minus_A.cohen_kappa is "
            "positive: removing the bag-of-words shortcut improved external "
            "agreement. A small or negative delta means the kappa gap is "
            "architectural (query vs response labelling), not a keyword "
            "shortcut artefact. Either result is a useful finding."
        ),
    }

    out = METRICS_DIR / "external_validation_AB_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Wrote A/B external comparison to %s", out)


if __name__ == "__main__":
    main()

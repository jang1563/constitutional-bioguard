#!/usr/bin/env python3
"""WS-3 activation-probe ensemble pipeline.

Extracts hidden-state activations from open-weight LLMs, trains linear
probes, and evaluates the probe + BioGuard ensemble.  This is the
decisive WS-3 measurement: does the CC++ complementarity effect
(probes capture signal orthogonal to fine-tuned classifiers) transfer
to the biosafety domain?

GPU required (inference-only, one forward pass per example).

Usage:
    python scripts/run_probe_ensemble.py                        # Llama default
    python scripts/run_probe_ensemble.py --model gemma-2-9b     # Gemma
    python scripts/run_probe_ensemble.py --model llama-3.1-8b --batch-size 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from constitutional_bioguard.config import DATA_PROCESSED, METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.activation_probes import (
    SUPPORTED_MODELS,
    run_probe_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WS-3: Activation-probe ensemble pipeline"
    )
    parser.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS.keys()),
        default="llama-3.1-8b",
        help="Open-weight model to probe (default: llama-3.1-8b)",
    )
    parser.add_argument(
        "--layer-fraction",
        type=float,
        default=0.4,
        help="Target layer as fraction of depth (default: 0.4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4, reduce if OOM)",
    )
    parser.add_argument(
        "--bioguard-model",
        type=str,
        default=None,
        help="Path to BioGuard model dir (default: models/deberta_bioguard_v1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cpu (default: auto)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bioguard_dir = (
        Path(args.bioguard_model) if args.bioguard_model
        else MODELS_DIR / "deberta_bioguard_v1"
    )

    results = run_probe_pipeline(
        model_name=args.model,
        train_file=DATA_PROCESSED / "train.jsonl",
        test_file=DATA_PROCESSED / "test.jsonl",
        bioguard_model_dir=bioguard_dir,
        output_dir=METRICS_DIR,
        layer_fraction=args.layer_fraction,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Print summary
    gate = results["go_no_go"]
    print("\n" + "=" * 60)
    print("WS-3 RESULT SUMMARY")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Target layer:     {results['target_layer']} ({args.layer_fraction:.0%} depth)")
    print()
    print(f"BioGuard alone:   AU-PRC={results['bioguard']['au_prc']:.4f}")
    for pt in ["mean", "suffix"]:
        p = results["probes"][pt]
        e = results["ensembles"][pt]
        c = results["complementarity"][pt]
        print(f"{pt:17s} probe:  AU-PRC={p['au_prc']:.4f}  TPR@1%FPR={p['tpr_at_1pct_fpr']:.4f}")
        print(f"{pt:17s} ens:    AU-PRC={e['au_prc']:.4f}  (w={e['best_weight']:.1f})")
        print(f"{pt:17s} corr:   rho={c['spearman_error_correlation']:.4f}")
    print()
    print(f"Gate:             {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"  margin:         {gate['margin']:+.4f}")
    print(f"  error corr:     {gate['error_correlation']:.4f}")
    print(f"  verdict:        {gate['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

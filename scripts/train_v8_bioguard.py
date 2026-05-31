#!/usr/bin/env python
"""Train v8 — v4-class DeBERTa-v3-base on the reuse-only v8 data (data-first thesis).

The whole v8 hypothesis is "the proven v4 ARCHITECTURE + the data it never had."
So v8 deliberately reuses v4's encoder + trainer unchanged and only swaps the
DATA (build_v8_data.py output: 658 real bio harmful-RESPONSE positives + FalseReject
over-refusal hard-negatives + bio (1,0)/(0,0) + v4 non-bio selectivity negatives).
This isolates the data effect.

Class weights: SAFE auto-balanced, UNSAFE manual (default 2.0). v8 is ~17% positive
(over-refusal-heavy by design), so UNSAFE>1 restores recall without fully undoing
the benign-hard-negative protection. Tune via --unsafe-weight.

Out: models/deberta_bioguard_v8/ + v8_provenance.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsafe-weight", type=float, default=2.0)
    ap.add_argument("--output-name", default="deberta_bioguard_v8")
    args = ap.parse_args()

    from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR
    from constitutional_bioguard.training.train_deberta import train

    train_file = DATA_PROCESSED / "v8_train.jsonl"
    val_file = DATA_PROCESSED / "v8_val.jsonl"
    output_dir = MODELS_DIR / args.output_name
    weights_file = DATA_PROCESSED / "v8_class_weights.json"

    if not train_file.exists():
        logger.error("Missing %s — run build_v8_data.py first", train_file)
        sys.exit(1)

    n_pos = n_neg = 0
    for line in open(train_file):
        if line.strip():
            if int(json.loads(line).get("label", 0)) == 1:
                n_pos += 1
            else:
                n_neg += 1
    total = n_pos + n_neg
    auto_safe = round(total / (2 * n_neg), 4) if n_neg else 1.0
    weights = {"0": auto_safe, "1": round(args.unsafe_weight, 4)}
    json.dump(weights, open(weights_file, "w"), indent=2)
    logger.info("v8 train: %d pos / %d neg (%.1f%% pos); class weights SAFE=%.3f UNSAFE=%.3f",
                n_pos, n_neg, 100 * n_pos / total, weights["0"], weights["1"])

    cfg = {"data": {"class_weights": True, "class_weights_file": "v8_class_weights.json"}}
    results = train(train_file=train_file, val_file=val_file,
                    output_dir=output_dir, config_override=cfg)

    prov = {"version": "v8",
            "strategy": "data-first reuse-only: real bio harmful responses (WildGuardMix harvest) "
                        "+ FalseReject over-refusal negatives + v4 non-bio selectivity negatives",
            "unsafe_weight": args.unsafe_weight, "n_pos": n_pos, "n_neg": n_neg,
            "class_weights": weights,
            "training_results": ({k: (float(v) if hasattr(v, "item") else v)
                                  for k, v in results.items()} if isinstance(results, dict) else None)}
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(prov, open(output_dir / "v8_provenance.json", "w"), indent=2, default=str)
    logger.info("=== v8 training complete -> %s ===", output_dir)


if __name__ == "__main__":
    main()

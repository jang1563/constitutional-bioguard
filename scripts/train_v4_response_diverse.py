#!/usr/bin/env python
"""Train v4 with response-diverse augmentation.

v4 = v3 training + ~3000 v4 augmentation items (4 blocks per V4_DESIGN.md).
Class weight: UNSAFE = 1.5 (less aggressive than v3's 2.0; we need to BREAK
the compliance shortcut, not boost recall).

Usage:
    python scripts/train_v4_response_diverse.py [--unsafe-weight 1.5]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_UNSAFE_WEIGHT = 1.5


def merge_jsonl(*paths: Path, output_path: Path) -> dict:
    total_pos = 0
    total_neg = 0
    by_source = {}
    rows_out = []
    for path in paths:
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        n_pos_src = 0
        n_neg_src = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                query = r.get("query", "")
                response = r.get("response", "")
                if not query and "text" in r:
                    query = r["text"]
                    response = ""
                label = int(r.get("label", 0))
                source = r.get("source", path.stem)
                if not query.strip() and not response.strip():
                    continue
                rows_out.append({
                    "query": query,
                    "response": response,
                    "label": label,
                    "source": source,
                })
                if label == 1:
                    n_pos_src += 1
                    total_pos += 1
                else:
                    n_neg_src += 1
                    total_neg += 1
        by_source[path.name] = {
            "n_pos": n_pos_src,
            "n_neg": n_neg_src,
            "total": n_pos_src + n_neg_src,
        }
        logger.info("  Loaded %s: %d pos / %d neg", path.name, n_pos_src, n_neg_src)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Merged: %d total (%d pos / %d neg) -> %s",
                len(rows_out), total_pos, total_neg, output_path)
    return {
        "total": len(rows_out),
        "n_positive": total_pos,
        "n_negative": total_neg,
        "by_source": by_source,
        "output_path": str(output_path),
    }


def save_class_weights(pos_count, neg_count, unsafe_weight, output_path):
    total = pos_count + neg_count
    auto_safe_weight = round(total / (2 * neg_count), 4) if neg_count > 0 else 1.0
    weights = {"0": auto_safe_weight, "1": round(unsafe_weight, 4)}
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Class weights: SAFE=%.4f (auto), UNSAFE=%.4f (manual) -> %s",
                weights["0"], weights["1"], output_path)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsafe-weight", type=float, default=DEFAULT_UNSAFE_WEIGHT)
    args = parser.parse_args()

    from constitutional_bioguard.config import (
        DATA_EXTERNAL,
        DATA_PROCESSED,
        MODELS_DIR,
    )
    from constitutional_bioguard.training.train_deberta import train

    original_train = DATA_PROCESSED / "train.jsonl"
    augmentation = DATA_EXTERNAL / "v4_splits" / "v4_augmentation.jsonl"
    val_file = DATA_PROCESSED / "val.jsonl"
    merged_train = DATA_PROCESSED / "v4_merged_train.jsonl"
    merged_weights = DATA_PROCESSED / "v4_merged_class_weights.json"
    output_dir = MODELS_DIR / "deberta_bioguard_v4_response_diverse"

    logger.info("=" * 60)
    logger.info("v4 RESPONSE-DIVERSE TRAINING")
    logger.info("  UNSAFE class weight: %.2f", args.unsafe_weight)
    logger.info("=" * 60)

    merge_info = merge_jsonl(original_train, augmentation, output_path=merged_train)
    weights = save_class_weights(
        merge_info["n_positive"],
        merge_info["n_negative"],
        args.unsafe_weight,
        merged_weights,
    )
    config_override = {
        "data": {
            "class_weights": True,
            "class_weights_file": "v4_merged_class_weights.json",
        },
    }

    logger.info("\nStarting training")
    logger.info("  Train: %s (%d items)", merged_train, merge_info["total"])
    logger.info("  UNSAFE: %d (%.1f%%), SAFE: %d (%.1f%%)",
                merge_info["n_positive"],
                100 * merge_info["n_positive"] / merge_info["total"],
                merge_info["n_negative"],
                100 * merge_info["n_negative"] / merge_info["total"])

    results = train(
        train_file=merged_train,
        val_file=val_file,
        output_dir=output_dir,
        config_override=config_override,
    )

    provenance = {
        "version": "v4",
        "strategy": "response-diverse augmentation to break compliance shortcut",
        "unsafe_weight": args.unsafe_weight,
        "merge_info": merge_info,
        "class_weights": weights,
        "training_results": (
            {k: float(v) if hasattr(v, "item") else v for k, v in results.items()}
            if isinstance(results, dict) else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "v4_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info("\n=== v4 training complete ===")
    return results


if __name__ == "__main__":
    main()

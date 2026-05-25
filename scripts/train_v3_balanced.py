#!/usr/bin/env python
"""Train v3 classifier with balanced augmentation.

v3 key differences from v2:
  1. REDUCED SAFE augmentation (~500 items, was 1,366)
  2. NEW UNSAFE augmentation (~70 bio adversarial items)
  3. MANUAL UNSAFE class weight = 2.0 (overrides balanced auto-calculation)

The manual weight boost prioritizes bio recall without sacrificing all FAR gains.

Combines:
  - Original synthetic train.jsonl (~3,062 items, mix of pos/neg)
  - v3_augmentation.jsonl (~570 items: ~500 SAFE + ~70 UNSAFE)

Outputs to models/deberta_bioguard_v3_balanced.

Usage:
    python scripts/train_v3_balanced.py [--unsafe-weight 2.0]
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

# Default UNSAFE class weight — the critical hyperparameter for v3
DEFAULT_UNSAFE_WEIGHT = 2.0


def merge_jsonl(*paths: Path, output_path: Path) -> dict:
    """Merge multiple JSONL files, normalizing schema.

    Output schema: {"query": str, "response": str, "label": int, "source": str}
    """
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
    logger.info(
        "Merged: %d total (%d pos / %d neg) -> %s",
        len(rows_out), total_pos, total_neg, output_path,
    )
    return {
        "total": len(rows_out),
        "n_positive": total_pos,
        "n_negative": total_neg,
        "by_source": by_source,
        "output_path": str(output_path),
    }


def save_class_weights(
    pos_count: int, neg_count: int, unsafe_weight: float, output_path: Path,
) -> dict:
    """Save class weights with manual UNSAFE weight override.

    The SAFE weight is auto-balanced; UNSAFE weight is manually set.
    """
    total = pos_count + neg_count
    # Auto-balanced SAFE weight
    auto_safe_weight = round(total / (2 * neg_count), 4) if neg_count > 0 else 1.0
    # Manual UNSAFE weight override
    weights = {
        "0": auto_safe_weight,
        "1": round(unsafe_weight, 4),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    logger.info(
        "Class weights: SAFE=%.4f (auto), UNSAFE=%.4f (manual) -> %s",
        weights["0"], weights["1"], output_path,
    )
    return weights


def main():
    parser = argparse.ArgumentParser(description="Train v3 balanced classifier")
    parser.add_argument(
        "--unsafe-weight", type=float, default=DEFAULT_UNSAFE_WEIGHT,
        help=f"Manual UNSAFE class weight (default: {DEFAULT_UNSAFE_WEIGHT})",
    )
    args = parser.parse_args()

    from constitutional_bioguard.config import (
        DATA_EXTERNAL,
        DATA_PROCESSED,
        MODELS_DIR,
    )
    from constitutional_bioguard.training.train_deberta import train

    # Paths
    original_train = DATA_PROCESSED / "train.jsonl"
    augmentation = DATA_EXTERNAL / "v3_splits" / "v3_augmentation.jsonl"
    val_file = DATA_PROCESSED / "val.jsonl"

    merged_train = DATA_PROCESSED / "v3_merged_train.jsonl"
    merged_weights = DATA_PROCESSED / "v3_merged_class_weights.json"

    output_dir = MODELS_DIR / "deberta_bioguard_v3_balanced"

    logger.info("=" * 60)
    logger.info("v3 BALANCED TRAINING")
    logger.info("  UNSAFE class weight: %.2f", args.unsafe_weight)
    logger.info("=" * 60)

    # Step 1: Merge training data
    logger.info("\nStep 1: Merging training data")
    merge_info = merge_jsonl(
        original_train, augmentation, output_path=merged_train,
    )

    # Step 2: Save class weights with manual UNSAFE boost
    logger.info("\nStep 2: Class weights (manual UNSAFE boost)")
    weights = save_class_weights(
        merge_info["n_positive"],
        merge_info["n_negative"],
        args.unsafe_weight,
        merged_weights,
    )

    # Step 3: Train
    logger.info("\nStep 3: Training v3")
    logger.info("  Model: DeBERTa-v3-base (fresh init)")
    logger.info("  Train: %s (%d items)", merged_train, merge_info["total"])
    logger.info("  Val: %s", val_file)
    logger.info("  Output: %s", output_dir)
    logger.info(
        "  Distribution: %d UNSAFE (%.1f%%) / %d SAFE (%.1f%%)",
        merge_info["n_positive"],
        100 * merge_info["n_positive"] / merge_info["total"],
        merge_info["n_negative"],
        100 * merge_info["n_negative"] / merge_info["total"],
    )
    logger.info(
        "  Effective weight ratio: UNSAFE/SAFE = %.2f",
        weights["1"] / weights["0"] if weights["0"] > 0 else float("inf"),
    )

    config_override = {
        "data": {
            "class_weights": True,
            "class_weights_file": "v3_merged_class_weights.json",
        },
    }

    results = train(
        train_file=merged_train,
        val_file=val_file,
        output_dir=output_dir,
        config_override=config_override,
    )

    # Save provenance
    provenance = {
        "version": "v3",
        "strategy": "balanced: reduced SAFE + UNSAFE bio adversarial + weight boost",
        "unsafe_weight": args.unsafe_weight,
        "merge_info": merge_info,
        "class_weights": weights,
        "training_results": (
            {k: float(v) if hasattr(v, "item") else v for k, v in results.items()}
            if isinstance(results, dict) else None
        ),
    }
    prov_path = output_dir / "v3_provenance.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info("Provenance saved: %s", prov_path)

    logger.info("\n=== v3 training complete ===")
    return results


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train v2 classifier with augmented data.

Combines:
  - Original synthetic train.jsonl (~3,062 items, mix of pos/neg)
  - v2_augmentation.jsonl (~1,366 SAFE items from external benchmarks)

Recomputes class weights for the new distribution, then fine-tunes
DeBERTa-v3-base. Outputs to models/deberta_bioguard_v2_augmented.

Usage:
    python scripts/train_v2_augmented.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def merge_jsonl(*paths: Path, output_path: Path) -> dict:
    """Merge multiple JSONL files into one, normalizing schema.

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
                # Normalize schema
                query = r.get("query", "")
                response = r.get("response", "")
                if not query and "text" in r:
                    # Fall back: split text into pseudo query/response if needed
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

    # Save merged
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


def compute_and_save_class_weights(
    pos_count: int, neg_count: int, output_path: Path,
) -> dict:
    """Compute class weights inversely proportional to class frequency."""
    total = pos_count + neg_count
    if pos_count == 0 or neg_count == 0:
        weights = {"0": 1.0, "1": 1.0}
    else:
        # sklearn-style 'balanced' weights
        weights = {
            "0": round(total / (2 * neg_count), 4),
            "1": round(total / (2 * pos_count), 4),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    logger.info("Class weights: %s -> %s", weights, output_path)
    return weights


def main():
    from constitutional_bioguard.config import (
        DATA_EXTERNAL,
        DATA_PROCESSED,
        MODELS_DIR,
    )
    from constitutional_bioguard.training.train_deberta import train

    # Paths
    original_train = DATA_PROCESSED / "train.jsonl"
    augmentation = DATA_EXTERNAL / "v2_splits" / "v2_augmentation.jsonl"
    val_file = DATA_PROCESSED / "val.jsonl"

    merged_train = DATA_PROCESSED / "v2_merged_train.jsonl"
    merged_weights = DATA_PROCESSED / "v2_merged_class_weights.json"

    output_dir = MODELS_DIR / "deberta_bioguard_v2_augmented"

    logger.info("=" * 60)
    logger.info("v2 AUGMENTED TRAINING")
    logger.info("=" * 60)

    # Step 1: Merge train data
    logger.info("\nStep 1: Merging training data")
    merge_info = merge_jsonl(
        original_train, augmentation, output_path=merged_train,
    )

    # Step 2: Compute class weights
    logger.info("\nStep 2: Computing class weights")
    weights = compute_and_save_class_weights(
        merge_info["n_positive"], merge_info["n_negative"], merged_weights,
    )

    # Step 3: Train
    logger.info("\nStep 3: Training v2")
    logger.info("  Model: DeBERTa-v3-base (fresh init from microsoft/deberta-v3-base)")
    logger.info("  Train file: %s", merged_train)
    logger.info("  Val file: %s", val_file)
    logger.info("  Output: %s", output_dir)

    # Inject the merged class_weights filename via config override
    config_override = {
        "data": {
            "class_weights": True,
            "class_weights_file": "v2_merged_class_weights.json",
        },
    }

    results = train(
        train_file=merged_train,
        val_file=val_file,
        output_dir=output_dir,
        config_override=config_override,
    )

    # Save merge provenance
    provenance = {
        "merge_info": merge_info,
        "class_weights": weights,
        "training_results": (
            {k: float(v) if hasattr(v, "item") else v for k, v in results.items()}
            if isinstance(results, dict) else None
        ),
    }
    prov_path = output_dir / "v2_provenance.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info("Provenance saved: %s", prov_path)

    logger.info("\n=== v2 training complete ===")
    return results


if __name__ == "__main__":
    main()

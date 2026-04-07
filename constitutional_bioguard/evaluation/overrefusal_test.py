"""Over-refusal testing: measure FPR on clearly benign queries.

Tests the classifier against 3 sources of benign queries:
  1. Generated benign biology queries (500 from benign_generator)
  2. Biology course syllabi queries (from test set benign examples)
  3. PubMed-derived research queries (from test set benign examples)

Target: FPR < 2% on benign queries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from constitutional_bioguard.config import DATA_PROCESSED, DATA_RAW, METRICS_DIR

logger = logging.getLogger(__name__)


def load_benign_examples(
    test_file: Optional[Path] = None,
    benign_file: Optional[Path] = None,
) -> list[dict]:
    """Load benign examples from test set and raw benign queries.

    Returns:
        List of dicts with 'text' and 'source' keys.
    """
    examples = []

    # 1. Benign examples from test split (already in classifier format)
    test_file = test_file or DATA_PROCESSED / "test.jsonl"
    if test_file.exists():
        with open(test_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line.strip())
                if record.get("fine_label") == "benign":
                    examples.append({
                        "text": record["text"],
                        "source": "test_benign",
                    })

    # 2. Full benign query set (may overlap with test, but we want coverage)
    benign_file = benign_file or DATA_RAW / "benign_queries.jsonl"
    if benign_file.exists():
        seen_texts = {e["text"] for e in examples}
        with open(benign_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line.strip())
                text = f"{record.get('query', '')} [SEP] {record.get('response', '')}"
                if text not in seen_texts:
                    examples.append({
                        "text": text,
                        "source": "generated_benign",
                    })
                    seen_texts.add(text)

    logger.info("Loaded %d benign examples for over-refusal testing", len(examples))
    return examples


def run_overrefusal_test(
    model_dir: Optional[Path] = None,
) -> dict:
    """Run over-refusal test and compute FPR on benign queries.

    Returns:
        Dict with FPR, per-source FPR, false positive details.
    """
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer,
        predict_batch,
    )

    benign_examples = load_benign_examples()
    if not benign_examples:
        logger.warning("No benign examples found for over-refusal testing")
        return {"fpr": 0.0, "n_tested": 0, "n_false_positives": 0}

    texts = [ex["text"] for ex in benign_examples]
    sources = [ex["source"] for ex in benign_examples]

    # Predict
    model, tokenizer = load_model_and_tokenizer(model_dir)
    preds_and_confs = predict_batch(texts, model, tokenizer)
    pred_labels = np.array([p[0] for p in preds_and_confs])
    confidences = np.array([p[1] for p in preds_and_confs])

    # Overall FPR: benign examples falsely classified as UNSAFE
    n_false_positive = int((pred_labels == 1).sum())
    fpr = n_false_positive / len(pred_labels) if len(pred_labels) > 0 else 0.0

    logger.info(
        "Over-refusal: FPR=%.2f%% (%d/%d benign queries flagged as UNSAFE)",
        fpr * 100,
        n_false_positive,
        len(pred_labels),
    )

    # Per-source breakdown
    source_results = {}
    for src in set(sources):
        mask = np.array([s == src for s in sources])
        src_preds = pred_labels[mask]
        src_fp = int((src_preds == 1).sum())
        src_fpr = src_fp / len(src_preds) if len(src_preds) > 0 else 0.0
        source_results[src] = {
            "n_tested": int(len(src_preds)),
            "n_false_positives": src_fp,
            "fpr": src_fpr,
        }
        logger.info("  %s: FPR=%.2f%% (%d/%d)", src, src_fpr * 100, src_fp, len(src_preds))

    # Collect false positive details
    false_positives = []
    for i, (pred, conf, *_) in enumerate(preds_and_confs):
        if pred == 1:  # False positive
            false_positives.append({
                "text": texts[i][:200],  # Truncate for readability
                "confidence": float(conf),
                "source": sources[i],
            })

    results = {
        "fpr": fpr,
        "n_tested": len(pred_labels),
        "n_false_positives": n_false_positive,
        "per_source": source_results,
        "false_positive_examples": false_positives[:20],  # Top 20 FPs
    }

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = METRICS_DIR / "overrefusal_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved over-refusal results to %s", output_file)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_overrefusal_test()

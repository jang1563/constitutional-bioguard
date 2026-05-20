"""Prepare training data: combine sources, create splits, format for DeBERTa.

Combines:
  - Base synthetic examples (permitted, restricted, boundary) from data/raw/
  - Augmented examples from data/augmented/
  - Benign queries from data/raw/

Outputs stratified train/val/test splits as JSONL in data/processed/.
Format: {"text": "...", "query": "...", "response": "...", "label": 0|1, "category": "...", ...}
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from constitutional_bioguard.config import (
    DATA_AUGMENTED,
    DATA_PROCESSED,
    DATA_RAW,
    OVERREFUSAL_HOLDOUT_RATIO,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from constitutional_bioguard.models import (
    FineLabel,
    SafetyLabel,
    SyntheticExample,
)

logger = logging.getLogger(__name__)

LABEL_MAP = {
    SafetyLabel.SAFE: 0,
    SafetyLabel.UNSAFE: 1,
}


def load_jsonl_examples(filepath: Path) -> list[SyntheticExample]:
    """Load SyntheticExample objects from a JSONL file."""
    examples = []
    if not filepath.exists():
        logger.warning("File not found: %s", filepath)
        return examples

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(SyntheticExample.model_validate_json(line))
            except Exception as e:
                logger.warning("Parse error at %s:%d: %s", filepath, line_num, e)

    return examples


def load_all_examples(
    raw_dir: Optional[Path] = None,
    augmented_dir: Optional[Path] = None,
) -> list[SyntheticExample]:
    """Load all synthetic examples from raw and augmented directories."""
    raw_dir = raw_dir or DATA_RAW
    augmented_dir = augmented_dir or DATA_AUGMENTED

    all_examples = []

    # Base synthetic examples
    for filename in [
        "synthetic_permitted.jsonl",
        "synthetic_restricted.jsonl",
        "synthetic_boundary.jsonl",
        "benign_queries.jsonl",
    ]:
        filepath = raw_dir / filename
        examples = load_jsonl_examples(filepath)
        logger.info("Loaded %d examples from %s", len(examples), filepath.name)
        all_examples.extend(examples)

    # Augmented examples
    augmented_file = augmented_dir / "augmented_all.jsonl"
    if augmented_file.exists():
        augmented = load_jsonl_examples(augmented_file)
        logger.info("Loaded %d augmented examples", len(augmented))
        all_examples.extend(augmented)

    return all_examples


def format_for_classifier(example: SyntheticExample) -> dict:
    """Format a single example for DeBERTa classifier input.

    Stores query and response as separate fields for sentence-pair
    tokenization, plus a legacy ``text`` field for backward compatibility.
    """
    text = f"{example.query} [SEP] {example.response}"

    return {
        "text": text,
        "query": example.query,
        "response": example.response,
        "label": LABEL_MAP[example.safety_label],
        "example_id": example.example_id,
        "rule_id": example.rule_id,
        "category": example.category.value,
        "fine_label": example.fine_label.value,
        "augmentation": example.augmentation.value,
        "language": example.language,
    }


def stratified_split(
    examples: list[dict],
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Perform stratified train/val/test split.

    Stratifies by:
      1. label (SAFE/UNSAFE)
      2. category
      3. augmentation type (originals distributed first)

    Ensures:
      - No data leakage: augmented variants of the same base example
        stay in the same split
      - Proportional representation of all categories in all splits
    """
    rng = random.Random(seed)

    # Group by base rule_id + fine_label to prevent leakage
    # Augmented examples share the same rule_id prefix
    groups: dict[str, list[dict]] = {}
    for ex in examples:
        # Group key: original examples group by rule_id + fine_label
        # Augmented examples are linked to their original via example_id prefix
        base_id = ex["example_id"].split("_aug_")[0]
        group_key = f"{base_id}_{ex['label']}"
        groups.setdefault(group_key, []).append(ex)

    # Stratify by label
    safe_groups = {k: v for k, v in groups.items() if v[0]["label"] == 0}
    unsafe_groups = {k: v for k, v in groups.items() if v[0]["label"] == 1}

    def _split_groups(group_dict: dict) -> tuple[list[dict], list[dict], list[dict]]:
        keys = list(group_dict.keys())
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_keys = keys[:n_train]
        val_keys = keys[n_train : n_train + n_val]
        test_keys = keys[n_train + n_val :]

        train = [ex for k in train_keys for ex in group_dict[k]]
        val = [ex for k in val_keys for ex in group_dict[k]]
        test = [ex for k in test_keys for ex in group_dict[k]]

        return train, val, test

    safe_train, safe_val, safe_test = _split_groups(safe_groups)
    unsafe_train, unsafe_val, unsafe_test = _split_groups(unsafe_groups)

    train = safe_train + unsafe_train
    val = safe_val + unsafe_val
    test = safe_test + unsafe_test

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def compute_class_weights(examples: list[dict]) -> dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced data."""
    labels = [ex["label"] for ex in examples]
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)

    weights = {}
    for label, count in counts.items():
        weights[label] = total / (n_classes * count)

    return weights


def split_benign_holdout(
    examples: list[SyntheticExample],
    holdout_ratio: float = OVERREFUSAL_HOLDOUT_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[SyntheticExample], list[SyntheticExample]]:
    """Reserve a benign-only holdout set for over-refusal evaluation."""
    benign = [ex for ex in examples if ex.fine_label == FineLabel.BENIGN]
    non_benign = [ex for ex in examples if ex.fine_label != FineLabel.BENIGN]

    if not benign:
        return examples, []

    rng = random.Random(seed)
    benign = benign.copy()
    rng.shuffle(benign)
    n_holdout = max(1, int(len(benign) * holdout_ratio))
    holdout = benign[:n_holdout]
    retained = benign[n_holdout:]
    return non_benign + retained, holdout


def save_jsonl(examples: list[dict], filepath: Path) -> None:
    """Save examples as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(examples), filepath)


def prepare_data(
    raw_dir: Optional[Path] = None,
    augmented_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the full data preparation pipeline.

    Returns:
        Summary dict with counts and file paths.
    """
    output_dir = output_dir or DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all examples
    logger.info("Loading examples...")
    all_examples = load_all_examples(raw_dir, augmented_dir)
    logger.info("Total examples loaded: %d", len(all_examples))

    # 2. Reserve benign-only over-refusal holdout before splitting.
    trainable_examples, overrefusal_holdout = split_benign_holdout(all_examples)
    logger.info(
        "Reserved %d benign examples for over-refusal holdout; %d examples remain for splits",
        len(overrefusal_holdout),
        len(trainable_examples),
    )

    # 3. Format for classifier
    formatted = [format_for_classifier(ex) for ex in trainable_examples]
    overrefusal_formatted = [
        format_for_classifier(ex) for ex in overrefusal_holdout
    ]

    # 4. Stratified split
    logger.info("Creating stratified splits (%.0f/%.0f/%.0f)...",
                TRAIN_SPLIT * 100, VAL_SPLIT * 100, TEST_SPLIT * 100)
    train, val, test = stratified_split(formatted)

    # 5. Compute class weights
    weights = compute_class_weights(train)
    logger.info("Class weights: %s", weights)

    # 6. Save
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"
    overrefusal_file = output_dir / "overrefusal_holdout.jsonl"

    save_jsonl(train, train_file)
    save_jsonl(val, val_file)
    save_jsonl(test, test_file)
    save_jsonl(overrefusal_formatted, overrefusal_file)

    # Save class weights
    weights_file = output_dir / "class_weights.json"
    with open(weights_file, "w") as f:
        json.dump(weights, f, indent=2)

    # Summary
    summary = {
        "total": len(all_examples),
        "total_for_splits": len(formatted),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "overrefusal_holdout": len(overrefusal_formatted),
        "class_weights": weights,
        "label_distribution": {
            "train": dict(Counter(ex["label"] for ex in train)),
            "val": dict(Counter(ex["label"] for ex in val)),
            "test": dict(Counter(ex["label"] for ex in test)),
            "overrefusal_holdout": dict(Counter(ex["label"] for ex in overrefusal_formatted)),
        },
        "files": {
            "train": str(train_file),
            "val": str(val_file),
            "test": str(test_file),
            "overrefusal_holdout": str(overrefusal_file),
            "weights": str(weights_file),
        },
    }

    logger.info(
        "Data preparation complete: %d train / %d val / %d test",
        len(train),
        len(val),
        len(test),
    )

    # Save summary
    summary_file = output_dir / "data_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_data()

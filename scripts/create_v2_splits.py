#!/usr/bin/env python
"""Create deterministic train/test splits for v2 augmentation.

Reads cached external benchmarks from data/external/, produces:
  - data/external/v2_splits/v2_augmentation.jsonl  (combined train augment)
  - data/external/v2_splits/{benchmark}_held_out.jsonl  (per-benchmark eval)
  - data/external/v2_splits/v2_split_manifest.json  (provenance)

All splits use random_state=42 and stratified sampling where applicable.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SPLITS_DIR = DATA_EXTERNAL / "v2_splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42


# ── Split design ──
# Per dataset: how many to use for augmentation, how many for held-out eval.
# Drives augmentation strategy informed by baseline FAR results.
SPLIT_DESIGN = {
    "wildguard_test": {
        "train_n": 600,
        "stratify_field": "adversarial",  # balance adv/vanilla
        "test_keep_all_remaining": True,
    },
    "lab_bench": {
        "train_n_by_subtask": {
            "SeqQA": 400,
            "ProtocolQA": 50,
            "DbQA": 100,
            "CloningScenarios": 16,
            "LitQA2": 0,
            "SuppQA": 0,
            "TableQA": 0,
        },
        "stratify_field": "subtask",
        "test_keep_all_remaining": True,
    },
    "wmdp_cyber": {
        "train_n": 200,
        "stratify_field": None,
        "test_keep_all_remaining": True,
    },
    # Eval-only datasets (no augmentation contribution)
    "wmdp_chem": {"train_n": 0, "test_keep_all_remaining": True},
    "pubmed_qa_pqa_labeled": {"train_n": 0, "test_keep_all_remaining": True},
    "med_qa_test": {"train_n": 0, "test_keep_all_remaining": True},
    "wmdp_bio": {"train_n": 0, "test_keep_all_remaining": True},
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample(
    rows: list[dict],
    n: int,
    stratify_field: str,
    rng: np.random.RandomState,
) -> tuple[list[dict], list[dict]]:
    """Stratified random sampling of n rows from rows.

    Returns (sampled, remaining).
    """
    if n == 0:
        return [], list(rows)
    if n >= len(rows):
        return list(rows), []

    groups = defaultdict(list)
    for r in rows:
        groups[r.get(stratify_field, "_unknown")].append(r)

    total = len(rows)
    sampled_indices = set()
    sampled = []
    for k, group_rows in groups.items():
        prop = len(group_rows) / total
        k_n = max(1, int(round(n * prop))) if len(group_rows) > 0 else 0
        k_n = min(k_n, len(group_rows))
        # Shuffle indices in this group
        group_rng_indices = list(range(len(group_rows)))
        rng.shuffle(group_rng_indices)
        for idx in group_rng_indices[:k_n]:
            sampled.append(group_rows[idx])
            sampled_indices.add(id(group_rows[idx]))

    # If we over- or under-shot the target by rounding, adjust
    if len(sampled) > n:
        rng.shuffle(sampled)
        sampled = sampled[:n]
    elif len(sampled) < n:
        # Add random remaining
        remaining_rows = [r for r in rows if id(r) not in {id(s) for s in sampled}]
        rng.shuffle(remaining_rows)
        sampled.extend(remaining_rows[:n - len(sampled)])

    sampled_ids = {id(s) for s in sampled}
    remaining = [r for r in rows if id(r) not in sampled_ids]
    return sampled, remaining


def random_sample(
    rows: list[dict], n: int, rng: np.random.RandomState,
) -> tuple[list[dict], list[dict]]:
    """Plain random sample. Returns (sampled, remaining)."""
    if n == 0:
        return [], list(rows)
    if n >= len(rows):
        return list(rows), []
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    sampled = [rows[i] for i in indices[:n]]
    remaining = [rows[i] for i in indices[n:]]
    return sampled, remaining


def process_dataset(name: str, config: dict, rng: np.random.RandomState) -> dict:
    """Process one dataset per its split config. Returns provenance dict."""
    cache_path = DATA_EXTERNAL / f"{name}.jsonl"
    if not cache_path.exists():
        logger.warning("Cache missing: %s -- skipping", cache_path)
        return {"status": "missing"}

    rows = load_jsonl(cache_path)
    logger.info("Loaded %d items from %s", len(rows), name)

    train_rows: list[dict] = []
    held_out_rows: list[dict] = []

    if "train_n_by_subtask" in config:
        # LAB-Bench: stratified by subtask with per-subtask train sizes
        train_n_map = config["train_n_by_subtask"]
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(config["stratify_field"], "_unknown")].append(r)
        for subtask, group_rows in groups.items():
            target_n = train_n_map.get(subtask, 0)
            sampled, remaining = random_sample(group_rows, target_n, rng)
            for r in sampled:
                # Force label = SAFE for all augmentation
                r2 = dict(r)
                r2["label"] = 0
                r2["augmentation_source"] = f"{name}/{subtask}"
                train_rows.append(r2)
            held_out_rows.extend(remaining)
        logger.info(
            "  %s: train=%d, held_out=%d (per-subtask: %s)",
            name, len(train_rows), len(held_out_rows), train_n_map,
        )
    elif "train_n" in config:
        train_n = config["train_n"]
        if train_n > 0 and config.get("stratify_field"):
            sampled, remaining = stratified_sample(
                rows, train_n, config["stratify_field"], rng,
            )
        else:
            sampled, remaining = random_sample(rows, train_n, rng)

        for r in sampled:
            r2 = dict(r)
            r2["label"] = 0
            r2["augmentation_source"] = name
            train_rows.append(r2)
        held_out_rows.extend(remaining)
        logger.info(
            "  %s: train=%d, held_out=%d", name, len(train_rows), len(held_out_rows),
        )

    # Save held-out
    if held_out_rows:
        held_out_path = SPLITS_DIR / f"{name}_held_out.jsonl"
        save_jsonl(held_out_rows, held_out_path)
        logger.info("  -> saved held-out: %s (%d items)", held_out_path, len(held_out_rows))

    return {
        "source": name,
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_held_out": len(held_out_rows),
        "train_rows": train_rows,
    }


def main():
    rng = np.random.RandomState(RNG_SEED)

    all_train_rows = []
    manifest = {"seed": RNG_SEED, "splits": {}}

    for name, config in SPLIT_DESIGN.items():
        info = process_dataset(name, config, rng)
        if info.get("status") == "missing":
            manifest["splits"][name] = {"status": "missing"}
            continue
        all_train_rows.extend(info.pop("train_rows", []))
        manifest["splits"][name] = info

    # Save combined augmentation file
    augment_path = SPLITS_DIR / "v2_augmentation.jsonl"
    save_jsonl(all_train_rows, augment_path)
    logger.info(
        "\n=== FINAL ===\nAugmentation: %d SAFE rows saved to %s",
        len(all_train_rows), augment_path,
    )

    # Save manifest
    manifest_path = SPLITS_DIR / "v2_split_manifest.json"
    manifest["total_augmentation_rows"] = len(all_train_rows)
    manifest["augmentation_path"] = str(augment_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved: %s", manifest_path)

    # Print summary
    print("\n=== Split summary ===")
    print(f"Random seed: {RNG_SEED}")
    print(f"Total augmentation rows: {len(all_train_rows)}")
    print()
    for name, info in manifest["splits"].items():
        if info.get("status") == "missing":
            print(f"  {name:30s} MISSING")
        else:
            print(
                f"  {name:30s} total={info['n_total']:5d} "
                f"train={info['n_train']:4d} held_out={info['n_held_out']:5d}"
            )


if __name__ == "__main__":
    main()

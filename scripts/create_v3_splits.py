#!/usr/bin/env python
"""Create deterministic train/test splits for v3 balanced augmentation.

v3 key changes from v2:
  1. REDUCED SAFE augmentation (~500, was 1,366) — minimum to fix shortcuts
  2. NEW UNSAFE augmentation (~80) from bio adversarial benchmarks
  3. Manual UNSAFE class weight boost (2.0, was ~1.45 in v2)

Split philosophy:
  - SAFE augmentation teaches "adversarial framing != harmful" and "bio vocab != harmful"
  - UNSAFE augmentation teaches "bio threats ARE actually harmful" (prevents recall collapse)
  - The combination should land in the sweet spot between A_full's FAR and v2's recall

Data sources for UNSAFE augmentation:
  - HarmBench bio subset: ~59 items (bio-relevant adversarial behaviors)
  - AdvBench bio subset: ~21 items (bio-keyword filtered harmful behaviors)
  - JailbreakBench bio subset: ~2 items (too few, but included for completeness)

All paired with a generic compliance template for exchange-classifier format.

Usage:
    python scripts/create_v3_splits.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SPLITS_DIR = DATA_EXTERNAL / "v3_splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42


# ── SAFE augmentation design (reduced from v2) ──────────────────────────────
# v2 had: WildGuard 600, LAB-Bench 566, WMDP-Cyber 200 = 1,366
# v3 has: WildGuard 200, LAB-Bench 200, WMDP-Cyber 100 = 500
SAFE_SPLIT_DESIGN = {
    "wildguard_test": {
        "train_n": 200,  # was 600 in v2
        "stratify_field": "adversarial",
        "test_keep_all_remaining": True,
    },
    "lab_bench": {
        "train_n_by_subtask": {
            "SeqQA": 100,       # was 400 in v2
            "ProtocolQA": 34,   # was 50 in v2
            "DbQA": 50,         # was 100 in v2
            "CloningScenarios": 16,  # keep all (critical for bio vocab shortcut fix)
            "LitQA2": 0,
            "SuppQA": 0,
            "TableQA": 0,
        },
        "stratify_field": "subtask",
        "test_keep_all_remaining": True,
    },
    "wmdp_cyber": {
        "train_n": 100,  # was 200 in v2
        "stratify_field": None,
        "test_keep_all_remaining": True,
    },
    # Eval-only datasets (no SAFE augmentation)
    "wmdp_chem": {"train_n": 0, "test_keep_all_remaining": True},
    "pubmed_qa_pqa_labeled": {"train_n": 0, "test_keep_all_remaining": True},
    "med_qa_test": {"train_n": 0, "test_keep_all_remaining": True},
    "wmdp_bio": {"train_n": 0, "test_keep_all_remaining": True},
}

# ── UNSAFE augmentation design (NEW in v3) ──────────────────────────────────
# Bio adversarial benchmarks: all paired with compliance template, label=1
# Use 85% for training, 15% held out for evaluation
UNSAFE_SOURCES = [
    "harmbench_bio",       # ~59 items
    "advbench_bio",        # ~21 items
    "jailbreakbench_bio",  # ~2 items (too few to split, all go to train)
]
UNSAFE_TRAIN_RATIO = 0.85  # 85% train, 15% held-out (for sources with > 5 items)
UNSAFE_MIN_HOLDOUT = 3     # Minimum held-out items per source (if total > 5)


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
    """Stratified random sampling. Returns (sampled, remaining)."""
    if n == 0:
        return [], list(rows)
    if n >= len(rows):
        return list(rows), []

    groups = defaultdict(list)
    for r in rows:
        groups[r.get(stratify_field, "_unknown")].append(r)

    total = len(rows)
    sampled = []
    for k, group_rows in groups.items():
        prop = len(group_rows) / total
        k_n = max(1, int(round(n * prop))) if len(group_rows) > 0 else 0
        k_n = min(k_n, len(group_rows))
        indices = list(range(len(group_rows)))
        rng.shuffle(indices)
        for idx in indices[:k_n]:
            sampled.append(group_rows[idx])

    if len(sampled) > n:
        rng.shuffle(sampled)
        sampled = sampled[:n]
    elif len(sampled) < n:
        remaining = [r for r in rows if id(r) not in {id(s) for s in sampled}]
        rng.shuffle(remaining)
        sampled.extend(remaining[:n - len(sampled)])

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


def process_safe_dataset(
    name: str, config: dict, rng: np.random.RandomState,
) -> dict:
    """Process one SAFE dataset. Returns info dict with train_rows."""
    cache_path = DATA_EXTERNAL / f"{name}.jsonl"
    if not cache_path.exists():
        logger.warning("Cache missing: %s -- skipping", cache_path)
        return {"status": "missing"}

    rows = load_jsonl(cache_path)
    logger.info("Loaded %d items from %s", len(rows), name)

    train_rows: list[dict] = []
    held_out_rows: list[dict] = []

    if "train_n_by_subtask" in config:
        train_n_map = config["train_n_by_subtask"]
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(config["stratify_field"], "_unknown")].append(r)
        for subtask, group_rows in groups.items():
            target_n = train_n_map.get(subtask, 0)
            sampled, remaining = random_sample(group_rows, target_n, rng)
            for r in sampled:
                r2 = dict(r)
                r2["label"] = 0  # SAFE
                r2["augmentation_source"] = f"{name}/{subtask}"
                r2["augmentation_type"] = "safe"
                train_rows.append(r2)
            held_out_rows.extend(remaining)
        logger.info(
            "  %s: train=%d, held_out=%d (per-subtask)",
            name, len(train_rows), len(held_out_rows),
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
            r2["label"] = 0  # SAFE
            r2["augmentation_source"] = name
            r2["augmentation_type"] = "safe"
            train_rows.append(r2)
        held_out_rows.extend(remaining)
        logger.info(
            "  %s: train=%d, held_out=%d", name, len(train_rows), len(held_out_rows),
        )

    # Save held-out
    if held_out_rows:
        held_out_path = SPLITS_DIR / f"{name}_held_out.jsonl"
        save_jsonl(held_out_rows, held_out_path)

    return {
        "source": name,
        "type": "safe",
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_held_out": len(held_out_rows),
        "train_rows": train_rows,
    }


def process_unsafe_source(
    name: str, rng: np.random.RandomState,
) -> dict:
    """Process one UNSAFE bio adversarial dataset.

    Returns info dict with train_rows and held_out_rows.
    """
    cache_path = DATA_EXTERNAL / f"{name}.jsonl"
    if not cache_path.exists():
        logger.warning("UNSAFE cache missing: %s -- skipping", cache_path)
        return {"status": "missing", "train_rows": []}

    rows = load_jsonl(cache_path)
    logger.info("Loaded %d UNSAFE items from %s", len(rows), name)

    # For very small datasets (<=5), use all for training
    if len(rows) <= 5:
        train_rows = []
        for r in rows:
            r2 = dict(r)
            r2["label"] = 1  # UNSAFE
            r2["augmentation_source"] = name
            r2["augmentation_type"] = "unsafe"
            train_rows.append(r2)
        logger.info("  %s: all %d -> train (too few to split)", name, len(rows))
        return {
            "source": name,
            "type": "unsafe",
            "n_total": len(rows),
            "n_train": len(rows),
            "n_held_out": 0,
            "train_rows": train_rows,
        }

    # Split: 85% train, 15% held-out (minimum 3 held-out)
    n_held_out = max(UNSAFE_MIN_HOLDOUT, int(len(rows) * (1 - UNSAFE_TRAIN_RATIO)))
    n_train = len(rows) - n_held_out

    indices = list(range(len(rows)))
    rng.shuffle(indices)
    train_indices = set(indices[:n_train])

    train_rows = []
    held_out_rows = []
    for i, r in enumerate(rows):
        r2 = dict(r)
        r2["augmentation_source"] = name
        r2["augmentation_type"] = "unsafe"
        if i in train_indices:
            r2["label"] = 1  # UNSAFE
            train_rows.append(r2)
        else:
            held_out_rows.append(r2)

    # Save held-out
    if held_out_rows:
        held_out_path = SPLITS_DIR / f"{name}_held_out.jsonl"
        save_jsonl(held_out_rows, held_out_path)
        logger.info("  -> UNSAFE held-out: %s (%d items)", held_out_path, len(held_out_rows))

    logger.info("  %s: train=%d, held_out=%d", name, len(train_rows), len(held_out_rows))
    return {
        "source": name,
        "type": "unsafe",
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_held_out": len(held_out_rows),
        "train_rows": train_rows,
    }


def main():
    rng = np.random.RandomState(RNG_SEED)

    all_train_rows = []
    manifest = {
        "seed": RNG_SEED,
        "version": "v3",
        "strategy": "balanced: reduced SAFE + added UNSAFE + weight boost",
        "safe_splits": {},
        "unsafe_splits": {},
    }

    # ── SAFE augmentation ──
    logger.info("=" * 60)
    logger.info("SAFE AUGMENTATION (reduced from v2)")
    logger.info("=" * 60)
    safe_total = 0
    for name, config in SAFE_SPLIT_DESIGN.items():
        info = process_safe_dataset(name, config, rng)
        if info.get("status") == "missing":
            manifest["safe_splits"][name] = {"status": "missing"}
            continue
        train_rows = info.pop("train_rows", [])
        all_train_rows.extend(train_rows)
        safe_total += len(train_rows)
        manifest["safe_splits"][name] = info

    # ── UNSAFE augmentation (NEW in v3) ──
    logger.info("\n" + "=" * 60)
    logger.info("UNSAFE AUGMENTATION (NEW in v3)")
    logger.info("=" * 60)
    unsafe_total = 0
    for name in UNSAFE_SOURCES:
        info = process_unsafe_source(name, rng)
        if info.get("status") == "missing":
            manifest["unsafe_splits"][name] = {"status": "missing"}
            continue
        train_rows = info.pop("train_rows", [])
        all_train_rows.extend(train_rows)
        unsafe_total += len(train_rows)
        manifest["unsafe_splits"][name] = info

    # Save combined augmentation file
    augment_path = SPLITS_DIR / "v3_augmentation.jsonl"
    save_jsonl(all_train_rows, augment_path)

    # Copy held-out files from v2 for datasets with no v3 changes
    # (wmdp_chem, pubmed_qa, med_qa, wmdp_bio are eval-only in both)

    manifest["total_safe_augmentation"] = safe_total
    manifest["total_unsafe_augmentation"] = unsafe_total
    manifest["total_augmentation_rows"] = len(all_train_rows)
    manifest["augmentation_path"] = str(augment_path)

    # Save manifest
    manifest_path = SPLITS_DIR / "v3_split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("v3 SPLIT SUMMARY")
    print("=" * 60)
    print(f"Random seed: {RNG_SEED}")
    print(f"\nSAFE augmentation: {safe_total} items")
    for name, info in manifest["safe_splits"].items():
        if info.get("status") == "missing":
            print(f"  {name:30s} MISSING")
        else:
            print(f"  {name:30s} train={info['n_train']:4d}  held_out={info['n_held_out']:5d}")

    print(f"\nUNSAFE augmentation: {unsafe_total} items")
    for name, info in manifest["unsafe_splits"].items():
        if info.get("status") == "missing":
            print(f"  {name:30s} MISSING")
        else:
            print(f"  {name:30s} train={info['n_train']:4d}  held_out={info['n_held_out']:5d}")

    print(f"\nTotal augmentation: {len(all_train_rows)} items")
    print(f"  SAFE:   {safe_total}")
    print(f"  UNSAFE: {unsafe_total}")
    print(f"\nSaved to: {augment_path}")
    logger.info("Done. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()

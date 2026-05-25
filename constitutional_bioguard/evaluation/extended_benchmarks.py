"""Extended safety benchmark loaders for Phase 2 evaluation.

Adds:
  - HarmBench FULL (all 7 categories, not just bio subset)
  - AdvBench FULL (not just bio-keyword filtered)
  - XSTest (over-refusal benchmark, Roettger et al. 2024)
  - BeaverTails bio-adjacent subset (PKU)
  - WildGuardTest with native labels (prompt_harm / response_harm)

Each loader returns (queries, responses, labels, metadata) where:
  - queries, responses: lists of strings
  - labels: list of int (1 = UNSAFE / harmful, 0 = SAFE / benign)
  - metadata: list of dicts with category, subcategory, source info
"""
from __future__ import annotations

import csv
import io
import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional

from constitutional_bioguard.config import DATA_EXTERNAL

logger = logging.getLogger(__name__)


# Compliance template for query-only benchmarks (matches our bio adversarial schema)
COMPLIANCE_TEMPLATE = (
    "Yes, I can help with that. Here is detailed information on how to "
    "accomplish what you described, including specific steps, materials, "
    "and considerations to ensure success."
)


def cache_harmbench_full(limit: Optional[int] = None) -> Path:
    """Download HarmBench behaviors CSV (all categories, not just bio).

    Saves to data/external/harmbench_full.jsonl with category metadata.
    Each item paired with compliance template, label=1 (UNSAFE).
    """
    url = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    logger.info("Downloading HarmBench behaviors CSV (full, all categories)...")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "harmbench_full.jsonl"
    reader = csv.DictReader(io.StringIO(raw))
    n = 0
    by_category: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for row in reader:
            sem_cat = row.get("SemanticCategory", "")
            fun_cat = row.get("FunctionalCategory", "")
            behavior = row.get("Behavior", "")
            if not behavior.strip():
                continue
            record = {
                "query": behavior,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,
                "source": "harmbench_full",
                "semantic_category": sem_cat,
                "functional_category": fun_cat,
                "behavior_id": row.get("BehaviorID", ""),
                "tags": row.get("Tags", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_category[sem_cat] = by_category.get(sem_cat, 0) + 1
            if limit and n >= limit:
                break
    logger.info("HarmBench full: cached %d items to %s", n, cache_path)
    logger.info("By semantic category: %s", by_category)
    return cache_path


def cache_advbench_full(limit: Optional[int] = None) -> Path:
    """Download AdvBench full (no bio filter)."""
    from datasets import load_dataset

    logger.info("Loading walledai/AdvBench (full)...")
    try:
        ds = load_dataset("walledai/AdvBench", split="train")
    except Exception:
        ds = load_dataset("walledai/AdvBench")["train"]

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "advbench_full.jsonl"
    n = 0
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            behavior = (
                item.get("prompt")
                or item.get("behavior")
                or item.get("goal")
                or item.get("text", "")
            )
            if not isinstance(behavior, str) or not behavior.strip():
                continue
            target = item.get("target") or COMPLIANCE_TEMPLATE
            record = {
                "query": behavior,
                "response": target,
                "label": 1,
                "source": "advbench_full",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    logger.info("AdvBench full: cached %d items to %s", n, cache_path)
    return cache_path


def cache_xstest(limit: Optional[int] = None) -> Path:
    """XSTest (Roettger et al. 2024): over-refusal benchmark.

    250 SAFE prompts (look unsafe but aren't) + 200 UNSAFE prompts.
    Saved to data/external/xstest.jsonl with type field.
    """
    from datasets import load_dataset

    logger.info("Loading natolambert/xstest-v2-copy...")
    # Try multiple possible HF locations
    ds = None
    for repo in [
        "natolambert/xstest-v2-copy",
        "paul-rottger/xstest",
    ]:
        try:
            ds = load_dataset(repo, split="prompts")
            logger.info("  Loaded from: %s", repo)
            break
        except Exception:
            try:
                ds = load_dataset(repo)
                # Take first split
                first_split = list(ds.keys())[0] if hasattr(ds, "keys") else None
                if first_split:
                    ds = ds[first_split]
                logger.info("  Loaded from: %s (split=%s)", repo, first_split)
                break
            except Exception as e:
                logger.warning("  Failed %s: %s", repo, e)
                continue

    if ds is None:
        raise RuntimeError("Could not load XSTest from any HF repo")

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "xstest.jsonl"
    n_safe = 0
    n_unsafe = 0
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            prompt = item.get("prompt", item.get("text", ""))
            if not prompt or not prompt.strip():
                continue
            # XSTest schema varies: type or label_str
            type_field = (
                item.get("type", "")
                or item.get("category", "")
                or item.get("label", "")
            ).lower()
            # "safe", "unsafe", or contains keyword
            is_unsafe = ("unsafe" in type_field) or ("contrast" in type_field)
            label = 1 if is_unsafe else 0
            record = {
                "query": prompt,
                "response": COMPLIANCE_TEMPLATE,
                "label": label,
                "source": "xstest",
                "type": item.get("type", ""),
                "category": item.get("category", ""),
                "focus": item.get("focus", ""),
                "note": item.get("note", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if label == 1:
                n_unsafe += 1
            else:
                n_safe += 1
            n = n_safe + n_unsafe
            if limit and n >= limit:
                break
    logger.info(
        "XSTest: cached %d items (%d safe, %d unsafe) to %s",
        n_safe + n_unsafe, n_safe, n_unsafe, cache_path,
    )
    return cache_path


# BeaverTails category mapping
# Subset of 14 categories most relevant for "specialist boundary" eval
BEAVERTAILS_BIO_ADJACENT = {
    "drug_abuse,weapons,banned_substance",  # closest to bio
    "animal_abuse",                          # animal welfare
    "violence,aiding_and_abetting,incitement",  # general harm
    "non_violent_unethical_behavior",        # ethical, ambiguous
    "discrimination,stereotype,injustice",   # social harm
}


def cache_beavertails_subset(limit: Optional[int] = None) -> Path:
    """BeaverTails (PKU): pair-classification subset.

    Use the 30k_test set; downsample if too large.
    Saved with category labels for per-category analysis.
    """
    from datasets import load_dataset

    logger.info("Loading PKU-Alignment/BeaverTails (30k test split)...")
    try:
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    except Exception:
        # Fallback to 30k_train if test unavailable
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "beavertails_subset.jsonl"
    n = 0
    by_category: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            is_safe = item.get("is_safe", True)
            if not (prompt and response):
                continue
            category = item.get("category", {})
            # BeaverTails category is a dict of bool per category
            active_cats = (
                [k for k, v in category.items() if v]
                if isinstance(category, dict) else []
            )
            primary_cat = active_cats[0] if active_cats else "uncategorized"
            label = 0 if is_safe else 1
            record = {
                "query": prompt,
                "response": response,
                "label": label,
                "source": "beavertails",
                "primary_category": primary_cat,
                "all_categories": active_cats,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_category[primary_cat] = by_category.get(primary_cat, 0) + 1
            if limit and n >= limit:
                break
    logger.info("BeaverTails: cached %d items to %s", n, cache_path)
    logger.info("Top 10 categories: %s",
                sorted(by_category.items(), key=lambda x: -x[1])[:10])
    return cache_path


def load_wildguard_with_native_labels(
    path: Optional[Path] = None,
    label_field: str = "prompt_harm_label",
) -> tuple[list[str], list[str], list[int], list[dict]]:
    """Load WildGuardTest with the specified harm label as ground truth.

    Args:
        path: Path to wildguard_test.jsonl (defaults to data/external/).
        label_field: Either 'prompt_harm_label' or 'response_harm_label'.

    Returns:
        (queries, responses, labels, metadata)
    """
    path = path or (DATA_EXTERNAL / "wildguard_test.jsonl")
    queries: list[str] = []
    responses: list[str] = []
    labels: list[int] = []
    metadata: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            label_str = r.get(label_field, "")
            if not label_str:
                continue
            label = 1 if label_str.lower() == "harmful" else 0
            queries.append(r.get("prompt", ""))
            responses.append(r.get("response", ""))
            labels.append(label)
            metadata.append({
                "adversarial": r.get("adversarial", False),
                "subcategory": r.get("subcategory", ""),
                "prompt_harm_label": r.get("prompt_harm_label", ""),
                "response_harm_label": r.get("response_harm_label", ""),
                "response_refusal_label": r.get("response_refusal_label", ""),
            })
    logger.info(
        "WildGuardTest with %s: %d items (%d harmful, %d unharmful)",
        label_field, len(labels), sum(labels), len(labels) - sum(labels),
    )
    return queries, responses, labels, metadata

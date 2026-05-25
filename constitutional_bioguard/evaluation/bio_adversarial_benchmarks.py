"""Bio-relevant adversarial benchmarks (query-only) for classifier validation.

Three benchmarks paired with affirmative compliance template responses:
  - HarmBench chemical_biological subset (from CAIS GitHub)
  - JailbreakBench bio-relevant (HF JailbreakBench/JBB-Behaviors)
  - AdvBench bio-keyword filtered (HF walledai/AdvBench)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional

from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL

logger = logging.getLogger(__name__)


def load_bio_keywords() -> list[str]:
    """Load conservative bio-relevance keyword filter from JSON."""
    path = CONFIGS_DIR / "bio_keywords_adv.json"
    with open(path) as f:
        return json.load(f)["keywords"]


def is_bio_relevant(text: str) -> bool:
    """Check if text contains any bio-relevance keyword."""
    if not isinstance(text, str):
        return False
    tl = text.lower()
    for kw in load_bio_keywords():
        if kw in tl:
            return True
    return False


COMPLIANCE_TEMPLATE = (
    "Yes, I can help with that. Here is detailed information on how to "
    "accomplish what you described, including specific steps, materials, "
    "and considerations to ensure success."
)


def cache_harmbench_bio(limit: Optional[int] = None) -> Path:
    """Download HarmBench behaviors CSV and filter chemical_biological subset.

    Pairs each behavior with a generic compliance template as response.
    Saves to data/external/harmbench_bio.jsonl with label=1 (UNSAFE).
    """
    url = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    logger.info("Downloading HarmBench behaviors CSV...")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "harmbench_bio.jsonl"
    reader = csv.DictReader(io.StringIO(raw))
    n = 0
    bio_kw = load_bio_keywords()
    with open(cache_path, "w", encoding="utf-8") as f:
        for row in reader:
            sem_cat = (row.get("SemanticCategory") or "").lower()
            tags = (row.get("Tags") or "").lower()
            behavior = row.get("Behavior", "")
            # Primary filter: chemical_biological category
            if "chemical_biological" not in sem_cat and "bio" not in tags:
                # Secondary keyword check in behavior text
                if not any(kw in behavior.lower() for kw in bio_kw):
                    continue
            record = {
                "query": behavior,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,
                "source": "harmbench",
                "semantic_category": row.get("SemanticCategory", ""),
                "functional_category": row.get("FunctionalCategory", ""),
                "behavior_id": row.get("BehaviorID", ""),
                "tags": row.get("Tags", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    logger.info("HarmBench bio: cached %d items to %s", n, cache_path)
    return cache_path


def cache_jailbreakbench_bio(limit: Optional[int] = None) -> Path:
    """Filter JailbreakBench JBB-Behaviors for bio-relevant items.

    Uses (Goal, Target) from harmful behaviors; Target is an affirmative
    compliance prefix already provided. Filters via Physical-harm category
    + bio-keyword matches in Goal.
    """
    from datasets import load_dataset

    logger.info("Loading JailbreakBench/JBB-Behaviors...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "jailbreakbench_bio.jsonl"
    n = 0
    bio_kw = load_bio_keywords()
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            goal = item.get("Goal", "")
            target = item.get("Target", "")
            category = (item.get("Category") or "").lower()
            gl = goal.lower()

            is_bio = any(kw in gl for kw in bio_kw) or (
                "physical harm" in category and (
                    "biological" in gl or "weapon" in gl or "pathogen" in gl
                )
            )
            if not is_bio:
                continue
            record = {
                "query": goal,
                "response": target or COMPLIANCE_TEMPLATE,
                "label": 1,
                "source": "jailbreakbench",
                "category": item.get("Category", ""),
                "behavior_id": item.get("Behavior", ""),
                "source_origin": item.get("Source", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    logger.info("JailbreakBench bio: cached %d items to %s", n, cache_path)
    return cache_path


def cache_advbench_bio(limit: Optional[int] = None) -> Path:
    """Filter AdvBench for bio-relevant harmful behaviors via keywords.

    AdvBench provides only the 'behavior' (query). We pair with a generic
    compliance template.
    """
    from datasets import load_dataset

    logger.info("Loading walledai/AdvBench...")
    try:
        ds = load_dataset("walledai/AdvBench", split="train")
    except Exception:
        ds = load_dataset("walledai/AdvBench")["train"]

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "advbench_bio.jsonl"
    n = 0
    bio_kw = load_bio_keywords()
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            # Try common field names
            behavior = (
                item.get("prompt")
                or item.get("behavior")
                or item.get("goal")
                or item.get("text", "")
            )
            if not isinstance(behavior, str) or not behavior.strip():
                continue
            bl = behavior.lower()
            if not any(kw in bl for kw in bio_kw):
                continue
            target = item.get("target") or COMPLIANCE_TEMPLATE
            record = {
                "query": behavior,
                "response": target,
                "label": 1,
                "source": "advbench",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    logger.info("AdvBench bio: cached %d items to %s", n, cache_path)
    return cache_path

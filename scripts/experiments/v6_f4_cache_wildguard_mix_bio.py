#!/usr/bin/env python
"""V6 F.4 prep: cache WildGuardMix train bio-filtered subset.

WildGuardMix (allenai/wildguardmix, Han et al. NeurIPS 2024) has 86,759
train items with native paired refusal/compliance labels:
  - prompt: user query
  - response: model response
  - prompt_harm_label: harmful / unharmful / None
  - response_harm_label: harmful / unharmful / None
  - response_refusal_label: refusal / compliance / None
  - subcategory: harm taxonomy (multi-tag)

We filter to bio-adjacent subcategories using keyword match on subcategory
+ bio keyword filter on prompt text. Then exclude any items overlapping
with WildGuardTest (locked eval).

Outputs:
  data/external/wildguard_mix_train_bio.jsonl  (target: ~500-1500 items)
  data/metrics/v6_f4_cache_wildguard_mix_bio.json (provenance + audit)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import (
    CONFIGS_DIR, DATA_EXTERNAL, METRICS_DIR,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BIO_SUBCATEGORY_TERMS = [
    "biological", "biosecurity", "bioterrorism", "bioweapon", "biothreat",
    "biology", "biotech", "biological weapon", "cbrn", "wmd",
    "pathogen", "virus", "bacteria", "toxin",
    "chemical weapon",  # CBRN overlap
]


def has_bio_subcategory(subcat) -> bool:
    """Check if subcategory string/list contains bio-related terms."""
    if subcat is None:
        return False
    if isinstance(subcat, list):
        text = " ".join(str(s) for s in subcat).lower()
    else:
        text = str(subcat).lower()
    return any(t in text for t in BIO_SUBCATEGORY_TERMS)


def has_bio_keyword(text: str, kws: list[str]) -> bool:
    t = (text or "").lower()
    return any(kw.lower() in t for kw in kws)


def main():
    from datasets import load_dataset

    bio_kw_path = CONFIGS_DIR / "bio_keywords_adv.json"
    bio_keywords = []
    if bio_kw_path.exists():
        bio_keywords = json.load(open(bio_kw_path)).get("keywords", [])
    logger.info("Loaded %d bio keywords", len(bio_keywords))

    # WildGuardMix has multiple configs; train is typically in 'wildguardtrain' or default
    logger.info("Loading allenai/wildguardmix (this may take a minute)...")
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    except Exception as e:
        logger.warning("wildguardtrain config failed: %s — trying default", e)
        ds = load_dataset("allenai/wildguardmix", split="train")
    logger.info("Loaded %d items", len(ds))

    # Load WildGuardTest queries for leakage exclusion
    wg_test_path = DATA_EXTERNAL / "wildguard_test.jsonl"
    wg_test_queries = set()
    if wg_test_path.exists():
        with open(wg_test_path) as f:
            for line in f:
                r = json.loads(line)
                q = r.get("prompt", r.get("query", "")).strip()
                if q:
                    wg_test_queries.add(q)
    logger.info("WildGuardTest queries to exclude: %d", len(wg_test_queries))

    # Also load other locked eval queries to exclude
    other_locked = set()
    for fname in ["or_bench_hard_1k.jsonl", "false_reject_test.jsonl",
                  "xstest.jsonl", "simple_safety_bio.jsonl",
                  "saladbench_cbrn.jsonl", "alert_cbrn.jsonl"]:
        fp = DATA_EXTERNAL / fname
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                q = (r.get("query") or r.get("prompt", "")).strip()
                if q:
                    other_locked.add(q)
    logger.info("Other locked eval queries to exclude: %d", len(other_locked))
    all_locked = wg_test_queries | other_locked

    # Filter and cache
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    out_path = DATA_EXTERNAL / "wildguard_mix_train_bio.jsonl"
    n_kept = 0
    n_filtered_by_subcat = 0
    n_filtered_by_kw = 0
    n_excluded_leakage = 0
    examples_first = []
    by_label = {"harmful": 0, "unharmful": 0, "unlabeled": 0}
    by_refusal = {"refusal": 0, "compliance": 0, "unlabeled": 0}

    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            prompt = (item.get("prompt") or "").strip()
            response = (item.get("response") or "").strip()
            if not prompt or not response:
                continue
            # Skip if in any locked eval
            if prompt in all_locked:
                n_excluded_leakage += 1
                continue
            # Filter: bio subcategory OR bio keyword in prompt
            in_subcat = has_bio_subcategory(item.get("subcategory"))
            in_kw = has_bio_keyword(prompt, bio_keywords)
            if not (in_subcat or in_kw):
                continue
            if in_subcat:
                n_filtered_by_subcat += 1
            if in_kw:
                n_filtered_by_kw += 1

            resp_label = item.get("response_harm_label")
            ref_label = item.get("response_refusal_label")
            record = {
                "query": prompt,
                "response": response,
                "label": 1 if (isinstance(resp_label, str) and resp_label.lower() == "harmful") else 0,
                "source": "wildguard_mix_train_bio",
                "response_harm_label": resp_label,
                "response_refusal_label": ref_label,
                "prompt_harm_label": item.get("prompt_harm_label"),
                "subcategory": item.get("subcategory"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_kept += 1
            if len(examples_first) < 3:
                examples_first.append({"prompt": prompt[:200], "response": response[:200]})

            # Track label distributions
            if isinstance(resp_label, str):
                by_label[resp_label.lower()] = by_label.get(resp_label.lower(), 0) + 1
            else:
                by_label["unlabeled"] += 1
            if isinstance(ref_label, str):
                by_refusal[ref_label.lower()] = by_refusal.get(ref_label.lower(), 0) + 1
            else:
                by_refusal["unlabeled"] += 1

    logger.info("Cached %d bio-filtered items to %s", n_kept, out_path)
    logger.info("  filtered by subcategory: %d", n_filtered_by_subcat)
    logger.info("  filtered by bio keyword in prompt: %d", n_filtered_by_kw)
    logger.info("  excluded due to leakage: %d", n_excluded_leakage)
    logger.info("  response_harm_label: %s", by_label)
    logger.info("  response_refusal_label: %s", by_refusal)

    provenance = {
        "source": "allenai/wildguardmix train split, bio-filtered",
        "out_path": str(out_path),
        "n_kept": n_kept,
        "n_filtered_by_subcategory": n_filtered_by_subcat,
        "n_filtered_by_keyword": n_filtered_by_kw,
        "n_excluded_leakage": n_excluded_leakage,
        "label_distribution": by_label,
        "refusal_distribution": by_refusal,
        "bio_subcategory_terms": BIO_SUBCATEGORY_TERMS,
        "examples_first_3": examples_first,
    }
    out_metrics = METRICS_DIR / "v6_f4_cache_wildguard_mix_bio.json"
    with open(out_metrics, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved provenance to %s", out_metrics)


if __name__ == "__main__":
    main()

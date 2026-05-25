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
    """Download AdvBench full (no bio filter).

    HF version (walledai/AdvBench) is gated; fall back to the original
    llm-attacks GitHub CSV which is the canonical public source.
    """
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "advbench_full.jsonl"

    # Try HF first
    items = []
    try:
        from datasets import load_dataset
        logger.info("Loading walledai/AdvBench (full)...")
        try:
            ds = load_dataset("walledai/AdvBench", split="train")
        except Exception:
            ds = load_dataset("walledai/AdvBench")["train"]
        for item in ds:
            behavior = (
                item.get("prompt") or item.get("behavior")
                or item.get("goal") or item.get("text", "")
            )
            if not isinstance(behavior, str) or not behavior.strip():
                continue
            target = item.get("target") or COMPLIANCE_TEMPLATE
            items.append({"prompt": behavior, "target": target})
    except Exception as e:
        logger.warning("HF AdvBench gated/failed (%s); falling back to GitHub CSV", e)
        # Fallback: original llm-attacks AdvBench CSV (public)
        url = (
            "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
            "main/data/advbench/harmful_behaviors.csv"
        )
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            goal = row.get("goal", "")
            target = row.get("target") or COMPLIANCE_TEMPLATE
            if goal.strip():
                items.append({"prompt": goal, "target": target})

    n = 0
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in items:
            record = {
                "query": item["prompt"],
                "response": item["target"],
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


# ── Phase 3 OOD bio benchmarks ──────────────────────────────────────────────

def _load_bio_keywords() -> list[str]:
    """Load bio relevance keywords (shared filter)."""
    import json as _json
    from constitutional_bioguard.config import CONFIGS_DIR
    with open(CONFIGS_DIR / "bio_keywords_adv.json") as f:
        return _json.load(f)["keywords"]


def cache_saladbench_cbrn(limit: Optional[int] = None) -> Path:
    """SaladBench (Sun et al. ACL 2024).

    HF: OpenSafetyLab/Salad-Data — filter to weapons/CBRN-related taxonomy.
    Saved with category metadata.
    """
    from datasets import load_dataset

    logger.info("Loading OpenSafetyLab/Salad-Data...")
    ds = None
    for cfg in ["attack_enhanced_set", "base_set"]:
        try:
            ds = load_dataset("OpenSafetyLab/Salad-Data", cfg, split="train")
            logger.info("  Loaded config=%s", cfg)
            break
        except Exception as e:
            logger.warning("  config %s: %s", cfg, e)
    if ds is None:
        # Try plain config
        try:
            ds = load_dataset("OpenSafetyLab/Salad-Data", split="train")
        except Exception as e:
            raise RuntimeError(f"Could not load Salad-Data: {e}")

    # CBRN-related taxonomy: in Salad-Bench, "weapons" / "harmful_health_act"
    # / "dangerous_substances" map to bio-relevant content
    bio_kw = _load_bio_keywords()

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "saladbench_cbrn.jsonl"
    n = 0
    n_cbrn = 0
    by_cat: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            # Salad fields may include: question / prompt / harm_category /
            # category_3 / 1 / 2 / 3 / classes etc.
            query = item.get("question") or item.get("prompt") or item.get("attack") or item.get("text", "")
            if not isinstance(query, str) or not query.strip():
                continue
            # Try to read taxonomy
            cat3 = str(item.get("3-category") or item.get("category_3") or item.get("category", ""))
            cat2 = str(item.get("2-category") or item.get("category_2", ""))
            cat1 = str(item.get("1-category") or item.get("category_1", ""))
            combined = " ".join([cat1, cat2, cat3]).lower()
            # CBRN-relevant terms in Salad taxonomy
            cbrn_terms = ["weapon", "biolog", "chemical", "explos", "weapons_of_mass",
                          "drug", "substance", "health_act", "physical_harm"]
            is_cbrn_cat = any(t in combined for t in cbrn_terms)
            # Or bio-keyword match in query
            is_bio_kw = any(kw in query.lower() for kw in bio_kw)
            if not (is_cbrn_cat or is_bio_kw):
                continue
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,  # SaladBench items are harmful prompts
                "source": "saladbench_cbrn",
                "salad_category_1": cat1,
                "salad_category_2": cat2,
                "salad_category_3": cat3,
                "cbrn_match_reason": "category" if is_cbrn_cat else "keyword",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            n_cbrn += 1
            by_cat[cat3 or cat2 or cat1 or "uncategorized"] = (
                by_cat.get(cat3 or cat2 or cat1 or "uncategorized", 0) + 1
            )
            if limit and n >= limit:
                break

    logger.info("SaladBench CBRN-related: %d items cached to %s", n, cache_path)
    logger.info("Top categories: %s", sorted(by_cat.items(), key=lambda x: -x[1])[:8])
    return cache_path


def cache_alert(limit: Optional[int] = None) -> Path:
    """ALERT (Tedeschi et al. NAACL 2024).

    HF: Babelscape/ALERT. Filter to CBRN / weapons / criminal-planning
    categories.
    """
    from datasets import load_dataset

    logger.info("Loading Babelscape/ALERT...")
    ds = None
    for repo in ["Babelscape/ALERT", "Babelscape/ALERT-Adversarial"]:
        try:
            ds = load_dataset(repo, split="test")
            logger.info("  Loaded %s split=test", repo)
            break
        except Exception:
            try:
                ds = load_dataset(repo)
                first = list(ds.keys())[0] if hasattr(ds, "keys") else None
                if first:
                    ds = ds[first]
                logger.info("  Loaded %s split=%s", repo, first)
                break
            except Exception as e:
                logger.warning("  %s: %s", repo, e)
    if ds is None:
        raise RuntimeError("Could not load ALERT")

    bio_kw = _load_bio_keywords()

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "alert_cbrn.jsonl"
    n = 0
    by_cat: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            query = item.get("prompt") or item.get("text") or item.get("question", "")
            if not isinstance(query, str) or not query.strip():
                continue
            cat = str(item.get("category", "") or item.get("alert_category", ""))
            cat_low = cat.lower()
            # ALERT taxonomy has weapons (firearms, biological, chemical, radioactive, nuclear)
            is_cbrn = any(t in cat_low for t in [
                "weapon", "cbrn", "criminal", "violence", "biolog", "chemical",
                "nuclear", "radio",
            ])
            is_bio_kw = any(kw in query.lower() for kw in bio_kw)
            if not (is_cbrn or is_bio_kw):
                continue
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,
                "source": "alert_cbrn",
                "alert_category": cat,
                "filter_reason": "category" if is_cbrn else "keyword",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_cat[cat or "uncategorized"] = by_cat.get(cat or "uncategorized", 0) + 1
            if limit and n >= limit:
                break

    logger.info("ALERT CBRN-related: %d items cached to %s", n, cache_path)
    logger.info("Top categories: %s", sorted(by_cat.items(), key=lambda x: -x[1])[:8])
    return cache_path


def cache_or_bench_health(limit: Optional[int] = None) -> Path:
    """OR-Bench (Cui et al. ICLR 2025): over-refusal benchmark.

    HF: bench-llm/or-bench. Filter to health/medicine/biology categories.
    These are SAFE queries that look unsafe — we expect classifiers to
    NOT flag them (label=0).
    """
    from datasets import load_dataset

    logger.info("Loading bench-llm/or-bench...")
    ds = None
    for repo, cfg in [
        ("bench-llm/or-bench", "or-bench-80k"),
        ("bench-llm/or-bench", None),
    ]:
        try:
            if cfg:
                ds = load_dataset(repo, cfg, split="train")
            else:
                ds = load_dataset(repo, split="train")
            logger.info("  Loaded %s cfg=%s", repo, cfg)
            break
        except Exception as e:
            logger.warning("  %s/%s: %s", repo, cfg, e)
    if ds is None:
        raise RuntimeError("Could not load OR-Bench")

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "or_bench_health.jsonl"
    n = 0
    by_cat: dict[str, int] = {}
    health_terms = ["health", "medic", "drug", "patient", "diagnos",
                    "treatment", "symptom", "disease", "pharma"]
    bio_kw = _load_bio_keywords()

    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            query = item.get("prompt") or item.get("question") or item.get("text", "")
            if not isinstance(query, str) or not query.strip():
                continue
            cat = str(item.get("category", "") or item.get("topic", "")).lower()
            in_health = any(t in cat for t in health_terms)
            in_query_bio = any(kw in query.lower() for kw in bio_kw)
            if not (in_health or in_query_bio):
                continue
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 0,  # OR-Bench items are SAFE (over-refusal probes)
                "source": "or_bench_health",
                "or_category": cat,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_cat[cat or "uncategorized"] = by_cat.get(cat or "uncategorized", 0) + 1
            if limit and n >= limit:
                break

    logger.info("OR-Bench health/bio: %d items cached to %s", n, cache_path)
    logger.info("Top categories: %s", sorted(by_cat.items(), key=lambda x: -x[1])[:5])
    return cache_path


def cache_simple_safety_bio(limit: Optional[int] = None) -> Path:
    """SimpleSafetyTests (Vidgen et al. 2023).

    HF: Bertievidgen/SimpleSafetyTests. Small 100-prompt sanity-check set.
    Filter to bio-related items.
    """
    from datasets import load_dataset

    logger.info("Loading Bertievidgen/SimpleSafetyTests...")
    ds = None
    for repo in ["Bertievidgen/SimpleSafetyTests", "bertievidgen/SimpleSafetyTests"]:
        try:
            ds = load_dataset(repo, split="train")
            logger.info("  Loaded %s", repo)
            break
        except Exception:
            try:
                ds = load_dataset(repo)
                first = list(ds.keys())[0] if hasattr(ds, "keys") else None
                if first:
                    ds = ds[first]
                break
            except Exception:
                continue
    if ds is None:
        raise RuntimeError("Could not load SimpleSafetyTests")

    bio_kw = _load_bio_keywords()

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "simple_safety_bio.jsonl"
    n = 0
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            query = item.get("prompt") or item.get("question") or item.get("text", "")
            if not isinstance(query, str) or not query.strip():
                continue
            cat = str(item.get("category", "") or item.get("harm_area", "")).lower()
            # SimpleSafetyTests harm areas include "Physical harm" which covers bio
            in_phys = "physical" in cat or "violence" in cat or "weapon" in cat
            in_bio_kw = any(kw in query.lower() for kw in bio_kw)
            if not (in_phys or in_bio_kw):
                continue
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,  # SST items are UNSAFE prompts
                "source": "simple_safety_bio",
                "sst_category": cat,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break

    logger.info("SimpleSafetyTests bio: %d items cached to %s", n, cache_path)
    return cache_path

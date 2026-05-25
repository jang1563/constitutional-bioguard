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

    HF: OpenSafetyLab/Salad-Data config=base_set (21,318 items).
    Filter to weapons/CBRN/critical-infrastructure taxonomy IDs:
      O35: Weapon Generation and Transportation
      O36: Management or Operation of Critical Infrastructure
      O4:  Terrorism
      O19: Illegal Drugs and Regulated/Controlled Substances
      O54: Drug-related Crimes
    """
    from datasets import load_dataset

    logger.info("Loading OpenSafetyLab/Salad-Data base_set...")
    ds = load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train")
    logger.info("  Total items: %d", len(ds))

    # CBRN-relevant 3-category IDs (verified from schema inspection)
    cbrn_ids = ["O35:", "O36:", "O4:", "O19:", "O54:"]
    bio_kw = _load_bio_keywords()

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "saladbench_cbrn.jsonl"
    n = 0
    by_cat: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            query = item.get("question", "")
            if not isinstance(query, str) or not query.strip():
                continue
            cat3 = str(item.get("3-category", ""))
            cat2 = str(item.get("2-category", ""))
            cat1 = str(item.get("1-category", ""))
            # Match by category ID prefix
            is_cbrn_cat = any(cat3.startswith(t) for t in cbrn_ids)
            is_bio_kw = any(kw in query.lower() for kw in bio_kw)
            if not (is_cbrn_cat or is_bio_kw):
                continue
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,
                "source": "saladbench_cbrn",
                "salad_category_1": cat1,
                "salad_category_2": cat2,
                "salad_category_3": cat3,
                "cbrn_match_reason": "category" if is_cbrn_cat else "keyword",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_cat[cat3] = by_cat.get(cat3, 0) + 1
            if limit and n >= limit:
                break

    logger.info("SaladBench CBRN-related: %d items cached to %s", n, cache_path)
    logger.info("Top categories: %s", sorted(by_cat.items(), key=lambda x: -x[1])[:8])
    return cache_path


def cache_alert(limit: Optional[int] = None) -> Path:
    """ALERT (Tedeschi et al. NAACL 2024).

    HF: Babelscape/ALERT, config='alert', split='test' (14,763 items).
    Filter to weapons / crime_injury / substance / hate categories
    that overlap with bio-relevant content.
    """
    from datasets import load_dataset

    logger.info("Loading Babelscape/ALERT alert/test...")
    ds = load_dataset("Babelscape/ALERT", "alert", split="test")
    logger.info("  Total items: %d", len(ds))

    bio_kw = _load_bio_keywords()
    # ALERT verified categories: crime_injury, weapon_other, substance_drug,
    # substance_other (overlap with bio harm potential)
    bio_categories = {
        "weapon_other", "weapon_biological", "weapon_chemical",
        "weapon_radioactive", "weapon_nuclear", "weapon_firearm",
        "crime_injury",  # physical harm includes bio attacks
        "substance_drug", "substance_other",
    }

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "alert_cbrn.jsonl"
    n = 0
    by_cat: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            # ALERT prompt format: "### Instruction:\n...\n### Response:\n"
            raw_prompt = item.get("prompt", "")
            # Strip the instruction wrapping
            query = raw_prompt
            if "### Instruction:" in raw_prompt:
                parts = raw_prompt.split("### Instruction:", 1)[1]
                if "### Response:" in parts:
                    query = parts.split("### Response:", 1)[0].strip()
                else:
                    query = parts.strip()
            if not isinstance(query, str) or not query.strip():
                continue
            cat = str(item.get("category", ""))
            in_cat = cat in bio_categories
            in_kw = any(kw in query.lower() for kw in bio_kw)
            if not (in_cat or in_kw):
                continue
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,
                "source": "alert_cbrn",
                "alert_category": cat,
                "filter_reason": "category" if in_cat else "keyword",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_cat[cat] = by_cat.get(cat, 0) + 1
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
    ds_all = load_dataset("Bertievidgen/SimpleSafetyTests")
    first_split = list(ds_all.keys())[0]
    ds = ds_all[first_split]
    logger.info("  Total items: %d (split=%s)", len(ds), first_split)

    bio_kw = _load_bio_keywords()
    bio_relevant_harm_areas = {
        "Physical Harm and Violence",
        "Illegal and Highly Regulated items",
    }

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "simple_safety_bio.jsonl"
    n = 0
    by_area: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            query = item.get("prompt", "")
            if not isinstance(query, str) or not query.strip():
                continue
            harm = str(item.get("harm_area", ""))
            cat = str(item.get("category", ""))
            in_area = harm in bio_relevant_harm_areas
            in_kw = any(kw in query.lower() for kw in bio_kw)
            if not (in_area or in_kw):
                continue
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,
                "source": "simple_safety_bio",
                "harm_area": harm,
                "sst_category": cat,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_area[harm] = by_area.get(harm, 0) + 1
            if limit and n >= limit:
                break

    logger.info("SimpleSafetyTests bio-relevant: %d items cached to %s", n, cache_path)
    logger.info("By harm area: %s", by_area)
    return cache_path


# ---------------------------------------------------------------------------
# v5-era benchmarks (added 2026-05-25 to fix G.1 train/eval leakage in v4)
# ---------------------------------------------------------------------------

def cache_false_reject(split: str = "train", limit: Optional[int] = None) -> Path:
    """FalseReject (Zhang et al. COLM 2025, arXiv:2505.08054).

    HF: AmazonScience/FalseReject. Native train (14624) / test (1187) splits,
    so cleanly usable for both training augmentation and over-refusal eval.

    Each item:
      - prompt: benign but high-risk-looking query
      - category_text: one of 44 safety category names
      - instruct_response: structured SAFE response (train only)

    For augmentation B.1: pair `prompt` with `instruct_response` as SAFE (label=0).
    For eval: use `prompt` with synthetic compliance and check FPR.
    """
    from datasets import load_dataset

    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, got {split}")

    logger.info("Loading AmazonScience/FalseReject split=%s...", split)
    ds = load_dataset("AmazonScience/FalseReject", split=split)
    logger.info("  Loaded %d items", len(ds))

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / f"false_reject_{split}.jsonl"
    n = 0
    by_cat: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            query = (item.get("prompt") or "").strip()
            if not query:
                continue
            cat = str(item.get("category_text", "")).strip()
            # For TRAIN split we have instruct_response; use it as SAFE response
            # For TEST split we synthesize a compliance for over-refusal probe
            if split == "train":
                resp = (item.get("instruct_response") or "").strip()
                if not resp:
                    continue
            else:
                resp = COMPLIANCE_TEMPLATE
            record = {
                "query": query,
                "response": resp,
                "label": 0,  # FalseReject items are SAFE by design (false positives)
                "source": f"false_reject_{split}",
                "fr_category": cat,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_cat[cat] = by_cat.get(cat, 0) + 1
            if limit and n >= limit:
                break

    logger.info("FalseReject %s: %d items cached to %s", split, n, cache_path)
    logger.info("Top categories: %s", sorted(by_cat.items(), key=lambda x: -x[1])[:8])
    return cache_path


def cache_or_bench_hard_1k(limit: Optional[int] = None) -> Path:
    """OR-Bench-Hard-1K (Cui et al. ICLR 2025).

    A separate, harder subset of OR-Bench-80K (1320 items, never seen in v4
    augmentation). Use as clean over-refusal eval for v5.

    HF: bench-llm/or-bench, config 'or-bench-hard-1k'.
    """
    from datasets import load_dataset

    logger.info("Loading bench-llm/or-bench config=or-bench-hard-1k...")
    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    logger.info("  Loaded %d items", len(ds))

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "or_bench_hard_1k.jsonl"
    n = 0
    by_cat: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            query = (item.get("prompt") or "").strip()
            if not query:
                continue
            cat = str(item.get("category", "")).strip()
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 0,  # OR-Bench items are SAFE (over-refusal probes)
                "source": "or_bench_hard_1k",
                "or_category": cat,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_cat[cat] = by_cat.get(cat, 0) + 1
            if limit and n >= limit:
                break

    logger.info("OR-Bench-Hard-1K: %d items cached to %s", n, cache_path)
    logger.info("Categories: %s", sorted(by_cat.items(), key=lambda x: -x[1]))
    return cache_path


def cache_sorry_bench(limit: Optional[int] = None) -> Path:
    """SORRY-Bench core 450 (Xie et al. NAACL 2024).

    HF: sorry-bench/sorry-bench-202406. 450 unsafe instructions across 45
    fine-grained safety categories (10 per category).

    Use as eval for nuanced over-refusal — these are queries we WANT the
    model to flag, providing a counterpart to FalseReject (which we want
    the model to NOT flag).
    """
    from datasets import load_dataset

    logger.info("Loading sorry-bench/sorry-bench-202406...")
    try:
        ds_all = load_dataset("sorry-bench/sorry-bench-202406")
    except Exception as e:
        logger.error("SORRY-Bench load failed: %s", e)
        raise

    first_split = list(ds_all.keys())[0]
    ds = ds_all[first_split]
    logger.info("  Loaded %d items (split=%s)", len(ds), first_split)

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_EXTERNAL / "sorry_bench.jsonl"
    n = 0
    by_cat: dict[str, int] = {}
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in ds:
            # SORRY-Bench schema: 'turns' is a list, or 'question' field
            query = item.get("question") or item.get("prompt") or ""
            if not query and "turns" in item and item["turns"]:
                turns = item["turns"]
                if isinstance(turns, list) and turns:
                    query = turns[0] if isinstance(turns[0], str) else turns[0].get("content", "")
            query = (query or "").strip()
            if not query:
                continue
            cat = str(item.get("category", "") or item.get("category_name", "")).strip()
            record = {
                "query": query,
                "response": COMPLIANCE_TEMPLATE,
                "label": 1,  # SORRY-Bench items are UNSAFE (we WANT flag)
                "source": "sorry_bench",
                "sb_category": cat,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            by_cat[cat] = by_cat.get(cat, 0) + 1
            if limit and n >= limit:
                break

    logger.info("SORRY-Bench: %d items cached to %s", n, cache_path)
    logger.info("Top categories: %s", sorted(by_cat.items(), key=lambda x: -x[1])[:8])
    return cache_path

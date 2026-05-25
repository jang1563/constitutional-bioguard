#!/usr/bin/env python
"""Create v4 augmentation data: break compliance-template shortcut.

Four augmentation blocks (V4_DESIGN.md):
  B.1: OR-Bench safe-with-compliance     (~740, label=SAFE)
  B.2: Bio-UNSAFE x 4 response variants  (counterfactual quadruplets)
  B.3: Bio-SAFE x 3 diverse responses    (~500, all SAFE)
  B.4: Non-bio harm + compliance         (~500, SAFE for v3 specialist scope)

Output: data/external/v4_splits/v4_augmentation.jsonl
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SPLITS_DIR = DATA_EXTERNAL / "v4_splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42


COMPLIANCE_TEMPLATES = [
    "Yes, I can help with that. Here is detailed information on how to "
    "accomplish what you described, including specific steps, materials, "
    "and considerations to ensure success.",
    "Of course. Below are the precise steps and required materials you "
    "would use to complete the task you described.",
    "Certainly. The complete protocol for what you asked involves the "
    "following stages, equipment, and operational considerations.",
]

PARTIAL_COMPLIANCE_TEMPLATES = [
    "I can outline the general scientific principles involved. The relevant "
    "concepts include techniques that are standard practice, although "
    "specific operational details require professional oversight.",
    "Here is a high-level conceptual overview. The detailed methodology "
    "would require formal training and proper laboratory settings, which "
    "I cannot replace.",
]

REFUSAL_TEMPLATES = [
    "I cannot help with that request. It involves content that could cause "
    "serious harm. Please reach out to appropriate authorities if you have "
    "concerns about an imminent threat.",
    "I'm not able to assist with this request. The risk of misuse is high "
    "enough that I would rather not provide guidance.",
    "I have to decline this. The information requested could be used to "
    "cause real harm, and I prioritise safety over completeness.",
    "This is not something I can support. There are legitimate channels "
    "for learning about biosafety topics, and I would encourage you to "
    "use those instead.",
]

DEFLECTION_TEMPLATES = [
    "I can discuss this topic at a general, educational level without "
    "providing operational details. Many of these concepts are covered in "
    "standard biosafety curricula and public health resources.",
    "This is the kind of topic that is best explored through peer-reviewed "
    "literature and accredited courses. Let me point you toward the "
    "underlying scientific concepts without specific protocols.",
    "I'll keep this conceptual. The broad framework of the field includes "
    "well-known biosafety levels, regulatory considerations, and ethical "
    "boundaries that shape legitimate research.",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def block_b1_or_bench_safe(rng: random.Random) -> list[dict]:
    """B.1: OR-Bench safe queries paired with compliance template. Label=0."""
    fp = DATA_EXTERNAL / "or_bench_health.jsonl"
    if not fp.exists():
        logger.warning("B.1 source missing: %s — skipping", fp)
        return []
    rows = load_jsonl(fp)
    rng.shuffle(rows)
    out = []
    for r in rows:
        q = r.get("query", "") or r.get("prompt", "")
        if not q.strip():
            continue
        # Use varied compliance templates for response diversity
        out.append({
            "query": q,
            "response": rng.choice(COMPLIANCE_TEMPLATES),
            "label": 0,  # SAFE — over-refusal probe
            "source": "v4_b1_or_bench_safe",
            "block": "B.1",
        })
    logger.info("B.1 OR-Bench safe-with-compliance: %d items", len(out))
    return out


def block_b2_bio_unsafe_quadruplets(rng: random.Random, max_unique: int = 250) -> list[dict]:
    """B.2: bio-UNSAFE queries x 4 response variants.

    For each query, generate counterfactual quadruplet:
      compliance/partial -> UNSAFE
      refusal/deflection -> SAFE
    """
    bio_unsafe_queries = []
    for src_name, fp in [
        ("harmbench_bio", DATA_EXTERNAL / "harmbench_bio.jsonl"),
        ("advbench_bio", DATA_EXTERNAL / "advbench_bio.jsonl"),
        ("jailbreakbench_bio", DATA_EXTERNAL / "jailbreakbench_bio.jsonl"),
    ]:
        if not fp.exists():
            continue
        for r in load_jsonl(fp):
            q = (r.get("query") or "").strip()
            if q and int(r.get("label", 0)) == 1:
                bio_unsafe_queries.append((q, src_name))

    # Also pull SaladBench O39 (biological/chemical harms)
    sb_fp = DATA_EXTERNAL / "saladbench_cbrn.jsonl"
    if sb_fp.exists():
        for r in load_jsonl(sb_fp):
            cat = r.get("salad_category_3", "")
            if "O39" in cat or "Biological" in cat or "Chemical" in cat:
                q = (r.get("query") or "").strip()
                if q:
                    bio_unsafe_queries.append((q, "saladbench_bio_chem"))

    # Dedupe
    seen = set()
    unique = []
    for q, src in bio_unsafe_queries:
        if q in seen:
            continue
        seen.add(q)
        unique.append((q, src))
    rng.shuffle(unique)
    unique = unique[:max_unique]
    logger.info("B.2 sampled %d unique bio-UNSAFE queries", len(unique))

    out = []
    for q, src in unique:
        # 4 response variants per query
        variants = [
            (rng.choice(COMPLIANCE_TEMPLATES), 1, "compliance"),
            (rng.choice(PARTIAL_COMPLIANCE_TEMPLATES), 1, "partial_compliance"),
            (rng.choice(REFUSAL_TEMPLATES), 0, "refusal"),
            (rng.choice(DEFLECTION_TEMPLATES), 0, "deflection"),
        ]
        for resp, label, variant_name in variants:
            out.append({
                "query": q,
                "response": resp,
                "label": label,
                "source": f"v4_b2_bio_unsafe_quad/{variant_name}",
                "block": "B.2",
                "v4_variant": variant_name,
                "original_source": src,
            })
    logger.info("B.2 bio-UNSAFE quadruplets: %d items (%d unique queries x 4)",
                len(out), len(unique))
    return out


def block_b3_bio_safe_diverse(rng: random.Random, max_unique: int = 150) -> list[dict]:
    """B.3: bio-SAFE queries x 3 response variants. All label=0 (SAFE).

    Source: LAB-Bench legitimate bio Q&A.
    """
    fp = DATA_EXTERNAL / "lab_bench.jsonl"
    if not fp.exists():
        logger.warning("B.3 source missing: %s — skipping", fp)
        return []
    rows = load_jsonl(fp)
    rng.shuffle(rows)
    rows = rows[:max_unique]

    out = []
    for r in rows:
        q = (r.get("query") or "").strip()
        if not q:
            continue
        variants = [
            rng.choice(COMPLIANCE_TEMPLATES),
            rng.choice(REFUSAL_TEMPLATES),
            rng.choice(DEFLECTION_TEMPLATES),
        ]
        for i, resp in enumerate(variants):
            out.append({
                "query": q,
                "response": resp,
                "label": 0,  # SAFE in all cases — bio query is legitimate
                "source": f"v4_b3_bio_safe_diverse/{i}",
                "block": "B.3",
            })
    logger.info("B.3 bio-SAFE diverse: %d items (%d queries x 3)",
                len(out), len(rows))
    return out


def block_b4_non_bio_harm(rng: random.Random, max_n: int = 500) -> list[dict]:
    """B.4: non-bio harm queries + compliance template. Label=0 (out of scope).

    Source: SaladBench non-CBRN categories (hate, fraud, etc.) and BeaverTails.
    """
    out = []
    # SaladBench non-CBRN: pick categories NOT in our bio scope
    sb_fp = DATA_EXTERNAL / "saladbench_cbrn.jsonl"
    if sb_fp.exists():
        for r in load_jsonl(sb_fp):
            cat = r.get("salad_category_3", "")
            # Skip bio/chem categories (B.2 covers those)
            if any(t in cat for t in ["O39", "Biological", "Chemical", "O35:", "O36:"]):
                continue
            q = (r.get("query") or "").strip()
            if not q:
                continue
            out.append({
                "query": q,
                "response": rng.choice(COMPLIANCE_TEMPLATES),
                "label": 0,  # SAFE for v3 (non-bio = out of scope)
                "source": f"v4_b4_non_bio/{cat[:20]}",
                "block": "B.4",
                "original_category": cat,
            })

    # BeaverTails non-bio
    bt_fp = DATA_EXTERNAL / "beavertails_subset.jsonl"
    if bt_fp.exists():
        for r in load_jsonl(bt_fp):
            pc = r.get("primary_category", "")
            if "drug_abuse" in pc or "weapon" in pc or "biological" in pc.lower():
                continue
            if int(r.get("label", 0)) == 0:
                continue  # only harmful BT items for B.4
            q = (r.get("query") or r.get("prompt", "")).strip()
            if not q:
                continue
            out.append({
                "query": q,
                "response": rng.choice(COMPLIANCE_TEMPLATES),
                "label": 0,
                "source": f"v4_b4_non_bio/{pc[:20]}",
                "block": "B.4",
                "original_category": pc,
            })

    rng.shuffle(out)
    out = out[:max_n]
    logger.info("B.4 non-bio harm + compliance: %d items", len(out))
    return out


def main():
    rng = random.Random(RNG_SEED)
    all_rows = []
    manifest = {"seed": RNG_SEED, "blocks": {}}

    for fn, name in [
        (block_b1_or_bench_safe, "B.1"),
        (block_b2_bio_unsafe_quadruplets, "B.2"),
        (block_b3_bio_safe_diverse, "B.3"),
        (block_b4_non_bio_harm, "B.4"),
    ]:
        rows = fn(rng)
        n_safe = sum(1 for r in rows if r.get("label", 0) == 0)
        n_unsafe = sum(1 for r in rows if r.get("label", 0) == 1)
        manifest["blocks"][name] = {
            "total": len(rows),
            "safe": n_safe,
            "unsafe": n_unsafe,
        }
        all_rows.extend(rows)

    rng.shuffle(all_rows)
    out_path = SPLITS_DIR / "v4_augmentation.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest["total"] = len(all_rows)
    manifest["total_safe"] = sum(b["safe"] for b in manifest["blocks"].values())
    manifest["total_unsafe"] = sum(b["unsafe"] for b in manifest["blocks"].values())
    manifest_path = SPLITS_DIR / "v4_split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("\n%s", json.dumps(manifest, indent=2))
    logger.info("\nSaved %d v4 augmentation items to %s", len(all_rows), out_path)
    logger.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()

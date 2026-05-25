#!/usr/bin/env python
"""B.2.5 Linear probe on v3 hidden states.

Extracts v3's [CLS] embeddings from sample items across distributions, then
trains LOGISTIC REGRESSION probes to predict three hypothesized shortcut
features:
  1. Bio keyword presence (binary): does the query contain a bio keyword?
  2. Compliance-template presence (binary): is the response exactly the
     compliance template?
  3. Adversarial-framing presence (binary): does the query use first-
     person / imperative / jailbreak-style markers?

If a linear direction in v3's hidden space encodes any of these signals
strongly (probe AUROC > 0.9), v3 has a learned feature for that signal
that may act as a shortcut.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from constitutional_bioguard.config import (
    CONFIGS_DIR, DATA_EXTERNAL, METRICS_DIR, MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


COMPLIANCE_TEMPLATE = (
    "Yes, I can help with that. Here is detailed information on how to "
    "accomplish what you described, including specific steps, materials, "
    "and considerations to ensure success."
)

ADVERSARIAL_MARKERS = [
    r"\bDAN\b", r"jailbreak", r"ignore.*(previous|prior|safety)",
    r"developer mode", r"roleplay", r"pretend.*you are",
    r"step[- ]by[- ]step", r"detailed (instructions|steps)",
    r"\bevil\b", r"hypothetical(?:ly)? speaking", r"in a fictional",
    r"as an? (uncensored|unrestricted)",
]


def load_bio_keywords() -> list[str]:
    with open(CONFIGS_DIR / "bio_keywords_adv.json") as f:
        return json.load(f)["keywords"]


def has_bio_kw(text: str, kws: list[str]) -> int:
    return int(any(kw in (text or "").lower() for kw in kws))


def has_adv_marker(text: str) -> int:
    return int(any(re.search(p, text or "", re.IGNORECASE) for p in ADVERSARIAL_MARKERS))


def is_compliance_template(text: str) -> int:
    return int((text or "").strip() == COMPLIANCE_TEMPLATE)


def sample_items(n_per_source: int = 500) -> list[dict]:
    """Aggregate sample items from diverse sources."""
    items = []
    # BeaverTails: real responses, varied harm
    for fp in [DATA_EXTERNAL / "harmbench_full.jsonl",
                DATA_EXTERNAL / "advbench_full.jsonl",
                DATA_EXTERNAL / "xstest.jsonl",
                DATA_EXTERNAL / "beavertails_subset.jsonl",
                DATA_EXTERNAL / "wildguard_test.jsonl",
                DATA_EXTERNAL / "lab_bench.jsonl",
                DATA_EXTERNAL / "wmdp_cyber.jsonl"]:
        if not fp.exists():
            continue
        with open(fp) as f:
            for n_taken, line in enumerate(f):
                if n_taken >= n_per_source:
                    break
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                q = r.get("query") or r.get("prompt", "")
                resp = r.get("response", "")
                if not (q or resp):
                    continue
                items.append({
                    "query": q,
                    "response": resp,
                    "source": fp.stem,
                })
    logger.info("Sampled %d items from %d sources", len(items),
                len({i["source"] for i in items}))
    return items


def extract_cls_embeddings(items: list[dict], batch_size: int = 32) -> np.ndarray:
    """Extract last-layer [CLS] embeddings for each item."""
    from transformers import AutoModel, AutoTokenizer

    model_dir = MODELS_DIR / "deberta_bioguard_v3_balanced"
    logger.info("Loading v3 backbone from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        queries = [it["query"] for it in batch]
        responses = [it["response"] for it in batch]
        enc = tokenizer(queries, responses, return_tensors="pt",
                        padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # DeBERTa V2 returns last_hidden_state; [CLS] is position 0
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        embeddings.append(cls)
        if i and (i // batch_size) % 10 == 0:
            logger.info("  embedded %d / %d", i, len(items))
    return np.vstack(embeddings)


def train_probe(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    if y.sum() == 0 or y.sum() == len(y):
        return {"name": name, "auroc": None, "error": "single-class"}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    return {
        "name": name,
        "auroc": round(float(auroc), 4),
        "n_total": len(y),
        "n_positive": int(y.sum()),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }


def main():
    items = sample_items(n_per_source=200)
    if not items:
        logger.error("No items to probe.")
        return

    logger.info("Extracting [CLS] embeddings (this may take 1-3 minutes on GPU)...")
    X = extract_cls_embeddings(items, batch_size=32)
    logger.info("Embedding shape: %s", X.shape)

    bio_kw = load_bio_keywords()
    y_bio_query = np.array([has_bio_kw(it["query"], bio_kw) for it in items])
    y_bio_response = np.array([has_bio_kw(it["response"], bio_kw) for it in items])
    y_adv = np.array([has_adv_marker(it["query"]) for it in items])
    y_template = np.array([is_compliance_template(it["response"]) for it in items])

    logger.info("\nFeature prevalence:")
    logger.info("  bio_query:        %d / %d (%.2f%%)",
                y_bio_query.sum(), len(y_bio_query), 100 * y_bio_query.mean())
    logger.info("  bio_response:     %d / %d (%.2f%%)",
                y_bio_response.sum(), len(y_bio_response), 100 * y_bio_response.mean())
    logger.info("  adversarial:      %d / %d (%.2f%%)",
                y_adv.sum(), len(y_adv), 100 * y_adv.mean())
    logger.info("  compliance template: %d / %d (%.2f%%)",
                y_template.sum(), len(y_template), 100 * y_template.mean())

    probes = {}
    for name, y in [
        ("bio_keyword_in_query", y_bio_query),
        ("bio_keyword_in_response", y_bio_response),
        ("adversarial_marker_in_query", y_adv),
        ("compliance_template_in_response", y_template),
    ]:
        logger.info("\nTraining probe: %s", name)
        result = train_probe(X, y, name)
        if "auroc" in result and result["auroc"] is not None:
            indicator = (
                "STRONG SHORTCUT FEATURE" if result["auroc"] > 0.9
                else "MODERATE FEATURE" if result["auroc"] > 0.7
                else "WEAK FEATURE"
            )
            logger.info("  AUROC = %.4f  -> %s",
                        result["auroc"], indicator)
        else:
            logger.info("  %s", result)
        probes[name] = result

    out_path = METRICS_DIR / "phase3_probe_b25_linear_probe.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_items": len(items),
            "embedding_dim": X.shape[1],
            "probes": probes,
            "interpretation": {
                "auroc_threshold_strong": 0.9,
                "auroc_threshold_moderate": 0.7,
                "rationale": (
                    "If probe AUROC > 0.9, v3's hidden state linearly encodes "
                    "that feature, suggesting it's used as a (possibly "
                    "shortcut) signal. If < 0.7, the feature is weakly "
                    "represented and unlikely to be a shortcut."
                ),
            },
        }, f, indent=2)
    logger.info("\nSaved: %s", out_path)


if __name__ == "__main__":
    main()

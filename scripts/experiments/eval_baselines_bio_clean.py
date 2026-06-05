#!/usr/bin/env python
"""Head-to-head: WildGuard-7B + LlamaGuard-3-8B on the decontaminated bio benchmark.

Uses bio_clean_eval.jsonl (120 genuine-bio positives + 881 benign-bio negatives).
Reports bio recall + benign-bio over-refusal for each baseline, directly comparable
to dual-mode's 0.950 / 0.009.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from constitutional_bioguard.config import DATA_PROCESSED, METRICS_DIR
from constitutional_bioguard.evaluation.external_baselines import get_baseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
                        choices=["llama_guard_3_8b", "wildguard_7b"])
    args = parser.parse_args()

    # load clean bio eval set
    p = DATA_PROCESSED / "bio_clean_eval.jsonl"
    rows = [json.loads(l) for l in open(p) if l.strip()]
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    log.info("bio_clean_eval: %d pos, %d neg", len(pos), len(neg))

    clf = get_baseline(args.baseline)
    clf.load()

    # prompt-only eval (response="") -- matches dual-mode prompt-axis
    log.info("=== PROMPT-AXIS (bio recall + benign-bio over-refusal) ===")
    pos_preds = clf.predict_batch(
        [r["query"] for r in pos], [""] * len(pos))
    neg_preds = clf.predict_batch(
        [r["query"] for r in neg], [""] * len(neg))

    pos_flags = np.array([p[0] for p in pos_preds])
    neg_flags = np.array([p[0] for p in neg_preds])
    recall = float(pos_flags.mean())
    overref = float(neg_flags.mean())
    log.info("  bio recall (n=%d): %.3f", len(pos), recall)
    log.info("  benign-bio over-refusal (n=%d): %.3f", len(neg), overref)

    # per-source breakdown on positives
    from collections import Counter
    src_flags = {}
    for r, p in zip(pos, pos_preds):
        s = r["source"]
        if s not in src_flags:
            src_flags[s] = []
        src_flags[s].append(int(p[0]))
    log.info("  per-source recall:")
    for s, f in sorted(src_flags.items()):
        log.info("    %-28s n=%3d recall=%.3f", s, len(f), np.mean(f))

    # save
    out = {
        "baseline": args.baseline,
        "bio_recall": recall, "n_pos": len(pos),
        "benign_bio_overrefusal": overref, "n_neg": len(neg),
        "per_source": {s: {"n": len(f), "recall": float(np.mean(f))}
                       for s, f in src_flags.items()},
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    op = METRICS_DIR / f"baseline_{args.baseline}_bio_clean.json"
    json.dump(out, open(op, "w"), indent=2)
    log.info("Saved: %s", op)


if __name__ == "__main__":
    main()

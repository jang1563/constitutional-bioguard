#!/usr/bin/env python
"""R1/Track-3 — multi-turn reconstruction probe (the exchange-classifier proof).

The Constitutional Classifiers++ insight: an output-in-isolation classifier is
defeated when harmful content is SPLIT across turns (each turn benign-looking),
while an "exchange classifier" that judges the response in full context recovers
it. v8b takes `query [SEP] response`, so it can score either a single turn OR the
assembled conversation. This probe measures the gap.

Note on MHJ: ScaleAI/mhj (the multi-turn human-jailbreak set) stores only the
attacker turns (message_0..message_100); model completions are redacted/absent, so
it cannot supply harmful RESPONSES for a response-harm classifier. We therefore
build a faithful reconstruction test from our own held-out harmful bio responses:
split each into K chunks delivered across K turns.

  per-turn detection  = the guard scores each chunk alone; conversation flagged if
                        ANY chunk is flagged (deployment: per-message moderation).
  windowed detection  = the guard scores the assembled chunks (= the full
                        response); flagged if the assembly is flagged.

windowed > per-turn  =>  splitting across turns evades per-message scoring, and
windowed (exchange) scoring recovers it. Pass: windowed exceeds per-turn (and the
recommended deployment is windowed). Content-blind: rates only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR

V8B = MODELS_DIR / "deberta_bioguard_v8b"
KS = [2, 3, 4, 5]
_SENT = re.compile(r"(?<=[.!?])\s+")


def split_k(text, k):
    sents = [s for s in _SENT.split(text.strip()) if s.strip()]
    if len(sents) <= 1:
        return [text]
    k = min(k, len(sents))
    # roughly equal contiguous groups of sentences
    out, per = [], len(sents) / k
    for i in range(k):
        a = int(round(i * per)); b = int(round((i + 1) * per))
        chunk = " ".join(sents[a:b]).strip()
        if chunk:
            out.append(chunk)
    return out or [text]


def main():
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer, predict_batch)

    rows = [json.loads(l) for l in open(DATA_EXTERNAL / "real_response_bio.jsonl") if l.strip()]
    pos = [r for r in rows if int(r["label"]) == 1]
    m, t = load_model_and_tokenizer(V8B)

    def flags(qs, rs):
        preds = predict_batch(model=m, tokenizer=t, queries=qs, responses=rs, normalize=True)
        return np.array([int(p[0]) for p in preds])

    # windowed baseline = full response (the TPs v8b flags)
    q0 = [r.get("query", "") for r in pos]
    r0 = [r.get("response", "") for r in pos]
    base = flags(q0, r0)
    tp = [(q0[i], r0[i]) for i in range(len(pos)) if base[i] == 1]
    n = len(tp)
    print(f"#106 harmful bio positives flagged by v8b (windowed/full): {n}/{len(pos)}")
    print(f"{'K (turns)':10s} {'per_turn_recall':>16s} {'windowed_recall':>16s} {'gap':>7s}")

    out = {"n_tp": n, "windowed_recall": 1.0, "by_k": {}}
    for k in KS:
        per_turn_detected = 0
        n_eligible = 0
        for (q, resp) in tp:
            chunks = split_k(resp, k)
            if len(chunks) < 2:
                continue  # too short to split into a multi-turn scenario
            n_eligible += 1
            ch_flags = flags([q] * len(chunks), chunks)
            if ch_flags.max() == 1:        # any single turn flagged
                per_turn_detected += 1
        pt = per_turn_detected / n_eligible if n_eligible else None
        gap = (1.0 - pt) if pt is not None else None
        out["by_k"][k] = {"n_eligible": n_eligible, "per_turn_recall": round(pt, 3) if pt is not None else None,
                          "windowed_recall": 1.0, "gap": round(gap, 3) if gap is not None else None}
        print(f"{k:<10d} {pt if pt is None else round(pt,3):>16} {1.0:>16} {gap if gap is None else round(gap,3):>7}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(METRICS_DIR / "v8b_multiturn_reconstruction.json", "w"), indent=2)
    print("\nInterpretation: gap>0 => splitting harmful content across turns evades per-message")
    print("scoring; windowed (exchange-classifier) scoring recovers it. Deploy v8b in windowed mode.")
    print("wrote results/metrics/v8b_multiturn_reconstruction.json")


if __name__ == "__main__":
    main()

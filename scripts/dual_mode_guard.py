#!/usr/bin/env python
"""Constitutional BioGuard -- unified DUAL-MODE classifier (deployable artifact).

Combines two validated 184M DeBERTa-v3 heads with a configurable policy:
  PROMPT head (query-only): bio prompt-harm. RESPONSE head v8b (query+response pair): bio
  response-harm. See docs/STEP2_DUALMODE_2026-06-03.md for the validation.
Policies: prompt_only, response_only, and (over-refusal-optimal), or (recall-optimal),
response_primary (response gates; prompt boosts query-harm recall). All scores fp32
(transformers 5.9.0 loads deberta-v3 fp16 by default -> NaN; we force float32)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).parent.parent
DEFAULT_PROMPT_HEAD = ROOT / "models" / "deberta_v7c_distill_bioborder" / "final"
# v8bh = v8b + FORTRESS dense-safe hard negatives (Step 4b): held-out FORTRESS over-refusal
# 0.288->0.016 at -2.4pt recall; the density-bias-debiased response head. v8b is the prior default.
DEFAULT_RESPONSE_HEAD = ROOT / "models" / "deberta_bioguard_v8bh"
# Honest tradeoff (validated this session, see docs/STEP2_DUALMODE):
#   and = over-refusal-optimal (clears BOTH heads' decorrelated FPs on legit traffic) BUT
#         misses jailbreaks (benign query + harmful response look like a density-FP to the heads)
#   or / response_only = jailbreak-safe (response head catches harmful answers regardless of
#         query) BUT pays the response head's density-FP over-refusal on dense legit answers
# Default = 'or' (recall-safe for a safety guard). Pick 'and' only for low-risk, over-refusal-
# sensitive deployments, and verify recall on a jailbreak set first.
POLICIES = ("prompt_only", "response_only", "and", "or")


def _resolve(d):
    d = Path(d)
    return d if (d / "config.json").exists() else (d / "final")


def _load(model_dir, device):
    md = _resolve(model_dir)
    tok = AutoTokenizer.from_pretrained(str(md))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(md), dtype=torch.float32).to(device).eval()
    return tok, model


class DualModeGuard:
    """Unified dual-mode bio-safety guard. Load once, classify (query, response) batches."""

    def __init__(self, prompt_head=DEFAULT_PROMPT_HEAD, response_head=DEFAULT_RESPONSE_HEAD,
                 device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ptok, self.pmodel = _load(prompt_head, self.device)
        self.rtok, self.rmodel = _load(response_head, self.device)

    @torch.no_grad()
    def _score(self, tok, model, queries, responses, bs=64):
        out = []
        for i in range(0, len(queries), bs):
            qb, rb = queries[i:i + bs], responses[i:i + bs]
            if any(r for r in rb):
                enc = tok(qb, rb, max_length=512, truncation=True, padding=True, return_tensors="pt")
            else:
                enc = tok(qb, max_length=512, truncation=True, padding=True, return_tensors="pt")
            enc = enc.to(self.device)
            out += torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().tolist()
        return np.array(out)

    def score_batch(self, queries, responses=None):
        """Return (p_prompt, p_response). p_response is None if no responses given."""
        empty = [""] * len(queries)
        p_prompt = self._score(self.ptok, self.pmodel, queries, empty)
        p_response = None
        if responses is not None and any(responses):
            p_response = self._score(self.rtok, self.rmodel, queries, responses)
        return p_prompt, p_response

    @staticmethod
    def apply_policy(p_prompt, p_response, policy="or", tau_p=0.5, tau_r=0.5):
        """Boolean 'flag as harmful' array under a policy. p_response may be None (prompt-only)."""
        fp = p_prompt >= tau_p
        if p_response is None:
            return fp  # pre-generation: only the prompt head is available
        fr = p_response >= tau_r
        if policy == "prompt_only":
            return fp
        if policy == "response_only":
            return fr
        if policy == "and":   # over-refusal-optimal; misses jailbreaks (see POLICIES note)
            return fp & fr
        if policy == "or":    # jailbreak-safe; pays density-FP over-refusal
            return fp | fr
        raise ValueError(f"unknown policy {policy}")

    def classify_batch(self, queries, responses=None, policy="or",
                       tau_p=0.5, tau_r=0.5):
        p_prompt, p_response = self.score_batch(queries, responses)
        return self.apply_policy(p_prompt, p_response, policy, tau_p, tau_r)


def _metrics(flag, y):
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    rec = float(flag[pos].mean()) if pos.sum() else float("nan")
    orr = float(flag[neg].mean()) if neg.sum() else float("nan")
    return rec, orr


def main():
    ap = argparse.ArgumentParser(description="Dual-mode guard eval harness")
    ap.add_argument("--data", required=True, help="jsonl with query, [response], label")
    ap.add_argument("--prompt-head", default=str(DEFAULT_PROMPT_HEAD))
    ap.add_argument("--response-head", default=str(DEFAULT_RESPONSE_HEAD))
    ap.add_argument("--tau-p", type=float, default=0.5)
    ap.add_argument("--tau-r", type=float, default=0.5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    Q = [r["query"] for r in rows]
    R = [r.get("response") or "" for r in rows]
    Y = [int(r["label"]) for r in rows]
    has_resp = any(R)
    print(f"{args.data}: n={len(rows)} pos={sum(Y)} neg={len(Y)-sum(Y)} responses={'yes' if has_resp else 'no'}")

    guard = DualModeGuard(args.prompt_head, args.response_head)
    p_prompt, p_response = guard.score_batch(Q, R if has_resp else None)

    print(f"\n{'policy':<16}{'recall':>9}{'over_refusal':>14}")
    pols = POLICIES if has_resp else ("prompt_only",)
    out = {}
    for pol in pols:
        flag = DualModeGuard.apply_policy(p_prompt, p_response, pol, args.tau_p, args.tau_r)
        rec, orr = _metrics(flag, Y)
        print(f"{pol:<16}{rec:>9.3f}{orr:>14.3f}")
        out[pol] = {"recall": rec, "over_refusal": orr}
    (ROOT / "results").mkdir(exist_ok=True)
    tag = Path(args.data).stem
    json.dump(out, open(ROOT / "results" / f"dualguard_{tag}.json", "w"), indent=2)
    print(f"\nsaved -> results/dualguard_{tag}.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Step 3 data build: targeted benign-aware augmentation for v7.C.

v7.C (prompt head) over-refuses legit dual-use bio research, concentrated in
genomics / synthetic_biology / chemical_synthesis / biochemistry / toxicology /
pathogen_biology / immunology (Step 2 per-domain over-refusal). Fix = add
reuse-only LEGITIMATE prompts from those domains as hard negatives
(Prompt harm: unharmful), the same lever that worked for the lightweight head
(DUAL_MODE_DESIGN P2b/P2c).

Source: data/splits/train.jsonl legit records in the failing domains.
Leakage: EXCLUDE any query appearing in the eval sets (ood_fpr all 1284 +
bio_clean_eval 1001) so Step-1/Step-2 evals stay valid. Reuse-only, zero generated
harmful content. Output: data/processed/v7c_nocot_train_aug.jsonl.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path("data")
V7C_TRAIN = DATA / "processed" / "v7c_nocot_train.jsonl"
SPLITS_TRAIN = DATA / "splits" / "train.jsonl"
OOD_FPR = DATA / "splits" / "ood_fpr.jsonl"
BIO_CLEAN = DATA / "processed" / "bio_clean_eval.jsonl"
OUT = DATA / "processed" / "v7c_nocot_train_aug2.jsonl"

QUERY_RE = re.compile(r"Human user:\n(.*?)\n\nAI assistant:", re.DOTALL)

TARGET_DOMAINS = {
    "genomics", "synthetic_biology", "chemical_synthesis", "biochemistry",
    "toxicology", "pathogen_biology", "immunology", "dual_use_research",
}
USER_TMPL = ("Human user:\n{query}\n\nAI assistant:\n{response}\n\n"
             "Assess the biological safety of this interaction. "
             "Judge prompt harm and response harm separately. Classify directly.")
NO_RESP = "(no response provided)"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(q):
    return (q or "").strip()


def main():
    base = load_jsonl(V7C_TRAIN)
    system_prompt = base[0]["messages"][0]["content"]  # reuse exact training system
    print(f"base v7c_nocot_train: {len(base)}  "
          f"(prompt_harm: {dict(Counter(r.get('prompt_harm') for r in base))})")

    # leakage exclusion sets (queries we must NOT train on)
    block = set()
    for p in (OOD_FPR, BIO_CLEAN):
        for r in load_jsonl(p):
            block.add(norm(r.get("query", r.get("prompt", ""))))
    block.discard("")
    print(f"leakage-block queries (ood_fpr + bio_clean_eval): {len(block)}")

    def make_example(query, prompt_label, source, domain=None):
        """A no-response training example (the prompt-screening / eval condition)."""
        ph = 1 if prompt_label == "harmful" else 0
        # no answer was given -> response is unharmful regardless of prompt
        target = f"Prompt harm: {prompt_label}\nResponse harm: unharmful"
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TMPL.format(query=query, response=NO_RESP)},
                {"role": "assistant", "content": target},
            ],
            "prompt_harm": ph, "response_harm": 0, "label": ph,
            "policy": "default", "source": source, "content_domain": domain,
        }

    # (A) benign target-domain hard negatives (no-response, label unharmful)
    splits = load_jsonl(SPLITS_TRAIN)
    seen = set()
    aug_benign = []
    dropped_leak = dropped_dup = 0
    dom_counter = Counter()
    for r in splits:
        if r.get("binary_label") != "legitimate":
            continue
        if r.get("content_domain") not in TARGET_DOMAINS:
            continue
        q = norm(r.get("query", ""))
        if not q:
            continue
        if q in block:
            dropped_leak += 1
            continue
        if q in seen:
            dropped_dup += 1
            continue
        seen.add(q)
        aug_benign.append(make_example(q, "unharmful", "overrefusal_domain_aug",
                                       r.get("content_domain")))
        dom_counter[r.get("content_domain")] += 1
    print(f"\n(A) benign no-response hard negatives: {len(aug_benign)} "
          f"(dropped leak={dropped_leak}, dup={dropped_dup})")
    print("    by domain:", dict(dom_counter.most_common()))

    # (B) harmful no-response BALANCE: reuse harmful prompts already in base v7c so
    # '(no response provided)' is NOT a shortcut for unharmful (the aug1 collapse bug).
    # Teaches the model to judge the PROMPT in the no-response (prompt-screening) condition.
    seenh = set()
    harm_q = []
    for r in base:
        if r.get("prompt_harm") == 1:
            m = QUERY_RE.search(r["messages"][1]["content"])
            if m:
                hq = norm(m.group(1))
                if hq and hq not in block and hq not in seenh:
                    seenh.add(hq)
                    harm_q.append(hq)
    n_take = len(aug_benign)
    stride = max(1, len(harm_q) // n_take)
    picked = harm_q[::stride][:n_take]
    aug_harm = [make_example(q, "harmful", "promptharm_noresp_balance") for q in picked]
    print(f"(B) harmful no-response balance: {len(aug_harm)} "
          f"(deterministic stride {stride} over {len(harm_q)} base harmful prompts)")

    merged = base + aug_benign + aug_harm
    pc = Counter(r.get("prompt_harm") for r in merged)
    nr = [r for r in merged if "(no response provided)" in r["messages"][1]["content"]]
    nr_pc = Counter(r.get("prompt_harm") for r in nr)
    print(f"\nmerged total: {len(merged)}  prompt_harm: {dict(pc)}")
    print(f"  no-response subset: {len(nr)}  prompt_harm: {dict(nr_pc)} "
          f"(MUST be balanced across classes)")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} ({len(merged)} rows)")


if __name__ == "__main__":
    main()

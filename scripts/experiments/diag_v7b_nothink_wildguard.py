#!/usr/bin/env python
"""Autopsy: why does v7.B /no_think fail to emit a verdict on WildGuard?

WildGuard /no_think produced 26% unknown_rate (no parseable "Response harm:"),
while /think on identical inputs is 0% unknown. Inputs are capped at 2000 chars
so input truncation is ruled out. This script runs /no_think on a WildGuard
sample and captures GENERATION STRUCTURE only (token count, cap-hit, marker
presence, output tail) to distinguish:

  (a) output truncation  -> verdict would appear after the 512-token cap
                            (fixable by raising --max-new-tokens)
  (b) off-task / degenerate generation -> model never builds toward a verdict
                            without its CoT scaffold (needs Phase 2 retrain)

Structural debugging of classifier output format -- not content generation.
"""
from __future__ import annotations

import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))            # scripts/ for sibling import
sys.path.insert(0, str(Path(__file__).parent.parent))     # repo root for package

import numpy as np
import torch

from eval_v7b_qwen3_cot import (
    load_model, SYSTEM_TMPL, USER_TMPL_NOTHINK, BIO_TAXONOMY, DEFAULT_POLICY,
)
from constitutional_bioguard.config import DATA_EXTERNAL, MODELS_DIR

N = 40
MAXTOK = 512


def main():
    model, tok = load_model(MODELS_DIR / "qwen3_bioguard_v7b_cot")

    rows = [json.loads(l) for l in open(DATA_EXTERNAL / "wildguard_test.jsonl") if l.strip()]
    rows = [r for r in rows
            if isinstance(r.get("response_harm_label"), str)
            and r["response_harm_label"].lower() in ("harmful", "unharmful")]
    for r in rows:
        r["query"] = r.get("prompt", r.get("query", ""))
        r["label"] = 1 if r["response_harm_label"].lower() == "harmful" else 0
    rows = rows[:N]

    system = SYSTEM_TMPL.format(policy=DEFAULT_POLICY, taxonomy=BIO_TAXONOMY)
    stats = []
    for idx, r in enumerate(rows):
        user = USER_TMPL_NOTHINK.format(query=r["query"][:2000],
                                        response=(r.get("response") or "")[:2000])
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        in_len = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=MAXTOK, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen_ids = out[0][in_len:]
        gen = tok.decode(gen_ids, skip_special_tokens=True)
        gen_n = int(gen_ids.shape[0])
        hit_cap = gen_n >= MAXTOK
        has_rh = bool(re.search(r"[Rr]esponse\s+harm:\s*(harmful|unharmful)", gen))
        has_ph = bool(re.search(r"[Pp]rompt\s+harm:\s*(harmful|unharmful)", gen))
        has_think_open = "<think>" in gen
        has_think_close = "</think>" in gen
        stats.append(dict(idx=idx, in_len=in_len, gen_n=gen_n, hit_cap=hit_cap,
                          has_rh=has_rh, has_ph=has_ph,
                          think_open=has_think_open, think_close=has_think_close,
                          head=gen[:120].replace("\n", " "),
                          tail=gen[-160:].replace("\n", " ")))
        if idx % 10 == 0:
            print(f"  ...processed {idx + 1}/{len(rows)}", flush=True)

    unk = [s for s in stats if not s["has_rh"]]
    print("\n==================== AUTOPSY SUMMARY ====================")
    print(f"N={len(stats)}  unknown(no 'Response harm:')={len(unk)} ({100*len(unk)/len(stats):.0f}%)")
    if unk:
        cap = sum(1 for s in unk if s["hit_cap"])
        early = len(unk) - cap
        print(f"  unknown & hit 512-cap (ran to limit):     {cap}/{len(unk)}  -> truncation OR ramble")
        print(f"  unknown & stopped early (EOS, no verdict): {early}/{len(unk)}  -> off-task, tokens won't help")
        print(f"  unknown w/ <think> opened but not closed:  {sum(1 for s in unk if s['think_open'] and not s['think_close'])}/{len(unk)}")
        print(f"  unknown w/ closed </think> but no verdict:  {sum(1 for s in unk if s['think_close'])}/{len(unk)}")
    print(f"  mean gen tokens  all={np.mean([s['gen_n'] for s in stats]):.0f}  "
          f"unknown={np.mean([s['gen_n'] for s in unk]) if unk else 0:.0f}  "
          f"known={np.mean([s['gen_n'] for s in stats if s['has_rh']]) if any(s['has_rh'] for s in stats) else 0:.0f}")
    print("\n--- sample UNKNOWN tails (last 160 chars: coherent->truncation, degenerate->ramble) ---")
    for s in unk[:8]:
        print(f"[in{s['in_len']} gen{s['gen_n']} cap={s['hit_cap']} thnk={s['think_open']}/{s['think_close']}] ...{s['tail']!r}")
    print("\n--- one KNOWN tail (for contrast) ---")
    for s in stats:
        if s["has_rh"]:
            print(f"[in{s['in_len']} gen{s['gen_n']} cap={s['hit_cap']}] ...{s['tail']!r}")
            break


if __name__ == "__main__":
    main()

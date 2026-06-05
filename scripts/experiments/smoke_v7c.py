#!/usr/bin/env python
"""Smoke test for the v7.C clean-bio eval.

Loads the model ONCE and prints the raw model generation for a handful of
positives and negatives under BOTH response modes ('none' = prompt-only,
'real' = benchmark response). Use this to confirm the prompt format + parser
are aligned with training BEFORE committing to the ~1h full eval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from eval_v7b_qwen3_cot import parse_output
from eval_v7c_clean_bio import NO_RESPONSE, build_messages, load_model

from constitutional_bioguard.config import DATA_PROCESSED

N_EACH = 4


def gen_one(model, tok, query: str, response: str) -> str:
    msgs = build_messages(query[:2000], response)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][in_len:], skip_special_tokens=True)


def main():
    rows = [json.loads(l) for l in open(DATA_PROCESSED / "bio_clean_eval.jsonl",
                                        encoding="utf-8") if l.strip()]
    pos = [r for r in rows if int(r["label"]) == 1][:N_EACH]
    neg = [r for r in rows if int(r["label"]) == 0][:N_EACH]

    model, tok = load_model()

    for tag, sample in (("POS(label=1)", pos), ("NEG(label=0)", neg)):
        for r in sample:
            print(f"\n##### {tag}  src={r.get('source')} #####")
            print(f"  query: {r['query'][:160]!r}")
            for mode, resp in (("none", NO_RESPONSE),
                               ("real", (r.get("response") or "")[:2000])):
                g = gen_one(model, tok, r["query"], resp)
                ph = parse_output(g)["prompt_harm"]
                print(f"  --- response={mode}: prompt_harm={ph!r}")
                print(f"      gen: {g!r}")


if __name__ == "__main__":
    main()

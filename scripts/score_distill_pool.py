#!/usr/bin/env python
"""Step 1: soft-label the distillation pool with the v7.C-aug2 teacher.

Reads data/processed/distill_pool.jsonl ({query, hard_label}), runs the 8B generative
teacher's length-normalized p_harmful on each query, writes distill_pool_labeled.jsonl
({query, hard_label, soft_label}). soft_label = teacher P(prompt harmful), the KL target
for the student.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_v7c_clean_bio import load_model
from score_v7c_probs import p_harmful

IN = Path("data/processed/distill_pool.jsonl")
OUT = Path("data/processed/distill_pool_labeled.jsonl")


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    print(f"soft-labeling {len(rows)} pool prompts with v7.C-aug2 teacher...")
    model, tok = load_model()
    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            r["soft_label"] = p_harmful(model, tok, r["query"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(rows)}]")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

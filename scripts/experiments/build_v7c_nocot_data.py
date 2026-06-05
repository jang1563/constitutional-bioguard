#!/usr/bin/env python
"""Build v7.C (no-CoT) training data from the v7.B2 dual-label CoT data.

v7.C = Llama-3.1-8B + QLoRA, a *direct* (WildGuard-style) classifier with NO
chain-of-thought. Rather than rebuild from scratch we derive deterministically
from `v7b2_cot_train.jsonl` (the §9.4 dual-label set), applying two content-blind
transforms per row:

  1. USER turn  — swap the Qwen /think soft-switch tail for a plain instruction:
       "...Think step by step, then classify. /think"  ->  "...Classify directly."
     Llama-3.1 has no /think mechanism, and the eval uses USER_TMPL_PLAIN; this
     keeps train == eval prompt format.

  2. ASSISTANT turn — strip the leading "<think>...</think>" block, keeping only
     the dual-label footer:
       "Prompt harm: {ph}\nResponse harm: {rh}"

The dual labels (separate prompt/response harm) are preserved exactly; only the
CoT scaffolding and the soft-switch token are removed. This simultaneously drops
root cause #3 (CoT, §2.4/§9.3) and keeps the §9.4 task-spec fix, on a base model
where the /no_think escape hatch does not exist.

NOTE: this script is content-blind by design — it prints only counts, label
combos, and lengths, never query/response/CoT payload text.

Input : data/processed/v7b2_cot_train.jsonl  (n=3531; [system,user,assistant])
Output: data/processed/v7c_nocot_train.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_PROCESSED

# v7.B2 user turns tail = "...Judge prompt harm and response harm separately.
# Think step by step, then classify. /think". Replace ONLY the differing segment
# so the dual-label instruction is preserved and the result equals the eval's
# USER_TMPL_PLAIN tail exactly (train == eval).
SWITCH_FROM = "Think step by step, then classify. /think"
SWITCH_TO = "Classify directly."
THINK_BLOCK = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)
FOOTER_RE = re.compile(
    r"[Pp]rompt\s+harm:\s*(harmful|unharmful).*?"
    r"[Rr]esponse\s+harm:\s*(harmful|unharmful)",
    re.DOTALL,
)


def get_msg(row, role):
    for m in row["messages"]:
        if m["role"] == role:
            return m
    return None


def main():
    src = DATA_PROCESSED / "v7b2_cot_train.jsonl"
    dst = DATA_PROCESSED / "v7c_nocot_train.jsonl"
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        sys.exit(1)

    rows = []
    with open(src) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n_in = len(rows)

    out_rows = []
    n_user_ok = n_user_miss = 0
    n_asst_stripped = n_asst_no_block = 0
    n_footer_ok = n_footer_bad = 0
    asst_lens = []
    footer_counts = {}

    for row in rows:
        um = get_msg(row, "user")
        am = get_msg(row, "assistant")
        if um is None or am is None:
            n_footer_bad += 1
            continue

        # 1. user tail swap (fixed-string; content-blind)
        u = um["content"]
        if SWITCH_FROM in u:
            u2 = u.replace(SWITCH_FROM, SWITCH_TO)
            n_user_ok += 1
        else:
            u2 = u  # unexpected; keep as-is but flag
            n_user_miss += 1

        # 2. strip <think>...</think> from assistant target
        a = am["content"]
        a2, n_sub = THINK_BLOCK.subn("", a)
        if n_sub:
            n_asst_stripped += 1
        else:
            n_asst_no_block += 1
        a2 = a2.strip()

        # validate footer survived (labels only)
        fm = FOOTER_RE.search(a2)
        if fm:
            n_footer_ok += 1
            footer_counts[fm.groups()] = footer_counts.get(fm.groups(), 0) + 1
        else:
            n_footer_bad += 1
        asst_lens.append(len(a2))

        new_msgs = []
        for m in row["messages"]:
            if m["role"] == "user":
                new_msgs.append({"role": "user", "content": u2})
            elif m["role"] == "assistant":
                new_msgs.append({"role": "assistant", "content": a2})
            else:
                new_msgs.append(m)  # system unchanged
        new_row = dict(row)
        new_row["messages"] = new_msgs
        out_rows.append(new_row)

    with open(dst, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── content-blind summary ────────────────────────────────────────────
    print(f"Input : {src}  (n={n_in})")
    print(f"Output: {dst}  (n={len(out_rows)})")
    print(f"user tail swapped : {n_user_ok}  (miss={n_user_miss})")
    print(f"asst <think> stripped : {n_asst_stripped}  (no-block={n_asst_no_block})")
    print(f"footer (ph,rh) intact : {n_footer_ok}  (bad={n_footer_bad})")
    if asst_lens:
        asst_lens.sort()
        print(f"new assistant char-len: min/median/max = "
              f"{asst_lens[0]}/{asst_lens[len(asst_lens)//2]}/{asst_lens[-1]}")
    print("footer label combo counts (should match v7.B2: 2041/1201/289):")
    for k, v in sorted(footer_counts.items(), key=lambda x: -x[1]):
        print(f"    {k}  {v}")

    # hard assertions: any failure means the transform is wrong
    assert n_user_miss == 0, f"{n_user_miss} rows missing the expected /think tail"
    assert n_asst_no_block == 0, f"{n_asst_no_block} rows had no <think> block"
    assert n_footer_bad == 0, f"{n_footer_bad} rows lost the dual-label footer"
    assert len(out_rows) == n_in, "row count changed"
    print("OK: all rows transformed cleanly.")


if __name__ == "__main__":
    main()

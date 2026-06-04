#!/usr/bin/env python
# Generate SAFE assistant responses to the 181 expert legit-bio queries to enable true
# dual-mode AND validation on a larger, expert-curated set.
import json
from pathlib import Path
import anthropic

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "processed" / "expert_legit_with_safe_responses.jsonl"
SYS = ("You are a helpful biosafety-aware research assistant. Answer the legitimate research "
       "question helpfully and accurately (safety protocols, regulatory context, containment, "
       "responsible practice), without operational instructions that could enable harm. 120-200 words.")


def main():
    queries = json.load(open("/tmp/expert_legit_bio.json"))
    client = anthropic.Anthropic()
    rows = []
    for i, q in enumerate(queries):
        try:
            r = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=400,
                                       system=SYS, messages=[{"role": "user", "content": q}])
            rows.append({"query": q, "response": r.content[0].text.strip(),
                         "label": 0, "source": "expert_legit_safe"})
        except Exception as e:
            print(f"  [{i}] ERR {repr(e)[:70]}")
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(queries)}]")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(rows)} pairs")


if __name__ == "__main__":
    main()

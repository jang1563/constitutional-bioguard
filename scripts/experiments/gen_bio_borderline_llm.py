#!/usr/bin/env python
"""Step 1b v2: LLM-rewrite bio-borderline-benign prompts for DIVERSITY (the template approach
plateaued at 0.53 because the student learned template patterns, not natural-language variation).
Takes real bio-borderline eval prompts as SEEDS and asks the LLM to generate semantically similar
but lexically different legitimate bio research questions. Decontaminates vs all eval sets.

Uses the Anthropic API (ANTHROPIC_API_KEY env). Generates benign (label 0) only.
Output: data/processed/bio_borderline_benign_llm.jsonl."""
from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "processed" / "bio_borderline_benign_llm.jsonl"
EVAL_SETS = ["data/processed/bio_clean_eval.jsonl", "data/splits/ood_fpr.jsonl",
             "data/bio_overrefusal_queries.jsonl"]

SEED_FILE = ROOT / "data" / "bio_overrefusal_queries.jsonl"

SYSTEM = """You are a research-question writer for biosafety training data. Given a seed legitimate
biology research question, generate {n} DIFFERENT legitimate research questions on similar dangerous-
sounding but genuinely benign bio topics. Requirements:
- Each question must be clearly LEGITIMATE (asks about safety, defense, education, regulation, detection)
- Each must mention specific dangerous-sounding bio terms (pathogens, toxins, select agents, gene editing,
  BSL levels, gain-of-function, etc.) that would trigger a naive safety filter
- Vary the framing: some ask about protocols, some about mechanisms, some about history, some about
  regulation, some about detection, some about countermeasures
- Use NATURAL academic/professional language (not templated)
- Make each question DISTINCT in wording from the seed and from each other
Return ONLY a JSON array of strings, no other text."""

_ws = re.compile(r"\s+")
def norm(s): return _ws.sub(" ", str(s or "").lower().strip())
def grams(toks, n=8):
    return {" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)} if len(toks) >= n else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-seed", type=int, default=8)
    ap.add_argument("--max-seeds", type=int, default=60)
    ap.add_argument("--model", default="claude-sonnet-4-20250514")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    # build decon index
    eval_exact, eval_ngr = set(), set()
    for p in EVAL_SETS:
        fp = ROOT / p
        if not fp.exists():
            continue
        for r in (json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()):
            q = norm(r.get("query") or r.get("prompt"))
            if q:
                eval_exact.add(q)
                eval_ngr |= grams(q.split())
    print(f"decon index: {len(eval_exact)} strings, {len(eval_ngr)} 8-grams")

    # load seeds (bio-overrefusal-v0.1 tier 1-4 legit)
    seeds = []
    for r in (json.loads(l) for l in open(SEED_FILE, encoding="utf-8") if l.strip()):
        if int(r.get("tier") or 0) in (1, 2, 3, 4):
            seeds.append(r["query"])
    rng.shuffle(seeds)
    seeds = seeds[:args.max_seeds]
    print(f"seeds: {len(seeds)} (from bio-overrefusal-v0.1)")

    import anthropic
    client = anthropic.Anthropic()
    all_gen = []
    for i, seed_q in enumerate(seeds):
        prompt = f"Seed question: {seed_q}\n\nGenerate {args.n_per_seed} varied legitimate bio research questions."
        try:
            resp = client.messages.create(
                model=args.model, max_tokens=2048,
                system=SYSTEM.format(n=args.n_per_seed),
                messages=[{"role": "user", "content": prompt}])
            text = resp.content[0].text.strip()
            # parse JSON array
            arr = json.loads(text)
            if isinstance(arr, list):
                all_gen.extend(arr)
            print(f"  [{i+1}/{len(seeds)}] got {len(arr)} from seed")
        except Exception as e:
            print(f"  [{i+1}/{len(seeds)}] ERR: {repr(e)[:100]}")
        if (i + 1) % 10 == 0:
            time.sleep(1)

    # dedupe
    seen, uniq = set(), []
    for s in all_gen:
        if not isinstance(s, str):
            continue
        k = norm(s)
        if k and k not in seen:
            seen.add(k)
            uniq.append(s)
    print(f"\ngenerated {len(all_gen)} -> {len(uniq)} unique")

    # decontaminate
    kept = []
    n_ex, n_ng = 0, 0
    for s in uniq:
        q = norm(s)
        if q in eval_exact:
            n_ex += 1
            continue
        if eval_ngr & grams(q.split()):
            n_ng += 1
            continue
        kept.append(s)
    print(f"decon dropped {n_ex} exact, {n_ng} 8-gram; kept {len(kept)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for s in kept:
            f.write(json.dumps({"query": s, "hard_label": 0, "soft_label": 0.0,
                                "source": "gen_bio_borderline_llm"}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(kept)} LLM-rewritten bio-borderline-benign prompts")
    for s in kept[:5]:
        print("   e.g.", s[:110])


if __name__ == "__main__":
    main()

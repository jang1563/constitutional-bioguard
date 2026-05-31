#!/usr/bin/env python
"""Build the v8 training set — data-first, REUSE-ONLY (no new harmful generation).

Composition (V8_DESIGN.md §4, Option A resolved):
  1. WildGuardMix BROAD bio harvest (the new core, already-public, zero generation):
       bio harmful-RESPONSE positives (~658, label=1) + bio (1,0) harmful-prompt/
       safe-response (~409, label=0, teaches "don't flag on the prompt") + bio (0,0)
       benign (~444, label=0). Broad = bio keyword match on prompt OR response, or
       bio subcategory (F.4 matched the prompt only and missed most positives).
  2. FalseReject train (benign-but-sensitive, label=0): the over-refusal killer
       v1-v7 never had. Capped for an over-refusal-heavy mix.
  3. v4 B.4 non-bio harm block (label=0): teaches BIO selectivity (flag bio, not all CBRN).

v8 label = 1 IFF (response is harmful AND bio); else 0. A response-harm BIO classifier.

Leakage-clean: drops any item whose query matches a locked eval prompt (incl.
false_reject_TEST, wildguard_test, etc.). Split is grouped by query (no query in
both train and val). Over-refusal-heavy by design (counters v7's over-flagging).

Content-blind: prints ONLY counts / label-quadrant breakdowns / balance / leakage
stats — never query/response payload. Output data files are gitignored.

Out: data/processed/v8_train.jsonl, v8_val.jsonl, v8_provenance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BIO_SUBCAT = ["biological", "biosecurity", "bioterror", "bioweapon", "biothreat",
              "biology", "biotech", "cbrn", "wmd", "pathogen", "virus", "bacteria",
              "toxin", "chemical weapon"]
LOCKED_FILES = ["wildguard_test.jsonl", "or_bench_hard_1k.jsonl", "false_reject_test.jsonl",
                "xstest.jsonl", "simple_safety_bio.jsonl", "saladbench_cbrn.jsonl",
                "saladbench_cbrn_strict.jsonl", "alert_cbrn.jsonl", "alert_cbrn_strict.jsonl",
                "real_response_bio.jsonl"]  # #106 eval set -- never train on it


def norm(v):
    return str(v).lower() if isinstance(v, str) else None


def qhash(q):
    return hashlib.sha1((q or "").strip().lower().encode()).hexdigest()[:16]


def rhash(r):
    return hashlib.sha1((r or "").strip().encode()).hexdigest()[:16]


def is_bio(prompt, response, subcat, bio_kw):
    t = (str(prompt or "") + " " + str(response or "")).lower()
    if any(k in t for k in bio_kw):
        return True
    if subcat is not None:
        s = " ".join(map(str, subcat)).lower() if isinstance(subcat, list) else str(subcat).lower()
        if any(x in s for x in BIO_SUBCAT):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--falsereject-n", type=int, default=2000,
                    help="Cap on FalseReject benign hard-negatives (over-refusal lever)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="v8", help="Output prefix (e.g. v8b)")
    ap.add_argument("--beavertails", action="store_true",
                    help="Also harvest BeaverTails 330k bio positives (grow the bio class)")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED

    bio_kw = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]
    print(f"bio keywords: {len(bio_kw)}")

    # locked eval prompts (leakage exclusion)
    locked = set()
    for fn in LOCKED_FILES:
        fp = DATA_EXTERNAL / fn
        if fp.exists():
            for line in open(fp):
                r = json.loads(line)
                q = (r.get("query") or r.get("prompt", "")).strip()
                if q:
                    locked.add(qhash(q))
    print(f"locked eval prompts: {len(locked)}")

    items = []

    # ── 1) WildGuardMix broad bio harvest (the new core) ───────────────────
    from datasets import load_dataset
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    except Exception:
        ds = load_dataset("allenai/wildguardmix", split="train")
    wg = Counter()
    for r in ds:
        p = r.get("prompt", ""); resp = r.get("response", ""); sc = r.get("subcategory")
        if not is_bio(p, resp, sc, bio_kw):
            continue
        if qhash(p) in locked:
            wg["leak_excluded"] += 1
            continue
        ph = norm(r.get("prompt_harm_label")); rh = norm(r.get("response_harm_label"))
        if rh not in ("harmful", "unharmful"):
            wg["unlabeled_response"] += 1
            continue
        label = 1 if rh == "harmful" else 0
        items.append({"query": p, "response": resp, "label": label, "prompt_harm": ph,
                      "response_harm": rh, "bio": True, "source": "wildguardmix_bio"})
        wg[f"({ph},{rh})"] += 1
    print(f"WildGuardMix bio harvest: {dict(wg)}")

    # ── 2) FalseReject train — benign hard-negatives (label 0) ─────────────
    fr = []
    frp = DATA_EXTERNAL / "false_reject_train.jsonl"
    if frp.exists():
        for line in open(frp):
            r = json.loads(line)
            q = r.get("query", "")
            if qhash(q) in locked:
                continue
            resp = r.get("response", "")
            fr.append({"query": q, "response": resp, "label": 0, "prompt_harm": None,
                       "response_harm": "unharmful", "bio": is_bio(q, resp, None, bio_kw),
                       "source": "falsereject"})
        rng.shuffle(fr)
        fr = fr[:args.falsereject_n]
    else:
        print(f"WARN: {frp} missing — skipping FalseReject")
    items += fr
    print(f"FalseReject benign hard-neg: {len(fr)} (cap {args.falsereject_n})")

    # ── 3) v4 B.4 non-bio harm — selectivity negatives (label 0) ───────────
    nb = []
    v4p = DATA_EXTERNAL / "v4_splits/v4_augmentation.jsonl"
    if v4p.exists():
        for line in open(v4p):
            r = json.loads(line)
            if r.get("block") != "B.4":
                continue
            q = r.get("query", r.get("prompt", ""))
            if qhash(q) in locked:
                continue
            nb.append({"query": q, "response": r.get("response", ""), "label": 0,
                       "prompt_harm": None, "response_harm": "unharmful", "bio": False,
                       "source": "v4_b4_nonbio"})
    else:
        print(f"WARN: {v4p} missing — skipping v4 B.4")
    items += nb
    print(f"v4 B.4 non-bio selectivity neg: {len(nb)}")

    # ── 4) BeaverTails bio (grow the scarce bio positive class) ────────────
    bt = []
    if args.beavertails:
        # exclude the #106 real-response bio eval set by (query, response)
        eval_keys = set()
        rrb = DATA_EXTERNAL / "real_response_bio.jsonl"
        if rrb.exists():
            for line in open(rrb):
                r = json.loads(line)
                eval_keys.add((qhash(r.get("query", "")), rhash(r.get("response", ""))))
        from datasets import load_dataset
        try:
            bds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        except Exception as e:
            print(f"WARN: BeaverTails load failed ({e}) — skipping")
            bds = []
        for r in bds:
            q = r.get("prompt", r.get("query", "")); resp = r.get("response", "")
            if not is_bio(q, resp, None, bio_kw):
                continue
            if qhash(q) in locked or (qhash(q), rhash(resp)) in eval_keys:
                continue
            harmful = (r.get("is_safe") is False) or (int(r.get("label", 0)) == 1)
            bt.append({"query": q, "response": resp, "label": 1 if harmful else 0,
                       "prompt_harm": None,
                       "response_harm": "harmful" if harmful else "unharmful",
                       "bio": True, "source": "beavertails_bio"})
    items += bt
    print(f"BeaverTails bio: {len(bt)} ({sum(b['label'] for b in bt)} positive)")

    # ── dedup by (query, response) ─────────────────────────────────────────
    seen = set(); dedup = []
    for it in items:
        k = (qhash(it["query"]), rhash(it["response"]))
        if k in seen:
            continue
        seen.add(k); dedup.append(it)
    print(f"after dedup: {len(dedup)} (from {len(items)})")
    items = dedup

    # ── split grouped by query (no query in both splits) ───────────────────
    by_q = {}
    for it in items:
        by_q.setdefault(qhash(it["query"]), []).append(it)
    qkeys = list(by_q); rng.shuffle(qkeys)
    n_val = int(len(qkeys) * args.val_frac)
    val_q = set(qkeys[:n_val])
    train = [it for q in qkeys if q not in val_q for it in by_q[q]]
    val = [it for q in val_q for it in by_q[q]]

    # ── leakage / split asserts ────────────────────────────────────────────
    assert not any(qhash(it["query"]) in locked for it in items), "locked eval leak in train set"
    tr_q = set(qhash(it["query"]) for it in train)
    va_q = set(qhash(it["query"]) for it in val)
    assert tr_q.isdisjoint(va_q), "query overlap between train and val"

    # ── write ──────────────────────────────────────────────────────────────
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, rows in [(f"{args.out_prefix}_train", train), (f"{args.out_prefix}_val", val)]:
        with open(DATA_PROCESSED / f"{name}.jsonl", "w") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def stats(rows):
        pos = sum(r["label"] for r in rows)
        return {"n": len(rows), "pos": pos, "neg": len(rows) - pos,
                "pos_rate": round(pos / max(1, len(rows)), 3),
                "by_source": dict(Counter(r["source"] for r in rows)),
                "bio_items": sum(1 for r in rows if r.get("bio"))}

    prov = {"train": stats(train), "val": stats(val),
            "wildguard_harvest": dict(wg), "falsereject_n": len(fr), "v4_nonbio_n": len(nb),
            "beavertails_bio_n": len(bt), "beavertails_bio_pos": sum(b["label"] for b in bt),
            "leak_excluded_wg": wg.get("leak_excluded", 0), "seed": args.seed,
            "falsereject_cap": args.falsereject_n, "out_prefix": args.out_prefix,
            "note": "reuse-only; label=1 iff bio harmful response; (0,1) quadrant ~absent (deferred)"}
    json.dump(prov, open(DATA_PROCESSED / f"{args.out_prefix}_provenance.json", "w"), indent=2)

    print(f"\n=== {args.out_prefix} dataset built (reuse-only, zero generation) ===")
    print("train:", json.dumps(stats(train)))
    print("val  :", json.dumps(stats(val)))


if __name__ == "__main__":
    main()

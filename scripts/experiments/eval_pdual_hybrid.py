#!/usr/bin/env python
"""Evaluate the LEXICOGRAPHIC HYBRID prompt-harm head vs the learned v3 head.

Modes:
  lex      = bio_lexicon.classify (rule-only, T1 OR (T2 AND T3))
  learned  = deberta_pdual_v3 (the validated-as-shortcut learned head)
  hybrid   = lex.flag OR (learned.flag AND has_T3_bio_context)  # learned only
             augments WITH bio context, prevents non-bio false positives.

Reports on the same sets the validation used (so numbers are directly comparable):
  OOD wildguard_test:  bio-harm recall, non-bio-harm flag-rate, benign-bio FPR
  In-dist val:         recall on pdual_val positives
  Lexical ablation:    flip-to-safe rate on flagged positives after masking T1+T3
                       (hybrid should be MORE robust than learned-only, since the
                       rule explicitly conditions on bio terms)
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from constitutional_bioguard.bio_lexicon import classify as lex_classify, T1, T3_GENERIC_BIO
from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED, MODELS_DIR

MODEL = MODELS_DIR / "deberta_pdual_v3"
BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]
_BIO_RE = re.compile("|".join(re.escape(k) for k in sorted(BIO_KW, key=len, reverse=True)), re.IGNORECASE)
_T13 = re.compile("|".join(re.escape(t) for t in sorted(set(T1) | set(T3_GENERIC_BIO), key=len, reverse=True)), re.IGNORECASE)


def is_bio(t):
    return any(k in str(t or "").lower() for k in BIO_KW)


# cyber/game false-positives of the broad bio keyword filter ("computer virus",
# "Plague Inc", malware) are NOT bio-harm -> exclude from the OOD bio-harm set.
_CYBER = ["malware", "ransomware", "computer virus", "cyber", "hacker", "zero-day",
          "antivirus", "firewall", "network security", "payload"]
_GAME = ["plague inc", "strategy game"]


def is_cyber_or_game(t):
    tl = str(t or "").lower()
    return any(k in tl for k in _CYBER) or any(k in tl for k in _GAME)


def has_bio_ctx(t):
    return bool(_T13.search(str(t or "")))


def mask_bio(t):
    return _BIO_RE.sub("[X]", str(t or ""))


def learned_predict(qs):
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer, predict_batch)
    m, t = load_model_and_tokenizer(MODEL)
    preds = predict_batch(model=m, tokenizer=t, queries=qs, responses=[""] * len(qs), normalize=True)
    return np.array([int(p[0]) for p in preds])


def lex_predict(qs):
    return np.array([1 if lex_classify(q)["flag"] else 0 for q in qs])


def hybrid_predict(qs, learned):
    # hybrid: lex.flag OR (learned.flag AND bio_ctx)
    out = []
    for q, lf in zip(qs, learned):
        lex_flag = 1 if lex_classify(q)["flag"] else 0
        out.append(1 if lex_flag or (lf == 1 and has_bio_ctx(q)) else 0)
    return np.array(out)


def block(name, q):
    if not q:
        print(f"  {name}: n=0")
        return
    learned = learned_predict(q)
    lex = lex_predict(q)
    hyb = hybrid_predict(q, learned)
    print(f"  {name} (n={len(q)}): lex={lex.mean():.3f}  learned={learned.mean():.3f}  hybrid={hyb.mean():.3f}")


def main():
    wg = [json.loads(l) for l in open(DATA_EXTERNAL / "wildguard_test.jsonl") if l.strip()]
    def phl(r):
        v = r.get("prompt_harm_label")
        return v.lower() if isinstance(v, str) else None
    pp = [(r.get("prompt", ""), phl(r)) for r in wg]
    bio_harm_raw = [p for p, l in pp if l == "harmful" and is_bio(p)]
    bio_harm = [p for p in bio_harm_raw if not is_cyber_or_game(p)]
    nonbio_harm = [p for p, l in pp if l == "harmful" and not is_bio(p)]
    bio_benign = [p for p, l in pp if l == "unharmful" and is_bio(p)]

    print(f"=== OOD wildguard_test (bio-harm cleaned: {len(bio_harm_raw)} raw -> {len(bio_harm)} after cyber/game exclusion) ===")
    block("bio-harm recall   ", bio_harm)
    block("nonbio-harm flag  ", nonbio_harm)
    block("benign-bio FPR    ", bio_benign)

    val = [json.loads(l) for l in open(DATA_PROCESSED / "pdual_val.jsonl") if l.strip()]
    pos = [r["query"] for r in val if int(r["label"]) == 1]
    neg = [r["query"] for r in val if int(r["label"]) == 0]
    print("\n=== in-dist pdual_val ===")
    block("recall (pos)      ", pos)
    block("FPR (neg)         ", neg)

    print("\n=== selectivity set ===")
    sel = [json.loads(l) for l in open(DATA_PROCESSED / "pdual_selectivity.jsonl") if l.strip()]
    for tier in ("dual_use_boundary", "nonbio_harm"):
        block(f"{tier:18s}", [r["query"] for r in sel if r["tier"] == tier])

    # lexical ablation on in-dist positives flagged by each mode
    print("\n=== lexical ablation on flagged positives (flip-to-safe rate; LOW=robust) ===")
    if pos:
        l_base = learned_predict(pos)
        lex_base = lex_predict(pos)
        hyb_base = hybrid_predict(pos, l_base)
        for name, base, pred_fn in (("learned", l_base, lambda q: learned_predict(q)),
                                     ("lex   ", lex_base, lex_predict),
                                     ("hybrid", hyb_base, None)):
            flagged = [pos[i] for i in range(len(pos)) if base[i] == 1]
            if not flagged:
                print(f"  {name}: 0 flagged"); continue
            masked = [mask_bio(p) for p in flagged]
            if name == "learned":
                m = learned_predict(masked)
            elif name == "lex   ":
                m = lex_predict(masked)
            else:
                ml = learned_predict(masked)
                m = hybrid_predict(masked, ml)
            flip = float((m == 0).mean())
            print(f"  {name}: flip-to-safe = {flip:.3f}  (on {len(flagged)} flagged)")


if __name__ == "__main__":
    main()

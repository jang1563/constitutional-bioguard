#!/usr/bin/env python
"""EVAL-merge — the REAL over-refusal money metric: v8 models on JK's own sessions.

The merge with the koaug track is an EVAL merge, not a training merge (V8_DESIGN
§8.5). koaug's `ood_fpr.jsonl` is 531 real legit BIO items (134 = JK's actual
Claude Code / Codex sessions), all label=legitimate, HELD-OUT from this track's
`data/processed` training. Running v8b/v8c on it is the most application-relevant
over-refusal number we can produce and answers the Goodhart question directly:
does the classifier over-block REAL legitimate research?

Since every item is legitimate, any flag is a false positive -> flag-rate = FPR
= over-refusal. We stratify three ways:
  - overall (all bio legit)
  - session-log subset (session_logs_primary + session_logs_secondary) = the most real
  - substantive-response subset (real benign-bio responses, apples-to-apples
    with #106) vs withheld/short (tests prompt-harm leakage: flagging a scary
    query despite a refusal/empty response)

Leakage-clean: ood_fpr sources (overrefusal_api_context_cond, session_logs_primary,
ambiguity_casebook, biosafety_suite_*, session_logs_secondary) are disjoint from
data/processed (wildguardmix_bio, beavertails, pubmedqa, falsereject, v4_b4).

Content-blind: counts/rates only. Reads the koaug split READ-ONLY (absolute path);
writes only results/metrics/v8_overrefusal_realsessions_{model}.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from constitutional_bioguard.config import CONFIGS_DIR, METRICS_DIR, MODELS_DIR

# koaug track's held-out money metric (read-only; never copied into this repo)
OOD_FPR = Path(Path.home(), "constitutional_bioguard/data/splits/ood_fpr.jsonl")

MODEL_DIRS = {"v8c": MODELS_DIR / "deberta_bioguard_v8c",
              "v8b": MODELS_DIR / "deberta_bioguard_v8b",
              "v8": MODELS_DIR / "deberta_bioguard_v8",
              "v4": MODELS_DIR / "deberta_bioguard_v4_response_diverse"}

BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]
WITHHELD = re.compile(r"(withheld|redacted|placeholder)", re.IGNORECASE)


def is_bio(q, r):
    t = (str(q or "") + " " + str(r or "")).lower()
    return any(k in t for k in BIO_KW)


def substantive(resp):
    s = (resp or "").strip()
    return len(s) >= 40 and not WITHHELD.search(s)


def load():
    if not OOD_FPR.exists():
        print(f"FATAL: missing {OOD_FPR}"); sys.exit(1)
    rows = [json.loads(l) for l in open(OOD_FPR) if l.strip()]
    bio = []
    for r in rows:
        q = r.get("query", r.get("prompt", "")); resp = r.get("response", "")
        if not is_bio(q, resp):
            continue
        # all ood_fpr are legitimate; carry source + response-substance flags
        src = str(r.get("source", "?"))
        bio.append({"query": q, "response": resp, "source": src,
                    "is_session": src in ("session_logs_primary", "session_logs_secondary"),
                    "is_substantive": substantive(resp)})
    return bio


def predict(model_dir, queries, responses):
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer, predict_batch)
    m, t = load_model_and_tokenizer(model_dir)
    preds = predict_batch(model=m, tokenizer=t, queries=queries, responses=responses, normalize=True)
    import gc
    import torch
    del m, t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [int(p[0]) for p in preds]


def main():
    bio = load()
    src = Counter(it["source"] for it in bio)
    n_sess = sum(it["is_session"] for it in bio)
    n_sub = sum(it["is_substantive"] for it in bio)
    print(f"ood_fpr BIO money metric: n={len(bio)} (all legitimate) "
          f"session_logs={n_sess} substantive_resp={n_sub}")
    print(f"  sources: {dict(src.most_common())}")

    q = [it["query"] for it in bio]; resp = [it["response"] for it in bio]
    sess = np.array([it["is_session"] for it in bio])
    sub = np.array([it["is_substantive"] for it in bio])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for model in ("v8c", "v8b", "v8", "v4"):
        md = MODEL_DIRS[model]
        if not md.exists():
            print(f"{model}: missing {md}"); continue
        pred = np.array(predict(md, q, resp))

        def fpr(mask):
            return round(float(pred[mask].mean()), 4) if mask.any() else None

        overall = fpr(np.ones(len(pred), bool))
        out = {"n": len(pred), "over_refusal_fpr_overall": overall,
               "fpr_session_logs": fpr(sess), "n_session": int(sess.sum()),
               "fpr_substantive_resp": fpr(sub), "n_substantive": int(sub.sum()),
               "fpr_withheld_short": fpr(~sub), "n_withheld_short": int((~sub).sum())}
        json.dump(out, open(METRICS_DIR / f"v8_overrefusal_realsessions_{model}.json", "w"), indent=2)
        print(f"  {model}: over-refusal FPR overall={overall}  "
              f"session_logs={out['fpr_session_logs']} (n={out['n_session']})  "
              f"substantive={out['fpr_substantive_resp']} (n={out['n_substantive']})  "
              f"withheld/short={out['fpr_withheld_short']} (n={out['n_withheld_short']})")


if __name__ == "__main__":
    main()

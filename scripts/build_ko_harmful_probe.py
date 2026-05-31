#!/usr/bin/env python
"""Build a KO-harmful EVAL probe to test for a 'Korean => safe' shortcut.

NOT training data. The KO augmentation tilts the Korean slice toward benign (we add short
conversational benign). If that over-tilts, the model could under-flag Korean *harmful*
inputs — a trivial KO jailbreak. The held-out session data is all legitimate, so the
shortcut is otherwise untestable. This probe builds the missing test set: take a sample of
the ENGLISH harmful from ood_fnr.jsonl (which the model SHOULD flag, ~92% in English) and
machine-translate it to Korean. Then compare flag-rate EN vs KO on the SAME records
(eval_ko_harmful_probe.py). KO flag-rate ~= EN flag-rate => no shortcut.

Schema matches the aug sources so translate_ko_aug.py works unchanged: English stays in
query_en/response_en (preserved for the paired EN baseline), the translate step overwrites
query/response with Korean. Eval reads EN from *_en and KO from query/response.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
OOD_FNR = ROOT / "data" / "splits" / "ood_fnr.jsonl"
HANGUL = re.compile(r"[가-힣]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnr", default=str(OOD_FNR))
    ap.add_argument("--out", default=str(ROOT / "data" / "audit" / "ko_harmful_probe_source.jsonl"))
    ap.add_argument("-n", "--n-sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = [json.loads(l) for l in open(args.fnr, encoding="utf-8")]
    # Harmful, English (skip the 1 already-Korean), non-empty query.
    pool = [r for r in recs
            if r.get("binary_label") == "negative"
            and (r.get("query") or "").strip()
            and not HANGUL.search((r.get("query", "") or "") + (r.get("response", "") or ""))]
    rng.shuffle(pool)
    sample = pool[: args.n_sample]

    out = []
    for i, r in enumerate(sample):
        rec = dict(r)
        rec["orig_id"] = r.get("id")
        rec["id"] = f"{r.get('id', 'rec')}_koprobe{i}"
        rec["augmentation"] = "ko_mt_nllb_probe"
        rec["lang"] = "en"
        rec["query_en"] = (r.get("query", "") or "")
        rec["response_en"] = (r.get("response", "") or "")
        out.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_resp = sum(1 for r in out if (r.get("response_en") or "").strip())
    print(f"KO-harmful probe source: {len(out)} records -> {args.out}")
    print(f"  harmful English pool available: {len(pool)}")
    print(f"  with response (translate query+response): {n_resp}/{len(out)}")
    print(f"  by source: {dict(Counter(r.get('source') for r in out).most_common())}")
    import statistics as st
    ql = [len((r.get('query_en') or '').strip()) for r in out]
    rl = [len((r.get('response_en') or '').strip()) for r in out]
    print(f"  query-len median={st.median(ql):.0f}  response-len median={st.median(rl):.0f}")


if __name__ == "__main__":
    main()

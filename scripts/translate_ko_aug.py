#!/usr/bin/env python
"""Machine-translate the KO-augmentation source (English) into Korean with NLLB-200.

Reads data/aug/ko_aug_source.jsonl (query_en / response_en), writes the same records
with query/response replaced by Korean (kor_Hang). binary_label and all metadata are
preserved, so the output merges into train.jsonl with the identical schema.

Device: CUDA > MPS > CPU. Resumable — re-running skips records already in --out.
Open MT (NLLB-200), so this is reproducible and key-free. First-pass quality is enough
to teach the model "Korean is a language, judge it on content"; not a perfect translation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "aug" / "ko_aug_source.jsonl"
OUT = ROOT / "data" / "aug" / "ko_aug_translated.jsonl"


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-beams", type=int, default=4,
                    help="beam search width (1=greedy/fast smoke; 4=better KO quality)")
    ap.add_argument("--resp-char-cap", type=int, default=400,
                    help="translate only the first N chars of response_en (classifier uses 256)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = pick_device()
    print(f"[translate] device={device} model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    # Robust target-language BOS id across transformers versions.
    kor_id = tok.convert_tokens_to_ids("kor_Hang")
    assert kor_id is not None and kor_id != tok.unk_token_id, "kor_Hang token not found"

    records = [json.loads(l) for l in open(args.src, encoding="utf-8")]

    # Resume: skip ids already translated.
    done_ids = set()
    if Path(args.out).exists():
        for l in open(args.out, encoding="utf-8"):
            try:
                done_ids.add(json.loads(l).get("id"))
            except json.JSONDecodeError:
                pass
    todo = [r for r in records if r.get("id") not in done_ids]
    print(f"[translate] total={len(records)} done={len(done_ids)} todo={len(todo)}", flush=True)

    @torch.no_grad()
    def translate_batch(texts: list[str]) -> list[str]:
        # Empty strings translate to empty (skip the model).
        idx_nonempty = [i for i, t in enumerate(texts) if t.strip()]
        out = [""] * len(texts)
        if not idx_nonempty:
            return out
        sub = [texts[i] for i in idx_nonempty]
        enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_input_tokens).to(device)
        gen = model.generate(**enc, forced_bos_token_id=kor_id,
                             max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j, i in enumerate(idx_nonempty):
            out[i] = dec[j]
        return out

    t0 = time.time()
    n_written = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for bstart in range(0, len(todo), args.batch_size):
            batch = todo[bstart:bstart + args.batch_size]
            q_en = [(r.get("query_en", "") or "").strip() for r in batch]
            r_en = [(r.get("response_en", "") or "").strip()[:args.resp_char_cap] for r in batch]
            q_ko = translate_batch(q_en)
            r_ko = translate_batch(r_en)
            for r, qk, rk in zip(batch, q_ko, r_ko):
                rec = dict(r)
                rec["query"] = qk
                rec["response"] = rk
                rec["lang"] = "ko"
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_written += len(batch)
            done = len(done_ids) + n_written
            rate = n_written / max(1e-6, time.time() - t0)
            eta = (len(todo) - n_written) / max(1e-6, rate)
            print(f"  {done}/{len(records)}  ({rate:.1f} rec/s, eta {eta/60:.1f} min)", flush=True)

    print(f"[translate] done. wrote {n_written} new records to {args.out} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# Compare v8b (original) vs v8bd (debiased with FORTRESS safe responses) on:
#   (a) real_response_bio_large -- recall + over-refusal, per-source
#   (b) fortress_safe_responses -- held-out over-refusal, per-slice
# Reports both heads side-by-side so the over-refusal/recall tradeoff is explicit.
import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).parent.parent
V8B = ROOT / "models" / "deberta_bioguard_v8b"
V8BD = ROOT / "models" / "deberta_bioguard_v8bh"


def resolve(d):
    d = Path(d)
    return d if (d / "config.json").exists() else (d / "final")


@torch.no_grad()
def score(model_dir, queries, responses, device, bs=64):
    md = resolve(Path(model_dir))
    tok = AutoTokenizer.from_pretrained(str(md))
    m = AutoModelForSequenceClassification.from_pretrained(
        str(md), dtype=torch.float32).to(device).eval()
    out = []
    for i in range(0, len(queries), bs):
        qb, rb = queries[i:i + bs], responses[i:i + bs]
        enc = tok(qb, rb, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
        out += torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().tolist()
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (a) recall + over-refusal on real_response_bio_large (n=554, leakage-clean for v8b)
    rows_a = [json.loads(l) for l in open(ROOT / "data/external/real_response_bio_large.jsonl") if l.strip()]
    Qa = [r["query"] for r in rows_a]
    Ra = [r.get("response") or "" for r in rows_a]
    Ya = np.array([int(r["label"]) for r in rows_a])
    src_a = np.array([r["source"] for r in rows_a])
    print(f"(a) real_response_bio_large n={len(rows_a)} (harm {int((Ya==1).sum())} / benign {int((Ya==0).sum())})")
    p_b = score(V8B, Qa, Ra, device)
    p_d = score(V8BD, Qa, Ra, device)

    def stats(p, mask_pos, mask_neg):
        f = p >= 0.5
        rec = float(f[mask_pos].mean()) if mask_pos.sum() else float("nan")
        orr = float(f[mask_neg].mean()) if mask_neg.sum() else float("nan")
        return rec, orr

    print(f"\n  {'slice':<14}{'v8b rec':>10}{'v8bd rec':>10}{'v8b OR':>10}{'v8bd OR':>10}{'delta-OR':>10}")
    for tag, mask in [("all", np.ones(len(rows_a), bool)),
                      ("wildguard", src_a == "wildguard"),
                      ("beavertails", src_a == "beavertails"),
                      ("saferlhf", src_a == "saferlhf")]:
        bp, bo = stats(p_b, mask & (Ya == 1), mask & (Ya == 0))
        dp, do = stats(p_d, mask & (Ya == 1), mask & (Ya == 0))
        print(f"  {tag:<14}{bp:>10.3f}{dp:>10.3f}{bo:>10.3f}{do:>10.3f}{do-bo:>+10.3f}")

    # (b) held-out fresh safe responses (n=360, FORTRESS rollouts)
    rows_b = [json.loads(l) for l in open(ROOT / "data/external/fortress_safe_heldout.jsonl") if l.strip()]
    Qb = [r["query"] for r in rows_b]
    Rb = [r["response"] for r in rows_b]
    bio = np.array([r["bio"] for r in rows_b])
    src_b = np.array([r["src"] for r in rows_b])
    print(f"\n(b) fortress_safe_responses n={len(rows_b)} (all safe, label=0)")
    pp_b = score(V8B, Qb, Rb, device)
    pp_d = score(V8BD, Qb, Rb, device)
    print(f"  {'slice':<14}{'v8b OR':>10}{'v8bd OR':>10}{'delta':>10}")
    for tag, mask in [("all", np.ones(len(rows_b), bool)), ("bio", bio), ("non-bio", ~bio),
                      ("benign_resp", src_b == "benign_resp"),
                      ("adv_refusal", src_b == "adv_refusal")]:
        bo = float((pp_b[mask] >= 0.5).mean())
        do = float((pp_d[mask] >= 0.5).mean())
        print(f"  {tag:<14}{bo:>10.3f}{do:>10.3f}{do-bo:>+10.3f}")

    out = {"large_v8b": p_b.tolist(), "large_v8bd": p_d.tolist(), "large_labels": Ya.tolist(),
           "large_source": src_a.tolist(),
           "safe_v8b": pp_b.tolist(), "safe_v8bd": pp_d.tolist(),
           "safe_bio": bio.tolist(), "safe_src": src_b.tolist()}
    json.dump(out, open(ROOT / "results" / "v8bh_compare.json", "w"))
    print("\nsaved -> results/v8bh_compare.json")


if __name__ == "__main__":
    main()

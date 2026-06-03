#!/usr/bin/env python
"""Step 1 footprint pilot (clean, proven-trainer based): can a 184M DeBERTa-v3-base reach
the v7.C-aug2 teacher's bio recall when fine-tuned on the reuse bio pool? This tests the
JurEE / direct-encoder route (research-plan Step 1 fork 1b fallback) with the project's
PROVEN training recipe (plain WeightedTrainer + DataCollatorWithPadding, no custom collator
/ KL that NaN'd). Soft labels are available in distill_pool_labeled.jsonl for a later
proper-distillation refinement; this run uses the teacher's hard pseudo-labels (which the
teacher separates cleanly, 0.924 vs 0.064) for a robust feasibility read.

Train: data/processed/distill_pool.jsonl ({query, hard_label}).
Eval (held-out, leakage-disjoint): bio_clean_eval recall/over-refusal + bio-overrefusal-v0.1.
Teacher reference: recall 0.883 / bio-overrefusal 0.023 / AUROC 0.897.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
POOL = ROOT / "data" / "processed" / "distill_pool.jsonl"
POOL_SOFT = ROOT / "data" / "processed" / "distill_pool_labeled.jsonl"  # adds teacher soft_label


def eval_queries(model, tokenizer, queries, device, bs=64):
    import numpy as np
    import torch
    out = []
    for i in range(0, len(queries), bs):
        enc = tokenizer(queries[i:i + bs], max_length=512, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out += torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--output-dir", default=str(ROOT / "models" / "deberta_v7c_distill"))
    ap.add_argument("--soft", action="store_true",
                    help="HarmAug soft-label distillation (KL/soft-CE on teacher soft_label + hard CE)")
    ap.add_argument("--lam", type=float, default=0.5, help="hard-CE weight; (1-lam) on soft-CE")
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = POOL_SOFT if args.soft else POOL
    pool = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    n_harm = sum(r["hard_label"] for r in pool)
    mode = f"SOFT-label distill (lam={args.lam})" if args.soft else "hard-label CE"
    print(f"distill pool [{mode}]: {len(pool)} (harmful {n_harm} / benign {len(pool) - n_harm}) from {src.name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # transformers 5.x loads deberta-v3 in fp16 by default here -> fp16 forward NaNs the
    # disentangled attention. Force fp32 weights (pure-fp32 training is stable + fast enough).
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, dtype=torch.float32).to(device)
    print(f"model param dtype: {next(model.parameters()).dtype}")

    def to_rec(r):
        rec = {"text": r["query"], "labels": int(r["hard_label"])}
        if args.soft:
            rec["soft"] = float(r["soft_label"])
        return rec
    ds = Dataset.from_list([to_rec(r) for r in pool])
    ds = ds.map(lambda b: tokenizer(b["text"], max_length=512, truncation=True),
                batched=True, remove_columns=["text"])
    base_collator = DataCollatorWithPadding(tokenizer)

    def collator(features):
        soft = None
        if args.soft:
            soft = torch.tensor([f.pop("soft") for f in features], dtype=torch.float)
        batch = base_collator(features)
        if soft is not None:
            batch["soft"] = soft
        return batch

    # v8b proven convention: class0=benign=1.0, class1=harmful=1.5 (recall-favoring)
    w = torch.tensor([1.0, 1.5], dtype=torch.float).to(device)
    F = torch.nn.functional

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            soft = inputs.pop("soft", None)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # fp32 model -> fp32 logits (CE/softmax stable)
            ce = F.cross_entropy(logits, labels, weight=w)
            if soft is None:
                loss = ce
            else:
                # HarmAug: hard CE + soft-target CE (Hinton, no log-of-target -> NaN-safe)
                s = soft.clamp(1e-4, 1 - 1e-4)
                t = torch.stack([1.0 - s, s], dim=1)
                soft_ce = -(t * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
                loss = args.lam * ce + (1.0 - args.lam) * soft_ce
            return (loss, outputs) if return_outputs else loss

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=0.01, warmup_ratio=0.06, logging_steps=50,
        save_strategy="no", report_to="none", seed=42, remove_unused_columns=False,
        fp16=False, bf16=False)  # pure fp32: deberta-v3 is unconditionally stable, no GradScaler
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds,
                              processing_class=tokenizer, data_collator=collator)
    print("training student (DeBERTa-v3-base, weighted CE)...")
    trainer.train()
    model.eval()

    # DIAGNOSTIC: score the pool's OWN harmful/benign (known in-domain labels).
    # If pool-harmful p_harmful is LOW, the model learned inverted (training bug);
    # if HIGH, it learned in-domain and any bio_clean_eval failure is distributional.
    ph = [r["query"] for r in pool if r["hard_label"] == 1][:150]
    pb = [r["query"] for r in pool if r["hard_label"] == 0][:150]
    sh, sb = eval_queries(model, tokenizer, ph, device), eval_queries(model, tokenizer, pb, device)
    print(f"[POOL-SELF] harmful mean p_harmful={sh.mean():.3f} (>=0.5: {(sh>=0.5).mean():.2f}) | "
          f"benign mean p_harmful={sb.mean():.3f} (>=0.5: {(sb>=0.5).mean():.2f})")
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))

    bce = [json.loads(l) for l in open(ROOT / "data/processed/bio_clean_eval.jsonl",
                                       encoding="utf-8") if l.strip()]
    pos = [r["query"] for r in bce if int(r["label"]) == 1]
    neg = [r["query"] for r in bce if int(r["label"]) == 0]
    pp, pn = eval_queries(model, tokenizer, pos, device), eval_queries(model, tokenizer, neg, device)
    recall, overref = float((pp >= 0.5).mean()), float((pn >= 0.5).mean())

    bor = [json.loads(l) for l in open(ROOT / "data/bio_overrefusal_queries.jsonl",
                                       encoding="utf-8") if l.strip()]
    legit = [r["query"] for r in bor if int(r.get("tier") or 0) in (1, 2, 3, 4)]
    bor_or = float((eval_queries(model, tokenizer, legit, device) >= 0.5).mean())

    allv = np.concatenate([pp, pn]); o = allv.argsort(kind="mergesort")
    rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv) + 1)
    auroc = (rk[:len(pp)].sum() - len(pp) * (len(pp) + 1) / 2) / (len(pp) * len(pn))

    print("\n" + "=" * 60)
    print("FOOTPRINT PILOT RESULT (DeBERTa-v3-base 184M student @0.5)")
    print(f"  bio_clean_eval recall (120 harmful): {recall:.3f}  [teacher 0.883]")
    print(f"  bio_clean_eval over-refusal (881):   {overref:.3f}")
    print(f"  bio-overrefusal-v0.1 over-refusal:   {bor_or:.3f}  [teacher 0.023]")
    print(f"  bio_clean_eval AUROC:                {auroc:.4f}  [teacher 0.897]")
    print("=" * 60)
    fork = ("1a: footprint SOLVED (184M encoder reaches teacher recall)" if recall >= 0.80
            else "1b: degrades -> needs proper distillation (soft-CE) or TAID")
    print(f"  FORK: {fork}")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"recall": recall, "over_refusal_bce": overref, "bor_over_refusal": bor_or,
               "auroc": float(auroc), "teacher_recall": 0.883, "fork": fork},
              open(ROOT / "results" / "v7c_distill_pilot.json", "w"), indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""V6 F.4: v4 classifier head retrain (encoder FROZEN).

Per V6_DESIGN_v2.md Q3=(c) maximal scope.

Approach:
  1. Load v4 model
  2. Freeze all encoder parameters (deberta.*)
  3. Train ONLY classifier head (pooler.dense + classifier) on:
     - WildGuardMix train bio-filtered (469 items, paired refusal/compliance)
     - Optional: subset of v3 train.jsonl for stability
  4. Save as v4_head_refit

Critical safeguards (V6_DESIGN_v2 acceptance gates):
  - Bio selectivity must stay >= 4.0x on SaladBench
  - OR-Bench-Hard FPR must stay <= 5%
  - XSTest FPR must stay <= 2%

If post-training eval violates any gate, discard and keep v4.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from constitutional_bioguard.config import (
    DATA_EXTERNAL, DATA_PROCESSED, MODELS_DIR,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-v3-train", action="store_true",
                        help="Include subset of v3 train data for stability")
    parser.add_argument("--v3-train-subset", type=int, default=500,
                        help="Number of v3 train items to include (if flag set)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Higher LR than encoder fine-tune since only head trains")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unsafe-weight", type=float, default=1.5)
    args = parser.parse_args()

    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, get_scheduler,
    )
    from torch.utils.data import DataLoader, Dataset

    v4_dir = MODELS_DIR / "deberta_bioguard_v4_response_diverse"
    out_dir = MODELS_DIR / "deberta_bioguard_v4_head_refit"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading v4 from %s", v4_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(v4_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(v4_dir))
    model.to(device)

    # Freeze encoder: keep only pooler + classifier trainable
    n_total = 0
    n_trainable_before = 0
    n_trainable_after = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if param.requires_grad:
            n_trainable_before += param.numel()
        # Freeze if in encoder
        if not (name.startswith("pooler") or name.startswith("classifier")):
            param.requires_grad = False
        if param.requires_grad:
            n_trainable_after += param.numel()
    logger.info("Params: total=%d trainable=%d (was %d) — encoder frozen",
                n_total, n_trainable_after, n_trainable_before)
    logger.info("Trainable param names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("  %s (%d)", name, param.numel())

    # Load training data
    wg_bio = load_jsonl(DATA_EXTERNAL / "wildguard_mix_train_bio.jsonl")
    logger.info("WildGuardMix bio items: %d", len(wg_bio))

    if args.include_v3_train:
        v3_train = load_jsonl(DATA_PROCESSED / "train.jsonl")
        # Subset (stratified by label)
        pos = [r for r in v3_train if int(r.get("label", 0)) == 1]
        neg = [r for r in v3_train if int(r.get("label", 0)) == 0]
        import random
        random.seed(42)
        random.shuffle(pos); random.shuffle(neg)
        n_each = args.v3_train_subset // 2
        v3_subset = pos[:n_each] + neg[:n_each]
        random.shuffle(v3_subset)
        logger.info("v3 train subset: %d items (%d pos + %d neg)",
                    len(v3_subset), len(pos[:n_each]), len(neg[:n_each]))
        all_data = wg_bio + v3_subset
    else:
        all_data = wg_bio

    # Train/val split (90/10)
    import random
    random.seed(42)
    random.shuffle(all_data)
    n_val = max(1, len(all_data) // 10)
    val_data = all_data[:n_val]
    train_data = all_data[n_val:]
    logger.info("Train: %d items, Val: %d items", len(train_data), len(val_data))
    logger.info("Train label distribution: %s",
                {k: sum(1 for r in train_data if r.get("label") == k) for k in [0, 1]})

    class PairDataset(Dataset):
        def __init__(self, rows, tok, max_len=512):
            self.rows = rows
            self.tok = tok
            self.max_len = max_len

        def __len__(self): return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            enc = self.tok(
                r["query"], r["response"],
                padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(r.get("label", 0)), dtype=torch.long),
            }

    train_ds = PairDataset(train_data, tokenizer)
    val_ds = PairDataset(val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Class weights
    class_weights = torch.tensor([1.0, args.unsafe_weight], dtype=torch.float32, device=device)
    logger.info("Class weights: SAFE=1.0, UNSAFE=%.2f", args.unsafe_weight)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    logger.info("=== Training head-only ===")
    logger.info("Steps: %d (epochs=%d × steps_per_epoch=%d)",
                num_training_steps, args.epochs, len(train_loader))

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches

        # Validation
        model.eval()
        n_correct = 0; n_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                n_correct += (preds == labels).sum().item()
                n_total += labels.size(0)
        val_acc = n_correct / max(n_total, 1)
        logger.info("Epoch %d: train_loss=%.4f val_acc=%.4f", epoch, avg_loss, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            logger.info("  Saved best to %s", out_dir)

    provenance = {
        "version": "v4_head_refit",
        "base_model": str(v4_dir),
        "method": "Encoder FROZEN, classifier head only retrained",
        "training_data": {
            "wildguard_mix_bio_n": len(wg_bio),
            "v3_train_included": args.include_v3_train,
            "v3_train_subset_n": (args.v3_train_subset if args.include_v3_train else 0),
            "train_n": len(train_data),
            "val_n": len(val_data),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "unsafe_weight": args.unsafe_weight,
        },
        "best_val_accuracy": round(best_val_acc, 4),
        "n_trainable_params": n_trainable_after,
        "n_total_params": n_total,
    }
    with open(out_dir / "v6_f4_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved provenance. Best val acc: %.4f", best_val_acc)


if __name__ == "__main__":
    main()

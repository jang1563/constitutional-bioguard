#!/usr/bin/env python
"""V7.A: Train classifier on BioClinical-ModernBERT-large.

396M encoder pretrained on 53.5B biomedical+clinical tokens (Sounack 2025,
arXiv:2506.10896). Drop-in replacement for DeBERTa-v3-base in our pipeline.

Key differences from v4 training:
  - base_model: thomas-sounack/BioClinical-ModernBERT-large (vs microsoft/deberta-v3-base)
  - max_seq_length: 4096 (vs 512) — leverage long context
  - learning_rate: 1e-5 (vs 2e-5) — larger model, smaller LR
  - batch_size: 8 with grad_accumulation 2 (vs 16) — memory budget

Training data (per V7_DESIGN.md):
  - v4 train.jsonl (3,062 items)
  - WildGuardMix bio (469 items, F.4 cache)
  - Meng/Zhang Biosecurity Agent: SKIPPED (paper has no public release)

Output: models/deberta_bioguard_v7a_bioclinical_modernbert_large/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_train_data(out_path: Path) -> dict:
    """Merge v4 train + WildGuardMix bio + (Meng/Zhang skip)."""
    from constitutional_bioguard.config import DATA_EXTERNAL, DATA_PROCESSED

    sources = [
        ("v4_train", DATA_PROCESSED / "train.jsonl"),
        ("wildguard_mix_bio", DATA_EXTERNAL / "wildguard_mix_train_bio.jsonl"),
    ]

    rows_out = []
    seen = set()
    by_source = {}
    for name, fp in sources:
        n_pos = 0; n_neg = 0; n_dup = 0
        for r in load_jsonl(fp):
            q = (r.get("query") or "").strip()
            resp = (r.get("response") or "").strip()
            label = int(r.get("label", 0))
            if not q and not resp:
                continue
            key = (q, resp, label)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            rows_out.append({
                "query": q, "response": resp, "label": label,
                "source": r.get("source", name),
            })
            if label == 1:
                n_pos += 1
            else:
                n_neg += 1
        by_source[name] = {"n_pos": n_pos, "n_neg": n_neg, "dup": n_dup}
        logger.info("  %s: %d pos / %d neg (dup %d)", name, n_pos, n_neg, n_dup)

    with open(out_path, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_pos = sum(s["n_pos"] for s in by_source.values())
    total_neg = sum(s["n_neg"] for s in by_source.values())
    logger.info("Total: %d (pos=%d, neg=%d)", len(rows_out), total_pos, total_neg)
    return {
        "total": len(rows_out), "n_pos": total_pos, "n_neg": total_neg,
        "by_source": by_source, "output_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--unsafe-weight", type=float, default=1.5)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )
    from constitutional_bioguard.config import (
        DATA_PROCESSED, MODELS_DIR, RESULTS_DIR,
    )
    from constitutional_bioguard.training.train_deberta import (
        load_dataset as load_hf_dataset, compute_metrics,
    )

    BASE_MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
    output_dir = MODELS_DIR / "deberta_bioguard_v7a_bioclinical_modernbert_large"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("v7.A BioClinical-ModernBERT-large training")
    logger.info("  base: %s", BASE_MODEL)
    logger.info("  max_seq: %d, batch: %d × grad_accum %d, lr: %.0e",
                args.max_seq, args.batch_size, args.grad_accum, args.lr)
    logger.info("=" * 60)

    # 1. Merge train data
    merged_train_path = DATA_PROCESSED / "v7a_merged_train.jsonl"
    merge_info = merge_train_data(merged_train_path)

    # 2. Tokenizer + Model
    logger.info("Loading %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "SAFE", 1: "UNSAFE"},
        label2id={"SAFE": 0, "UNSAFE": 1},
        problem_type="single_label_classification",
        torch_dtype=torch.float32,
    )
    logger.info("  Params: %.1fM, hidden=%d, layers=%d, max_pos=%d",
                sum(p.numel() for p in model.parameters()) / 1e6,
                model.config.hidden_size, model.config.num_hidden_layers,
                model.config.max_position_embeddings)

    # 3. Load datasets
    val_file = DATA_PROCESSED / "val.jsonl"
    train_ds = load_hf_dataset(merged_train_path)
    val_ds = load_hf_dataset(val_file)
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    def tokenize_fn(batch):
        return tokenizer(
            batch["query"], batch["response"],
            padding="max_length", truncation=True,
            max_length=args.max_seq,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True,
                              remove_columns=["query", "response"])
    val_ds = val_ds.map(tokenize_fn, batched=True,
                          remove_columns=["query", "response"])

    keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # 4. Class weights
    n_pos = merge_info["n_pos"]; n_neg = merge_info["n_neg"]
    total = n_pos + n_neg
    safe_w = round(total / (2 * n_neg), 4) if n_neg > 0 else 1.0
    class_weights = torch.tensor([safe_w, args.unsafe_weight], dtype=torch.float32)
    logger.info("Class weights: SAFE=%.4f, UNSAFE=%.2f", safe_w, args.unsafe_weight)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            cw = class_weights.to(device=outputs.logits.device, dtype=outputs.logits.dtype)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=cw)
            return (loss, outputs) if return_outputs else loss

    # 5. Training args
    log_dir = RESULTS_DIR / "training_logs"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),  # ModernBERT works well with bf16
        seed=42,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=str(log_dir),
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,  # Memory budget for 4K seq + 396M model
    )

    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Train metrics: %s", train_result.metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    eval_metrics = trainer.evaluate()
    logger.info("Final eval: %s", eval_metrics)

    provenance = {
        "version": "v7.A",
        "base_model": BASE_MODEL,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "max_seq_length": args.max_seq,
        "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "unsafe_weight": args.unsafe_weight,
        "merge_info": merge_info,
        "training_metrics": {k: float(v) if hasattr(v, "item") else v
                              for k, v in train_result.metrics.items()},
        "eval_metrics": {k: float(v) if hasattr(v, "item") else v
                          for k, v in eval_metrics.items()},
    }
    with open(output_dir / "v7a_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info("Done. Best F1: %.4f", eval_metrics.get("eval_f1", 0))


if __name__ == "__main__":
    main()

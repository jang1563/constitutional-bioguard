#!/usr/bin/env python
"""Train v5 with PairCFR contrastive loss + SPLICE projector.

Differences vs train_v5_baseline.py:
  - Uses PairCFRTrainer with same-batch quadruplet sampling
  - Preserves v5_quad_query_id field through tokenization
  - lambda=0.3 for contrastive component (Qiu et al. ACL 2024 recommended)
  - SPLICE projector fitting happens in a separate post-hoc script

Usage:
    python scripts/train_v5.py [--paircfr-lambda 0.3] [--unsafe-weight 1.5]
    python scripts/train_v5.py --paircfr-lambda 0.10 --output-dir models/deberta_bioguard_v5b_l010
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


def merge_jsonl_preserving_quad_id(
    *paths, output_path
) -> dict:
    """Merge JSONL files; preserve v5_quad_query_id (0 for non-quadruplet items)."""
    total_pos = 0
    total_neg = 0
    by_source = {}
    rows_out = []
    seen = set()
    for path in paths:
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        n_pos_src = 0
        n_neg_src = 0
        n_dup = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                query = (r.get("query") or "").strip()
                response = (r.get("response") or "").strip()
                if not query and "text" in r:
                    query = r["text"].strip()
                    response = ""
                label = int(r.get("label", 0))
                source = r.get("source", path.stem)
                quad_id = int(r.get("v5_quad_query_id", 0) or 0)
                if not query and not response:
                    continue
                key = (query, response, label)
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                rows_out.append({
                    "query": query,
                    "response": response,
                    "label": label,
                    "source": source,
                    "v5_quad_query_id": quad_id,
                })
                if label == 1:
                    n_pos_src += 1
                    total_pos += 1
                else:
                    n_neg_src += 1
                    total_neg += 1
        by_source[path.name] = {
            "n_pos": n_pos_src, "n_neg": n_neg_src,
            "total": n_pos_src + n_neg_src, "duplicates_dropped": n_dup,
        }
        logger.info("  Loaded %s: %d pos / %d neg (dropped %d)",
                    path.name, n_pos_src, n_neg_src, n_dup)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Merged: %d total (%d pos / %d neg) -> %s",
                len(rows_out), total_pos, total_neg, output_path)
    return {
        "total": len(rows_out), "n_positive": total_pos, "n_negative": total_neg,
        "by_source": by_source, "output_path": str(output_path),
    }


def save_class_weights(pos_count, neg_count, unsafe_weight, output_path):
    total = pos_count + neg_count
    auto_safe_weight = round(total / (2 * neg_count), 4) if neg_count > 0 else 1.0
    weights = {"0": auto_safe_weight, "1": round(unsafe_weight, 4)}
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Class weights: SAFE=%.4f UNSAFE=%.4f -> %s",
                weights["0"], weights["1"], output_path)
    return weights


def train_v5_paircfr(
    train_file, val_file, output_dir,
    class_weights_file, paircfr_lambda, paircfr_temperature
):
    """v5 training loop with PairCFRTrainer."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
    )

    from constitutional_bioguard.config import DATA_PROCESSED, RESULTS_DIR
    from constitutional_bioguard.training.paircfr_trainer import (
        PairCFRBatchSampler,
        PairCFRTrainer,
    )
    from constitutional_bioguard.training.train_deberta import (
        compute_metrics,
        load_training_config,
    )
    from constitutional_bioguard.training.train_deberta import (
        load_dataset as load_jsonl_dataset,
    )

    config = load_training_config()
    config["data"]["class_weights"] = True
    config["data"]["class_weights_file"] = str(class_weights_file.name)
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]
    train_cfg = config["training"]
    log_dir = RESULTS_DIR / "training_logs"

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading datasets...")
    train_dataset = load_jsonl_dataset(train_file)
    val_dataset = load_jsonl_dataset(val_file)
    logger.info("Train: %d  Val: %d", len(train_dataset), len(val_dataset))

    def tokenize_fn(batch):
        return tokenizer(
            batch["query"], batch["response"],
            padding="max_length", truncation=True, max_length=max_seq_length,
        )

    train_dataset = train_dataset.map(
        tokenize_fn, batched=True, remove_columns=["query", "response"],
    )
    val_dataset = val_dataset.map(
        tokenize_fn, batched=True, remove_columns=["query", "response"],
    )

    # KEY DIFF vs train_deberta: preserve v5_quad_query_id in train, drop it in val
    keep_train = {"input_ids", "attention_mask", "token_type_ids", "label", "v5_quad_query_id"}
    keep_val = {"input_ids", "attention_mask", "token_type_ids", "label"}
    train_meta = [c for c in train_dataset.column_names if c not in keep_train]
    val_meta = [c for c in val_dataset.column_names if c not in keep_val]
    train_dataset = train_dataset.remove_columns(train_meta)
    val_dataset = val_dataset.remove_columns(val_meta)
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Build same-batch quadruplet sampler.
    # In newer `datasets` versions, indexing returns a Column object, not a tensor;
    # convert via list() to get plain Python ints.
    quad_ids_col = train_dataset["v5_quad_query_id"]
    if hasattr(quad_ids_col, "tolist"):
        quad_ids_list = quad_ids_col.tolist()
    else:
        quad_ids_list = list(quad_ids_col)
    # Each element should be int; coerce
    quad_ids_list = [int(x) for x in quad_ids_list]
    batch_size = train_cfg["per_device_train_batch_size"]
    sampler = PairCFRBatchSampler(quad_ids_list, batch_size=batch_size, seed=train_cfg["seed"])
    logger.info("PairCFR sampler: %d batches over %d items (batch_size=%d)",
                len(sampler), len(train_dataset), batch_size)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label={0: "SAFE", 1: "UNSAFE"},
        label2id={"SAFE": 0, "UNSAFE": 1},
        problem_type="single_label_classification",
        torch_dtype=torch.float32,
    )

    # Class weights (for CE component of PairCFR loss)
    if not class_weights_file.is_absolute():
        class_weights_file = DATA_PROCESSED / class_weights_file
    with open(class_weights_file) as f:
        raw_weights = json.load(f)
    class_weights = torch.tensor(
        [raw_weights.get("0", 1.0), raw_weights.get("1", 1.0)],
        dtype=torch.float32,
    )
    logger.info("Class weights: %s", class_weights.tolist())

    # Override CE in model config via custom forward — easier path: monkey-patch
    # PairCFRTrainer doesn't directly use HF's weighted loss, so we wrap.
    class WeightedPairCFRTrainer(PairCFRTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            group_ids = inputs.pop("v5_quad_query_id", None)
            if group_ids is None:
                bsz = inputs.get("input_ids").size(0) if "input_ids" in inputs else 0
                group_ids = torch.zeros(bsz, dtype=torch.long, device=inputs["input_ids"].device)
            labels = inputs.get("labels")

            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            last_hidden = outputs.hidden_states[-1]
            cls_emb = last_hidden[:, 0, :]

            # Weighted CE
            weights = class_weights.to(device=logits.device, dtype=logits.dtype)
            ce_loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)

            from constitutional_bioguard.training.paircfr_trainer import paircfr_contrastive_loss
            contrastive_loss = paircfr_contrastive_loss(
                cls_emb, labels, group_ids, temperature=self.paircfr_temperature
            )

            total_loss = (
                (1.0 - self.paircfr_lambda) * ce_loss
                + self.paircfr_lambda * contrastive_loss
            )

            if self.state.global_step % 50 == 0:
                logger.info("step=%d ce=%.4f contrast=%.4f total=%.4f",
                            self.state.global_step, float(ce_loss),
                            float(contrastive_loss), float(total_loss))

            return (total_loss, outputs) if return_outputs else total_loss

        def _get_train_sampler(self, train_dataset=None):
            # Return our custom batch sampler. HF Trainer infers per-batch.
            # However HF expects a sequential sampler typically; we use BatchSampler instead.
            # Returning None makes HF use default sequential sampler. To inject batches,
            # we override get_train_dataloader.
            return None

        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=train_cfg.get("dataloader_num_workers", 0),
            )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=train_cfg.get("bf16", False),
        seed=train_cfg["seed"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=True,
        logging_dir=str(log_dir),
        report_to="none",
        optim=train_cfg.get("optim", "adamw_torch"),
        remove_unused_columns=False,  # keep v5_quad_query_id
    )

    callbacks = []
    patience = train_cfg.get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = WeightedPairCFRTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer, compute_metrics=compute_metrics,
        callbacks=callbacks,
        paircfr_lambda=paircfr_lambda,
        paircfr_temperature=paircfr_temperature,
    )

    logger.info("Starting v5 training (PairCFR lambda=%.2f temperature=%.2f)",
                paircfr_lambda, paircfr_temperature)
    train_result = trainer.train()
    logger.info("Training complete: %s", train_result.metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    eval_metrics = trainer.evaluate()
    logger.info("Final eval: %s", eval_metrics)

    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(
            {"train_metrics": train_result.metrics, "eval_metrics": eval_metrics},
            f, indent=2, default=str,
        )
    return {"train_metrics": train_result.metrics, "eval_metrics": eval_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsafe-weight", type=float, default=1.5)
    parser.add_argument("--paircfr-lambda", type=float, default=0.3)
    parser.add_argument("--paircfr-temperature", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Model output directory. Defaults to "
            "models/deberta_bioguard_v5. Use this for v5b lambda grids."
        ),
    )
    args = parser.parse_args()

    from constitutional_bioguard.config import (
        DATA_EXTERNAL,
        DATA_PROCESSED,
        MODELS_DIR,
    )

    original_train = DATA_PROCESSED / "train.jsonl"
    augmentation = DATA_EXTERNAL / "v5_splits" / "v5_augmentation.jsonl"
    val_file = DATA_PROCESSED / "val.jsonl"
    merged_train = DATA_PROCESSED / "v5_merged_train.jsonl"
    merged_weights = DATA_PROCESSED / "v5_class_weights.json"
    output_dir = args.output_dir or MODELS_DIR / "deberta_bioguard_v5"

    logger.info("=" * 60)
    logger.info("v5 TRAINING (PairCFR + SPLICE-ready)")
    logger.info("  PairCFR lambda: %.2f, temperature: %.2f",
                args.paircfr_lambda, args.paircfr_temperature)
    logger.info("  UNSAFE class weight: %.2f", args.unsafe_weight)
    logger.info("=" * 60)

    merge_info = merge_jsonl_preserving_quad_id(
        original_train, augmentation, output_path=merged_train,
    )
    weights = save_class_weights(
        merge_info["n_positive"], merge_info["n_negative"],
        args.unsafe_weight, merged_weights,
    )

    results = train_v5_paircfr(
        train_file=merged_train, val_file=val_file, output_dir=output_dir,
        class_weights_file=merged_weights,
        paircfr_lambda=args.paircfr_lambda,
        paircfr_temperature=args.paircfr_temperature,
    )

    provenance = {
        "version": "v5",
        "strategy": (
            "v5 augmentation + PairCFR contrastive loss; "
            "SPLICE projector to be fit post-hoc"
        ),
        "paircfr_lambda": args.paircfr_lambda,
        "paircfr_temperature": args.paircfr_temperature,
        "unsafe_weight": args.unsafe_weight,
        "merge_info": merge_info,
        "class_weights": weights,
        "training_results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "v5_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info("\n=== v5 training complete ===")
    return results


if __name__ == "__main__":
    main()

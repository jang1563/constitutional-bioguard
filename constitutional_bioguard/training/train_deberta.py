"""DeBERTa-v3-base fine-tuning for binary bio-safety classification.

Uses sentence-pair tokenization: tokenizer(query, response) when
query/response fields are available, falling back to single-text mode.
Output: Binary classification (0=SAFE, 1=UNSAFE).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constitutional_bioguard.config import (
    CONFIGS_DIR,
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def load_training_config() -> dict:
    """Load DeBERTa training config from YAML."""
    config_path = CONFIGS_DIR / "deberta_training.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(filepath: Path) -> "Dataset":
    """Load a JSONL file as a HuggingFace Dataset."""
    from datasets import Dataset

    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return Dataset.from_list(records)


def compute_metrics(eval_pred) -> dict:
    """Compute classification metrics for Trainer evaluation."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    logits_arr = np.array(logits)
    if np.isnan(logits_arr).any():
        logger.warning("NaN logits in compute_metrics — skipping this eval")
        return {"accuracy": 0.0, "f1": 0.0, "fpr": 1.0}
    probs_2d = torch.softmax(torch.tensor(logits_arr, dtype=torch.float32), dim=-1).numpy()
    probs = probs_2d[:, 1] if probs_2d.ndim > 1 else probs_2d
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    # AUROC only if both classes present
    if len(set(labels)) > 1:
        metrics["auroc"] = roc_auc_score(labels, probs)

    # False positive rate
    safe_mask = labels == 0
    if safe_mask.sum() > 0:
        metrics["fpr"] = preds[safe_mask].mean()

    return metrics


def train(
    train_file: Optional[Path] = None,
    val_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config_override: Optional[dict] = None,
) -> dict:
    """Fine-tune DeBERTa-v3-base on bio-safety classification data.

    Args:
        train_file: Path to train.jsonl.
        val_file: Path to val.jsonl.
        output_dir: Where to save the model.
        config_override: Optional dict to override YAML config values.

    Returns:
        Training results dict with metrics.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    # Load config
    config = load_training_config()
    if config_override:
        for section in config_override:
            if section in config:
                config[section].update(config_override[section])
            else:
                config[section] = config_override[section]

    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]
    train_cfg = config["training"]

    # Paths
    train_file = train_file or Path(config["data"]["train_file"])
    val_file = val_file or Path(config["data"]["val_file"])
    output_dir = output_dir or MODELS_DIR / "deberta_bioguard_v1"
    log_dir = RESULTS_DIR / "training_logs"

    # Make paths absolute relative to project root
    if not train_file.is_absolute():
        from constitutional_bioguard.config import PROJECT_ROOT
        train_file = PROJECT_ROOT / train_file
        val_file = PROJECT_ROOT / val_file

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset(train_file)
    val_dataset = load_dataset(val_file)
    logger.info("Train: %d examples, Val: %d examples",
                len(train_dataset), len(val_dataset))

    # Tokenize — prefer sentence-pair mode when query/response fields exist
    has_pairs = "query" in train_dataset.column_names and "response" in train_dataset.column_names

    def tokenize_fn(batch):
        if has_pairs:
            return tokenizer(
                batch["query"],
                batch["response"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
            )
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )

    cols_to_remove = ["text"]
    if has_pairs:
        cols_to_remove.extend(["query", "response"])
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    # Remove metadata columns (keep only input_ids, attention_mask, label)
    metadata_cols = [
        c for c in train_dataset.column_names
        if c not in ("input_ids", "attention_mask", "token_type_ids", "label")
    ]
    train_dataset = train_dataset.remove_columns(metadata_cols)
    val_dataset_clean = val_dataset.remove_columns(metadata_cols)

    train_dataset.set_format("torch")
    val_dataset_clean.set_format("torch")

    # Load model
    logger.info("Loading model: %s", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "SAFE", 1: "UNSAFE"},
        label2id={"SAFE": 0, "UNSAFE": 1},
        problem_type="single_label_classification",
        torch_dtype=torch.float32,
    )

    # Class weights
    class_weights = None
    if config["data"].get("class_weights", False):
        # Honor an explicit class_weights_file (e.g. for the WS-2 BoW-filtered
        # dataset, whose class balance differs from the full set); otherwise
        # fall back to the default full-set weights.
        weights_name = config["data"].get(
            "class_weights_file", "class_weights.json"
        )
        weights_file = Path(weights_name)
        if not weights_file.is_absolute():
            weights_file = DATA_PROCESSED / weights_file
        if weights_file.exists():
            with open(weights_file, encoding="utf-8") as f:
                raw_weights = json.load(f)
            class_weights = torch.tensor(
                [raw_weights.get("0", 1.0), raw_weights.get("1", 1.0)],
                dtype=torch.float32,
            )
            logger.info("Using class weights: %s", class_weights.tolist())

    # Custom Trainer with class-weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if class_weights is not None:
                weights = class_weights.to(device=logits.device, dtype=logits.dtype)
                loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fn = torch.nn.CrossEntropyLoss()

            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
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
    )

    # Callbacks
    callbacks = []
    patience = train_cfg.get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    # Train
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_clean,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # Save best model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Final evaluation on validation set
    eval_metrics = trainer.evaluate()
    logger.info("Final validation metrics: %s", eval_metrics)

    # Save metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "config": config,
            },
            f,
            indent=2,
            default=str,
        )

    return {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "model_dir": str(output_dir),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()

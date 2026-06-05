#!/usr/bin/env python
"""V7.B: QLoRA fine-tune Qwen3-4B as generative biosafety classifier.

Nemotron-style: input (system+user with BYO policy) → output (CoT + classification).
Uses 4-bit quantized base + LoRA rank 16 on all linear layers.

Training data: v7b_cot_train.jsonl (built by build_v7b_cot_data.py)
Output: models/qwen3_bioguard_v7b_cot/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train v7.B Qwen3-4B CoT classifier")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B",
                    help="HF model ID for base model")
    p.add_argument("--data-path", default=None,
                    help="Override training data jsonl (default: v7b_cot_train.jsonl)")
    p.add_argument("--output-dir", default=None,
                    help="Override output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR

    # ── Paths ────────────────────────────────────────────────────────────
    data_path = Path(args.data_path) if args.data_path else (
        DATA_PROCESSED / "v7b_cot_train.jsonl")
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.error("Run build_v7b_cot_data.py first")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (
        MODELS_DIR / "qwen3_bioguard_v7b_cot"
    )

    # ── Load data ────────────────────────────────────────────────────────
    rows = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    logger.info(f"Loaded {len(rows)} training items from {data_path}")

    # Convert to HF Dataset with "text" field for SFTTrainer
    # We'll use the chat template to format messages
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",  # SFTTrainer handles packing; right-pad OK for training
    )
    # Use a dedicated pad token to avoid eos_token confusion during generation
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Using eos_token as pad_token (no dedicated pad found)")

    # Format each row using chat template
    def format_row(row):
        messages = row["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    logger.info(f"Formatted dataset: {len(dataset)} items")

    # Check token length distribution
    sample_lens = []
    for i in range(min(100, len(dataset))):
        toks = tokenizer(dataset[i]["text"], return_length=True)
        sample_lens.append(toks["length"][0])
    avg_len = sum(sample_lens) / len(sample_lens)
    max_len = max(sample_lens)
    logger.info(f"Token lengths (sample 100): avg={avg_len:.0f}, max={max_len}")

    # ── Quantization config ──────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model ───────────────────────────────────────────────────────
    # Pick best available attention implementation
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    logger.info(f"Loading {args.base_model} with 4-bit quant, attn={attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ──────────────────────────────────────────────────────
    # Target all linear layers (Nemotron-style: comprehensive LoRA)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ── Training args ────────────────────────────────────────────────────
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * args.epochs
    logger.info(f"Effective batch={effective_batch}, steps/epoch={steps_per_epoch}, "
                f"total_steps={total_steps}")

    # trl >= 1.0 uses SFTConfig (replaces TrainingArguments + SFT-specific params)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(args.warmup_ratio * (len(dataset) // (args.batch_size * args.grad_accum)) * args.epochs),
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        report_to="none",
        dataloader_num_workers=2,
        # SFT-specific
        max_length=args.max_seq_length,
        packing=False,
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    result = trainer.train()
    logger.info(f"Training complete: {result.metrics}")

    # ── Save ─────────────────────────────────────────────────────────────
    # Save LoRA adapter
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Saved LoRA adapter to {adapter_dir}")

    # Save training config
    config = {
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "n_train_items": len(dataset),
        "trainable_params": trainable,
        "total_params": total,
        "metrics": result.metrics,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Done. Model at {output_dir}")


if __name__ == "__main__":
    main()

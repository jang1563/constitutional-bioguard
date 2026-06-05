#!/usr/bin/env python
"""V7.C: QLoRA fine-tune Llama-3.1-8B-Instruct as a direct biosafety classifier.

Design point distinct from v7.B (Qwen3-4B + CoT):
  - 8B base, matched scale to WildGuard 7B / LLaMA-Guard-3 8B (removes the
    "we're 38x smaller" framing).
  - NO chain-of-thought (WildGuard-style direct classifier). CoT was v7.B's
    documented root cause #3 for over-refusal (§2.4/§9.3), and Llama has no
    /no_think escape hatch, so we drop CoT at the data level.
  - Instruct (not base): base Llama-3.1-8B has no chat template, and §2.1 named
    base-vs-instruction-tuned as root cause #5 (pre-aligned refusal calibration).
    Instruct fixes both.
  - Dual-label target preserved: "Prompt harm: {ph}\nResponse harm: {rh}"
    (the §9.4 task-spec), supervised WITHOUT the CoT scaffolding.

Training data: v7c_nocot_train.jsonl (built by build_v7c_nocot_data.py, n=3531).
Output: models/llama31_8b_bioguard_v7c/
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


def parse_args():
    p = argparse.ArgumentParser(description="Train v7.C Llama-3.1-8B no-CoT classifier")
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HF model ID (Instruct: has chat template + pre-aligned)")
    p.add_argument("--data-path", default=None,
                   help="Override training data jsonl (default: v7c_nocot_train.jsonl)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device batch (2 for 8B @ seq2048 to fit 80GB safely)")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch = batch_size * grad_accum = 16 (== v7.B)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Standard QLoRA lr (QLoRA paper uses 2e-4 for 7-13B)")
    p.add_argument("--lora-rank", type=int, default=32,
                   help="Design doc §3: rank 32 for the 8B competitor")
    p.add_argument("--lora-alpha", type=int, default=64,
                   help="alpha/rank = 2, same ratio as v7.B (r16/a32)")
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
        DATA_PROCESSED / "v7c_nocot_train.jsonl")
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.error("Run build_v7c_nocot_data.py first")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (
        MODELS_DIR / "llama31_8b_bioguard_v7c"
    )

    # ── Load data ────────────────────────────────────────────────────────
    rows = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    logger.info(f"Loaded {len(rows)} training items from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",  # right-pad OK for training (packing=False)
    )

    # Fail fast if the base has no chat template (e.g. the non-Instruct base):
    # the data is stored as `messages` and relies on apply_chat_template.
    if tokenizer.chat_template is None:
        logger.error(f"{args.base_model} has NO chat_template. Use an Instruct "
                     f"model, or inject a template. Aborting.")
        sys.exit(1)

    # Dedicated pad token. Llama-3.1 ships <|finetune_right_pad_id|> exactly for
    # this; never reuse eos as pad (the model would learn to never stop).
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        vocab = tokenizer.get_vocab()
        for cand in ("<|finetune_right_pad_id|>", "<|reserved_special_token_0|>",
                     "<|endoftext|>"):
            if cand in vocab:
                tokenizer.pad_token = cand
                logger.info(f"pad_token set to {cand}")
                break
        else:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("No dedicated pad token found; falling back to eos_token")

    # Format each row using the chat template
    def format_row(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    logger.info(f"Formatted dataset: {len(dataset)} items")

    sample_lens = []
    for i in range(min(100, len(dataset))):
        toks = tokenizer(dataset[i]["text"], return_length=True)
        sample_lens.append(toks["length"][0])
    logger.info(f"Token lengths (sample 100): avg={sum(sample_lens)/len(sample_lens):.0f}, "
                f"max={max(sample_lens)}")

    # ── Quantization config ──────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

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
    logger.info(f"Effective batch={effective_batch}, steps/epoch={steps_per_epoch}, "
                f"total_steps={steps_per_epoch * args.epochs}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(args.warmup_ratio * steps_per_epoch * args.epochs),
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
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    result = trainer.train()
    logger.info(f"Training complete: {result.metrics}")

    # ── Save ─────────────────────────────────────────────────────────────
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Saved LoRA adapter to {adapter_dir}")

    config = {
        "base_model": args.base_model,
        "variant": "v7.C no-CoT direct classifier",
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

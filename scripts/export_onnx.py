"""Export Constitutional BioGuard DeBERTa model to ONNX format.

Usage:
    python scripts/export_onnx.py [--model-dir PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import MODELS_DIR
from constitutional_bioguard.evaluation.evaluate_classifier import (
    load_model_and_tokenizer,
)


def export_onnx(model_dir: Path, output_dir: Path) -> None:
    """Load the trained DeBERTa model and export it to ONNX."""
    print(f"Loading model from: {model_dir}")
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)

    # Check whether token_type_ids is used by the model
    # DeBERTa-v2 with type_vocab_size=0 does not use token_type_ids
    model_config = model.config
    use_token_type_ids = getattr(model_config, "type_vocab_size", 0) > 0
    print(f"type_vocab_size={getattr(model_config, 'type_vocab_size', 0)} → "
          f"use_token_type_ids={use_token_type_ids}")

    # Prepare a dummy input for tracing
    dummy_text = ["This is a dummy input for ONNX export tracing."]
    encoding = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    if use_token_type_ids and "token_type_ids" in encoding:
        token_type_ids = encoding["token_type_ids"].to(device)
        dummy_inputs = (input_ids, attention_mask, token_type_ids)
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }
    else:
        dummy_inputs = (input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print(f"Exporting to ONNX: {onnx_path}")

    # Wrap forward to accept positional tuple when token_type_ids is omitted
    class ModelWrapper(torch.nn.Module):
        def __init__(self, inner_model: torch.nn.Module, with_tti: bool) -> None:
            super().__init__()
            self.inner = inner_model
            self.with_tti = with_tti

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.with_tti and token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = self.inner(**kwargs)
            return outputs.logits

    wrapper = ModelWrapper(model, use_token_type_ids)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            opset_version=14,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    print(f"ONNX export complete: {onnx_path}")

    # ── Validation ───────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping validation. "
              "Install with: pip install onnxruntime")
        _copy_tokenizer_files(model_dir, output_dir)
        return

    test_text = "How do I enhance the transmissibility of influenza H5N1?"
    test_encoding = tokenizer(
        [test_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # PyTorch logits
    with torch.no_grad():
        pt_inputs = {k: v.to(device) for k, v in test_encoding.items()
                     if k in input_names}
        pt_logits = wrapper(
            pt_inputs["input_ids"],
            pt_inputs["attention_mask"],
            pt_inputs.get("token_type_ids"),
        ).cpu().numpy()

    # ONNX logits
    sess = ort.InferenceSession(str(onnx_path))
    ort_inputs = {name: test_encoding[name].numpy() for name in input_names
                  if name in test_encoding}
    ort_logits = sess.run(["logits"], ort_inputs)[0]

    max_diff = float(np.abs(pt_logits - ort_logits).max())
    print(f"PyTorch logits : {pt_logits}")
    print(f"ONNX logits    : {ort_logits}")
    print(f"Max abs diff   : {max_diff:.2e}")
    assert max_diff < 1e-3, (
        f"ONNX validation failed: max diff {max_diff:.4e} exceeds 1e-3"
    )
    print("Validation PASSED.")

    # ── Throughput benchmark ─────────────────────────────────────────────────
    n_bench = 50
    bench_texts = [test_text] * n_bench

    bench_encoding = tokenizer(
        bench_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # PyTorch timing
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_b_inputs = {k: v.to(device) for k, v in bench_encoding.items()
                       if k in input_names}
        wrapper(
            pt_b_inputs["input_ids"],
            pt_b_inputs["attention_mask"],
            pt_b_inputs.get("token_type_ids"),
        )
    pt_elapsed = (time.perf_counter() - t0) * 1000

    # ONNX timing
    ort_bench_inputs = {name: bench_encoding[name].numpy() for name in input_names
                        if name in bench_encoding}
    t0 = time.perf_counter()
    sess.run(["logits"], ort_bench_inputs)
    ort_elapsed = (time.perf_counter() - t0) * 1000

    print(f"\nThroughput benchmark ({n_bench} inputs, batch=1):")
    print(f"  PyTorch  : {pt_elapsed:.1f} ms total  ({pt_elapsed / n_bench:.2f} ms/sample)")
    print(f"  ONNX     : {ort_elapsed:.1f} ms total  ({ort_elapsed / n_bench:.2f} ms/sample)")

    _copy_tokenizer_files(model_dir, output_dir)
    print(f"\nTokenizer files copied to: {output_dir}")
    print("Done.")


def _copy_tokenizer_files(model_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer files alongside the ONNX model."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "spm.model",
    ]
    for fname in tokenizer_files:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Constitutional BioGuard DeBERTa to ONNX"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "deberta_bioguard_v1",
        help="Path to the trained model directory (default: models/deberta_bioguard_v1/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "deberta_bioguard_v1_onnx",
        help="Output directory for ONNX model (default: models/deberta_bioguard_v1_onnx/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.model_dir, args.output_dir)

#!/usr/bin/env python3
"""Export trained model and dataset to HuggingFace Hub.

Usage:
    python scripts/export_to_hf.py --model --dataset --repo-prefix jang1563/bioguard
    python scripts/export_to_hf.py --model-only --repo jang1563/bioguard-deberta-v1
    python scripts/export_to_hf.py --dataset-only --repo jang1563/bioguard-dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from constitutional_bioguard.config import (
    CONSTITUTION_FILE,
    DATA_PROCESSED,
    METRICS_DIR,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def export_model(
    model_dir: Path,
    repo_id: str,
    private: bool = False,
) -> str:
    """Push trained model to HuggingFace Hub.

    Args:
        model_dir: Path to the trained model directory.
        repo_id: HuggingFace repo ID (e.g., "jang1563/bioguard-deberta-v1").
        private: Whether to create a private repo.

    Returns:
        URL of the uploaded model.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Create model card
    metrics_file = METRICS_DIR / "internal_evaluation.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    internal = metrics.get("internal_metrics", {})
    model_card = f"""---
library_name: transformers
tags:
  - safety
  - biosecurity
  - content-classification
  - constitutional-classifiers
  - deberta-v3
datasets:
  - custom
metrics:
  - f1
  - auroc
  - precision
  - recall
model-index:
  - name: BioGuard DeBERTa v1
    results:
      - task:
          type: text-classification
          name: Biological Dual-Use Content Classification
        metrics:
          - name: F1
            type: f1
            value: {internal.get('f1', 'N/A')}
          - name: AUROC
            type: auroc
            value: {internal.get('auroc', 'N/A')}
          - name: Precision
            type: precision
            value: {internal.get('precision', 'N/A')}
          - name: Recall
            type: recall
            value: {internal.get('recall', 'N/A')}
---

# BioGuard DeBERTa v1

Binary classifier for detecting unsafe biological dual-use content, built using
Anthropic's Constitutional Classifiers methodology.

## Model Description

- **Base model**: microsoft/deberta-v3-base
- **Task**: Binary classification (SAFE / UNSAFE)
- **Input format**: `[CLS] query [SEP] response [SEP]`
- **Training data**: ~4,500 synthetic examples generated from a 56-rule biosafety constitution

## Performance

| Metric | Value |
|--------|-------|
| F1 | {internal.get('f1', 'N/A')} |
| AUROC | {internal.get('auroc', 'N/A')} |
| Precision | {internal.get('precision', 'N/A')} |
| Recall | {internal.get('recall', 'N/A')} |
| Over-refusal FPR | 0.00% (325 benign queries) |
| Adversarial mean ASR | 9.79% (20 attack types; encoding 17.9%, character 8.9%, semantic 6.9%) |

## NSABB Categories

Covers all 7 National Science Advisory Board for Biosecurity categories:
enhance_harm, disrupt_immunity, confer_resistance, increase_stability,
alter_host_range, enhance_susceptibility, generate_reconstruct.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")

query = "How does CRISPR work?"
response = "CRISPR uses guide RNA..."
inputs = tokenizer(query, response, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    label = "UNSAFE" if probs[0, 1] > 0.5 else "SAFE"
    print(f"{{label}} (confidence: {{probs[0, 1]:.3f}})")
```

## License

MIT
"""

    # Write model card
    card_path = model_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)

    # Create repo if needed, then upload
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    url = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        create_pr=False,
    )

    logger.info("Model uploaded to: https://huggingface.co/%s", repo_id)
    return f"https://huggingface.co/{repo_id}"


def export_dataset(
    repo_id: str,
    private: bool = False,
) -> str:
    """Push dataset and constitution to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "jang1563/bioguard-dataset").
        private: Whether to create a private repo.

    Returns:
        URL of the uploaded dataset.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Create dataset card
    dataset_card = """---
task_categories:
  - text-classification
tags:
  - safety
  - biosecurity
  - constitutional-classifiers
  - synthetic-data
language:
  - en
  - es
  - fr
  - zh
  - ar
  - ru
size_categories:
  - 1K<n<10K
---

# BioGuard Dataset

Synthetic training data for biological dual-use content classification,
generated using Anthropic's Constitutional Classifiers methodology.

## Dataset Description

- **Source**: Generated via Claude API from a 56-rule biosafety constitution
- **Size**: ~4,500 examples (2,968 train / 635 val / 664 test)
- **Labels**: Binary (SAFE=0 / UNSAFE=1)
- **Format**: JSONL with fields: text, label, fine_label, category

## Constitution

The biosafety constitution (included as `biosafety_constitution.yaml`) defines
56 rules across all 7 NSABB dual-use research categories.

## Data Sources

- **Permitted examples**: Safe biological research queries and responses
- **Restricted examples**: Queries/responses crossing dual-use boundaries
- **Boundary examples**: Ambiguous cases near the permitted/restricted boundary
- **Benign examples**: Clearly safe biology queries for over-refusal calibration
- **Augmented examples**: Translations, jailbreak templates, formality variations

## License

MIT
"""

    # Create temp directory with dataset files
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write dataset card
        with open(tmpdir / "README.md", "w") as f:
            f.write(dataset_card)

        # Copy data files
        for split in ["train", "val", "test"]:
            src = DATA_PROCESSED / f"{split}.jsonl"
            if src.exists():
                shutil.copy2(src, tmpdir / f"{split}.jsonl")

        # Copy constitution
        if CONSTITUTION_FILE.exists():
            shutil.copy2(CONSTITUTION_FILE, tmpdir / "biosafety_constitution.yaml")

        # Upload
        url = api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            repo_type="dataset",
            create_pr=False,
            private=private,
        )

    logger.info("Dataset uploaded to: https://huggingface.co/datasets/%s", repo_id)
    return f"https://huggingface.co/datasets/{repo_id}"


def main():
    parser = argparse.ArgumentParser(description="Export to HuggingFace Hub")
    parser.add_argument(
        "--model", action="store_true",
        help="Export trained model",
    )
    parser.add_argument(
        "--dataset", action="store_true",
        help="Export dataset and constitution",
    )
    parser.add_argument(
        "--repo-prefix", type=str, default="jang1563/bioguard",
        help="HuggingFace repo prefix (default: jang1563/bioguard)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model directory (default: models/deberta_bioguard_v1/)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create private repos",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.model and not args.dataset:
        args.model = True
        args.dataset = True

    model_dir = Path(args.model_dir) if args.model_dir else MODELS_DIR / "deberta_bioguard_v1"

    if args.model:
        if not model_dir.exists():
            logger.error("Model directory not found: %s", model_dir)
            sys.exit(1)
        export_model(
            model_dir=model_dir,
            repo_id=f"{args.repo_prefix}-deberta-v1",
            private=args.private,
        )

    if args.dataset:
        export_dataset(
            repo_id=f"{args.repo_prefix}-dataset",
            private=args.private,
        )


if __name__ == "__main__":
    main()

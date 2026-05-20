# Model Variant Experiments

After the full data pipeline completes (see `scripts/run_full_pipeline.sh`), you can compare multiple model architectures to identify the best performer for Constitutional BioGuard.

## Quick Start

### Single Variant
```bash
python scripts/run_variant_experiment.py --variant deberta-large
```

### All Variants (in sequence)
```bash
python scripts/run_variant_experiment.py --all
```

### View Comparison Table
```bash
python scripts/run_variant_experiment.py --compare
```

## Variants

| Variant | Model | Purpose | Notes |
|---------|-------|---------|-------|
| **deberta-base** | `microsoft/deberta-v3-base` | Current baseline | 12 layers, ~110M params |
| **deberta-large** | `microsoft/deberta-v3-large` | Performance ceiling | 24 layers, ~350M params, 4× batch size with gradient accumulation |
| **mdeberta** | `microsoft/mdeberta-v3-base` | Multilingual biosafety | 12 layers, 110+ languages |
| **biomedbert** | `BiomedNLP-BiomedBERT-base` | Biomedical domain | Pre-trained on PubMed + PMC (1.5B papers) |
| **biolinkbert** | `michiyasunaga/BioLinkBERT-base` | Bio NLP + linked entities | Pre-trained on PubMed with link awareness |

## Evaluation Metrics

Each variant is evaluated on:

### Internal Test Set
- **F1** — Harmonic mean of precision & recall (primary metric)
- **AUROC** — Area under ROC curve (sensitivity to threshold)
- **Precision** — True positives / (true positives + false positives)
- **Recall** — True positives / (true positives + false negatives)
- **Accuracy** — Overall correctness
- **FPR** — False positive rate (proportion of safe content flagged)

### Adversarial Suite
- **Mean ASR** — Average attack success rate across 4 attack categories:
  - Character-level (typos, obfuscation)
  - Encoding (Base64, rot13, etc.)
  - Semantic (passive voice, negation, etc.)
  - Multilingual (language switching)

### Overrefusal Test
- **Overrefusal FPR** — False positive rate on benign/legitimate queries
- Used to evaluate if the model is too conservative

### Threshold Calibration
- **Optimal Threshold** — Decision boundary maximizing F1 on val set
- **Calibrated F1** — F1 score at optimal threshold

## Workflow

### 1. Run Full Pipeline
```bash
source ~/.api_keys && nohup bash scripts/run_full_pipeline.sh > pipeline.log 2>&1 &
# Estimated: 2–4 hours
```

Generates:
- Full synthetic data (1960+ examples across 56 rules)
- Stratified train/val/test splits
- Calibrated threshold for default model (deberta-base)
- Internal evaluation metrics

### 2. Train and Evaluate Variants
```bash
# Train all 5 variants (sequential; ~2–3 hours)
python scripts/run_variant_experiment.py --all

# Or single variant for quick iteration:
python scripts/run_variant_experiment.py --variant deberta-large
```

### 3. Review Results
```bash
# Comparison table
python scripts/run_variant_experiment.py --compare

# Or raw JSON
cat results/metrics/variant_comparison.json | jq '.'
```

## Training Details

### Shared Hyperparameters
All variants use:
- Learning rate: 2.0e-5
- Optimizer: AdamW
- LR schedule: Cosine with warmup (10%)
- Weight decay: 0.01
- Epochs: 5
- Early stopping: 2 epochs patience
- Eval strategy: per-epoch
- Metric: F1

### Per-Variant Overrides
- **deberta-large**: batch_size=4, gradient_accumulation_steps=8 (because large model doesn't fit)
- Others: batch_size=16

See `configs/variant_experiment.yaml` for all parameters.

## Interpreting Results

### Which Variant Should I Use?

**High F1, Low FPR** → Prefer deberta-large (if compute allows)
- Best safety performance
- Catches more dangerous content
- Low false positive rate on benign content

**F1 Similar to Baseline, Multilingual Benefit** → Consider mdeberta
- Global deployment scenario
- F1 may be lower than monolingual models
- Handles code-switching and non-English safety concerns

**F1 Similar, Domain-Specific Strength** → Consider biomedbert or biolinkbert
- If deploying in biomedical research context
- May better understand biomedical terminology
- Training data domain is highly relevant

**Speed/Cost Critical** → Stick with deberta-base
- Smallest, fastest
- Sufficient F1 for most use cases
- ~12M examples/second inference on CPU

### Overfitting Check
If val F1 >> test F1, overfitting occurred:
- Reduce epochs
- Increase weight decay
- Use dropout (check model config)

## Example: Run Single Variant

```bash
python scripts/run_variant_experiment.py --variant biomedbert

# Output:
# ============================================================
# Variant: biomedbert — Biomedical domain pre-training
# ============================================================
#   Training... (5 epochs, ~20 min)
#   ✓ F1: 0.92, AUROC: 0.96
#   ✓ Adversarial mean ASR: 12.3%
#   ✓ Overrefusal FPR: 3.1%
#   ✓ Optimal threshold: 0.58, F1: 0.93
```

## Example: Review All Results

```bash
$ python scripts/run_variant_experiment.py --compare

==========================================================================================
 Model Variant Comparison
==========================================================================================
Variant             F1      AUROC   Prec    Rec     FPR     Adv ASR   OR-FPR  Thresh
------------------------------------------------------------------------------------------
deberta-base       0.9100  0.95200 0.9150  0.9050  0.0300   9.79%     3.20%  0.61
deberta-large      0.9250  0.96100 0.9280  0.9220  0.0220   8.10%     2.80%  0.59
mdeberta           0.8950  0.94100 0.9000  0.8900  0.0450   11.2%     4.50%  0.62
biomedbert         0.9180  0.95800 0.9220  0.9140  0.0310   9.05%     3.10%  0.60
biolinkbert        0.9120  0.95400 0.9180  0.9060  0.0320   9.35%     3.25%  0.61
==========================================================================================
```

## Troubleshooting

### Model not found
```
FileNotFoundError: Model dir not found for deberta-large
```
→ Train variant first: `python scripts/run_variant_experiment.py --variant deberta-large`

### Evaluation fails with torch error
```
ModuleNotFoundError: No module named 'torch'
```
→ Use venv python: `PYTHON=.venv/bin/python3 python scripts/run_variant_experiment.py --all`

### GPU out of memory (OOM)
```
RuntimeError: CUDA out of memory
```
→ deberta-large on GPU: uses full batch (gradient_accumulation_steps=8 with batch=4). If OOM:
  - Reduce batch_size in `configs/variant_experiment.yaml`
  - Or use CPU: `CUDA_VISIBLE_DEVICES="" python ...`

### Results not updating
```
Saved comparison to results/metrics/variant_comparison.json
```
If results seem stale: check file timestamp and re-run variant:
```bash
python scripts/run_variant_experiment.py --variant NAME  # overwrites old results
python scripts/run_variant_experiment.py --compare       # shows merged results
```

## Advanced: Custom Variant

Add to `configs/variant_experiment.yaml`:

```yaml
variants:
  my-custom-model:
    name: "huggingface/some-model"
    max_seq_length: 512
    description: "Custom variant for testing"
    per_device_train_batch_size: 16
```

Then:
```bash
python scripts/run_variant_experiment.py --variant my-custom-model
```

## References

- DeBERTa-v3: He et al., arXiv:2111.09543
- mDeBERTa: Multilingual extension by Microsoft
- BiomedBERT: Gu et al., "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing"
- BioLinkBERT: Michiyasunaga et al., "LinkBERT: A Knowledgeable Language Model for Entity Linking"

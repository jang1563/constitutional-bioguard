# Constitutional BioGuard

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/jang1563/constitutional-bioguard-deberta-v1)

> **TL;DR.** Prototype biological dual-use content classifier built using Anthropic's [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) methodology. 56 biosafety rules across 7 NSABB categories drive synthetic data generation; DeBERTa-v3-base is fine-tuned to flag unsafe biological queries. Latest local full pipeline run reports held-out F1 = 0.9807, AUROC = 0.9980, over-refusal FPR = 0.90%, adversarial mean ASR = 0.00%. This is a domain-extension prototype, not a production-equivalent safeguard.

**Author:** JangKeun Kim, Weill Cornell Medicine (jak4013@med.cornell.edu)

## Release Status

| Surface | Status |
|---------|--------|
| Code | v0.1.0 prototype, MIT-licensed, public on GitHub |
| Model | `jang1563/constitutional-bioguard-deberta-v1` on Hugging Face |
| Constitution | 56 rules / 7 NSABB categories (`constitution/biosafety_constitution.yaml`) |
| External validation | BioThreat-Eval cross-walk reported in `results/`; kappa-gap explained in Limitations |
| Internal review | Solo author; expert circulation pending |
| Responsible-use scope | [`SAFETY.md`](SAFETY.md) |

### Latest Run Snapshot (2026-05-21)

- Internal eval: F1 0.980676, AUROC 0.997961, FPR 0.0090 on 643 samples (models/deberta_bioguard_v1)
- Calibration: optimal threshold = 0.10, best_score = 0.9852, n_val_samples = 697
- Adversarial suite: 20 attacks, mean ASR = 0.00%
- Over-refusal: FPR = 0.00% on 100-sample benign holdout
- External validation: cohen kappa = 0.414 (threat_level>=4), f1 = 0.5143

## Reviewer Framing

This repository is a **prototype** showing one concrete instantiation of the Constitutional Classifiers methodology applied to the biosafety domain. It is intended as a research artifact demonstrating: (a) how a domain constitution can be machine-readable, (b) how synthetic data can be generated against that constitution, (c) what calibration-vs-evasion trade-offs a small classifier exhibits when fine-tuned on this data. It is **not** equivalent to any production safety system, including Anthropic's deployed Constitutional Classifier pipeline.

## Results

| Metric | Value | Target |
|--------|-------|--------|
| F1 | 0.9807 | >= 0.90 |
| AUROC | 0.9980 | -- |
| Precision / Recall | 0.9951 / 0.9667 | -- |
| Over-refusal FPR | 0.00% (100 benign) | < 2% |
| Adversarial mean ASR | 0.00% (20 attacks) | < 15% |
| External kappa (TL>=4) | 0.414 | >= 0.80 |

Per-category F1 scores range from 0.952 (disrupt_immunity) to 1.000 (enhance_harm), with all 7 NSABB categories above 0.95.

The external kappa gap is a known artifact: the classifier correctly flags dangerous *queries* even when LLMs refuse to answer, while the external benchmark (BioThreat-Eval) uses response-based labeling that penalizes this conservative behavior.

## Limitations

**External kappa (0.414) falls short of the 0.80 target.** This is the most important limitation and warrants transparency:

The BioThreat-Eval external benchmark scores LLM *responses* on four dimensions (refusal rate, information completeness, operational specificity, safety awareness). Constitutional BioGuard classifies *queries* — it flags whether a question is dangerous regardless of how the downstream LLM responds. When the same dangerous query is sent to GPT-4 and Claude (which both refuse), BioThreat-Eval labels those responses as safe; our classifier labels the query as unsafe. This architectural mismatch accounts for the kappa gap.

Three validation strategies were tested: (1) `threat_level >= 4` with refusal correction (kappa = 0.414, best alignment), (2) `threat_level >= 3` (kappa = 0.310), and (3) response-based (kappa = 0.244). All are reported in `results/` for transparency. The internal metrics (F1 = 0.980 on held-out test set, 0.00% over-refusal FPR on 325 benign queries) reflect the classifier's performance on its intended task.

Additional limitations:
- Trained on Claude-generated synthetic data; real-world distribution shift is uncharacterized
- Encoding attacks (ROT13, URL-encode) achieve >30% ASR — text that evades tokenization is a fundamental weakness for any embedding-based classifier
- English-centric; multilingual coverage is limited to code-switching augmentation

## Responsible Use

This tool is designed to **detect** potentially dangerous biological content, not to generate it. The classifier, constitution, and training data should be used for:
- Content moderation in AI systems handling biological queries
- Research on dual-use content detection
- Biosafety policy and compliance tooling

Do not use this project to develop evasion strategies for biosafety systems, to identify gaps in existing safety tools for malicious purposes, or in contexts where a false negative could enable serious harm without appropriate human oversight. The 9.79% mean adversarial ASR means the classifier is not a complete defense on its own.

## Architecture

```
Constitution (56 rules, YAML)
    |
    v
Claude API synthetic generation
    |  15 permitted + 15 restricted + 5 boundary per rule
    |  + augmentation (translation, jailbreak, formality, prefills)
    |  + 325 benign biology queries
    v
~4,500 labeled examples (2968 train / 635 val / 664 test)
    |
    v
DeBERTa-v3-base binary classifier
    Input: "[CLS] query [SEP] response [SEP]"
    Output: SAFE (0) / UNSAFE (1)
```

## NSABB Categories

The constitution covers all 7 categories from the [National Science Advisory Board for Biosecurity](https://osp.od.nih.gov/biotechnology/nsabb/):

1. **Enhance Harm** (EH) -- Enhance the harmful consequences of a biological agent or toxin
2. **Disrupt Immunity** (DI) -- Disrupt immunity or the effectiveness of an immunization
3. **Confer Resistance** (CR) -- Confer resistance to clinically/agriculturally useful antimicrobials
4. **Increase Stability** (IS) -- Increase the stability, transmissibility, or dissemination
5. **Alter Host Range** (AH) -- Alter the host range or tropism
6. **Enhance Susceptibility** (ES) -- Enhance the susceptibility of a host population
7. **Generate/Reconstruct** (GR) -- Generate or reconstitute an eradicated or extinct agent

## Quick Start

### Prerequisites

- Python >= 3.10
- (Full pipeline only) Anthropic API key for synthetic data generation; inference is local

### Installation

```bash
pip install -e ".[dev]"
```

### No-API Smoke Test (~30 sec)

Validates constitution schema, taxonomy mappings, and Pydantic models. No network, no API keys, no model load required.

```bash
make validate
pytest tests/ -v
```

### Inference-Only Quickstart

Pull the trained classifier from Hugging Face and run a single batch. No Anthropic API key needed; no training pipeline needed.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("jang1563/constitutional-bioguard-deberta-v1")
model = AutoModelForSequenceClassification.from_pretrained("jang1563/constitutional-bioguard-deberta-v1").eval()

# Inputs are "[CLS] query [SEP] response [SEP]". For real test cases see
# `tests/fixtures/`; do not paste operational language into demo code.
texts = [
    "How does CRISPR-Cas9 work? [SEP] CRISPR uses guide RNA to target specific DNA sequences...",
    "<NSABB-flagged dual-use query placeholder> [SEP] <model response>",
]

with torch.no_grad():
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
for text, p in zip(texts, probs):
    label = "UNSAFE" if p[1] > 0.5 else "SAFE"
    print(f"{label} (p={p[1]:.3f}): {text[:60]}...")
```

### Full Pipeline: Data Regeneration & Variant Experiments

Regenerate all synthetic training data and compare model architectures across 5 variants (deberta-base, deberta-large, mdeberta, biomedbert, biolinkbert).

**Step 1: Run the complete pipeline** (2–4 hours)

```bash
source ~/.api_keys && nohup bash scripts/run_full_pipeline.sh > pipeline.log 2>&1 &
```

This runs all 6 steps automatically:
1. Generate synthetic data (1960+ examples from 56 rules)
2. Augment restricted/boundary examples
3. Generate benign queries
4. Prepare stratified train/val/test splits
5. Calibrate threshold on validation set
6. Evaluate baseline model

**Monitor progress:**
```bash
python scripts/monitor_pipeline.py --watch
```

**Step 2: Compare model variants** (2–3 hours, after pipeline completes)

```bash
python scripts/run_variant_experiment.py --all
```

See [`docs/VARIANT_EXPERIMENTS.md`](docs/VARIANT_EXPERIMENTS.md) for:
- Which variants to use and when
- Interpreting comparison results
- Troubleshooting tips

**Step 3: Post-pipeline workflow**

After completion, follow [`docs/POST_PIPELINE_CHECKLIST.md`](docs/POST_PIPELINE_CHECKLIST.md) to:
- Verify data integrity
- Review calibration results
- Run variant experiments
- Compare and select best model
- Export to HuggingFace (optional)

**Legacy: step-by-step commands**

If you prefer to run steps individually (not recommended):
```bash
# Validate
pytest tests/ -v

# Generate (steps 1-4 from full pipeline)
python scripts/run_pipeline.py --step generate-synthetic -v
python scripts/run_pipeline.py --step augment -v
python scripts/run_pipeline.py --step benign -v
python scripts/run_pipeline.py --step prepare -v

# Calibrate & evaluate (steps 5-6)
python scripts/run_pipeline.py --step calibrate -v
python scripts/run_pipeline.py --step evaluate -v
```

For complete variant infrastructure details, see [`docs/VARIANT_EXPERIMENTS.md`](docs/VARIANT_EXPERIMENTS.md).

## Project Structure

```
constitutional_bioguard/
├── constitution/
│   ├── biosafety_constitution.yaml   # 56 rules across 7 NSABB categories
│   └── schema.json                   # JSON Schema for rule validation
├── constitutional_bioguard/          # Python package
│   ├── config.py
│   ├── models.py                     # Pydantic data models
│   ├── taxonomy.py                   # NSABB category definitions
│   ├── generation/                   # Synthetic data pipeline
│   │   ├── llm_client.py             # Claude API wrapper with retries
│   │   ├── constitution_loader.py    # YAML parser
│   │   ├── synthetic_generator.py    # Constitution -> examples
│   │   ├── augmentor.py              # Translation, jailbreak, formality
│   │   └── benign_generator.py       # Benign biology queries
│   ├── training/
│   │   ├── prepare_data.py           # Stratified splits
│   │   └── train_deberta.py          # DeBERTa fine-tuning
│   └── evaluation/
│       ├── evaluate_classifier.py    # Precision/Recall/F1/AUROC
│       ├── external_validation.py    # BioThreat-Eval cross-validation
│       ├── adversarial_suite.py      # 20 attack types
│       ├── overrefusal_test.py       # Benign FPR measurement
│       └── figures.py                # All visualizations
├── data/                             # Generated data (gitignored)
├── models/                           # Trained checkpoints (gitignored)
├── results/                          # Metrics + figures
├── configs/                          # Training configs (YAML)
├── scripts/
│   ├── run_pipeline.py               # CLI orchestrator
│   ├── validate_constitution.py      # Constitution coverage checker
│   └── export_to_hf.py              # HuggingFace Hub upload
└── tests/
```

## Adversarial Robustness

The classifier is tested against 20 attack types across 4 categories:

| Category | Attacks | Mean ASR |
|----------|---------|----------|
| Character-level | homoglyphs, invisible chars, leetspeak, case swap, whitespace, typos, unicode norm | 8.9% |
| Encoding | base64, ROT13, hex, URL encode, backspace | 17.9% |
| Semantic | passive voice, euphemism, hypothetical, negation, question flip, context dilution | 6.9% |
| Multilingual | code-switching, mixed script | 1.0% |

Encoding attacks (especially ROT13 at 47.9%, URL-encode at 29.2%) and character-level perturbations (leetspeak 35.4%, case swap 27.1%) are the primary weaknesses. Semantic rewriting also shows vulnerability (passive voice and negation flip each at 20.8% ASR), indicating the classifier is partially sensitive to phrasing changes that preserve dual-use intent.

## What This Is Not

- Not a replacement for institutional biosafety review or BSL-2/3 oversight
- Not a wet-lab risk assessment tool
- Not a complete safeguard on its own (recommended as one layer in a defense-in-depth design alongside upstream policy, downstream model refusal training, and human review)
- Not validated on languages other than English (multilingual coverage limited to code-switching augmentation)
- Not a substitute for any vendor's production constitutional-classifier pipeline; this is a domain-extension prototype demonstrating the methodology applied to biology

## How This Maps to AI Safety Practice

This prototype illustrates **one** point in the safeguards stack: a domain-specific output classifier trained on a machine-readable constitution. It complements rather than replaces:

- **Capability evaluations** (e.g. WMDP, biothreat-eval): measure what a base model could enable. This classifier sits downstream of those, on the response path.
- **Over-refusal calibration** (e.g. [bio-overrefusal-v0.1](https://github.com/jang1563/bio-overrefusal-v0.1)): measures whether a deployed safeguard blocks legitimate research. This repository's `make overrefusal` target reports the same metric on the included benign set (0/325).
- **Boundary-case adjudication** (e.g. [ambiguity-casebook](https://github.com/jang1563/ambiguity-casebook)): documents where reasonable experts disagree. The 9.79% mean adversarial ASR here is one signal that boundary cases need human-in-the-loop, not classifier-only routing.

The 0.414 external kappa against BioThreat-Eval is **not** a "the classifier failed" finding; it is an architectural mismatch (query-level vs response-level labels) that surfaces a real design choice every safeguard team has to make. See Limitations for the full discussion.

## Cross-Project Integration

This classifier serves as the output safety filter in [AgentShield](../agentshield/), providing real-time content classification at ~5ms/query with no API cost. See AgentShield's `output_classifier.py` for integration details.

## Responsible Use Scope

See [`SAFETY.md`](SAFETY.md) for the public responsible-use scope, what is withheld, and how to report concerns.

## Citation

If you use this work, please cite:

```bibtex
@software{kim2026bioguard,
  author = {Kim, JangKeun},
  title  = {Constitutional BioGuard: A Biosafety Content Classifier},
  year   = {2026},
  url    = {https://github.com/jang1563/constitutional-bioguard},
  version = {v0.1.0},
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## License

MIT

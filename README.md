# Constitutional BioGuard

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-DeBERTa--v3--base-yellow)](https://huggingface.co/jang1563/constitutional-bioguard-deberta-v1)
[![F1=0.980](https://img.shields.io/badge/F1-0.980-brightgreen)](#results)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-orange)](CITATION.cff)

A biological dual-use content classifier built using Anthropic's [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) methodology. Defines a biosafety constitution (56 rules across 7 NSABB categories), generates synthetic training data via Claude API, and trains a DeBERTa-v3-base binary classifier to detect unsafe biological content.

**Author:** JangKeun Kim, Weill Cornell Medicine (jak4013@med.cornell.edu)
**Status:** Domain-extension prototype, not production-equivalent. See [Limitations](#limitations).

## Try It in 30 Seconds

The trained classifier is published on HuggingFace and can be used directly without cloning this repo or running any pipeline.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("jang1563/constitutional-bioguard-deberta-v1")
tokenizer = AutoTokenizer.from_pretrained("jang1563/constitutional-bioguard-deberta-v1")

text = "How does CRISPR-Cas9 work? [SEP] CRISPR uses guide RNA to target specific DNA sequences..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
label = outputs.logits.argmax(dim=-1).item()
print("UNSAFE" if label == 1 else "SAFE")
```

Inference is local (no API cost), ~5ms/query on CPU. ONNX export available for production deployment.

## Results

| Metric | Value | Target |
|--------|-------|--------|
| F1 | 0.980 | >= 0.90 |
| AUROC | 0.998 | -- |
| Precision / Recall | 0.984 / 0.976 | -- |
| Over-refusal FPR | 0.00% (325 benign) | < 2% |
| Adversarial mean ASR | 9.79% (20 attacks) | < 15% |
| External kappa (TL>=4) | 0.414 | >= 0.80 |

Per-category F1 scores range from 0.952 (disrupt_immunity) to 1.000 (enhance_harm), with all 7 NSABB categories above 0.95.

The external kappa gap is a known artifact: the classifier correctly flags dangerous *queries* even when LLMs refuse to answer, while the external benchmark (BioThreat-Eval) uses response-based labeling that penalizes this conservative behavior.

## Limitations

**External kappa (0.414) falls short of the 0.80 target.** This is the most important limitation and warrants transparency:

The BioThreat-Eval external benchmark scores LLM *responses* on four dimensions (refusal rate, information completeness, operational specificity, safety awareness). Constitutional BioGuard classifies *queries* (it flags whether a question is dangerous regardless of how the downstream LLM responds). When the same dangerous query is sent to GPT-4 and Claude (which both refuse), BioThreat-Eval labels those responses as safe; our classifier labels the query as unsafe. This architectural mismatch accounts for the kappa gap.

Three validation strategies were tested: (1) `threat_level >= 4` with refusal correction (kappa = 0.414, best alignment), (2) `threat_level >= 3` (kappa = 0.310), and (3) response-based (kappa = 0.244). All are reported in `results/` for transparency. The internal metrics (F1 = 0.980 on held-out test set, 0.00% over-refusal FPR on 325 benign queries) reflect the classifier's performance on its intended task.

Additional limitations:
- Trained on Claude-generated synthetic data; real-world distribution shift is uncharacterized
- Encoding attacks (ROT13, URL-encode) achieve >30% ASR; text that evades tokenization is a fundamental weakness for any embedding-based classifier
- English-centric; multilingual coverage is limited to code-switching augmentation

## Responsible Use

This tool is designed to **detect** potentially dangerous biological content, not to generate it. The classifier, constitution, and training data should be used for:
- Content moderation in AI systems handling biological queries
- Research on dual-use content detection
- Biosafety policy and compliance tooling

Do not use this project to develop evasion strategies for biosafety systems, to identify gaps in existing safety tools for malicious purposes, or in contexts where a false negative could enable serious harm without appropriate human oversight. The 9.79% mean adversarial ASR means the classifier is not a complete defense on its own.

## What This Is Not

- Not a replacement for institutional biosafety review or BSL-2/3 oversight
- Not a wet-lab risk assessment tool
- Not a complete safeguard on its own (recommended as one layer in a defense-in-depth design alongside upstream policy, downstream model refusal training, and human review)
- Not validated on languages other than English (multilingual coverage limited to code-switching augmentation)
- Not a substitute for the Anthropic Constitutional Classifiers production pipeline; this is a domain-extension prototype demonstrating the methodology applied to biology

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

1. **Enhance Harm** (EH): Enhance the harmful consequences of a biological agent or toxin
2. **Disrupt Immunity** (DI): Disrupt immunity or the effectiveness of an immunization
3. **Confer Resistance** (CR): Confer resistance to clinically/agriculturally useful antimicrobials
4. **Increase Stability** (IS): Increase the stability, transmissibility, or dissemination
5. **Alter Host Range** (AH): Alter the host range or tropism
6. **Enhance Susceptibility** (ES): Enhance the susceptibility of a host population
7. **Generate/Reconstruct** (GR): Generate or reconstitute an eradicated or extinct agent

## Quick Start

### Inference Only (no training, no API key)

See [Try It in 30 Seconds](#try-it-in-30-seconds) above. Just `pip install transformers torch` and load from HuggingFace.

### Full Reproducibility Pipeline

If you want to regenerate the training data or retrain from scratch:

**Prerequisites:**
- Python >= 3.10
- Anthropic API key (for synthetic data generation only; inference is local)

**Installation:**
```bash
git clone https://github.com/jang1563/constitutional-bioguard
cd constitutional-bioguard
pip install -e ".[dev]"
```

**Run the full pipeline:**
```bash
# 1. Validate constitution coverage
make validate

# 2. Generate synthetic data (~$15 API cost)
make generate
make augment
make benign

# 3. Prepare train/val/test splits
make prepare

# 4. Train DeBERTa classifier (GPU recommended)
make train

# 5. Evaluate
make evaluate      # Internal metrics (F1, AUROC)
make external      # External validation against BioThreat-Eval
make adversarial   # 20-attack robustness suite
make overrefusal   # FPR on benign queries
make figures       # Generate all visualizations

# Or run everything:
make all
```

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
│   └── export_to_hf.py               # HuggingFace Hub upload
└── tests/
```

## Adversarial Robustness

The classifier is tested against 20 attack types across 4 categories:

| Category | Attacks | Mean ASR |
|----------|---------|----------|
| Character-level | homoglyphs, invisible chars, leetspeak, case swap, whitespace, typos, unicode norm | 5.1% |
| Encoding | base64, ROT13, hex, URL encode, backspace | 21.5% |
| Semantic | passive voice, euphemism, hypothetical, negation, question flip, context dilution | 0.0% |
| Multilingual | code-switching, mixed script | 0.0% |

Encoding attacks (especially ROT13 at 47.9%) are the primary weakness, which is expected since encoded text is fundamentally different from natural language. All semantic and multilingual attacks achieve 0% ASR.

## Cross-Project Integration

This classifier serves as the output safety filter in [AgentShield](https://github.com/jang1563/agentshield), providing real-time content classification at ~5ms/query with no API cost. See AgentShield's `output_classifier.py` for integration details.

Related projects:
- [bio-overrefusal-v0.1](https://github.com/jang1563/bio-overrefusal-v0.1): 201-query expert-annotated dataset measuring legitimate-biology FPR for frontier models
- [ambiguity-casebook](https://github.com/jang1563/ambiguity-casebook): 36 dual-use boundary cases for classifier stress-testing
- [bio-constitution-rules](https://github.com/jang1563/bio-constitution-rules): 30 rules library covering 6 bio domains, validated by 5-fold CV

## Citation

See [CITATION.cff](CITATION.cff) for the structured citation, or use:

```
@software{kim2026bioguard,
  author = {Kim, JangKeun},
  title  = {Constitutional BioGuard: A Biosafety Content Classifier},
  year   = {2026},
  url    = {https://github.com/jang1563/constitutional-bioguard},
  version = {0.1.0},
}
```

## License

MIT (see [LICENSE](LICENSE)).

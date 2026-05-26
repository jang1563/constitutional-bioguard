# Constitutional BioGuard

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v4-yellow)](https://huggingface.co/jang1563/constitutional-bioguard-v4)

> **TL;DR.** Research prototype biological dual-use content classifier built using Anthropic's [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) methodology. A 56-rule biosafety constitution across 7 NSABB categories drives synthetic data generation and DeBERTa-v3-base fine-tuning. **v4 response-diverse is the recommended release checkpoint**: it breaks v3's compliance-template shortcut, reaches 2.1% FPR on truly held-out OR-Bench-Hard-1K, 0% XSTest FPR, 32% WildGuard native bio recall, and 0.45 BioThreat-Eval F1 at 184M parameters. **v5 was tested but not released**: PairCFR fixed one artificial hybrid-response Goodhart case but collapsed key bio recall. The project is a transparent case study in shortcut diagnosis, data-centric remediation, leakage auditing, and honest non-release decisions. It is not a production-equivalent safeguard.

> **Portfolio context.** This DeBERTa-v3 prototype is trained on the [ConstitutionRules](https://github.com/jang1563/bio-constitution-rules) 56-rule constitution and evaluated alongside [OverRefusal](https://github.com/jang1563/bio-overrefusal-v0.1) (FPR finding) and [AmbiguityCasebook](https://github.com/jang1563/ambiguity-casebook) (DURC boundary).

**Author:** JangKeun Kim, Weill Cornell Medicine (jak4013@med.cornell.edu)

## Release Status

| Surface | Status |
|---------|--------|
| Code | v0.2.0 research prototype, MIT-licensed on GitHub |
| Model | `jang1563/constitutional-bioguard-v4` private Hugging Face preview |
| Constitution | 56 rules / 7 NSABB categories (`constitution/biosafety_constitution.yaml`) |
| External validation | v4/v5 gate metrics and leakage audit reported in `data/metrics/` and `docs/TECHNICAL_REPORT.md` |
| Independent review | Not yet externally audited |
| Responsible-use scope | [`SAFETY.md`](SAFETY.md) |

### Latest Run Snapshot (2026-05-25)

- **Recommended checkpoint: v4 response-diverse.** v4 keeps the v3 bio-specialist scope while breaking a phrase-specific compliance-template shortcut: CRT compliance flag rate drops from 100% to 29%, with content discrimination restored (44% UNSAFE vs 14% SAFE under identical template).
- **Clean held-out gates.** v4 reaches 2.1% FPR on OR-Bench-Hard-1K, 0% FPR on XSTest, 32% recall / 0.43 F1 on WildGuard native bio, and 0.45 F1 on BioThreat-Eval.
- **Goodhart audit.** The earlier OR-Bench-Health 1.22% number was 100% train/eval overlap and is now treated as training-distribution evidence only. HarmBench/AdvBench "held-out" recall from v3-era reporting is also restated as training-distribution recall.
- **v5 non-release.** v5 PairCFR fixes the artificial refusal+compliance hybrid FPR (68% -> 10%) but fails the specialist recall gate (WildGuard native recall 17.1%, SimpleSafetyTests/SaladBench/ALERT CBRN recall 0%). v4 remains the release model.
- **Efficiency.** v4 is reported at 15.6x faster than WildGuard 7B and 6.7x faster than LLaMA-Guard 3 8B at batch=1, with roughly 7x lower GPU memory use.

## Reviewer Framing

This repository is a **prototype** showing one concrete instantiation of the Constitutional Classifiers methodology applied to the biosafety domain. It is intended as a research artifact demonstrating: (a) how a domain constitution can be machine-readable, (b) how synthetic data can be generated against that constitution, (c) what calibration-vs-evasion trade-offs a small classifier exhibits when fine-tuned on this data. It is **not** equivalent to any production safety system, including Anthropic's deployed Constitutional Classifier pipeline.

## Results

### Recommended: v4 Response-Diverse (2026-05-25)

After diagnosing shortcut learning in v1, recall collapse in v2, and a
phrase-specific compliance-template shortcut in v3, **v4 response-diverse**
is the recommended checkpoint. It was trained with four augmentation blocks
that decouple response style from unsafe labels while preserving the model's
bio-specialist boundary.

**Clean behavioral gates and mechanism probes:**

| Gate / Probe | v3 | **v4** | Interpretation |
|---|---:|---:|---|
| OR-Bench-Hard-1K FPR | n/a | **2.1%** | Clean held-out over-refusal gate |
| XSTest FPR | 94.0% | **0.0%** | Clean transfer beyond v4 augmentation |
| WildGuard native bio recall | 2.0% | **32.0%** | Real-response OOD bio recall |
| WildGuard native F1 | 0.04 | **0.43** | Specialist utility on native labels |
| BioThreat-Eval F1 | 0.43 | **0.45** | Preserved despite shortcut fix |
| CRT compliance flag rate | 100% | **29%** | Template no longer sufficient |
| CRT compliance TPR / FPR | 100% / 100% | **44% / 14%** | Content discrimination restored |
| Refusal+compliance UNSAFE recall | n/a | **64%** | No refusal-prefix bypass observed |
| Refusal+compliance SAFE FPR | n/a | 68% | Artificial hybrid Goodhart caveat |

**v5 release decision:**

| Gate | Target | v4 | v5_baseline | v5 PairCFR |
|---|---:|---:|---:|---:|
| OR-Bench-Hard-1K FPR | < 5% | **2.1%** | 55.3% | **0.0%** |
| XSTest FPR | 0% | **0.0%** | 16.0% | **0.0%** |
| WildGuard native bio recall | >= 28% | **32.0%** | **62.5%** | 17.1% |
| CRT hybrid FPR | < 35% | 68% | 100% | **10%** |

v5 fixes the artificial hybrid-response failure but loses too much bio recall,
so it is documented as an honest negative result rather than released. The next
useful experiment is a lower PairCFR weight (`lambda=0.1` or `0.15`) or a
cascade-first v6 design.

### Historical Baselines

| Version | Primary fix | Main failure mode | Status |
|---|---|---|---|
| v1 A_full | Synthetic-only baseline | Adversarial-framing shortcut; cross-domain FAR up to 73% | Deprecated |
| v2 augmented | SAFE augmentation | Bio recall collapsed to ~0% | Diagnostic |
| v3 balanced | Reduced SAFE + targeted UNSAFE + weight 2.0 | Compliance-template shortcut; OR-Bench-Health leakage in old reporting | Diagnostic |
| **v4 response-diverse** | Response-style diversity + label decoupling | Artificial refusal+compliance hybrid FPR | **Recommended** |
| v5 PairCFR | Clean splits + contrastive loss | Bio recall collapse at lambda=0.3 | Not released |

### v1 (original, synthetic-only training)

| Metric | Value | Target |
|--------|-------|--------|
| F1 (synthetic test) | 0.9807 | >= 0.90 |
| AUROC (synthetic) | 0.9980 | -- |
| Precision / Recall (synthetic) | 0.9951 / 0.9667 | -- |
| Over-refusal FPR | 0.00% (100 benign) | < 2% |
| Adversarial mean ASR | 9.79% (20 attacks, pre-preprocessing) | < 15% |
| External kappa (BioThreat-Eval, TL>=4) | 0.414 | >= 0.80 |

These v1 internal metrics were misleading: the model achieved them by
learning a **shortcut feature** (adversarial framing) rather than the
intended bio-harm concept. See Section 6.8/6.8b of the technical
report for the diagnostic chain that exposed this.

Per-category F1 scores range from 0.931 (confer_resistance) to 1.000 (enhance_harm, alter_host_range, enhance_susceptibility). 6 of 7 NSABB categories are above 0.95; confer_resistance is 0.931.

The external kappa gap is a known artifact: the classifier correctly flags dangerous *queries* even when LLMs refuse to answer, while the external benchmark (BioThreat-Eval) uses response-based labeling that penalizes this conservative behavior.

## Limitations

**This is a research prototype, not a production safeguard.** The repository is
useful as an auditable classifier-building case study, but any deployment that
handles real biology workflows needs upstream policy, normalization, general
safety models, human escalation, and domain expert review.

Additional limitations:
- Trained on Claude-generated synthetic data; real-world distribution shift is uncharacterized
- v4 still over-flags an artificial refusal+compliance hybrid response pattern
- Historical OR-Bench-Health and HarmBench/AdvBench "held-out" claims required restatement after leakage audit
- Encoding attacks remain a general weakness for embedding-based classifiers and should be handled upstream
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
    |  + v4/v5 corrective augmentation (response diversity, clean splits, PairCFR probes)
    |  + 325 benign biology queries
    v
Baseline: ~4,500 labeled examples (3062 train / 697 val / 643 test)
v4/v5: baseline train split + corrective augmentation blocks
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

Pull the trained classifier from Hugging Face and run a single batch. No Anthropic API key needed; no training pipeline needed. The v4 checkpoint is currently a private preview, so the Hugging Face account running this snippet must have access.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "jang1563/constitutional-bioguard-v4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

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
    label = "UNSAFE" if p[1] >= 0.5 else "SAFE"
    print(f"{label} (p={p[1]:.3f}): {text[:60]}...")
```

### Full Pipeline: Data Regeneration

Regenerate the original synthetic corpus and retrain the baseline classifier.
The v4/v5 release experiments are built as explicit follow-on scripts so their
data discipline and acceptance gates stay auditable.

**Step 1: Run the complete pipeline** (2–4 hours)

```bash
ANTHROPIC_API_KEY=your_key nohup bash scripts/run_full_pipeline.sh > pipeline.log 2>&1 &
```

`pipeline.log` is treated as a local, ignored run artifact.

This runs the original 7-step baseline automatically:
1. Generate synthetic data (1960+ examples from 56 rules)
2. Augment restricted/boundary examples
3. Generate benign queries
4. Prepare stratified train/val/test splits
5. Train the DeBERTa-v3-base classifier
6. Calibrate threshold on validation set
7. Evaluate the trained model

**Monitor progress:**
```bash
python scripts/monitor_pipeline.py --watch
```

**Step 2: Reproduce v4/v5 corrective experiments**

```bash
python scripts/create_v4_splits.py
python scripts/train_v4_response_diverse.py --unsafe-weight 1.5
python scripts/create_v5_splits.py
python scripts/train_v5_baseline.py --unsafe-weight 1.5
python scripts/train_v5.py --unsafe-weight 1.5 --paircfr-lambda 0.3 --paircfr-temperature 0.1
python scripts/v5_eval_all_gates.py
```

On Cayuga, use the matching SLURM wrappers in `scripts/cayuga_v4_*.slurm` and
`scripts/cayuga_v5_*.slurm`.

**Step 3: Post-pipeline workflow**

After completion, follow [`docs/POST_PIPELINE_CHECKLIST.md`](docs/POST_PIPELINE_CHECKLIST.md) to:
- Verify data integrity
- Review calibration results
- Run variant experiments
- Compare and select best model
- Export to Hugging Face (optional)

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
│   │   ├── train_deberta.py          # DeBERTa fine-tuning
│   │   ├── paircfr_trainer.py        # v5 PairCFR contrastive trainer
│   │   └── splice_projector.py       # v5/v6 linear concept-erasure utility
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
│   ├── train_v4_response_diverse.py  # Recommended v4 training
│   ├── train_v5.py                   # v5 PairCFR non-release experiment
│   └── export_to_hf.py               # Hugging Face Hub upload
└── tests/
```

## Robustness Audits

BioGuard is evaluated with both adversarial text perturbations and
representation-level shortcut probes. The most important v4 finding is not
"higher score everywhere"; it is a more specific mechanism result: the
compliance-template feature remains encoded in hidden state, but it is no
longer sufficient to trigger UNSAFE by itself.

| Audit | Finding |
|---|---|
| v1 adversarial suite | 9.79% mean ASR before normalization; encoding attacks remain a known weakness |
| v3 CRT | canonical compliance template caused 100% flag rate regardless of content |
| v4 CRT | same template drops to 29% flag rate with 44%/14% UNSAFE/SAFE discrimination |
| v4 refusal-prefix probe | no bypass: 64% UNSAFE recall even with refusal+compliance prefix |
| v4 hybrid caveat | artificial refusal+compliance hybrids over-flag SAFE items at 68% FPR |
| v5 PairCFR | hybrid FPR improves to 10%, but bio recall collapse prevents release |

## What This Is Not

- Not a replacement for institutional biosafety review or BSL-2/3 oversight
- Not a wet-lab risk assessment tool
- Not a complete safeguard on its own (recommended as one layer in a defense-in-depth design alongside upstream policy, downstream model refusal training, and human review)
- Not validated on languages other than English (multilingual coverage limited to code-switching augmentation)
- Not a substitute for any vendor's production constitutional-classifier pipeline; this is a domain-extension prototype demonstrating the methodology applied to biology

## How This Maps to AI Safety Practice

This prototype illustrates **one** point in the safeguards stack: a domain-specific output classifier trained on a machine-readable constitution. It complements rather than replaces:

- **Capability evaluations** (e.g. WMDP, biothreat-eval): measure what a base model could enable. This classifier sits downstream of those, on the response path.
- **Over-refusal calibration** (e.g. [bio-overrefusal-v0.1](https://github.com/jang1563/bio-overrefusal-v0.1)): measures whether a deployed safeguard blocks legitimate research. This repository's `make overrefusal` target reports the same metric on the included benign holdout (0/100).
- **Boundary-case adjudication** (e.g. [ambiguity-casebook](https://github.com/jang1563/ambiguity-casebook)): documents where reasonable experts disagree. The shortcut and leakage audits here are one signal that boundary cases need human-in-the-loop, not classifier-only routing.

The v4/v5 story is intentionally audit-heavy: several older measurements were
restated after overlap checks, and v5 was held back despite improving one gate.
That discipline is the main intended contribution of this repository.

## Cross-Project Integration

This classifier can serve as an output safety filter in downstream agent stacks, providing local content classification with no per-query API cost. Keep integration-specific code in the downstream application so this repository remains a focused classifier artifact.

## Responsible Use Scope

See [`SAFETY.md`](SAFETY.md) for the public responsible-use scope, what is withheld, and how to report concerns.
See [`docs/REPOSITORY_QUALITY_CHECKLIST.md`](docs/REPOSITORY_QUALITY_CHECKLIST.md) for the release-readiness checklist used before GitHub/Hugging Face updates.

## Citation

If you use this work, please cite:

```bibtex
@software{kim2026bioguard,
  author = {Kim, JangKeun},
  title  = {Constitutional BioGuard: A Biosafety Content Classifier},
  year   = {2026},
  url    = {https://github.com/jang1563/constitutional-bioguard},
  version = {v0.2.0},
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## License

MIT

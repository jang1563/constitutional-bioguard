---
license: mit
language:
  - en
library_name: transformers
tags:
  - safety
  - classifier
  - biosecurity
  - biosafety
  - deberta
  - domain-specialist
  - constitutional-classifiers
  - shortcut-mitigation
  - leakage-audit
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
model-index:
  - name: Constitutional BioGuard v4
    results:
      - task:
          type: text-classification
          name: Biological dual-use content classification
        metrics:
          - name: OR-Bench-Hard-1K FPR
            type: fpr
            value: 0.0212
          - name: XSTest FPR
            type: fpr
            value: 0.0
          - name: WildGuard native bio recall
            type: recall
            value: 0.32
          - name: WildGuard native F1
            type: f1
            value: 0.426
          - name: BioThreat-Eval F1
            type: f1
            value: 0.45
---

# Constitutional BioGuard v4

A 184M-parameter domain-specialist classifier for biological dual-use content.
It fine-tunes `microsoft/deberta-v3-base` on a 56-rule biosafety constitution
plus corrective augmentation designed to diagnose and reduce shortcut learning.

**Recommended checkpoint:** v4 response-diverse.

**Not released:** v5 PairCFR. v5 fixed one artificial hybrid-response Goodhart
case but lost too much specialist bio recall, so it is documented as a negative
result rather than shipped as the public model.

## Intended Scope

BioGuard v4 is a bio-specialist classifier for query/response pairs. It is not
a general-purpose safety model. For cyber, harassment, self-harm, misinformation,
or other non-bio categories, use a general safety classifier and route bio-like
content to BioGuard as a second stage.

```
general safety model -> if bio-like / needs biosafety review -> BioGuard v4
```

## Audited Results

The public headline uses clean held-out gates after the Goodhart/leakage audit.

| Metric | v3 | v4 |
|---|---:|---:|
| OR-Bench-Hard-1K FPR | n/a | 2.1% |
| XSTest FPR | 94.0% | 0.0% |
| WildGuard native bio recall | 2.0% | 32.0% |
| WildGuard native F1 | 0.04 | 0.43 |
| BioThreat-Eval F1 | 0.43 | 0.45 |
| CRT compliance flag rate | 100% | 29% |
| CRT compliance TPR / FPR | 100% / 100% | 44% / 14% |

### What Changed From Earlier Reporting

The earlier OR-Bench-Health result (v3 98.5% FPR -> v4 1.22% FPR) is now
treated as training-distribution evidence because the v4 audit found 100%
overlap with the v4 augmentation block. The clean replacement headline is
OR-Bench-Hard-1K FPR = 2.1%.

HarmBench/AdvBench bio "held-out" recall from earlier v3-era reporting is also
restated as training-distribution recall after overlap analysis. The strongest
clean transfer evidence is XSTest, WildGuard native labels, BioThreat-Eval,
SaladBench/ALERT CBRN selectivity, and the BioThreat-based CRT probes.

## v5 Non-Release Decision

| Gate | Target | v4 | v5_baseline | v5 PairCFR |
|---|---:|---:|---:|---:|
| OR-Bench-Hard-1K FPR | < 5% | 2.1% | 55.3% | 0.0% |
| XSTest FPR | 0% | 0.0% | 16.0% | 0.0% |
| WildGuard native bio recall | >= 28% | 32.0% | 62.5% | 17.1% |
| CRT refusal+compliance FPR | < 35% | 68% | 100% | 10% |

v5 PairCFR solved the artificial refusal+compliance hybrid FPR but failed the
specialist bio recall gate. v4 remains the recommended public checkpoint.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "jang1563/constitutional-bioguard-v4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()


def classify(query: str, response: str = "") -> dict:
    inputs = tokenizer(
        query,
        response,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    prob_unsafe = float(torch.softmax(logits, dim=-1)[0, 1])
    return {
        "label": "UNSAFE" if prob_unsafe >= 0.5 else "SAFE",
        "prob_unsafe": prob_unsafe,
    }


print(classify(
    query="What is the role of polymerase in DNA replication?",
    response="DNA polymerase synthesizes new DNA strands by adding nucleotides.",
))
```

## Training Summary

- Base model: `microsoft/deberta-v3-base`
- Architecture: binary text-pair classifier, `(query, response) -> SAFE/UNSAFE`
- Constitution: 56 rules across 7 NSABB categories
- v4 augmentation: response-diverse blocks that decouple compliance/refusal style
  from the unsafe label
- Class weighting: UNSAFE = 1.5
- Max sequence length: 512
- Seed: 42
- Hardware: single A100 GPU on Weill Cornell Cayuga HPC

## Limitations

- Research prototype, not a production safeguard.
- English-centric; multilingual behavior is not broadly validated.
- Synthetic and curated data dominate training; real-world distribution shift
  remains under-characterized.
- v4 over-flags an artificial refusal+compliance hybrid response pattern.
- It is intentionally bio-specialist and can miss non-bio harmful content.
- Use with upstream normalization and broader safety filters; do not rely on
  this model alone for high-stakes decisions.

## Responsible Use

This model is intended to detect biological dual-use risk signals and support
safety research. Do not use it to develop or optimize evasion strategies against
biosafety systems. Do not paste operational harmful biology content into public
issues or examples.

For the full safety scope, see `SAFETY.md` in the repository:
https://github.com/jang1563/constitutional-bioguard

## Citation

```bibtex
@software{kim2026bioguard,
  author = {Kim, JangKeun},
  title = {Constitutional BioGuard: A Biosafety Content Classifier},
  year = {2026},
  url = {https://github.com/jang1563/constitutional-bioguard},
  version = {v0.2.0},
}
```

## License

MIT.

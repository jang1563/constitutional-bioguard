---
license: mit
language:
- en
library_name: transformers
tags:
- safety
- classifier
- biosecurity
- deberta
- domain-specialist
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
---

# Constitutional BioGuard v3 (balanced)

A 184M-parameter **domain-specialist safety classifier** for biological dual-use content. Built using a domain-extended adaptation of Anthropic's [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) methodology. Fine-tunes `microsoft/deberta-v3-base` on constitution-driven synthetic data plus a small targeted augmentation that fixed a shortcut-learning failure mode in earlier iterations.

> **Designed scope.** Bio-harm detection (NSABB 7 categories). v3 is *not* a general-purpose safety classifier — for non-bio harm types (cybercrime, harassment, misinformation, etc.) use a generalist model like LLaMA-Guard 3 or WildGuard, ideally in cascade with this model.

## Versions

| Model | Status | Description |
|-------|--------|-------------|
| **v3 (balanced)** | **Recommended** | Reduced SAFE augmentation (500) + UNSAFE bio adversarial augmentation (71) + manual UNSAFE class weight 2.0. Fixes v2's recall collapse while preserving v2's FAR reduction. |
| v2 (augmented) | Diagnostic | Added 1,366 SAFE items to fix v1's shortcut FAR; collapsed bio recall to ~0% on adversarial bio. Released for completeness. |
| v1 (A_full) | Deprecated | Original synthetic-only model. Learned "adversarial framing" shortcut feature (Geirhos et al. 2020) rather than bio harm content. |

## Headline Results

**Bio recall on out-of-distribution adversarial benchmarks (higher = better):**

| Benchmark | v1 (A_full) | v2 | **v3** |
|---|---|---|---|
| HarmBench bio (held-out, n=8) | 87.5% | 0% | **100%** |
| AdvBench bio (held-out, n=3) | 100% | 0% | **100%** |
| BioThreat-Eval recall (TL=4, n=558) | 37.8% | 12.2% | 27.2% |
| BioThreat-Eval **AUROC** | 0.7196 | 0.6965 | **0.7650** |

v3 has the highest AUROC across the three iterations on the expert-labelled BioThreat-Eval set, indicating better calibration than the prior shortcut-driven model.

**Cross-domain false alarm rate (lower = better):**

| Benchmark | v1 (A_full) | v2 | **v3** |
|---|---|---|---|
| WildGuardMix held-out (n=1509) | 51.1% | 0.0% | **0.9%** |
| LAB-Bench held-out (n=1342) | 49.1% | 0.0% | **0.0%** |
| WMDP-Cyber held-out (n=1887) | 73.2% | 0.2% | **0.3%** |
| WMDP-Chem held-out (n=408) | 28.4% | 0.2% | **0.5%** |
| MedQA held-out (n=1273) | 0.9% | 0.0% | **0.0%** |
| PubMedQA held-out (n=1000) | 0.0% | 0.0% | 0.0% |

**v3 strictly Pareto-dominates both prior models:**
- vs v1: same bio recall (~75% avg), ~100× lower cross-domain FAR
- vs v2: same cross-domain FAR (~0.3% avg), ~20× higher bio recall

**Cost efficiency (parameter count):**

| Model | Params | BioThreat-Eval F1 | BioThreat-Eval AUROC |
|---|---|---|---|
| **v3 (this)** | **184M** | **0.4279** | **0.7650** |
| WildGuard 7B (AI2) | 7B (38× larger) | 0.3636 | 0.6111 |
| LLaMA-Guard 3 8B (Meta) | 8B (43× larger) | (pending) | (pending) |

v3 outperforms 7B WildGuard on bio domain at ~38× smaller scale, demonstrating the value of domain specialisation in safety classifier design.

## Quick Start

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "jang1563/constitutional-bioguard-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

def classify(query: str, response: str = "") -> dict:
    """Returns {label: 'SAFE'|'UNSAFE', prob_unsafe: float}."""
    inputs = tokenizer(query, response, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    prob_unsafe = float(probs[0, 1])
    return {
        "label": "UNSAFE" if prob_unsafe >= 0.5 else "SAFE",
        "prob_unsafe": prob_unsafe,
    }

# Example
print(classify(
    query="What is the role of polymerase in DNA replication?",
    response="DNA polymerase synthesizes new DNA strands by adding nucleotides..."
))
# {'label': 'SAFE', 'prob_unsafe': 0.01}
```

## Training Details

**Base model:** `microsoft/deberta-v3-base` (184M params)

**Architecture:** Binary text-pair classifier (query, response) → {SAFE, UNSAFE}

**Training data:** 3,633 items total (2,064 UNSAFE / 1,569 SAFE)
- 3,062 synthetic items from constitution-driven generation (1,993 UNSAFE / 1,069 SAFE)
- 571 augmentation items (71 UNSAFE bio adversarial + 500 SAFE cross-domain):
  - WildGuard: 200 (SAFE; teaches "adversarial framing ≠ bio harmful")
  - LAB-Bench: 200 (SAFE; teaches "bio technical vocab ≠ harmful")
  - WMDP-Cyber: 100 (SAFE; cross-domain technical content)
  - HarmBench bio: 51 (UNSAFE; bio adversarial behaviours)
  - AdvBench bio: 18 (UNSAFE; bio harmful behaviours)
  - JailbreakBench bio: 2 (UNSAFE)

**Hyperparameters:**
- Epochs: 5 (best at epoch 3 by validation F1)
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: AdamW with cosine schedule
- Class weights: SAFE=1.16 (auto-balanced), UNSAFE=2.0 (manual boost for recall priority)
- Max sequence length: 512
- Seed: 42

**Hardware:** Single A100 GPU (Weill Cornell Cayuga HPC), ~10 minutes training

## Domain Boundary

v3 was deliberately trained as a **bio specialist**. It will:
- ✅ Detect bio harm content at high recall (matches/exceeds 7-8B baselines)
- ❌ Likely *miss* non-bio harm content (cyber, harassment, misinformation, etc.)
- ✅ Show low false alarm rate on cross-domain non-bio content

This is by design, not a defect. For a production safety stack, deploy v3 in **cascade** with a general-purpose classifier:

```
[Stage 1: General safety] → if SAFE or non-bio harm → use Stage 1 decision
                          → if "looks bio" → [Stage 2: v3 BioGuard] (final)
```

This pattern follows the Constitutional Classifiers ++ architecture (Cunningham et al. 2026).

## Limitations

1. **Synthetic training data ceiling.** Training relies on Claude-generated synthetic examples; real-world distribution shift is uncharacterised beyond the external benchmarks listed above.

2. **Bio scope only.** As noted above, do not use as a general safety classifier.

3. **English-centric.** Multilingual coverage is limited to code-switching augmentation in training. For multilingual deployment, evaluate per target language before relying on v3.

4. **Case-swap regression.** WS-4 adversarial suite shows a tokenisation regression on case-swapped input (`hOw To SyNtHeSiZe...`) with 26.5% ASR. Mitigation: case-normalise text before passing to v3, or augment future training with case-swapped variants.

5. **Calibration threshold.** Default threshold 0.5 gives FPR 0% on BioThreat-Eval but lower recall (27%). For higher recall settings, calibrate threshold per deployment.

6. **Not equivalent to production systems.** This is a research prototype evaluating CC++ methodology transfer. Not equivalent to deployed Constitutional Classifiers in Anthropic's production stack.

## Responsible Use

This model is intended to **detect** bio dual-use content, not to generate it. Use cases:
- Content moderation in AI systems handling biological queries
- Research on dual-use content detection methodology
- Biosafety policy and compliance tooling
- Component in a multi-classifier safety cascade

See `SAFETY.md` in the project repo for full responsible-use scope.

## Citation

```bibtex
@misc{kim2026bioguard,
  title={Extending Constitutional Classifiers++ to Biosafety:
         What Transfers and What Does Not},
  author={Kim, JangKeun},
  year={2026},
  institution={Weill Cornell Medicine},
  url={https://github.com/jang1563/constitutional-bioguard}
}
```

## Acknowledgements

- Anthropic Constitutional Classifiers methodology (Sharma et al. 2025; Cunningham et al. 2026)
- HarmBench (Mazeika et al. ICML 2024)
- WildGuardMix (Han et al. NeurIPS 2024)
- BioThreat-Eval (provided by collaborators)
- Weill Cornell Medicine Cayuga HPC for compute

## License

MIT. See `LICENSE` in the project repository.

## Project Links

- **Code & docs:** https://github.com/jang1563/constitutional-bioguard
- **Technical report:** `docs/TECHNICAL_REPORT.md` (v1.8, includes WS-1/2/3/4 + Section 6 corrective experiments 6.1-6.11)
- **Safety policy:** `SAFETY.md`

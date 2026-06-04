---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
tags:
- safety
- moderation
- guard-model
- biosafety
- response-harm
- deberta-v3
datasets:
- allenai/wildguardmix
- PKU-Alignment/BeaverTails
- AmazonScience/FalseReject
metrics:
- recall
- f1
- auprc
model-index:
- name: constitutional-bioguard-response
  results:
  - task:
      type: text-classification
      name: Bio response-harm detection
    metrics:
    - type: recall
      value: 0.919
      name: Real bio-response recall (n=62, directional, Wilson 95% CI 0.825-0.965)
    - type: auroc
      value: 0.970
      name: AUROC (harmful bio vs real legitimate research)
    - type: auprc
      value: 0.938
      name: AUPRC
    - type: recall
      value: 0.694
      name: General response-harm recall (WildGuard-native, n=1709, better-powered)
    - type: fpr
      value: 0.021
      name: Over-refusal on real legitimate bio research (n=531)
extra_gated_prompt: >-
  constitutional-bioguard-response is a defensive bio-safety research artifact, released for
  non-commercial research only. By requesting access you agree to the
  responsible-use terms in the model card: use it solely for defensive evaluation
  and moderation research; do not use it as a reward, discriminator, or filter to
  generate, refine, or evade detection of harmful biological content; do not probe
  it to construct evasion strategies; do not redistribute the weights outside this
  gated channel.
extra_gated_fields:
  Name: text
  Affiliation: text
  Email: text
  Intended use: text
  I agree to the responsible-use terms (defensive evaluation only): checkbox
---

# constitutional-bioguard-response: a bio response-harm classifier

> **Note (2026-06):** the HF repo `jang1563/constitutional-bioguard-response` now ships the
> **density-debiased v8bh** checkpoint. This package documents the v8b lineage; for the
> integrity-corrected v8bh figures (recall 0.921, over-refusal 0.194, AUROC 0.952,
> Pareto-dominated by Qwen3Guard-0.6B) see `MODEL_CARD.md` and the HF model card.

A small encoder (DeBERTa-v3-base, ~184M params) that reads a `query [SEP] response`
pair and decides whether the **response** delivers harmful biological content.
Built to reduce over-refusal of legitimate bench research while catching genuinely
harmful bio completions. This card states scope and limits as plainly as results.
All numbers are held-out and leakage-audited (training queries are byte-disjoint
from every test set). This is the successor to
`jang1563/constitutional-bioguard-deberta-v1` (this line drops the "constitutional"
label: the model is standard supervised classification, not Constitutional AI;
Constitutional Classifiers is cited as a methodological influence, not a basis).

## Model details

- **Architecture:** DeBERTa-v3-base (12 layers, hidden 768, ~184M params).
- **Input:** `query [SEP] response`. **Output:** binary (harmful response vs not) + probability.
- **Class of model:** response-harm classifier. It judges the response, not the request.
- **Preprocessing (`preprocessing.py`, shipped in this repo):** an input normalization
  layer (`normalize_text`) that strips invisible/zero-width/tag/variation-selector
  characters, folds homoglyphs, decodes URL/base64/hex/ROT13, removes combining
  marks, and applies NFKC. This is a hardened adversarial-robustness defense (see
  Evaluation).
- **Decision threshold:** default 0.5 (near-optimal, see Operating point). Probabilities
  can be temperature-scaled (T=0.24) for calibration.

## Intended use

- **In scope:** monitoring model **outputs** in a bio-aware setting, as a
  research-grade second-stage filter or offline auditing tool, where the goal is
  to flag harmful bio responses without over-blocking legitimate research.
- **Out of scope:**
  - **Prompt/input filtering.** The model judges responses, not requests; it scores ~0
    on prompt-only benchmarks by design.
  - **Sole production safeguard.** Production use needs serving infrastructure,
    real-traffic threshold recalibration, and continuous monitoring first (see Limitations).
  - **High-consequence gating** without a human in the loop.

## How to use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("jang1563/constitutional-bioguard-response")
model = AutoModelForSequenceClassification.from_pretrained("jang1563/constitutional-bioguard-response").eval()

# apply the shipped input normalization first (preprocessing.py is in this repo;
# it reproduces the reported char-injection robustness)
from preprocessing import normalize_text
query, response = normalize_text(query), normalize_text(response)

# pair encoding tok(query, response) matches training/eval; do NOT concat with [SEP]
inp = tok(query, response, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    p_harmful = model(**inp).logits.softmax(-1)[0, 1].item()  # class 1 = UNSAFE
flag = p_harmful >= 0.5
```

## Training data

Reuse-only, with **zero newly generated harmful content** (3,507 examples;
1,163 positive / 2,344 negative):

| Source | n | License | Role |
|--------|---|---------|------|
| WildGuardMix (bio responses) | 1,350 | ODC-BY-1.0 | real harmful/benign responses |
| BeaverTails (bio) | 1,024 | CC-BY-NC-4.0 | real responses, response-harm labels |
| FalseReject (benign hard-negatives) | 891 | CC-BY-NC-4.0 | reduce over-refusal |
| non-bio control | 242 | mixed | selectivity control |

The model license (CC-BY-NC-4.0) is inherited from the two NonCommercial sources.

## Evaluation

**Discrimination (harmful bio vs real legitimate research):**

| Metric | Value |
|--------|-------|
| AUROC | 0.970 |
| AUPRC | 0.938 |
| Real bio-response recall | 0.919 (57/62; Wilson 95% CI 0.825-0.965; **directional**, small n) |
| General response-harm (WildGuard-native, n=1709) | recall 0.694 / F1 0.667 / FPR 0.077 |

**Over-refusal on real legitimate bio research** (the deployment-critical number):

| Set | n | Over-refusal (FPR) |
|-----|---|--------------------|
| All real legit bio | 531 | 0.021 |
| The author's own assistant sessions | 134 | 0.060 |
| Substantive-response subset | 68 | 0.015 |

**Calibration and operating point.** Temperature scaling (T=0.24, held-out cal
split) cuts ECE 0.137 to 0.042 (Brier 0.053 to 0.035). The default threshold 0.5
is near-optimal: **recall 0.919 at 2.1% over-refusal**; raising recall to ~0.95
costs ~15% over-refusal.

**Adversarial robustness (char-level, the primary classifier-evasion threat).**
On responses the model flags as harmful, evasion success rate (ESR) after the shipped
normalization:

| Attack | ESR |
|--------|-----|
| zero-width / fullwidth / Cyrillic homoglyph | 0.000 / 0.000 / 0.035 |
| combining diacritic / tag chars / variation selector | 0.000 / 0.000 / 0.000 |
| adversarial word perturbation (greedy char-swap) | 0.123 |
| intra-word spacing (residual) | 0.211 |
| Greek homoglyph (residual, not folded) | 0.105 |

7 of 8 char-injection attacks and the word-perturbation attack pass (ESR < 0.20 /
0.40). Two deliberate residuals are documented in Limitations.

**Scope boundary.** On stub-response benchmarks (SaladBench O39, ALERT-CBRN,
SimpleSafety-bio) the model flags ~0 by design: those attach a constant template
response and test prompt harm, which a response-harm classifier correctly ignores.

**Not a lexical shortcut.** Masking every bio keyword changes predictions by ~1%.

## Limitations

1. **Small bio sample / directional recall.** The headline bio recall (n=62) is
   directional. Public real-response bio-harmful data is scarce; growing it under
   reuse-only is capped near ~75 to 100 items, so a tighter CI would require gated-access
   data or generation (deliberately avoided). Report and read recall with its CI.
2. **Multi-turn: robust (tested).** Splitting harmful content across 2 to 5 turns
   is caught per-turn at 0.964; LLM-paraphrasing each turn then reconstructing
   gives per-turn 0.945 equal to windowed (no exchange-classifier gap). The model does
   not collapse under multi-turn delivery.
3. **Obfuscation: resisted (tested).** Benign framing wrappers evade at worst 0.14;
   a full neutral LLM paraphrase (Qwen2.5-7B, semantics preserved, surface fully
   rewritten) evades at only 0.07. The model judges content, not surface form.
4. **Spacing and Greek-homoglyph residuals.** Intra-word spacing (ESR 0.21) cannot
   be fixed by character stripping without breaking legitimate bio notation (e.g.
   spaced sequences like "A T G C"); needs adversarial training. Greek homoglyph
   (0.105, passing) is intentionally not folded to avoid corrupting legitimate bio
   notation (e.g. alpha-helix).
5. **Dual-use tail uncovered.** Ambiguous dual-use bio has no labeled harmful
   examples in public sources and is absent from training/eval.
6. **Distribution scope.** Trained on WildGuardMix-family responses; transfer to
   very different response styles is unmeasured.

## Responsible use, ethics, and data privacy

This is defensive biosafety research: the aim is to reduce over-refusal for
legitimate research while flagging harmful outputs. Withheld by design: the harmful
(positive) training examples, the exact production threshold, and any companion
attack harness. The real over-refusal evaluation uses the author's own assistant
session logs and is **not released** for privacy. Anyone deploying the model should
re-validate on their own traffic, add adversarial and multi-turn testing, and keep
a human in the loop for any consequential decision. Report misclassifications,
false negatives, or jailbreaks to the maintainer at silveray1563@gmail.com
(responsible disclosure welcome).

## Influences and citation

Methodological influences (cited, not a basis): Anthropic Constitutional
Classifiers (Sharma et al., arXiv:2501.18837) and its exchange-classifier
successor (arXiv:2601.04603); WildGuard (Han et al., arXiv:2406.18495). Training
data: WildGuardMix, BeaverTails (Ji et al., arXiv:2307.04657), FalseReject
(Zhang et al., arXiv:2505.08054).

Cite this model by its repository id `jang1563/constitutional-bioguard-response`.

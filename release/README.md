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
- name: bioguard-v8b
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
    - type: fpr
      value: 0.021
      name: Over-refusal on real legitimate bio research (n=531)
extra_gated_prompt: >-
  bioguard-v8b is a defensive bio-safety research artifact, released for
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

# bioguard-v8b: a bio response-harm classifier

A small encoder (DeBERTa-v3-base, ~184M params) that reads a `query [SEP] response`
pair and decides whether the **response** delivers harmful biological content.
Built to reduce over-refusal of legitimate bench research while catching genuinely
harmful bio completions. This card states scope and limits as plainly as results.
All numbers are held-out and leakage-audited (training queries are byte-disjoint
from every test set).

## Model details

- **Architecture:** DeBERTa-v3-base (12 layers, hidden 768, ~184M params).
- **Input:** `query [SEP] response`. **Output:** binary (harmful response vs not) + probability.
- **Class of model:** response-harm classifier. It judges the response, not the request.
- **Preprocessing (ships with the model):** an input normalization layer
  (`normalize_text`) that strips invisible/zero-width/tag/variation-selector
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
  - **Prompt/input filtering.** v8b judges responses, not requests; it scores ~0
    on prompt-only benchmarks by design.
  - **Sole production safeguard.** Multi-turn and content-obfuscation robustness
    are not yet validated (see Limitations).
  - **High-consequence gating** without a human in the loop.

## How to use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("bioguard-v8b")
model = AutoModelForSequenceClassification.from_pretrained("bioguard-v8b").eval()

# apply the shipped input normalization first (mitigates char-injection attacks)
# from constitutional_bioguard.preprocessing import normalize_text
# query, response = normalize_text(query), normalize_text(response)

text = query + tok.sep_token + response
inp = tok(text, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    p_harmful = model(**inp).logits.softmax(-1)[0, 1].item()
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
| General response-harm (WildGuard-native, n=1709) | recall 0.69 / F1 0.667 / FPR 0.077 |

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
On responses v8b flags as harmful, evasion success rate (ESR) after the shipped
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
SimpleSafety-bio) v8b flags ~0 by design: those attach a constant template
response and test prompt harm, which a response-harm classifier correctly ignores.

**Not a lexical shortcut.** Masking every bio keyword changes predictions by ~1%.

## Limitations

1. **Small bio sample / directional recall.** The headline bio recall (n=62) is
   directional. Public real-response bio-harmful data is scarce; growing it under
   reuse-only is capped near ~100 items, so a tighter CI would require gated-access
   data or generation (deliberately avoided). Report and read recall with its CI.
2. **Multi-turn: robust to naive splitting; adversarial reconstruction untested.**
   Splitting a harmful response across 2 to 5 turns is still caught per-turn at
   0.964 recall (windowed scoring recovers the rest, 1.0). The model does not
   collapse under naive multi-turn splitting. Not yet tested: adversarial
   reconstruction where each turn is rewritten to read benign so harm emerges only
   on assembly (this would require windowed/exchange scoring and is the safer
   deployment mode).
3. **Framing obfuscation tested and resisted.** Benign fiction/roleplay/educational/
   disclaimer wrappers around harmful content evade at worst 0.14 (most 0.02 to
   0.05); v8b judges the content, not the frame. Not tested: LLM-rewrite
   reconstruction where the harmful core itself is reworded to read benign.
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
session logs and is **not released** for privacy. Anyone deploying v8b should
re-validate on their own traffic, add adversarial and multi-turn testing, and keep
a human in the loop for any consequential decision. Report misclassifications,
false negatives, or jailbreaks to the maintainer at silveray1563@gmail.com
(responsible disclosure welcome).

## Citation

Part of the constitutional-bioguard line (v2 through v8). See the project's
`docs/V8_DESIGN.md`, `docs/V8B_SHIP_EVIDENCE.md`, and `docs/V8B_RELEASE_PLAN.md`
for design, gates, robustness, and the full result trail.

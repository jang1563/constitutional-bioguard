# Model Card: bioguard-v8b (bio response-harm classifier)

> **⚠️ Superseded (2026-06).** `bioguard-v8b` was renamed to `constitutional-bioguard-response`, and the
> shipped checkpoint is now the density-debiased **v8bh**. The authoritative card is
> [`MODEL_CARD.md`](MODEL_CARD.md); this file documents the earlier v8b lineage only.

A small encoder that judges whether a model **response** delivers harmful
biological content, built to reduce over-refusal of legitimate bench research
while catching genuinely harmful bio completions. This card states scope and
limits as plainly as results. Numbers are held-out and leakage-audited.

## Model details

- **Architecture:** DeBERTa-v3-base (12 layers, hidden 768, about 184M params).
- **Input:** `query [SEP] response`. **Output:** binary (harmful response vs not),
  with a probability. Default decision threshold 0.5.
- **Class of model:** response-harm classifier. It reads the response and decides
  whether *that response* is harmful. It is **not** a prompt/intent classifier.
- **Training:** supervised fine-tune, 3,507 examples, single seed.

## Intended use

- **In scope:** monitoring model **outputs** in a bio-aware setting, as a
  research-grade second-stage filter or an offline auditing tool, where the goal
  is to flag harmful bio responses without over-blocking legitimate research.
- **Out of scope:**
  - **Prompt/input filtering.** v8b judges responses, not requests. It scores
    near zero on prompt-only benchmarks by design (see Evaluation).
  - **Production deployment as a sole safeguard.** Not yet validated for
    adversarial, multi-turn, or out-of-distribution traffic (see Limitations).
  - **High-consequence gating** without a human in the loop.

## How to use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("bioguard-v8b")
model = AutoModelForSequenceClassification.from_pretrained("bioguard-v8b").eval()

text = query + tok.sep_token + response
inp = tok(text, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    p_harmful = model(**inp).logits.softmax(-1)[0, 1].item()
flag = p_harmful >= 0.5
```

## Training data

Reuse-only, with **zero newly generated harmful content**:

| Source | n | Role |
|--------|---|------|
| WildGuardMix (bio responses) | 1,350 | real harmful/benign responses |
| BeaverTails (bio) | 1,024 | real responses, response-harm labels |
| FalseReject (benign hard-negatives) | 891 | reduce over-refusal |
| non-bio control | 242 | selectivity control |

Total 3,507 (1,163 positive / 2,344 negative). All sources are public datasets;
no harmful examples were synthesized.

## Evaluation

All sets are held out, with training queries byte-disjoint from every test set.

**Real-response harm detection (the model's actual job):**

| Benchmark | n | Recall | Precision | FPR | Note |
|-----------|---|--------|-----------|-----|------|
| WildGuard-native (general response-harm) | 1,709 | 0.69 | 0.64 | 0.077 | well powered |
| Bio subset, held-out real responses | 62 pos / 75 neg | 0.92 | 0.76 | 0.27\* | bio-specific, small n |

Bio recall is 57/62, Wilson 95% CI about [0.82, 0.97]. Prior best on the same bio
task: v4 at 0.29. So the gain on bio recall is large, but the bio sample is small;
treat 0.92 as directional and 0.69 (n=1,709) as the better-powered estimate.

\*The 0.27 FPR here is on adversarial benchmark negatives, which are noisy. On
**real** legitimate research it does not reproduce (next table).

**Over-refusal on real legitimate bio research:**

| Set | n | Over-refusal (FPR) |
|-----|---|--------------------|
| All real legit bio | 531 | 0.021 (11/531) |
| The user's own Claude Code / Codex sessions | 134 | 0.060 (8/134) |
| Substantive-response subset | 68 | 0.015 (1/68) |

This clears the pre-registered over-refusal target of 10%. It is the most
deployment-relevant number, and it is small-sample, so read it as directional.

**Scope boundary (important):** on stub-response benchmarks (SaladBench O39,
ALERT-CBRN, SimpleSafety-bio), v8b flags ~0. Those benchmarks attach a constant
template response, so they test **prompt** harm; a response-harm classifier
correctly sees no harmful response and abstains. This is by design, not a recall
failure, but it means v8b does not cover prompt-side risk at all.

**Robustness check:** masking every bio keyword in the input changes predictions
by about 1%, so the model is not relying on a bio word list (not a lexical shortcut).

## Limitations

1. **Small bio sample.** The headline bio recall (n=62) and substantive
   over-refusal (n=68) are encouraging but not tightly estimated.
2. **Adversarial and multi-turn untested.** No jailbreak, obfuscation, or
   multi-turn evaluation on this checkpoint. Published work shows guards can be
   bypassed at high rates under multi-turn pressure; assume v8b is vulnerable
   until measured.
3. **Distribution scope.** Trained on WildGuardMix-family responses. Transfer to
   very different response styles is unmeasured and a related cross-source test
   was weak.
4. **The hard dual-use tail is not covered.** Ambiguous dual-use bio (for example
   immune-evasion vector design) has no labeled harmful examples in any public
   source, so it is absent from training and evaluation. These are the highest
   consequence cases.
5. **No deployment threshold calibration.** The 0.5 threshold is not tuned to a
   target operating point on representative traffic.
6. **Single checkpoint, single seed.** No variance estimate.

## Responsible use and data privacy

This is defensive biosafety research: the aim is to reduce over-refusal for
legitimate research while flagging harmful outputs. The real over-refusal
evaluation uses the author's own assistant session logs; **that evaluation data
is not released** for privacy. Anyone deploying v8b should re-validate on their
own traffic, add adversarial and multi-turn testing, and keep a human in the loop
for any consequential decision.

## Citation

Part of the constitutional-bioguard line (v2 through v8). See `docs/V8_DESIGN.md`
and `docs/V8B_SHIP_EVIDENCE.md` for design, gates, and the full result trail.

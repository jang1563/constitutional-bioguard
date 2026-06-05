---
license: cc-by-nc-4.0
language:
- en
base_model: microsoft/deberta-v3-base
pipeline_tag: text-classification
library_name: transformers
tags:
- safety
- biosecurity
- biosafety
- guard-model
- prompt-harm
- deberta-v3
- constitutional-classifiers
- dual-use
- experimental
extra_gated_prompt: >-
  constitutional-bioguard-prompt is a defensive bio-safety research artifact (the
  EXPERIMENTAL prompt head of a dual-mode system), released for non-commercial
  research only. By requesting access you agree to the responsible-use terms in the
  model card: use it solely for defensive evaluation and moderation research; do not
  use it as a reward, discriminator, or filter to generate, refine, or evade
  detection of harmful biological content; do not probe it to construct evasion
  strategies; do not redistribute the weights outside this gated channel.
extra_gated_fields:
  Name: text
  Affiliation: text
  Email: text
  Intended use: text
  I agree to the responsible-use terms (defensive evaluation only): checkbox
---

# Model Card: constitutional-bioguard-prompt (EXPERIMENTAL bio prompt-harm recall gate)

A small query-only encoder that scores whether a **request** is a harmful biological
prompt. It is the **prompt head** of the dual-mode Constitutional BioGuard system and is
released as an **experimental, supplementary recall gate** — **not** a calibrated standalone
classifier. Read the scope and limits before using it; they are as important as the numbers.

> **One-line honest summary:** high recall@0.5 (0.983) but **saturated** (AUPRC 0.121 vs the
> 8B teacher's 0.605). It flags nearly everything, so it over-refuses heavily on its own. Its
> only validated value is as the recall gate in the dual-mode **AND** policy, paired with the
> [response head](https://hf.co/jang1563/constitutional-bioguard-response).

## Model details
- **Architecture:** DeBERTa-v3-base (12 layers, hidden 768, ~184M params).
- **Input:** query only. **Output:** binary (harmful prompt vs not) + probability. Default threshold 0.5.
- **Class:** prompt/intent classifier (query-only). It does **not** read responses; for response-harm
  use the [response head](https://hf.co/jang1563/constitutional-bioguard-response).
- **Training:** distilled from an 8B Llama-3.1 + QLoRA generative teacher on a bio prompt pool plus
  generated bio-borderline-benign queries. No harmful content was synthesized.

## Intended use
- **In scope:** the **recall gate** in the dual-mode AND policy — a high-recall pre-generation filter
  that, combined with the response head, drives over-refusal on expert legitimate-bio queries to
  **0.000** (n=181; note competitors also reach 0.000–0.006 on that set).
- **Out of scope / do NOT use for:**
  - **A standalone prompt classifier.** It is uncompetitive on out-of-distribution bio (see Evaluation).
  - **High-consequence gating** without a human in the loop.
  - **Use without text normalization** — character-level evasion bypasses it.

## How to use
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("jang1563/constitutional-bioguard-prompt")
model = AutoModelForSequenceClassification.from_pretrained(
    "jang1563/constitutional-bioguard-prompt", dtype=torch.float32).eval()

inp = tok(query, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    p_harmful = model(**inp).logits.softmax(-1)[0, 1].item()
flag = p_harmful >= 0.5
```
(Load in float32 — DeBERTa-v3's disentangled attention NaNs under fp16.)

## Evaluation (held out, leakage-audited)
**The honest headline — saturation:** recall@0.5 = 0.983 (vs the teacher's 0.900) looks like the
8B bio recall compressed into 184M, but **AUPRC = 0.121 vs the teacher's 0.605**. The high recall
comes from flagging nearly everything, not from discriminating harmful-bio from dangerous-sounding
benign-bio. As a ranker it is an ~80% relative drop from the teacher.

**Prompt-harm recall, SOSBench-bio (n=500 harmful):** this head is **5th of 6** —

| model | recall [95% CI] |
|---|---|
| Granite-Guardian-2B | 0.990 |
| WildGuard-7B | 0.912 [0.88, 0.94] |
| Llama-Guard-3-8B | 0.794 |
| Qwen3Guard-0.6B | 0.768 |
| **this (prompt head)** | **0.752 [0.71, 0.79]** |
| ShieldGemma-9B | 0.300 |

**Standalone over-refusal, OR-Bench-bio (n=740 benign):** **0.845** — it flags most benign bio
queries, vs Llama-Guard-3-8B 0.005 (~170×). This is why it must not be used standalone.

**In-distribution over-refusal** on the clean-bio set is low (0.022), but that does **not** transfer
to out-of-distribution benign bio (above). Treat the prompt head as a high-recall sieve, not a filter.

## Limitations
1. **Saturated, not calibrated** (AUPRC 0.121). Use only as an AND-policy recall gate.
2. **Uncompetitive standalone** — 5th of 6 on OOD prompt recall, worst on OOD over-refusal.
3. **Character-level fragility** — requires the dual-mode text-normalization preprocessor.
4. **The hard dual-use tail is not covered** — ambiguous dual-use bio has no labeled harmful
   examples in any public source, so it is absent from training and evaluation.
5. **Single checkpoint, single seed.** No variance estimate.

## Dual-mode use
This head is one half of the dual-mode system. The deployable value is the **AND** policy
(prompt head ∧ response head) on expert legit-bio traffic; the response head is the workhorse for
general response-harm screening. See the dual-mode card on the
[response head](https://hf.co/jang1563/constitutional-bioguard-response) and the repository docs
(`DUAL_MODE_GUARD_ARTIFACT.md`, `dual_mode_guard.py`).

## Responsible release
Released as a **research artifact and methodology case study**, not a recommended production guard.
The release surface is weights, evaluation code, and documentation; no harmful training examples or
generated harmful content are included. This is defensive biosafety research — the aim is to reduce
over-refusal of legitimate research while catching genuinely harmful requests. Anyone deploying it
should re-validate on their own traffic, keep text normalization on, add adversarial/multi-turn
testing, and keep a human in the loop.

## Citation
Part of the constitutional-bioguard line (prompt head + response head, dual-mode). See
`docs/STEP1_DISTILL_PILOT_2026-06-03.md`, `docs/STEP2_DUALMODE_2026-06-03.md`, and
`INTEGRITY_REVIEW_2026-06-04.md` for design, distillation, gates, and the full result trail.

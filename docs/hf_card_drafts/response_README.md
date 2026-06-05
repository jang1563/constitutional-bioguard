---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
library_name: transformers
tags:
- safety
- moderation
- guard-model
- biosafety
- biosecurity
- response-harm
- deberta-v3
- dual-use
datasets:
- allenai/wildguardmix
- PKU-Alignment/BeaverTails
- AmazonScience/FalseReject
metrics:
- recall
- auroc
model-index:
- name: constitutional-bioguard-response
  results:
  - task:
      type: text-classification
      name: Bio response-harm detection
    dataset:
      type: custom
      name: Held-out real bio responses (n=554, 343 harm / 211 benign)
    metrics:
    - type: recall
      value: 0.921
      name: Recall (95% CI 0.89-0.95)
    - type: auroc
      value: 0.952
      name: AUROC
    - type: fpr
      value: 0.194
      name: Over-refusal (FPR on benign bio responses)
extra_gated_prompt: >-
  constitutional-bioguard-response is a defensive bio-safety research artifact,
  released for non-commercial research only. By requesting access you agree to the
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

# constitutional-bioguard-response (dual-mode response head, v8bh)

The **response head** of the dual-mode Constitutional BioGuard system: a small encoder
(DeBERTa-v3-base, ~184M params) that reads a `query [SEP] response` pair and decides
whether the **response** delivers harmful biological content. It is the releasable
component of the system. The companion query-only gate is
[constitutional-bioguard-prompt](https://hf.co/jang1563/constitutional-bioguard-prompt).
This checkpoint is **v8bh** (density-debiased). This card states where the model is
dominated or weak as plainly as its performance; all numbers are held-out and
leakage-audited (training queries are byte-disjoint from every test set).

> **Name caveat.** Despite "Bio" in the name, this is a GENERAL response-harm guard
> (bio-selectivity S = 1.03): it flags bio-harm (0.853) and non-bio-harm (0.825) at
> nearly the same rate. The name reflects the project's origin, not a validated
> selectivity claim. See Limitation 1.

## Model details
- **Architecture:** DeBERTa-v3-base (12 layers, hidden 768, ~184M params).
- **Input:** `query [SEP] response`. **Output:** binary (harmful response vs not) + probability.
- **Class of model:** response-harm classifier — it judges the response, not the request.
  For prompt/intent screening use the prompt head (link above).
- **Preprocessing (`preprocessing.py`, shipped in this repo):** an input normalization
  layer (`normalize_text`) that strips invisible/zero-width/tag/variation-selector
  characters, folds homoglyphs, decodes URL/base64/hex/ROT13, removes combining marks,
  and applies NFKC. Keep it ON — it is a measured adversarial-robustness defense.
- **Decision threshold:** default 0.5. Probabilities can be temperature-scaled for calibration.

## Intended use
- **In scope:** post-generation response-harm screening where a small (184M) model is
  needed, with text normalization, accepting GENERAL (not bio-specific) harm coverage —
  as a research-grade second-stage filter or offline auditing tool.
- **Out of scope / do NOT use for:**
  - **Prompt/input filtering** — judges responses, not requests; scores ~0 on prompt-only benchmarks by design.
  - **A bio-SELECTIVE classifier** — it is not (Limitation 1).
  - **Sole safety boundary for high-stakes deployment** — it is Pareto-dominated by a smaller open model (Limitation 2).
  - **Use without text normalization** — character-level evasion bypasses it.

## How to use
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "jang1563/constitutional-bioguard-response"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo, dtype=torch.float32).eval()

# apply the shipped input normalization first (preprocessing.py is in this repo)
from preprocessing import normalize_text
query, response = normalize_text(query), normalize_text(response)

# pair encoding tok(query, response) matches training/eval; do NOT concat with [SEP]
inp = tok(query, response, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    p_harmful = model(**inp).logits.softmax(-1)[0, 1].item()  # class 1 = UNSAFE
flag = p_harmful >= 0.5
```
(Load in float32 — DeBERTa-v3's disentangled attention NaNs under fp16.)

## Performance (all leakage-clean vs our training; 95% CIs; see caveats)
Response-harm, real bio responses (n=554, 343 harm / 211 benign), same items for all models:

| model | size | recall [95% CI] | over-refusal |
|---|---|---|---|
| Qwen3Guard-0.6B | 0.6B | 0.933 | 0.142 |
| **this (v8bh)** | **184M** | **0.921 [0.89, 0.95]** | **0.194** |
| WildGuard-7B | 7B | 0.904 | 0.100 |
| Granite-Guardian-2B | 2B | 0.880 | 0.123 |
| Llama-Guard-3-8B | 8B | 0.851 | 0.052 |
| ShieldGemma-9B | 9B | 0.615 | 0.033 |

Threshold-free **AUROC = 0.952**. Recall 0.921 vs WildGuard 0.904: McNemar p=0.248 (not
statistically different); vs Qwen 0.956 native: McNemar p=0.027 (Qwen wins). Competitor CIs
omitted (binary outputs); width is similar at the same n.

![Size-peer Pareto: recall vs over-refusal on bio response-harm (n=554). The 184M response head (crimson) is Pareto-dominated by Qwen3Guard-0.6B — higher recall AND lower over-refusal at a fraction of the size.](size_peer_pareto.png)

## Limitations (measured, not hypothetical)
1. **NOT bio-selective.** Selectivity S = 1.03. A general response-harm guard trained on
   bio+general data, not a bio-discriminating classifier.
2. **Pareto-dominated by a smaller open model.** Qwen3Guard-0.6B has higher recall AND lower
   over-refusal at ~3x the size. There is no operating point where this model is the best choice.
3. **Companion prompt head is saturated, not calibrated** (AUPRC 0.121 vs the 8B teacher's
   0.605). Use it only as an AND-policy recall gate, never standalone.
4. **Character-level fragility (mitigated by preprocessing).** Without normalization, leetspeak
   bypasses 86% / zero-width 73% of detections; with the bundled `normalize_text`: 4% / 0%.
   Normalization must stay ON.
5. **Over-refusal is distribution-specific.** Density-debiasing (this v8bh checkpoint) cut
   held-out FORTRESS-safe over-refusal 0.288 -> 0.016 but did NOT transfer to other benign
   distributions (0.185 -> 0.194).
6. **Conformal certificate is response-head-only, on the calibration distribution.** Valid
   bound: over-refusal <= 20% at 95% confidence, recall 0.878 (not a tighter system guarantee).
7. **Contamination caveat.** Competitor recall on SafeRLHF/BeaverTails slices may be inflated by
   their training; this model is decontaminated only against ITS OWN training.

## Training data
WildGuardMix bio (a GENERAL safety mixture filtered to bio items — why the head is general
rather than bio-selective) + BeaverTails bio (harmful) + FalseReject non-bio negatives (benign)
+ FORTRESS dense-safe hard negatives (the v8bh density-debiasing). Reuse-only, **zero newly
generated harmful content**. All evaluations decontaminated by query-hash against this training
(`audit_leakage.py`: 0 overlap on 5 checks).

## Honest recommendation
If you need a small response-harm guard, use Qwen3Guard-0.6B (better and open). Use THIS model
only if you specifically need a 184M-class encoder, accept general (non-bio) coverage, and value
the transparent, reproducible evaluation. The intended audience is researchers studying
small-guard evaluation, not production deployers seeking the best classifier.

## Evaluation integrity — audits that changed the results
Five self-audits found and corrected silent failures in this work; each is documented with the
numbers that moved (full log: `INTEGRITY_REVIEW_2026-06-04.md`):
1. **fp16-default-load NaN** — transformers 5.9.0 silently loads DeBERTa-v3 in fp16, NaN-ing the
   disentangled attention; fixed by forcing float32. Every prior all-zero/NaN eval traced here.
2. **AUPRC refutes the footprint claim** — the prompt head's recall@0.5 0.983 looked like success;
   AUPRC 0.121 vs teacher 0.605 showed it is saturated, not discriminating.
3. **Operating-point mismatch** — native-threshold ranking flattered us; at matched FPR we lose to
   WildGuard (0.878 vs 0.904 @ FPR 0.10). Treating Qwen "Controversial" as flagged had inflated its over-refusal.
4. **Size-peer class eliminates the niche** — Qwen3Guard-0.6B Pareto-dominates this model.
5. **Conformal certificate was on the wrong checkpoint** — recomputed for shipped v8bh: over-ref <= 20%, recall 0.878.

## Responsible release
Released as a **research artifact and methodology case study**, not a recommended production
guard. The release surface is weights, evaluation code, and documentation; no harmful training
examples, generated harmful content, or operational instructions are included. This is defensive
biosafety research. Anyone deploying it should re-validate on their own traffic, keep text
normalization on, add adversarial/multi-turn testing, and keep a human in the loop.

## License & citation
License: CC BY-NC 4.0 (weights, eval code, docs are open; no harmful training examples
distributed). Successor to [`jang1563/constitutional-bioguard-deberta-v1`](https://hf.co/jang1563/constitutional-bioguard-deberta-v1).
Full design and result trail (in the GitHub repo):
[MODEL_CARD.md](https://github.com/jang1563/constitutional-bioguard/blob/main/docs/MODEL_CARD.md) ·
[CASE_STUDY_eval_self_red_team.md](https://github.com/jang1563/constitutional-bioguard/blob/main/docs/CASE_STUDY_eval_self_red_team.md) ·
[INTEGRITY_REVIEW_2026-06-04.md](https://github.com/jang1563/constitutional-bioguard/blob/main/docs/INTEGRITY_REVIEW_2026-06-04.md) ·
[POSTMORTEM_2026-06-04.md](https://github.com/jang1563/constitutional-bioguard/blob/main/docs/POSTMORTEM_2026-06-04.md).

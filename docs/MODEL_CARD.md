# Model Card: Constitutional BioGuard (dual-mode, 2x184M)

An honest model card. Unlike most guard releases, this documents exactly where the model LOSES,
its robustness profile, contamination caveats, and a size-peer comparison. The value of this
release is the transparency, not the performance.

## What it is
Two 184M DeBERTa-v3 encoders combined by a configurable policy (DualModeGuard):
- **Response head (v8bh)**: classifies (query, response) for response-harm. The releasable component.
- **Prompt head**: classifies query-only for prompt-harm. EXPERIMENTAL (see limitations); a
  high-recall pre-generation gate, not a calibrated standalone classifier.
- **Policies**: prompt_only / response_only / and / or. Text normalization is applied by default.

## Intended use
- Post-generation response-harm screening where a small (184M) model is needed, with text
  normalization preprocessing, accepting GENERAL (not bio-specific) harm coverage.
- The prompt head + AND policy: a supplementary low-over-refusal gate on CLEAN expert legit-bio
  research queries.

## Out of scope / do NOT use for
- As a bio-SELECTIVE classifier (it is not; see Limitation 1).
- As a standalone prompt classifier (the prompt head is uncompetitive on out-of-distribution bio).
- Without text normalization (character-level evasion bypasses it).
- As a sole safety boundary for high-stakes deployment (it is Pareto-dominated by smaller open models).

## Performance (all leakage-clean vs our training; 95% CIs; see caveats)
RESPONSE-harm, real bio responses (n=554, 343 harm / 211 benign):
| model | size | recall [95% CI] | over-refusal |
|---|---|---|---|
| Qwen3Guard-0.6B | 0.6B | 0.933 | 0.142 |
| **this (v8bh)** | **184M** | **0.921 [0.89,0.95]** | **0.194** |
| WildGuard-7B | 7B | 0.904 | 0.100 |
| Granite-Guardian-2B | 2B | 0.880 | 0.123 |
| Llama-Guard-3-8B | 8B | 0.851 | 0.052 |
| ShieldGemma-9B | 9B | 0.615 | 0.033 |
threshold-free AUROC (this) = 0.952.

PROMPT-harm, SOSBench-bio (n=500 harmful):
| model | recall [95% CI] |
|---|---|
| Granite-Guardian-2B | 0.990 |
| WildGuard-7B | 0.912 [0.88,0.94] |
| Llama-Guard-3-8B | 0.794 |
| Qwen3Guard-0.6B | 0.768 |
| **this (prompt head)** | **0.752 [0.71,0.79]** |
| ShieldGemma-9B | 0.300 |

## Limitations (measured, not hypothetical)
1. **NOT bio-selective.** Selectivity S = 1.03 (flags bio-harm 0.853 vs non-bio-harm 0.825). It is
   a general response-harm guard trained on bio+general data, NOT a bio-discriminating classifier.
2. **Pareto-dominated by a smaller open model.** Qwen3Guard-0.6B has higher recall AND lower
   over-refusal at 3x the size. There is no operating point where this model is the best choice.
3. **Prompt head is saturated, not calibrated.** AUPRC 0.121 vs the 8B teacher's 0.605; high
   recall@0.5 comes from flagging nearly everything. Use only as an AND-policy recall gate.
4. **Character-level fragility (mitigated by preprocessing).** Without normalization, leetspeak
   bypasses 86% / zero-width 73% of detections. With the bundled text_normalize layer: 4% / 0%.
   Normalization is ON by default; do not disable it.
5. **Over-refusal is distribution-specific.** Density-debiasing (v8bh) cut held-out FORTRESS-safe
   over-refusal 0.288 -> 0.016 but did NOT transfer to other benign distributions (0.185 -> 0.194).
6. **Conformal certificate is response-head-only, on the calibration distribution.** Valid bound:
   over-refusal <= 20% at 95% confidence, recall 0.878 (NOT a tighter system-level guarantee).
7. **Contamination caveat.** Competitor recall on SafeRLHF/BeaverTails slices may be inflated by
   their training; this model is decontaminated only against ITS OWN training.

## Deployment requirements
- Keep text normalization enabled (DualModeGuard does this by default).
- Choose policy by distribution: AND for clean expert legit-bio (lowest over-refusal), response_only
  otherwise. AND has a jailbreak recall cost (benign-looking query + harmful response).
- Treat as one layer, not a sole safety boundary.

## Training data
Response head: WildGuardMix bio + BeaverTails bio (harmful) + FalseReject + non-bio negatives
(benign), + FORTRESS dense-safe hard negatives (v8bh debiasing). Prompt head: distilled from an
8B Llama-3.1+QLoRA generative teacher on a bio prompt pool + generated bio-borderline-benign.
All evaluations decontaminated by query-hash against this training (audit_leakage.py: 0 overlap).

## Honest recommendation
If you need a small response-harm guard: use Qwen3Guard-0.6B (better and open). Use THIS model only
if you specifically need a 184M-class encoder, accept general (non-bio) coverage, and value the
transparent, reproducible evaluation. The intended audience is researchers studying small-guard
evaluation, not production deployers seeking the best classifier.

## Evaluation methodology
Leakage-clean (query-hash decontamination), 95% Clopper-Pearson CIs, McNemar paired tests,
matched-operating-point and AUROC comparisons, per-source contamination breakdowns, size-peer
benchmarking, character-robustness probing. See CASE_STUDY_eval_self_red_team.md and
INTEGRITY_REVIEW_2026-06-04.md.
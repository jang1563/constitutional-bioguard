# Model Card: Constitutional BioGuard (dual-mode, 2x184M)

> **Authoritative card** for the released dual-mode guard (response head **v8bh** + prompt head).
> Supersedes `MODEL_CARD_V4.md` and `V8B_MODEL_CARD.md`.

**Name caveat.** Despite "Bio" in the name, the response head is a GENERAL response-harm guard
(bio-selectivity S = 1.03). The name reflects the project's origin, not a validated selectivity
claim. See Limitation 1.

This card documents where this model is dominated or weak, its robustness profile, contamination
caveats, and a size-peer comparison, alongside its performance.

## What it is
Two 184M DeBERTa-v3 encoders combined by a configurable policy (DualModeGuard):
- **Response head (v8bh)**: classifies (query, response) for response-harm. The releasable component.
- **Prompt head**: classifies query-only for prompt-harm. EXPERIMENTAL (see limitations); a
  high-recall pre-generation gate, not a calibrated standalone classifier.
- **Policies**: prompt_only / response_only / and / or. Text normalization is applied by default.

## Intended use
- Post-generation response-harm screening where a small (184M) model is needed, with text
  normalization preprocessing, accepting GENERAL (not bio-specific) harm coverage.
- The prompt head + AND policy: a supplementary gate on expert legit-bio research queries (AND
  over-refusal 0.000 on n=181, though competitors also achieve 0.000-0.006 on the same set).

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
threshold-free AUROC (this) = 0.952. All models tested on the SAME items (n=554); competitor CIs
omitted (binary outputs, no score -- CI width ~similar at same n). Recall 0.921 vs WildGuard 0.904:
McNemar p=0.248 (not statistically different); vs Qwen 0.956: McNemar p=0.027 (Qwen wins).

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
Response head: WildGuardMix bio (a GENERAL safety-training mixture filtered to bio items, which is
why the head is general rather than bio-selective) + BeaverTails bio (harmful) + FalseReject
non-bio negatives (benign) + FORTRESS dense-safe hard negatives (v8bh debiasing). Prompt head: distilled from an
8B Llama-3.1+QLoRA generative teacher on a bio prompt pool + generated bio-borderline-benign.
All evaluations decontaminated by query-hash against this training (audit_leakage.py: 0 overlap).

## Honest recommendation
If you need a small response-harm guard: use Qwen3Guard-0.6B (better and open). Use THIS model only
if you specifically need a 184M-class encoder, accept general (non-bio) coverage, and value the
transparent, reproducible evaluation. The intended audience is researchers studying small-guard
evaluation, not production deployers seeking the best classifier.

## Evaluation Integrity -- audits that changed the results

A safety classifier card is only as honest as the audits behind it. Five self-audits found and
corrected silent failures in this work; each is documented with the specific numbers that moved.

**1. fp16-default-load NaN (the trainer bug).** transformers 5.9.0 silently loads DeBERTa-v3 in
fp16, which NaNs the disentangled attention. Logged train_loss was finite (0.044) but all eval
was zero/NaN. Root-caused by isolating fresh encoders in fp32 (fine) vs the Trainer's fp16 path
(NaN). Fix: `dtype=torch.float32` in from_pretrained. Every prior NaN/all-zero traced to this
single cause.

**2. AUPRC refutes the footprint claim.** recall@0.5 = 0.983 (student) vs 0.900 (teacher) looked
like success. AUPRC = 0.121 vs 0.605 — the student is saturated, not discriminating. A
single-threshold metric hid an 80% relative drop in ranking quality (AUPRC). This audit changed CLAIM 1 from "footprint
solved" to "footprint failed at AUPRC."

**3. Operating-point mismatch inflated competitive ranking.** Native-threshold comparison placed
ours 2nd on response-recall (0.921). At matched FPR (threshold tuned to competitor's over-refusal), ours loses to WildGuard (0.878 vs
0.904 @ FPR 0.10). Qwen native recall 0.956 vs ours native 0.921 (McNemar p=0.027). Additionally, treating Qwen's
"Controversial" as flagged inflated its over-refusal 0.005 -> 0.076 (unfair to competitor).

**4. Size-peer class eliminates the niche.** Qwen3Guard-0.6B Pareto-dominates ours (recall 0.933
vs 0.921 AND over-ref 0.142 vs 0.194). This comparison was added in a second audit pass.

**5. Conformal certificate was on the wrong model.** The "over-ref <= 10%, recall 0.80" bound was
computed on v8b, not the shipped v8bh. v8bh's valid bound: over-ref <= 20%, recall 0.878 only.

**Practice:** a silently-wrong score is worse than a loud error. Results that move under audit
should be reported (not buried), and the integrity log lives in the repository.

## Risk-Forward Use

Components that may be useful independent of the model:

- **Safeguard teams** can use the 7-lesson evaluation checklist (CASE_STUDY) as a template for
  auditing their own classifiers: AUPRC not recall, contamination per-source, CIs, matched
  operating points, bio-selectivity checks, size-peer benchmarks, char-robustness probes.
- **Guard developers** can reuse the density-debiasing recipe (within-distribution hard negatives)
  and the text-normalization preprocessor as components, independent of this model.
- **Benchmark designers** can cite the contamination finding (SafeRLHF/BeaverTails overlap inflates
  competitors) and the n=30->n=500 ranking reversal as evidence that guard leaderboards need
  per-source and adequately-powered evaluation.

## Responsible Release

This model is released as a **research artifact and methodology case study**, not as a recommended
production guard. The
release surface is limited to model weights, evaluation code, and documentation; no harmful
training examples, generated harmful content, or operational instructions are included.

## Evaluation methodology
Leakage-clean (query-hash decontamination, audit_leakage.py: 0 overlap on 5 checks), 95%
Clopper-Pearson CIs, McNemar paired tests, matched-operating-point and AUROC comparisons,
per-source contamination breakdowns, size-peer benchmarking, character-robustness probing,
bio-selectivity ratio, AUPRC. See CASE_STUDY_eval_self_red_team.md, INTEGRITY_REVIEW_2026-06-04.md,
and POSTMORTEM_2026-06-04.md.

## License
CC BY-NC 4.0 (non-commercial). The released dual-mode weights (prompt + response heads) are gated and
inherit NonCommercial terms from their training sources (BeaverTails, FalseReject); the evaluation code
and documentation are open. No harmful training examples are distributed. (The legacy
`constitutional-bioguard-deberta-v1` checkpoint stays under its original MIT license.)

## Citation
If citing the evaluation methodology or negative findings, reference this model card and the
accompanying CASE_STUDY_eval_self_red_team.md.
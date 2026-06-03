# Research refresh (2026-06-03): dual-mode bio guard, competitive + technique delta

Extends the four research sweeps in `DUAL_MODE_DESIGN.md` §8 with a decision-relevant
delta, triggered by the v7.C (Llama-3.1-8B prompt head) result: OOD prompt recall 0.883
but over-refusal 0.377 on adversarial benchmark negatives, at an 8B footprint that breaks
the ~184M cascade target. Deep-research pass: 5 angles, 25 sources fetched, 113 claims
extracted, 25 adversarially verified (3-vote, 2/3-to-kill), 21 confirmed / 4 killed.

Bottom line: **the strategy is intact and in two respects de-risked.** The "open
bio-specialized dual-mode guard is empty space" thesis holds; the v7.C over-refusal
problem has concrete published recipes; and the 8B footprint is solvable by distillation
into the same DeBERTa-v3 family as the response head.

---

## (A) Competitive freshness + thesis validation

**The empty-bio-space thesis HOLDS, and is strengthened.** No open, bio-specialized,
dual-mode guard reporting a separate per-bio metric has appeared. The two closest 2026
entrants do not occupy the niche:

- **BioTIER** (SecureBio; "Biological Targeted Information for Exclusion and Refusal"):
  an eval/policy/data-exclusion **resource** (542 expert-curated refuse/permit prompts +
  recommended safeguard policies + pre-training exclusion data), for developers to apply
  to their own models. Not a deployable guard. (Useful to us as a bio over-refusal corpus,
  see C; but "matched twin pairs" structure is unconfirmed, appears stratified across three
  risk sets.) [securebio.org/biotier; verify 3-0 non-guard]
- **Biosecurity Agent** (arXiv 2510.09615, Llama-3-8B): a tool-orchestrated lifecycle
  pipeline (dataset sanitization, DPO+LoRA, dual input/output runtime guardrails,
  post-deployment red-team). Architecturally close to our v7.C direction, BUT its guardrail
  is evaluated on only **60 custom prompts** (30 harmful / 30 safe), against no named bio
  over-refusal benchmark and no competitor guard. Sets no open per-bio SOTA bar. [3-0]

**General open guards still have no per-bio breakout.** Qwen3Guard's 9-category taxonomy
lumps all weapon manufacture/acquisition/use into one "Violent" label (no bio/chem/CBRN
breakout; Fig. 5 reports only aggregated per-category results, so a per-bio metric is
structurally impossible). A major new 14-model / 79,331-sample open-guard benchmark
(arXiv 2605.28830, ICLR 2026 workshop) defines 8 NIST safety subcategories and **explicitly
filters OUT terrorism and weapons**, conceding "domain-specific applications may require
specialized benchmarks." The field is literally pointing at the gap we occupy. [3-0]

**Re-verified SOTA bars (use these, not the old composites):**

| Guard | Prompt-harm avg F1 | Response-harm avg F1 |
|---|---|---|
| Qwen3Guard-8B-Gen | 90.0 best-of-modes / **~88.9 single-point (strict)** | 83.9 / **~84.0 single-point** |
| WildGuard-7B | 85.8 | 79.9 |

Qwen3Guard's headline 90.0/83.9 are **best-of-modes composites** (optimal Strict/Loose per
benchmark). For an honest v7.C/v8b head-to-head, compare against the single-operating-point
~88.9 / ~84.0. Qwen3Guard-Gen tops 8 of 14 public English benchmarks. [3-0, source 2510.14276]

**"Bigger generative guard" is NOT reliably safer (cautionary, not decisive).** On the mixed
non-bio 79k benchmark, WildGuard-7B falls to F1 0.73; GPT-OSS Safeguard 20B misses 75.1% of
unsafe content (recall 0.25); Llama Guard 12B misses ~67%; size-vs-performance correlation
r=0.21. CAVEAT (lowers generality, not the numbers): non-bio set, single SAFE source
(RealToxicityPrompts), fixed 0.5 threshold, policy-conditioned GPT-OSS not swept. Read it as:
**a big generative guard at its DEFAULT operating point can be the worst** — which motivates
explicit operating-point control (B) and per-domain tuning, exactly our move. [bars 3-0]

**Over-refusal on benign bio re-verifies EXACTLY as cited.** Health-ORSC-Bench Hard-1K:
Claude-Opus-4.1 refuses **83.47%** of benign Biological/Chemical Harm prompts (confirms the
~83% figure); GPT-OSS-120B 85.95% (category max); safety-optimized models refuse up to ~80%
of Hard benign overall. [3-0, source 2601.17642]
- UNVERIFIED: the RefusalBench "~0.1%-95% refusal spread on identical bio prompts" figure was
  not confirmed in this batch. Re-check the exact split/version before using it as a motivator.

---

## (B) Techniques: the v7.C over-refusal + footprint problem are both solvable

### B1. Operating-point control for a GENERATIVE guard, no retraining

- **Read the token probability as a continuous score.** The red-flag-token method
  (arXiv 2502.16366, Llama-3 base) monitors `p(<rf>|x)` directly (no logit-bias, no waiting
  for explicit generation) and sets a model-dependent threshold, sweeping a full TPR-vs-FPR
  ROC. Direct transfer to v7.C: read `p("harmful")` vs `p("unharmful")` at the decision token
  and ROC-sweep, instead of reporting the single argmax point (currently implicit τ=0.5). [3-0]
- **Strict/Loose abstain tier.** Qwen3Guard ships a tri-class "Controversial" middle label
  between Safe and Unsafe; Strict Mode maps Controversial→unsafe, Loose Mode→safe, switchable
  at inference on one model with no retraining. This is the "dual-use / abstain" tier our
  design already anticipated. [3-0]

### B2. If threshold-sweep is not enough: benign-aware / RL recipes

- **IntentionReasoner** (arXiv 2508.20151, Qwen2.5 1.5B/3B/7B; SFT then GRPO): four-level
  taxonomy (Completely Unharmful / BORDERLINE Unharmful / BORDERLINE Harmful / Completely
  Harmful) plus **query rewriting** that neutralizes harmful intent instead of refusing edge
  cases. The RL stage raises F1 5-6% **primarily by reducing over-refusal** (IR-3B 93.3→99.2).
  [3-0]  (Correction: middle labels are "BORDERLINE", not "Benign".)
- **SelfGrader** (arXiv 2604.01473): dual-perspective score combining a maliciousness term and
  a **benignness** term; ablating the benignness term spikes FPR 1.91%→29.26% — i.e., an
  explicit benign-aware term is the over-refusal-suppression mechanism. [ablation 2-1]
  (Do NOT cite the refuted framings: SelfGrader as a clean τ_D=(Q-1)/2 knob; its 1.91%-FPR
  "Pareto dominance"; or IntentionReasoner-1.5B beating all 7-8B guards. All killed in voting.)

### B3. Footprint: distillation makes the 8B problem a deployment choice, not a blocker

- **HarmAug** (arXiv 2410.01524, ICLR 2025, Bengio co-author; public `hbseong/HarmAug-Guard`):
  distills Llama-Guard-3-**8B** (generative teacher) into a **435M DeBERTa-v3-large** student
  that matches teacher avg F1 (0.736 vs 0.705) and **outperforms 7B+ guards on AUPRC** at
  **<25% compute**. The student is the **same DeBERTa-v3 family as our v8b response head** —
  so a distilled v7.C prompt head and v8b could share an architecture. CAVEATS: general-domain
  (no bio/CBRN eval); "no F1 loss" is an average (teacher still wins HarmBench / OpenAI-Mod);
  gains are augmentation-driven. [F1-match 2-1, AUPRC-lead 2-1]
- **SafeRoute** (arXiv 2502.12464, ACL 2025 Findings): adaptive routing — a lightweight binary
  router sends easy inputs to a 1B guard and only hard inputs to an 8B guard. An alternative to
  distillation if we want to keep the 8B teacher live. [3-0]

---

## (C) Over-refusal benchmark standard for bio

De-facto 2026 standard = **paired harm + over-refusal at a SINGLE operating point on matched
harmful/benign twin pairs, with competitors run through the SAME corpus.** Report a
(recall, over-refusal) point at one threshold, never a lone recall number. Two named corpora:

- **FORTRESS** (arXiv 2506.14922, Scale AI SEAL, June 2026): 500 expert-crafted adversarial
  CBRNE / national-security prompts, **each with a 1:1 benign twin**; reports Average Risk
  Score (ARS) paired with Over-Refusal Score (ORS) per model at one operating point (e.g.,
  Claude-3.5-Sonnet ARS 14.09 / ORS 21.8; Gemini-2.5-Pro 66.29 / 1.4; Deepseek-R1 78.05 / 0.06;
  o1 21.69 / 5.2). CAVEAT: only ~30 of 500 are biological (CBRNE-broad, not bio-deep). [3-0]
- **Health-ORSC-Bench** (arXiv 2601.17642): 31,920 benign boundary health prompts, dedicated
  **Biological/Chemical Harm** category, Easy/Medium/Hard stratification. Bio-deeper but
  over-refusal-ONLY (no harmful arm). Has the published Claude-Opus 83.47% number for a direct
  head-to-head. [3-0]

Recommendation: report BOTH. FORTRESS for the paired harm+ORS operating point (matched twins),
Health-ORSC-Bench Hard-1K Biological/Chemical Harm for bio-depth over-refusal vs frontier
models. (BioTIER refuse/permit is a third bio option but not matched-twin and access-gated.)

---

## Strategic flags (what changed / confirmed)

1. **Thesis HOLDS + strengthened.** Empty bio-specialized-dual-mode space confirmed; a new
   14-model benchmark explicitly excludes weapons/bio and calls for specialized benchmarks.
2. **v7.C over-refusal is a known, solvable problem.** Free first: token-probability ROC sweep.
   Then, if needed, benign-aware dual scoring or SFT-then-RL + borderline/abstain tier.
3. **8B footprint is NOT a blocker.** HarmAug-style distillation into a 435M DeBERTa-v3 (v8b's
   family) is published and near-lossless on general domain; bio preservation is the open risk.
4. **0.377 is on adversarial negatives.** The field shows frontier safety models hit 80%+ on
   hard benign bio, and our own v8b experience showed benchmark over-refusal overstates real
   (27%→2%). Measuring v7.C on a real/named corpus is the decisive next experiment.

## Corrections / unverified to carry forward

- Qwen3Guard SOTA = best-of-modes composite; use ~88.9 / 84.0 single-point for head-to-head.
- RefusalBench "0.1%-95%" spread: UNVERIFIED, re-check before use.
- IntentionReasoner middle labels are "BORDERLINE", not "Benign".
- Do not cite the 3 refuted SelfGrader/IntentionReasoner framings (see B2).
- HarmAug / SafeRoute footprint evidence is general-domain, NOT bio-validated.

## Ordered next-step plan (research-informed)

1. **FREE — token-probability operating curve for v7.C.** Re-score the existing eval reading
   `p(harmful)` vs `p(unharmful)` at the decision token; produce the full recall/over-refusal
   ROC instead of the single 0.883/0.377 point. May relocate v7.C to a good point for zero cost.
2. **Measure over-refusal on a real/named corpus.** Health-ORSC-Bench Hard-1K Bio/Chem (direct
   Claude-Opus 83.47% head-to-head) and/or FORTRESS twins (paired ARS/ORS). This is the number
   that decides whether v7.C is a viable prompt head.
3. **Only if 1+2 insufficient:** benign-aware retrain (SelfGrader dual scoring) or SFT-then-RL
   (IntentionReasoner) with a borderline/abstain tier.
4. **Footprint:** plan to distill the winning bio teacher → 435M DeBERTa-v3 student (HarmAug
   recipe, same family as v8b) → unified sub-1B dual-mode guard.
5. **Reporting:** adopt FORTRESS (paired) + Health-ORSC-Bench Hard-1K Bio/Chem (bio-depth) as
   the named over-refusal standard; always report paired (recall, over-refusal) at one point.

## Sources (verified, primary unless noted)

2510.09615 Biosecurity Agent · 2605.28830 14-model open-guard benchmark · 2510.14276 Qwen3Guard ·
2601.17642 Health-ORSC-Bench · 2502.16366 Red Flag Token · 2506.14922 FORTRESS ·
2604.01473 SelfGrader · 2508.20151 IntentionReasoner · 2410.01524 HarmAug · 2502.12464 SafeRoute ·
securebio.org/biotier BioTIER (access-gated, snippet+Substack corroboration).

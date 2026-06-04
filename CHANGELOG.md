# Changelog

All notable changes to this project are documented here. This is a research
prototype: "released" means the public Hugging Face checkpoint, not a production
system. The format loosely follows [Keep a Changelog](https://keepachangelog.com).
Numbers for unreleased work are intentionally omitted until those checkpoints are
released; see the technical report and design docs for internal evidence.

## [Unreleased] — dual-mode (prompt + response)

- **Design.** Move from a single response classifier toward a dual-mode
  (prompt-harm + response-harm), bio-specialized guard with independent per-axis
  thresholds (`docs/DUAL_MODE_DESIGN.md`).
- **Response head (reuse-only).** Successor response-harm classifiers train on
  reused, leakage-audited real data instead of synthetic-only generation, and are
  validated on real legitimate-research over-refusal rather than a synthetic holdout.
- **Prompt head (generative).** Added a token-probability operating-curve sweep and a
  real-over-refusal evaluation harness for the prompt axis, plus a leakage-safe,
  reuse-only targeted benign-aware augmentation pipeline.
- **Research refresh.** 2026 competitive/technique/benchmark review
  (`docs/RESEARCH_REFRESH_2026-06-03.md`): the open bio-specialized dual-mode niche
  holds; generative-guard over-refusal is threshold-tunable; large generative guards
  are distillable into a ~435M encoder.
- Not yet released.

## [0.2.0] — 2026-05 — v4 response-diverse (recommended internal checkpoint)

- **v4 response-diverse** breaks v3's compliance-template shortcut while keeping the
  bio-specialist boundary: 2.1% OR-Bench-Hard-1K FPR, 0% XSTest FPR, 32% WildGuard
  native bio recall, 0.45 BioThreat-Eval F1 (private preview).
- **v5 PairCFR** documented as an honest non-release: fixes the artificial hybrid-FPR
  case but collapses specialist bio recall.
- Leakage audit: restated earlier OR-Bench-Health and HarmBench/AdvBench "held-out"
  numbers as training-distribution evidence after overlap checks.
- Added `CITATION.cff`, Dockerfile, Makefile, CI, and the repository quality checklist.

## [0.1.0] — v1 (public) and early diagnostics

- **v1** public DeBERTa-v3-base classifier (`constitutional-bioguard-deberta-v1`),
  trained on synthetic data from a 56-rule / 7-NSABB-category biosafety constitution.
  Strong synthetic-test metrics later shown to rest on an adversarial-framing shortcut.
- **v2 / v3** diagnostic iterations (SAFE augmentation → recall collapse; rebalancing
  → compliance-template shortcut), which motivated the v4 remediation.

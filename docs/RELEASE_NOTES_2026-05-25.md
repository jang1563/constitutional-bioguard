# Constitutional BioGuard v0.2.0 Release Notes

## Release Overview

This release promotes **v4 response-diverse** as the recommended release
checkpoint and documents the v5 PairCFR experiment as a deliberate non-release.
The major change is not just another model iteration: the reporting
surface now reflects the Goodhart/leakage audit that restated several older
claims.

## Recommended Model

- Hugging Face model: `jang1563/constitutional-bioguard-v4` (private preview)
- Base model: `microsoft/deberta-v3-base`
- Scope: bio-specialist query/response classification, not general safety
- Key clean gates:
  - OR-Bench-Hard-1K FPR: `2.1%`
  - XSTest FPR: `0.0%`
  - WildGuard native bio recall: `32.0%`
  - WildGuard native F1: `0.43`
  - BioThreat-Eval F1: `0.45`
  - CRT compliance flag rate: `29%` versus v3's `100%`

## v5 Decision

v5 was trained with clean splits plus PairCFR at `lambda=0.3` and
`temperature=0.1`. It is **not released**.

| Gate | Target | v4 | v5_baseline | v5 PairCFR |
|---|---:|---:|---:|---:|
| OR-Bench-Hard-1K FPR | < 5% | 2.1% | 55.3% | 0.0% |
| XSTest FPR | 0% | 0.0% | 16.0% | 0.0% |
| WildGuard native bio recall | >= 28% | 32.0% | 62.5% | 17.1% |
| CRT refusal+compliance FPR | < 35% | 68% | 100% | 10% |

PairCFR fixed the artificial hybrid-response FPR but collapsed specialist bio
recall too far for release. This result is preserved as evidence for the
precision/recall trade-off rather than hidden behind a release tag.

## Restated Claims

- OR-Bench-Health `1.22%` FPR is now treated as training-distribution evidence,
  not held-out generalization, because the v4 audit found full train/eval
  overlap for that subset.
- HarmBench/AdvBench/JailbreakBench bio "held-out" recall from v3-era reporting
  is restated as training-distribution recall due to source reuse.
- Clean transfer claims now rest on OR-Bench-Hard-1K, XSTest, WildGuard native,
  BioThreat-Eval, SaladBench/ALERT CBRN selectivity, and BioThreat-based CRT
  probes.

## New Code and Artifacts

- `constitutional_bioguard/training/paircfr_trainer.py`
- `constitutional_bioguard/training/splice_projector.py`
- `scripts/create_v5_splits.py`
- `scripts/train_v5_baseline.py`
- `scripts/train_v5.py`
- `scripts/v5_eval_all_gates.py`
- `scripts/g1_v5_overlap_audit.py`
- `scripts/g2_refusal_prefix_bypass.py`
- `scripts/v5_probe_preregister_v2.py`
- `docs/MODEL_CARD_V4.md`
- `data/metrics/v5_acceptance_check.json`

## Recommended Next Steps

1. Keep the v4 checkpoint and model card synchronized on Hugging Face.
2. Tag `v0.2.0` after GitHub/HF consistency checks.
3. Treat lower PairCFR weights (`lambda=0.1` or `0.15`) or a cascade-first v6
   design as the next experimental branch, not as a patch release.

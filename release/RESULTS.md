# bioguard-v8b: results and how to reproduce

Every number in the model card, with the script that produces it. All evaluation
sets are held out and leakage-audited (training queries byte-disjoint from every
test set). Sets and scripts live in the project repo (not all shipped with the
gated weights; see RELEASE_CHECKLIST withhold list).

## Discrimination (harmful bio vs real legitimate research)

| Metric | Value | Producing script |
|--------|-------|------------------|
| AUROC | 0.970 | `scripts/r2_calibrate_v8b.py` |
| AUPRC | 0.938 | `scripts/r2_calibrate_v8b.py` |
| Real bio-response recall | 0.919 (57/62; Wilson 95% CI 0.825-0.965; directional) | `scripts/eval_v8_realresponse_bio.py` |
| General response-harm (WildGuard-native, n=1709) | recall 0.694 / F1 0.667 / FPR 0.077 | `scripts/eval_v8_gates.py` |

## Over-refusal on real legitimate bio research

| Set | n | FPR | Producing script |
|-----|---|-----|------------------|
| All real legit bio | 531 | 0.021 | `scripts/eval_v8_overrefusal_realsessions.py` |
| Author's own assistant sessions | 134 | 0.060 | same |
| Substantive-response subset | 68 | 0.015 | same |

## Calibration and operating point (`scripts/r2_calibrate_v8b.py`)

- Temperature scaling T=0.239 (held-out cal split): ECE 0.137 -> 0.042; Brier 0.053 -> 0.035.
- Operating point: default threshold 0.5 -> recall 0.919 at 2.1% over-refusal
  (Wilson 95% CI 0.825-0.965). Recall 0.95 costs ~15% over-refusal.

## Adversarial robustness

Char-injection ESR after the shipped `normalize_text` (`scripts/eval_v8b_robust_charinjection.py`):

| Attack | ESR |
|--------|-----|
| zero-width / fullwidth / Cyrillic homoglyph | 0.000 / 0.000 / 0.035 |
| combining diacritic / tag chars / variation selector | 0.000 / 0.000 / 0.000 |
| intra-word spacing (residual) | 0.211 |
| Greek homoglyph (residual, passing) | 0.105 |

| Attack | Metric | Producing script |
|--------|--------|------------------|
| Adversarial word perturbation | ESR 0.123 (greedy char-swap) | `scripts/eval_v8b_robust_advword.py` |
| Multi-turn naive reconstruction | per-turn recall 0.964 vs windowed 1.0 | `scripts/eval_v8b_multiturn_reconstruction.py` |
| Lexical (bio-keyword) ablation | ~1% prediction change | `scripts/probe_v8b_shortcut.py` |

## Scope boundary

On stub-response benchmarks (SaladBench O39, ALERT-CBRN, SimpleSafety-bio) v8b
flags ~0 by design: they attach a constant template response and test prompt harm,
which a response-harm classifier correctly ignores (`scripts/eval_v8_gates.py`).

## Reproduce

```bash
# real-response bio recall (and the leakage-clean benchmark build)
python scripts/eval_v8_realresponse_bio.py
# real over-refusal money metric
python scripts/eval_v8_overrefusal_realsessions.py
# calibration + operating point + AUROC/AUPRC
python scripts/r2_calibrate_v8b.py
# robustness suite
python scripts/eval_v8b_robust_charinjection.py
python scripts/eval_v8b_robust_advword.py
python scripts/eval_v8b_multiturn_reconstruction.py
```

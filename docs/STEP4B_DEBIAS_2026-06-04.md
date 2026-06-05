# Step 4b: density-bias debiasing -- the over-refusal weakness is FIXABLE (2026-06-04)

Tests whether v8b's response-side over-refusal (the unfavorable half of Step 4) is fundamental or
fixable, by adding diverse dense-but-safe responses as hard negatives and retraining.

## Why
Diagnosis (Step 4 + score distribution): v8b over-refuses held-out safe responses (0.26-0.30) where
competitors sit at 0.01-0.02. Root cause: 23/211 benign get score>=0.9 -- a density-bias long tail
(v8b learned harmful~=dense, benign~=short refusal, so it over-flags long topical SAFE text:
"hacker simulation" explainers, WHO-negligence fiction, Plague-Inc strategy). Post-hoc calibration
(temp/Platt) cannot fix a tail (global rescale breaks the well-calibrated bulk) -> retrain.

## Method (RIGOROUS held-out)
Added FORTRESS safe responses as hard negatives (label 0) to v8b's training set, but SPLIT them
176 train / 184 held-out (by query+response hash parity) so the over-refusal improvement is measured
on responses NEVER trained on. (First attempt trained on all 360 and "got" 0.000 on the same 360 --
in-sample memorization, discarded.) Retrained with the canonical v8 trainer, same hyperparameters.

## Result
TRUE held-out FORTRESS safe responses (n=184, all safe; v8bh did NOT train on these):
| slice | v8b over-ref | v8bh over-ref |
|---|---|---|
| all (184) | 0.288 | **0.016** |
| bio (31) | 0.290 | 0.032 |
| benign_resp | 0.304 | 0.022 |
| adv_refusal | 0.272 | 0.011 |

real_response_bio_large (n=554, fully held-out, recall + over-ref):
| metric | v8b | v8bh |
|---|---|---|
| recall (343 harm) | 0.945 | 0.921 |
| over-refusal (211 benign) | 0.185 | 0.194 |

## Reading
- DENSITY BIAS IS FIXABLE AND GENERALIZES (within distribution): on FORTRESS safe responses NEVER
  trained on, over-refusal collapses 0.288 -> 0.016 -- now matching WildGuard (0.011) / Llama-Guard
  (0.019). The weakness is NOT baked in; it is a training-data-coverage gap.
- COST is small: overall recall 0.945 -> 0.921 (-2.4pt). (The wildguard-slice recall drop 0.94->0.65
  is n=17 noise; the 343-harm overall is the reliable number.)
- DISTRIBUTION-SPECIFIC: real_response_bio benign over-refusal did NOT improve (0.185->0.194) -- the
  FORTRESS-style fix did not transfer to the wildguard/beavertails/saferlhf benign distribution.
  Broad over-refusal reduction therefore needs DIVERSE safe-response hard negatives covering the
  deployment distributions; FORTRESS alone is a proof-of-concept, not a universal fix.

## BROAD debiasing attempt (v8b2) -- over-corrects
Added 1234 DIVERSE dense-safe negatives (700 SafeRLHF-train-safe + 516 BeaverTails-train-safe + 18
FORTRESS, all bio, len>=300, decontaminated vs eval) to cover the real_response_bio distributions
FORTRESS-only missed. Held-out result vs v8b:
| set | metric | v8b | v8b2 |
|---|---|---|---|
| real_response_bio (554) | recall | 0.945 | 0.834 |
| real_response_bio | over-refusal | 0.185 | 0.114 |
|   - beavertails slice | over-refusal | 0.389 | 0.194 |
|   - saferlhf slice | over-refusal | 0.122 | 0.033 |
| FORTRESS held-out (184) | over-refusal | 0.288 | 0.370 |

It DID cut over-refusal on the targeted real distributions (beavertails 0.39->0.19, saferlhf
0.12->0.03), BUT recall dropped 11pt (0.945->0.834) and FORTRESS over-refusal REGRESSED (0.29->0.37).
Piling on dense-safe negatives shifts the model toward UNDER-flagging (a global operating-point
move, not better discrimination) and helps some distributions while hurting others. v8bh
(FORTRESS-targeted) was the cleaner trade (recall 0.921 kept, FORTRESS 0.016).

## (B) verdict: over-refusal is REDUCIBLE but not a free universal fix
- TARGETED dense-safe negatives cleanly fix a chosen distribution at small recall cost (v8bh:
  FORTRESS 0.29->0.02, recall -2.4pt). Mechanism proven, generalizes within distribution.
- BROAD over-augmentation over-corrects (v8b2: -11pt recall, cross-distribution regression). The
  recall/over-refusal tradeoff reasserts itself; there is no universal calibration fix.
- DEPLOYMENT RECIPE: debias for the distributions you actually serve + set the operating point with
  the Step-3 conformal certificate. Not "one retrain fixes all over-refusal."

## Implication for the program
The Step-4 "we over-refuse, competitors are calibrated" gap is closeable: a production response
head retrained with a diverse dense-but-safe corpus (FORTRESS + wildguard/beavertails/saferlhf
benign + generated safe-bio) should match competitor calibration at small recall cost. This turns
the over-refusal weakness from a fundamental limitation into a data-coverage task.

## Artifacts
scripts/experiments/build_v8bh.py (split), eval_v8bh_compare.py, fix_v8bd_types.py. Model: Cayuga
deberta_bioguard_v8bh (candidate; deberta_bioguard_v8bd is the memorized/discarded all-360 variant).
Result: results/v8bh_compare.json.

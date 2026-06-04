# Step 3: distribution-free over-refusal bound (2026-06-03)

Turns the response head's continuous score into a deployable threshold with a STATISTICAL
GUARANTEE on over-refusal, via Learn-then-Test (LTT) with an exact Clopper-Pearson upper bound.

## Method
FPR(tau) (benign flag rate) is monotone in the threshold tau. Scan tau from high (low FPR) down;
at each, compute the Clopper-Pearson upper confidence bound on the calibration-set FPR at
confidence 1-delta. Keep lowering tau (raising recall) while UCB(FPR) <= alpha; stop at the first
failure (fixed-sequence test on a monotone family -> no multiplicity correction needed). The
selected tau* guarantees: P(true over-refusal <= alpha) >= 1-delta.

## Result (response head on real_response_bio_large; calib benign=105, test benign=106, harm=343; delta=0.05)
| alpha (target) | tau* | UCB(FPR) | held-out over-refusal | recall @tau* |
|---|---|---|---|---|
| 0.05 | 0.985 | 0.044 | 0.000 | 0.732 |
| 0.10 | 0.970 | 0.072 | 0.047 | 0.799 |
| 0.15 | 0.940 | 0.145 | 0.066 | 0.834 |
| 0.20 | 0.655 | 0.190 | 0.179 | 0.910 |

At every level UCB(FPR) <= alpha (the guarantee holds) and the held-out over-refusal is BELOW
alpha (the bound is conservative, as expected). Deployable statement, e.g.: "with 95% confidence
the guard's over-refusal is at most 10%, at recall 0.80."

## Reading
- This converts the ROC into a CERTIFIED operating point: pick your tolerated over-refusal alpha,
  get a threshold with a finite-sample, distribution-free guarantee and a known recall.
- The bound is loose at calib n=105 (UCB 0.072 vs empirical 0.047 at alpha=0.10); more benign
  calibration data tightens it -> higher recall at the same guarantee.

## Caveats
- Conformal validity is exchangeability-based: the guarantee holds for benign drawn from the SAME
  distribution as the calibration set. Deployment distribution shift weakens it (standard caveat).
- Applied to the RESPONSE head (the primary gate). Extending the certificate to the full dual-mode
  policy (joint prompt+response) is future work; the response head governs the system over-refusal
  under the deployed 'or'/'response-primary' policy (Step 2/artifact).
- Calibration benign here includes SafeRLHF/BeaverTails benign; a deployment-representative legit-bio
  calibration set is the right production choice.

## Artifacts
scripts/conformal_bound.py (LTT + Clopper-Pearson). Result: results/conformal_bound.json.

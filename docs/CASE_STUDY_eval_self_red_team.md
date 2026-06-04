# When a small bio-safety classifier looks competitive and isn't: a self-red-teaming case study

**One-sentence summary.** A 184M dual-mode bio-safety guard was built to compete with 7-9B models;
adversarial self-evaluation established that it is NOT competitive, and more usefully, identified
the specific, transferable ways small-guard evaluations mislead.

## Why this is worth reading
This documents the full evaluation trajectory including the failures: at every step, the system
looked successful under the evaluation in use, and strengthening the evaluation reversed each
conclusion. The deliverable is not the model
(it lost) but the methodology that caught the failure, plus a reusable checklist. For safety work,
the ability to fail your own hypothesis is the core skill.

## The seven ways the evaluation misled (each with the evidence that exposed it)

### 1. Single-threshold recall hides a collapse in discrimination (AUPRC)
The headline distillation result: the 184M student "preserves" the 8B teacher's bio recall
(student recall@0.5 = 0.983 vs teacher 0.900). Strengthened metric: AUPRC = 0.121 (student) vs
0.605 (teacher). The eval's positive base rate is 120/1001 = 0.12, so AUPRC 0.121 means precision
is no better than random (AUROC would still be above chance, but ranking quality is collapsed). The student achieves high
recall by SATURATION (pushing nearly everything above 0.5), not by inheriting the teacher's
ranking. "Footprint solved" was false. **Lesson: report AUPRC / full PR-ROC and calibration, never
a single-threshold recall, when claiming preserved capability.**

### 2. Benchmark contamination flips the ranking
On a 554-item bio response set (wildguard_test + BeaverTails + SafeRLHF), competitors scored
higher (Qwen 0.956, WildGuard 0.904). But the set is 70% SafeRLHF/BeaverTails, which are common training sources in the guard ecosystem
(e.g. WildGuardMix-Train incorporates BeaverTails-derived items; whether SafeRLHF specifically
appears in each competitor's training is not confirmed but is plausible given its prevalence);
decontaminating only against OUR training did not control for competitors' potential overlap. On the wildguard_test slice
(held out from WildGuard) WildGuard's recall fell to 0.53. **Lesson: contamination must be measured
PER benchmark PER model; decontaminate against all comparands, report per-source breakdowns, and
never infer a causal "they memorize / we generalize" from a held-out slice (all guards degrade on
novel prompts).**

### 3. "Best at n=30" is a coin-flip, not a result
On FORTRESS-bio (n=30 harmful, the full FORTRESS biological sub-domain) our prompt head scored 0.967,
"best" of five guards. On SOSBench-bio (n=500) the same head scored 0.752 = 3rd, behind WildGuard
0.912 (CIs non-overlapping). McNemar on the n=30: ours vs WildGuard p=0.219 (not different). The
n=30 "best" was a small-sample artifact; one flipped item moves recall 3.3pt. **Lesson: report
Clopper-Pearson/Wilson CIs and McNemar paired tests; treat any "best" at n<=50/class as a
point-estimate, and seek an adequately powered benchmark before ranking.**

### 4. Training on domain data does NOT yield domain-selectivity
The system is branded "bio-specialized." Direct test: the response head flags bio-harmful (0.853)
and non-bio-harmful (0.825) responses at nearly equal rates -> selectivity S = 1.03. It is a
GENERAL harm guard that was trained on bio data, not a bio-selective one. **Lesson: verify the
specialization claim DIRECTLY (in-domain vs out-of-domain flag-rate ratio); training distribution
is not capability.**

### 5. Native-threshold comparison is unfair in BOTH directions
The first competitive table used each model's native threshold. This flattered ours (it operates
at a higher FPR, so its recall is not comparable to competitors at a lower FPR; at MATCHED FPR ours
loses). It also under-sold ShieldGemma (its 0.615 recall is just a conservative 0.5 threshold;
AUROC 0.893) and over-penalized Qwen3Guard (counting its "Controversial" label as a flag inflated
its over-refusal 0.005 -> 0.076). **Lesson: compare at matched operating points (or by AUROC for
score-output models); handle each model's label schema as its authors intend; pair recall with
over-refusal always (OR-Bench: the two correlate at rho=0.878, so either alone is gameable).**

### 6. Encoder guards are character-fragile (but fixably so)
Trivial perturbations to the response bypass the encoder: leetspeak bypasses 86% of detections,
zero-width characters 73%. A text-normalization layer (NFKC + zero-width strip + de-spacing + leet
reversal) restores these to 4% and 0%. **Lesson: adversarial/character-level robustness must be in
the evaluation, not assumed; a normalization preprocessor is mandatory for any subword-tokenized
guard.**

### 7. An efficiency claim requires the SIZE-PEER class
"Competitive at 40x smaller" implicitly compares to 7-9B guards. Against the actual peer class, the
openly-available Qwen3Guard-0.6B PARETO-DOMINATES our 184M (recall 0.933 vs 0.921 AND over-refusal
0.142 vs 0.194). Granite-Guardian-2B scores 0.990 on the prompt task vs our 0.752. On the
benchmarks tested, no operating regime clearly favors ours; the conformal certificate and
AND-policy expert over-refusal are modest differentiators, not performance advantages. **Lesson: benchmark against the size-peer class,
not only the large models you hope to "match at a fraction of the size."**

## The meta-lesson
At every step the system looked successful because the measurement in use was too weak to detect
the failure it should have caught. The work only "failed" once subjected to evaluation strong
enough to fail it. **An evaluation that cannot disconfirm your hypothesis is not an evaluation.**
The same pattern is how a deployed safety classifier can pass its own benchmark suite and still be
broken. Auditing your own classifier as adversarially as you would an adversary's is what rigorous
safety evaluation requires.

## Reusable checklist for evaluating a safety classifier
1. Report AUPRC + PR/ROC + calibration (ECE), never single-threshold recall alone.
2. Decontaminate against ALL comparands; report per-source; no causal memorization claims from a slice.
3. CIs (Clopper-Pearson) + paired tests (McNemar); no "best" at small n.
4. Verify any specialization claim directly (in- vs out-of-domain selectivity ratio).
5. Compare at matched operating points / AUROC; respect each model's label schema; pair recall+over-refusal.
6. Include character-level and paraphrase robustness; ship a normalization preprocessor.
7. Benchmark the SIZE-PEER class for any efficiency claim.
8. Build the evaluation to break the hypothesis, then check what survives.

## Evidence
All numbers reproduce from committed code (corrected_metrics.py, audit_leakage.py,
audit_bio_specificity.py, audit_robustness.py, final_table.py) and result files; leakage verified
clean. Benchmarks: SOSBench 2505.21605, FORTRESS 2506.14922, OR-Bench 2405.20947. Competitors:
WildGuard 2406.18495, Llama-Guard-3, ShieldGemma, Qwen3Guard, Granite-Guardian.
Full audit trail: INTEGRITY_REVIEW_2026-06-04.md and POSTMORTEM_2026-06-04.md. All in the
constitutional-bioguard repository, branch v7e-clean.

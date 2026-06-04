# Integrity & completeness review (2026-06-04)

Self-audit of the dual-mode bio guard program against (a) direct code/data verification and (b) a
literature audit (deep-research, 17 claims confirmed / 8 refuted). Honest verdict: the CORE
engineering result is defensible, but SEVERAL HEADLINE CLAIMS were OVERCLAIMED and must be reframed
before any release.

## A. Direct verification (my own checks)
- **LEAKAGE: CLEAN.** 0 train/eval query-hash overlap on all 5 checks (audit_leakage.py): prompt-head
  pool vs FORTRESS-CBRN (0), vs bio_clean_eval (0); v8b train vs real_response_bio_large (0,
  decontamination worked); v8bh FORTRESS train-half vs held-out eval (0 -- the 0.016 IS held-out);
  generated bio-borderline vs FORTRESS (0). No train/eval leakage in my evals.
- **fp32 CONSISTENT.** All production deberta loads use dtype=torch.float32 (the fp16-NaN fix is applied
  in dual_mode_guard.py + every eval script).
- **FAIRNESS BUGS in my competitor comparison (found by audit):**
  1. My table reported guards at NATIVE thresholds, which FLATTERED ours -- ours operates at a higher
     FPR (over-ref 0.194) so its recall (0.921) is not comparable to competitors at lower FPR. At
     MATCHED FPR ours LOSES on the contaminated set: vs WildGuard 0.878 vs 0.904 @0.10; vs Llama
     0.767 vs 0.854 @0.05; vs Qwen 0.921 vs 0.956 @0.176.
  2. ShieldGemma was UNDERSOLD: its 0.615 recall is just its conservative native 0.5 threshold; its
     AUROC is 0.893 (ours v8bh 0.952) -- decent discrimination, not a weak model.
  3. Qwen was treated UNFAIRLY: my "Controversial=flagged" choice inflated Qwen over-refusal to 0.076;
     with Controversial=safe it is 0.005 (13/184 Controversial on held-out safe). Qwen is actually
     BETTER-calibrated than I reported (0.005 vs our 0.016).
  => Net: the competitive comparison must use MATCHED operating points + AUROC (threshold-free) +
     consistent multi-label handling, and PAIR recall with over-refusal. My "best/2nd-best" framing
     at native thresholds was misleading.

## B. Literature-audited integrity issues (prioritized)

### BLOCKERS (fix before release)
1. **CLAIM 4 "we generalize, they memorize" is NOT defensible (highest-priority fix).** Inferring a
   memorization MECHANISM from one competitor's drop on n=17 exceeds the literature. WildGuard's own
   paper makes only DESCRIPTIVE edge claims, never causal. A 2025 study (arXiv 2511.22047, unrefereed)
   shows ALL guards degrade on novel prompts -- Qwen3Guard WORST (91.0->33.8). REFRAME: "benchmark->
   novel-prompt degradation afflicts all guards; the n=17 slice cannot separate memorization from
   distribution shift; 'held-out' means 'not from a published split', not 'absent from any training
   corpus'." Do NOT single out WildGuard.
2. **Pair every over-refusal number with its recall on the same items.** OR-Bench: safety/over-refusal
   correlate at Spearman 0.878 -- an isolated over-refusal is gameable by under-flagging. My v8b2 case
   PROVES it: over-refusal 0.185->0.114 came WITH recall 0.945->0.834. CLAIM 2 (0.076 vs 0.532) and
   CLAIM 5 (0.288->0.016) must show paired recall. Report a 2D safety-utility Pareto, not standalone
   numbers. (Source: OR-Bench 2405.20947, FORTRESS benign-twin protocol 2506.14922.)
3. **Asymmetric decontamination caveat.** I decontaminated only vs MY training; SafeRLHF/BeaverTails are
   ALSO competitor training data, so the raw 554-item comparison can flatter competitors who memorized
   those items. Report PER-SOURCE breakdown (already have: wildguard_test / beavertails / saferlhf) +
   the caveat. (Field-wide blind spot -- even WildGuard does only within-pipeline minhash dedup.)
4. **CLAIM 5 density debiasing = WITHIN-DISTRIBUTION only.** FORTRESS 0.288->0.016 did NOT transfer to
   real_response_bio (0.185->0.194). Never headline 0.016 as a general over-refusal fix. (Already framed
   honestly in STEP4B; must stay that way.) (Source: contrast sets 2004.02709; embedding-drift 2603.01297.)
5. **Cite WildGuard as prior single-model dual/tri-mode guard** (prompt+response+refusal, June 2024,
   predates Qwen3Guard). Reframe novelty as the SMALL-FOOTPRINT two-encoder configurable-policy design,
   NOT the dual-mode idea itself. (Source: WildGuard 2406.18495.)

### SHOULD-FIX
6. **n=30 bio recall "best" needs CIs + McNemar.** FORTRESS authors themselves flag the bio sub-domain
   (n=30) as "limited statistical robustness." At n=30 one flipped item = 3.3pt; 0.967 (29/30) vs 0.933
   (28/30) is ONE item. Report Clopper-Pearson/Wilson CIs + McNemar paired tests; downgrade "best" to
   "highest point-estimate, CIs overlap." Also: ours prompt-head AUROC on FORTRESS-bio is only 0.682
   (saturated) -- the 0.967 is at over-ref 0.533 (worst) and cannot trade down. (Source: FORTRESS 2506.14922.)
7. **CLAIM 1 footprint / CLAIM 7 per-parameter overclaim from single-point recall.** Report AUPRC + full
   PR/ROC + calibration (ECE) for teacher vs student; reframe CLAIM 1 as "recall preserved at the chosen
   operating point", not "footprint solved." (Source: contrast sets 2004.02709; OR-Bench Pareto.)
8. **Benign-twin / over-refusal benchmarks have mislabeling + saturation** (ORFuzz 2508.11222: ~51% of
   OR-Bench "benign" rated harmful by humans; static sets fail to elicit over-refusal on resilient models).
   My self-generated safe responses sidestep mislabeling but inherit the generalization concern.

### NICE-TO-HAVE
9. **Conformal certificate scope.** It covers the RESPONSE HEAD ONLY on the SafeRLHF-mixed calibration
   distribution, NOT the deployed joint DualModeGuard policy. Label it "response-head, calibration
   distribution, exchangeability assumed"; re-run on the joint policy + a deployment-matched set to claim
   a system guarantee.
10. **Dual-mode thesis tension.** "response_only is now best" (post-v8bh) undercuts the dual-mode value.
    Honest framing: "a configurable policy lets a deployer pick prompt_only (pre-gen filter) vs
    response_only (post-gen audit)", not "dual-mode beats both heads." Show all 4 policies with paired
    recall+over-refusal on one eval.

## C. Refuted -- do NOT use (failed verification 0-3)
- "Generalization not accuracy should be the primary metric" (2511.22047 does NOT say this).
- "Frontier LLMs-as-classifiers beat all specialized guards / specialized guards hit 0% CBRN" (2507.06282) -- REFUTED, do not cite against our own value prop.
- "Model size does not correlate with safety performance" (2605.28830) -- REFUTED, do NOT use to argue small=competitive.

## D. The single highest-priority fix
Reframe CLAIM 4 (drop the causal "they memorize" narrative; present benchmark->novel degradation as a
shared property of all guards, with the n=17 limitation stated). Then globally: PAIR every over-refusal
with recall, report CIs + per-source breakdowns, and reframe dual-mode/footprint novelty.

## E. What remains genuinely defensible (the honest claim)
A 184M two-head configurable guard COMPETITIVE with 8-9B guards on bio response-harm (AUROC 0.952,
discrimination on par/above; recall in the same band as Qwen/WildGuard) at 40x smaller, with a
within-distribution density-debiasing recipe and a response-head over-refusal certificate -- IF reported
with paired metrics, matched operating points, per-source decontamination, confidence intervals, and the
reframed (non-causal, no-unqualified-novelty) narrative. The audit's instincts (leakage, Goodhart, small-n,
missed prior art) were all correct and match published standards. Leakage is clean; the overclaiming was
in FRAMING (operating points, causal narrative, small-n "best", novelty), not in fabricated numbers.

## Sources
WildGuard 2406.18495 · FORTRESS 2506.14922 · OR-Bench 2405.20947 · Contrast Sets 2004.02709 ·
ORFuzz 2508.11222 · Sainz et al. contamination EMNLP'23 · guard-degradation 2511.22047 (unrefereed,
directional) · embedding-drift 2603.01297 (motivation only).

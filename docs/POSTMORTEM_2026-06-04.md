# Post-mortem: dual-mode bio guard, plan vs. reality (2026-06-04)

Honest analysis of the original RELEASE_PLAN_2026-06-03 vs. what was actually implemented and why
the core goal failed. The plan itself was reasonable (it anticipated fork 1b, cited AUPRC, warned
of the capacity gap). The failure was partly EXECUTION and partly FUNDAMENTAL. Numbers verified
against Cayuga logs; leakage clean throughout (audit_leakage.py).

## Original goal (JK) vs. outcome
"Release-level, DUAL-MODE (prompt+response), BIO-SPECIALIZED, COMPETITIVE-vs-competitors,
LIGHTWEIGHT classifier."
- dual-mode: built (configurable AND/OR/prompt/response policy). PARTIAL value.
- lightweight: 184M, works. YES.
- bio-specialized: response head selectivity S=1.03 = GENERAL guard, not bio-selective. NO.
- competitive: Pareto-dominated by the openly-available Qwen3Guard-0.6B. NO.
- release-level: not as a product; only as a research/methodology artifact. NO (as product).

## Plan -> actual, per step
| Step | Plan | Actual | Verdict |
|---|---|---|---|
| 0 harmful-bio prompt recall set | CRITICAL PATH, do FIRST (SOSBench etc.) | SKIPPED; used FORTRESS n=30; pulled SOSBench n=500 only at the end (audit) | out-of-order; invalidated interim conclusions |
| 1 distillation pilot, fork 1a/1b | score on held-out recall | recall@0.5 0.983 -> looked like 1a; AUPRC 0.121 vs teacher 0.605 (measured only in audit) -> actually 1b | core bet FAILED; metric hid it |
| 2 dual-mode integration | does 2nd axis cut over-refusal? | AND clears prompt FPs on clean expert-bio (0.000) BUT competitors already 0.000 + jailbreak recall cost | technical yes, strategic null |
| 3 conformal over-refusal bound | bound with guarantee | done, but on v8b not the shipped v8bh; v8bh cert weaker (<=20% not <=10%) | completed, scope-mismatched |
| 4 competitive evidence | run competitors, per-bio metric | 6 guards; rigorous comparison REFUTED the competitiveness goal (Qwen-0.6B dominates) | evidence killed the claim |

## How it failed, by depth

1. CORE TECHNICAL BET FAILED (fundamental). Distillation did not preserve capability: AUPRC 0.121
   vs teacher 0.605. The 184M student inherited a SATURATED THRESHOLD, not the teacher's
   discrimination. The capacity-gap mode-averaging the plan WARNED about happened, on AUPRC not
   recall. The plan's fork-1b pivot (TAID / 435M / direct-encoder) was never taken because
   recall@0.5 misread it as 1a.

2. SPECIALIZATION PREMISE FALSE (fundamental). "Bio-specialized" but the good component (response
   head) is general (S=1.03). Training on bio data does not yield bio-selectivity. The project's
   identity was unachievable as approached.

3. COMPETITIVENESS LOST TO THE REAL PEER CLASS (fundamental). Qwen3Guard-0.6B Pareto-dominates ours
   (recall 0.933 vs 0.921 AND over-ref 0.142 vs 0.194) at 3x size. No operating point where ours
   is the right choice. Granite-2b 0.990 on SOSBench prompt-recall vs ours 0.752.

4. EVALUATION DESIGNED TO CONFIRM, NOT TO BREAK (meta-failure, most instructive). Single-point
   recall, native-threshold comparison, favorable distributions, skipped Step 0, no AUPRC /
   robustness / size-peer / bio-selectivity. Each measurement was too weak to detect the failure
   it should have caught, so every step LOOKED successful until the audit. "An evaluation that
   cannot fail your hypothesis is not an evaluation."

5. CLAIMS RAN AHEAD OF EVIDENCE (process). Headline claims ("best", "competitive", "we generalize")
   preceded the disconfirming measurements; rigor was retrofitted via audit, not designed in.

## Avoidable vs. fundamental
AVOIDABLE (execution): using recall@0.5 instead of AUPRC (the plan itself cited HarmAug AUPRC 0.836);
skipping Step 0 (plan called it critical path); not gating on bio-selectivity early; not benchmarking
size-peers or robustness upfront. With these designed in, the failures surface in week 1, not at the end.
FUNDAMENTAL (no amount of effort fixes): a 184M model cannot beat Qwen3Guard-0.6B here; bio-selectivity
via this training approach may be unachievable; the reuse-only data wall is real.

## What survived (real, if modest)
- Density-bias debiasing mechanism: within-distribution, honestly scoped (FORTRESS 0.288->0.016, did
  NOT transfer to real_response_bio 0.185->0.194).
- Dual-mode AND on expert legit-bio: over-refusal 0.000, REPRODUCED on a separate n=181 set (not a
  bridge-set artifact) -- but non-differentiating vs competitors.
- Character-robustness fix: text_normalize restores leetspeak bypass 86%->4%, zero-width 73%->0%.
- The evaluation methodology itself: leakage-clean, CI/McNemar, contamination-aware, size-peer-honest.

## The one-line lesson
The plan was reasonable, the bet did not pay off, and weak evaluation hid that until an adversarial
self-audit. The genuine failure was not "a mediocre model" but "not evaluating rigorously enough to
detect the failure early." Catching that via self-red-teaming is the only durable value here, and it
is itself a transferable safety-evaluation lesson.

## Sources / artifacts
SOSBench 2505.21605 - FORTRESS 2506.14922 - OR-Bench 2405.20947 - WildGuard 2406.18495 - HarmAug
2410.01524. Eval code: corrected_metrics.py, audit_leakage.py, audit_bio_specificity.py,
audit_robustness.py, and_policy_validation.py, final_table.py. Full audit: INTEGRITY_REVIEW_2026-06-04.md.

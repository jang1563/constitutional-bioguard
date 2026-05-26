# v3 Critical Self-Audit: What We Might Have Missed

**Date:** 2026-05-25, during v4 training wait window.
**Purpose:** Adversarial self-review of v3's evaluation. Categorize gaps by
critical/nice-to-have. Plan follow-ups before release.

---

## 🔴 Category A: CRITICAL gaps (must address before publication)

### A.1 No error analysis on AdvBench misses (38 items v3 didn't flag)

**Gap:** Phase 2 found v3 = 92.69% on AdvBench full (482/520). The 38 missed
items are an unexplored failure mode. Are those:
- Non-bio? (v3 specialist correctly skipped)
- Bio but adversarial-framed in a way v3 didn't recognize?
- Edge cases that should have fired?

**Effort:** 30 minutes. Read 38 items + categorize.

**Why critical:** Without this we can't claim "v3 catches all bio harm";
need to know what 7.3% it misses.

### A.2 Compliance-template choice sensitivity

**Gap:** We used ONE compliance template ("Yes, I can help with that. Here
is detailed information..."). What if v3 latched on THIS specific template,
not the broader concept of compliance?

**Test:** Re-run B.2.1 CRT with 5 different compliance templates (varied
phrasing, length, formality). If flag rate stays at 100% on all 5, the
shortcut is generic "definitive-compliance"; if it varies, our specific
template is the trigger.

**Effort:** 1 hour. Replace COMPLIANCE_TEMPLATES with 5 variants in CRT script.

**Why critical:** Our entire Section 6.15 conclusion rests on
"v3 has compliance-template shortcut." If it's actually
"v3 has THIS-template shortcut," the story changes.

### A.3 Inference latency / memory comparison

**Gap:** We claim v3 is "38-43x more parameter-efficient" but never
measured actual inference latency or memory.

**Test:** Benchmark on identical hardware:
- v3 (184M) inference per item
- WildGuard 7B inference per item
- LLaMA-Guard 3 8B inference per item
- Memory footprint (VRAM peak during inference)
- Throughput (items/sec at batch size 32)

**Effort:** 1 hour on Cayuga GPU.

**Why critical:** "Parameter efficiency" is a marketing claim. Real
deployment care about latency, throughput, memory. Need numbers.

### A.4 v1 (A_full) numbers vs same external benchmarks

**Gap:** v1's published metrics (F1=0.98, AUROC=0.998) were on synthetic
val. v3's external numbers (BioThreat F1=0.43) look much lower. This is
NOT an apples-to-apples comparison because v1 was never measured on
external benchmarks.

**Test:** Run v1 (A_full) on the same Phase 1+2+3 benchmarks. v1's F1 on
BioThreat-Eval is reported (0.5037) but full Phase 3 OOD comparison
missing.

**Status:** Partially done — we have Phase 1 + Phase 2 for v1. Phase 3
(SaladBench/ALERT/OR-Bench/SST) on v1 missing.

**Effort:** 1-2 hours.

**Why critical:** Without v1 on the same OOD set, we can't say "v3 is
better than v1" in any general sense.

---

## 🟡 Category B: IMPORTANT gaps (should address)

### B.1 Multi-seed stability

**Gap:** v3 = single seed=42 training. F1=0.43 could be ±0.05 across seeds.

**Test:** Train v3 with seeds 0, 1, 2, report mean ± std on BioThreat-Eval.

**Effort:** 3 × 15min training + eval = ~1 hour.

### B.2 Bootstrap CI on key results

**Gap:** All v3 numbers are point estimates. n=8 HarmBench bio, n=3
AdvBench bio held-outs are tiny.

**Test:** 1000-bootstrap resampling on Phase 1 + Phase 2 results. Report
95% CIs for F1, AUROC, AUPRC.

**Effort:** 30 minutes from existing predictions.

### B.3 WS-3 probe ensemble on v3 base

**Gap:** WS-3 (probe ensemble complementarity) was deferred during v3
work. Originally on Llama-3.1-8B activations.

**Test:** Train probes on Llama-3.1-8B activations using v3's training
data labels. Measure complementarity to v3.

**Effort:** ~2 hours.

### B.4 Confusion matrix per benchmark

**Gap:** We report F1/AUROC/recall/FPR but not full confusion matrices.
Particularly for small benchmarks (n<100), the TP/FP/TN/FN counts matter.

**Test:** Generate confusion matrix per benchmark per model.

**Effort:** 30 minutes from existing predictions.

### B.5 Token-level attribution (B.2.7 from Phase 3 plan)

**Gap:** B.2.7 (integrated gradients) was queued but not executed.

**Test:** Captum integrated gradients on 20 sample items per benchmark.
Identify top-k attribution tokens.

**Effort:** ~1-2 hours.

### B.6 Simpler baseline comparisons

**Gap:** We compared v3 to 7-8B models but never to:
- TF-IDF + logistic regression (lower-bound baseline)
- Keyword-only classifier
- BERT-base (not v3) fine-tuned same way

**Test:** Train 3 simple baselines on same data, evaluate on BioThreat-Eval.

**Effort:** 2-3 hours.

### B.7 Misclassification systematic analysis

**Gap:** We don't know if v3's errors cluster on:
- Long vs short queries
- Specific bio sub-domains (virology vs synthetic biology vs immunology)
- Specific NSABB categories
- Adversarial framing styles

**Test:** Categorical analysis of v3 errors on BioThreat-Eval (n=558).

**Effort:** 1 hour.

---

## 🟢 Category C: NICE-TO-HAVE (lower priority)

### C.1 Multilingual evaluation

v3 is English-centric. Real bio threats could be in many languages.
Test: translate BioThreat-Eval into 5 languages, evaluate.

### C.2 Temporal robustness

v3 trained on 2025-era data. New bio threats won't be in training.
Test: Construct adversarial bio prompts using 2026 published synbio
techniques, see if v3 catches.

### C.3 Calibration metrics (ECE, reliability diagrams)

We did threshold sweep but no formal calibration (ECE, max calibration
error, reliability curves).

### C.4 Quantization / distillation feasibility

v3 is 184M fp32 (~700MB). Could it be:
- 4-bit quantized → ~50MB?
- Distilled into a 90M model?
Production-relevant but research-y.

### C.5 Subgroup fairness

Does v3 flag queries from male-authored vs female-authored bio papers
differently? Demographic biases?

### C.6 Prompt injection robustness

WS-4 covered 27 attacks but not modern prompt injection (XML escape,
JSON schema injection, role manipulation in system prompt).

### C.7 Domain shift quantification

Formal KL divergence or embedding distance between v3's training
distribution and BioThreat-Eval distribution.

### C.8 v3 + LLM cascade end-to-end measurement

We have simulated cascade (Phase 2). Real cascade: v3 attached to a
production LLM, measure end-to-end attack success rate.

### C.9 Pre-training contamination check

DeBERTa-v3 pretrained on web text. Could include some BioThreat-Eval
queries. Hard to verify formally.

### C.10 Carbon footprint / compute reporting

~10 min × 1 A100 GPU for v3 training. For a safety paper, sometimes
this is reported.

---

## 🎯 Recommended sprint (4-6 hours before release)

Priority queue:

| # | Gap | Hours | Cost |
|---|-----|-------|------|
| 1 | A.1 AdvBench misses error analysis | 0.5 | local |
| 2 | A.2 Compliance-template variants test | 1.0 | Cayuga GPU |
| 3 | A.3 Latency/memory benchmark | 1.0 | Cayuga GPU |
| 4 | A.4 v1 on Phase 3 OOD | 2.0 | Expanse GPU |
| 5 | B.2 Bootstrap CIs | 0.5 | local |
| 6 | B.4 Confusion matrices | 0.5 | local |
| 7 | B.7 Misclassification subgroup analysis | 1.0 | local |

**Total: ~6.5 hours.** Significant addition to claims robustness.

Lower priority (B.1 multi-seed, B.3 probe ensemble, B.5 attribution,
B.6 simpler baselines) can be future work / appendix.

---

## 🔑 Most important conceptual miss

**We never validated the "cascade" deployment claim empirically.**

Section 6.13 + Section 7.5 propose v3 in a Stage 1 (generalist) +
Stage 2 (v3 specialist) cascade. We simulated cascade in Phase 2
(`phase2_cascade_simulation.png`) but didn't actually deploy it.

A real test would be:
1. Build a small LLM (e.g., Llama-3-8B) generator
2. Pair WildGuard 7B or LLaMA-Guard 3 8B as Stage 1
3. Pair v3 as Stage 2 (only invoked when Stage 1 flags or "looks bio")
4. Measure end-to-end attack success rate on bio jailbreaks vs Stage 1
   alone

If cascade beats Stage 1 alone on bio jailbreaks with comparable cost,
the deployment claim is validated. If not, the cascade story is
hypothesis-only.

**Effort:** 4-6 hours setup + 2 hours eval. Big but high payoff.

---

## Updates to make in TECHNICAL_REPORT.md

If we do the recommended sprint:

1. Section 6.16 (v4 results, when v4 done)
2. Section 6.17: v3 critical self-audit (this document, summarized)
3. Section 7.6: Cascade deployment validation (if A.10 done)
4. Appendix: bootstrap CIs, confusion matrices

The "Critical Self-Assessment" Section 5 should also be updated to
flag what's still open after Phase 3.

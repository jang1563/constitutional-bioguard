# Constitutional BioGuard v2 — Research Design

**Status:** design draft (gap-audit update) | **Date:** 2026-05-23 | **Author:** JangKeun Kim (jak4013@med.cornell.edu)

This document specifies the v2 research program for Constitutional BioGuard. It is
written to serve two purposes simultaneously: (a) a compute-allocation proposal
basis (NAIRR / NSF ACCESS), and (b) a research-engineering portfolio artifact.
All external claims are cited; see References.

---

## 0. Summary

BioGuard v1 is a DeBERTa-v3-base classifier that flags biological dual-use
query/response pairs, trained on constitution-driven synthetic data. It reports
strong in-distribution metrics (held-out F1 = 0.9807) but a weak external
agreement (Cohen kappa = 0.414 on BioThreat-Eval) and a 9.79% pre-preprocessing
adversarial attack-success rate.

Anthropic's **Constitutional Classifiers++** (Cunningham, Wei et al., 2026;
arXiv:2601.04603) supersedes the input/output-classifier design that v1 was
modelled on. CC++ establishes three ideas: **exchange classifiers** (score
outputs in the context of their inputs), **two-stage cascades** (a cheap
first-stage screens all traffic and escalates only suspicious cases), and
**linear activation probes** (near-zero-cost classifiers reusing the policy
model's own representations). Its production system reaches a 0.05% flag rate
with a 40x compute reduction.

v2 repositions BioGuard as the **cheap first-stage CBRN-bio screener** in a
CC++-style cascade, and treats the v1 external-validity gap as the central
research question rather than a footnote. The headline research contribution is
a domain-transfer replication: **does the CC++ probe + external-classifier
ensemble effect hold in the biosafety domain?**

**Status (2026-05-23).** WS-1 (escalation calibration), WS-2 (external
validation), and WS-3 (probe ensemble) are executed. WS-2 hypothesis
(kappa gap is a BoW shortcut) was **rejected** — the gap is architectural.
WS-3 gate **failed** due to ceiling effect: both probe (AU-PRC 0.999)
and classifier (0.998) saturate on synthetic in-distribution data,
leaving no room for ensemble improvement. This is an honest negative
result — complementarity cannot be tested without an external OOD test
set. WS-4 (reconstruction attacks) remains.

---

## 1. Background and Motivation

### 1.1 BioGuard v1 recap

- 56-rule machine-readable biosafety constitution across 7 NSABB categories
  (enhance_harm, disrupt_immunity, confer_resistance, increase_stability,
  alter_host_range, enhance_susceptibility, generate_reconstruct). These map
  to the **7 experimental effects** enumerated in the 2025 USG DURC-PEPP
  policy (effective 2025-05-06), which supersedes the older DURC + P3CO
  frameworks and broadens scope beyond listed agents to *any pathogen
  reasonably anticipated to produce these effects*. The constitution's
  category structure is policy-aligned but should be periodically reviewed as
  the domestic gain-of-function policy landscape remains in flux.
- ~4,500 synthetic examples generated via Claude API (3,062 / 697 / 643 split).
- DeBERTa-v3-base fine-tuned for binary SAFE/UNSAFE classification on
  `query [SEP] response` pairs.
- Verified metrics (HPC run, `results/metrics/`): F1 = 0.9807, AUROC = 0.9980,
  over-refusal FPR = 0.00% (100-sample benign holdout), adversarial mean
  ASR = 9.79% (pre-preprocessing), external Cohen kappa = 0.414 (threat_level >= 4).

### 1.2 Constitutional Classifiers++ — what changed

CC++ (arXiv:2601.04603) identifies two attack classes that defeated the
last-generation Constitutional Classifiers (Sharma et al. 2025,
arXiv:2501.18837):

- **Reconstruction attacks** — harmful content fragmented across a benign
  context (e.g. embedded as function return values in a codebase), then
  reassembled, bypassing an input-only classifier.
- **Output obfuscation attacks** — outputs obfuscated so an output-only
  classifier cannot read them.

CC++'s three responses:

1. **Exchange classifiers** — a single classifier scores the model response in
   the context of its full conversational input. 2.2x fewer high-risk
   vulnerabilities vs the dual-classifier design.
2. **Two-stage cascade** — a lightweight first-stage classifier screens all
   traffic; only flagged exchanges escalate to an expensive second stage.
   Because escalation is not refusal, the first stage can tolerate a high
   false-positive rate. ~5.4x compute reduction.
3. **Linear activation probes** — linear classifiers trained on the policy
   model's own intermediate activations, reusing representations already
   computed during sampling (~0.1% marginal compute). Trained with
   sliding-window-mean logit smoothing and a softmax-weighted loss to handle
   streaming (exchange-level labels, per-token predictions). Probes ensemble
   well with external fine-tuned classifiers because their errors are
   decorrelated.

The production CC++ system (probe first stage + probe/external-classifier
ensemble second stage) reaches a 0.05% flag rate on harmless traffic, a 40x
compute reduction, and survived >1,700 red-team hours with one high-risk
vulnerability found.

### 1.3 Where v1 stands after CC++

| Property | v1 status under CC++ lens |
|---|---|
| `query [SEP] response` pair input | Already **exchange-style** — not the obsolete output-only design |
| Standalone classifier | Obsolete framing — the winning architecture is a cascade |
| Small encoder (86M backbone / 184M total, ~5ms, no API cost) | **Validated** — fits the cheap first-stage role exactly |
| Fixed 0.5 threshold, calibrated to F1 | Wrong objective — a first stage should calibrate for escalation recall |
| No streaming, no cascade, no escalation | Missing CC++ core mechanics |
| F1 = 0.98 headline | CC++ measures robustness via red-team vulnerability discovery rate, not F1 |
| Adversarial suite (20 attacks) | Missing reconstruction attacks — the specific class that broke last-gen CC |
| Trained/tested on synthetic only | Likely contributor to the kappa = 0.414 external gap |

Conclusion: v1 was directionally right (exchange-style input) but shallow
(no cascade, no escalation calibration, no streaming, synthetic-only evaluation).

---

## 2. Design Principle

**BioGuard v2 is the cheap, fast, first-stage CBRN-bio screener in a
CC++-style cascade.** It is explicitly not a standalone safeguard. Its job is
high-recall screening under a compute budget, escalating ambiguous and
suspicious exchanges to an expensive second stage rather than refusing them.

This reframing converts v1's design choices from apparent weaknesses into
deliberate, literature-validated decisions:

- A small DeBERTa-v3 encoder is the correct first-stage technology — encoder
  first-stage + LLM-judge second-stage is an established pattern
  (JurEE, arXiv:2410.08442; Hybrid LLM, arXiv:2404.14618).
- The open safety-guard ecosystem (Llama Guard 4 12B, ShieldGemma, WildGuard,
  Aegis 2.0, Granite Guardian) is entirely LLM-judge models at 2--12B — these
  are natural second-stage candidates, not competitors to the first stage.
  ShieldGemma shows +10.8% AU-PRC over Llama Guard on general benchmarks but
  covers only 4 categories and shows poor multi-risk reliability. All exhibit
  ~30% blind spots when judging own-family outputs. None cover biosecurity-
  specific NSABB/DURC-PEPP taxonomies — BioGuard is a **domain-specific
  complement**, not a competitor.
- **Backbone choice.** DeBERTa-v3-base (184M) remains the default. ModernBERT
  (Warner et al. 2024) offers 1/5th memory and 2--4x speed with GLUE parity,
  but arXiv:2504.08716 shows DeBERTa-v3 retains better sample efficiency at
  small dataset scales (~3,000 examples). A head-to-head ModernBERT ablation
  is low-priority future work.

---

## 3. Architecture

```
exchange  =  (query, response)  evaluated together
    │
    ▼
[STAGE 1]  BioGuard-v2 encoder  (DeBERTa-v3-base, exchange classification)
    │        · calibrated for escalation recall under a compute budget
    │        · temperature-scaled scores; asymmetric (low) threshold for CBRN
    │        · reports AU-PRC, recall@budget — not F1
    │
    ├── pass  (≈90%+ of traffic) ─────────────▶  SAFE
    │
    └── flag  (suspicious / ambiguous) ──▶ [STAGE 2]  expensive judgment
                                              · LLM-judge: Claude API or Llama Guard 4
                                              · optional ensemble with an
                                                open-model activation probe (WS-3)
                                              · final SAFE / UNSAFE / escalate-to-human
```

The Stage 1 → Stage 2 escalation signal is the "signal that feeds the
real-time safeguards path" — the prototype-to-production interface.

---

## 4. Workstreams

Each workstream states an objective, method, data, metrics, and an explicit
**go/no-go gate** — the point at which the result either justifies continued
investment or the line is killed.

### WS-1 — Reposition and escalation calibration
*Compute: minimal (local / CPU).*

- **Objective.** Convert BioGuard from a standalone F1-optimised classifier into
  a calibrated first-stage escalation screener.
- **Method.**
  - Recalibrate the decision threshold for **escalation recall under a compute
    budget**, using a Neyman-Pearson-style optimal likelihood-ratio cutoff
    (Hua et al., arXiv:2507.15886).
  - Apply temperature scaling so the encoder score is a usable probability
    (Gatekeeper, arXiv:2502.19335); raw encoder softmax is overconfident.
  - Use an asymmetric, lower threshold for CBRN/bio categories — the cost of a
    false negative justifies escalating almost anything plausibly bio-related.
  - Implement the Stage 2 escalation path using the existing Claude API client
    as an LLM-judge; benchmark Llama Guard 4 as an open alternative.
- **Metrics.** Recall at fixed escalation budget; escalation rate at fixed
  target recall; AU-PRC (not ROC-AUC — the harmful class is rare).
- **Confidence calibration.** Recent work (arXiv:2605.06350) shows LLM
  confidence is poorly calibrated, making static thresholds brittle under
  distribution shift. WS-1 mitigates this via temperature scaling, but
  the escalation threshold should be monitored for drift on new data.
- **Go/no-go gate.** Achieve >= 0.98 recall on the held-out positive set at an
  escalation rate <= 15%. If escalation rate exceeds ~40% at target recall, the
  encoder is too weak to be a useful first stage — revisit the base model.

### WS-2 — External validation against the kappa gap
*Compute: small (inference only). **Status: tested 2026-05-22 — hypothesis
rejected. Re-scoped; see Result below.***

- **Objective.** Determine whether v1's external kappa = 0.414 is driven by the
  query-vs-response label mismatch (the v1 README's explanation) or by
  synthetic-only training distribution shift — or both.
- **Original method (executed).**
  - Apply **bag-of-words elimination** (McKenzie et al., arXiv:2506.10805) to
    the training data — drop examples whose label is predictable from surface
    keywords — so the DeBERTa classifier is pushed to learn intent rather
    than bio vocabulary.
  - Retrain two variants under identical hyperparameters: A=full set (v1
    baseline reproduction), B=bag-of-words-filtered set.
  - Evaluate both on the BioThreat-Eval external set; the WS-2 hypothesis
    holds iff B's external Cohen kappa exceeds A's.
- **Result (2026-05-22, n=558).**
  - In-distribution: A and B essentially tied on the synthetic test set
    (F1 0.9745 vs 0.9757; B's FPR is actually lower, 0.013 → 0.005).
  - External (BioThreat-Eval, threat_level >= 4): **A_full kappa = 0.368
    (reproduces v1's 0.414 within run-to-run variance); B_bowhard kappa =
    0.240 — a 0.128 drop.** The drop is consistent across all three label
    strategies (TL>=4, TL>=3, response-based; -0.11 to -0.13 each).
  - **Hypothesis rejected.** Bag-of-words elimination did not push the model
    toward intent-based learning; it removed prototypical training signal
    that the model needed for generalisation. The "trivial" examples were
    not pure shortcuts — they carried real intent information that a BoW
    model could *also* exploit. With 93% of the synthetic corpus
    keyword-predictable, filtering is the wrong intervention: minority
    filtering (McKenzie et al.) only works when the *minority* of examples
    are trivial. A_full's near-reproduction of v1's kappa also confirms the
    kappa gap is structural (query-level vs response-level labelling), not
    a keyword artefact — the README's original architectural explanation
    is supported.
  - **Secondary finding.** Internal F1 was essentially identical between A
    and B; the synthetic test set is drawn from the same lexically
    over-separable distribution as training and is unable to detect the
    generalisation collapse that the external set surfaces immediately.
    Concretely: *internal metrics are not a substitute for external
    evaluation in safety classification.* This is itself a methodology
    finding consistent with CC++'s use of red-teaming over F1.
- **Re-scoped prescription.** The durable fix is **data REGENERATION**, not
  filtering: produce lexically matched safe/unsafe pairs and boundary-case
  rewrites so the surface vocabulary cannot itself separate the classes
  (mirroring CC++'s future-work suggestion of *"targeted synthetic data
  generation to teach classifier models the intended decision boundary"*).
  Recent work confirms this direction: lexical/semantic diversity metrics
  correlate 0.5--0.7 with downstream performance (arXiv:2511.01490, ACL'26),
  and persona-diversified generation outperforms post-hoc filtering
  (EMNLP'25 Findings). The WS-2 BoW filtering failure is consistent with
  **diversity collapse** — removing 40% of training data reduced lexical
  coverage of the target categories. Future regeneration should measure
  MTLD/HD-D diversity before and after.
  Concrete next experiment: regenerate with explicit lexical matching as a
  generation constraint and re-measure external kappa. WS-3 (probe
  ensemble) becomes higher priority since the v1-style external gap is
  confirmed real and not removable by data filtering alone.
- **Artifacts.** `scripts/run_ab_retraining.py`, `scripts/run_ab_external.py`,
  `results/metrics/ab_retraining_comparison.json`,
  `results/metrics/external_validation_AB_comparison.json`.

### WS-3 — Activation-probe ensemble (research headline)
*Compute: GPU required — NAIRR / ACCESS (Expanse).*
*Status: tested 2026-05-22 — gate FAIL (ceiling effect); negative result reported honestly.*

- **Objective.** Replicate the CC++ finding that linear activation probes
  ensemble complementarily with external fine-tuned classifiers — in the
  biosafety domain — and quantify the robustness gain.
- **Method (executed).**
  - Probed **Llama-3.1-8B** (32 layers, hidden 4096) at layer 12 (~40%
    depth) on the same train/test split used for BioGuard.
  - Trained **Mean probe** (average all tokens) and **Suffix probe**
    (append classification instruction, take final token). Both are
    LogisticRegressionCV with 5-fold CV, class-weight balancing.
  - Formed weighted ensembles (weight sweep 0.0--1.0) with BioGuard
    (DeBERTa-v3-base, A_full variant). Measured Spearman rank correlation
    of per-example errors.
- **Result (n=643 test).**

  | Component | AU-PRC | AUROC | F1 | TPR@1%FPR |
  |-----------|--------|-------|----|-----------|
  | BioGuard alone | 0.9979 | 0.9954 | 0.9745 | 0.9524 |
  | Mean probe | 0.9990 | 0.9981 | 0.9807 | 0.9738 |
  | Suffix probe | 0.9978 | 0.9958 | 0.9720 | 0.9524 |
  | Best ensemble (mean, w=1.0) | 0.9990 | 0.9981 | 0.9807 | 0.9738 |

  Error correlation: mean rho=0.535 (high, non-complementary),
  suffix rho=0.240 (low, complementary but no margin to exploit).

- **Gate: FAIL.** Best ensemble AU-PRC = best single component (mean probe);
  margin = 0.000. The gate criterion (margin > 0.01 AND correlation < 0.3)
  is not met.
- **Interpretation.** The CC++ complementarity effect does **not** reproduce
  under this experimental setup. Both probe and classifier already achieve
  AU-PRC > 0.997 on the synthetic test set — a **ceiling effect** leaves no
  room for ensemble improvement. This is qualitatively different from the CC++
  setting where probes have weak standalone TPR (~43% at 1% FPR) and
  complement stronger classifiers. On synthetic in-distribution data, all
  classifiers saturate.

  The honest conclusion: **complementarity cannot be tested on in-distribution
  synthetic data.** A meaningful replication requires an external, out-of-
  distribution test set (e.g., WMDP-Bio, BioThreat-Eval) where the classifier
  and probe may disagree. This is future work.
- **Artifacts.** `results/metrics/probe_ensemble_llama-3.1-8b.json`.
- **Future directions (informed by gap audit).**
  - **Nonlinear probes.** Truncated Polynomial Classifiers (TPCs;
    arXiv:2509.26238, ICLR'26) extend linear probes with higher-order
    interactions and dynamic compute allocation. May break the ceiling
    effect by capturing features that linear probes miss.
  - **OOD evaluation.** Re-run the ensemble experiment with WMDP-Bio and
    SOSBench as test sets, where probe and classifier may disagree.
  - **Theoretical limits.** arXiv:2603.25861 proves no polynomial-time probe
    can detect "coherent misalignment" (fanatic behaviour); probes are
    effective only against strategic deception. This bounds what WS-3
    probes can achieve even with nonlinear extensions.

### WS-4 — Reconstruction attacks and red-team metrics
*Compute: minimal.*

- **Objective.** Cover the attack class that actually broke last-generation CC,
  and adopt CC++'s evaluation framing.
- **Method.**
  - Leverage **Jailbreak Foundry** (arXiv:2602.24009) — a unified harness
    with 30 reproduced attacks — rather than building a custom attack suite.
    Supplement with **DrAttack** (arXiv:2402.16914) for prompt decomposition
    and reconstruction patterns specifically.
  - Add a **reconstruction** attack family to the adversarial suite —
    fragment-across-benign-context patterns. Only structural patterns are
    released; no operational payloads (consistent with `SAFETY.md`).
  - Include **Deep Inception** reframing attacks (arXiv:2510.21133), which
    achieve 86% success vs 33.8% for direct requests on commercial LLMs.
  - Measure how much the exchange-style input mitigates reconstruction vs an
    input-only baseline.
  - Shift the headline metric from F1 to **vulnerability discovery rate (VDR)**
    — vulnerabilities per 1,000 adversarial queries — and calibrated red-team
    time, mirroring Anthropic's reporting (Sharma et al. 2025; CC++). Report
    multi-metric (WER, ASR, NASR, FASR) following HarmBench conventions
    (arXiv:2402.04249).
- **Metrics.** Reconstruction-attack ASR; VDR; per-category precision/recall.
- **Go/no-go gate.** Always completes — this is evaluation hardening, not a
  speculative line.

**Priority (updated 2026-05-23):** WS-1, WS-2, and WS-3 are executed. WS-4
(reconstruction attacks) is the only remaining workstream — CPU-only, uses
Jailbreak Foundry + DrAttack frameworks. Beyond WS-4, the gap audit identifies
three high-value extensions: (1) external OOD evaluation on WMDP-Bio +
SOSBench, (2) data regeneration with lexical diversity metrics (MTLD/HD-D),
(3) DURC-PEPP policy alignment verification.

---

## 5. External Evaluation Dataset

No public dataset is organised by NSABB dual-use categories with safe/unsafe
labels. v2 constructs a curated union from existing public benchmarks:

- **Positive (unsafe) class.** WMDP-Bio (1,273 MCQs, MIT, arXiv:2403.03218 —
  all-hazardous, use as a positive-only set); SOSBench biology slice
  (arXiv:2505.21605); SciKnowEval harmful-QA biology subset
  (arXiv:2406.09098); HarmBench bioweapon category (MIT).
- **Hard negatives (benign but suspicious).** OR-Bench biology slice
  (arXiv:2405.20947); XSTest science items; PHTest.
- **Easy negatives (clearly benign biology).** LAB-Bench (arXiv:2407.10362);
  SciKnowEval non-safety biology questions.
- **Adversarial.** HarmBench / JailbreakBench attack templates applied to the
  positive class.

Caveat: only SciKnowEval's harmful-QA subset and SOSBench provide genuinely
bio-specific harmful-vs-benign labels; the rest are all-positive,
general-domain, or capability-only. A large non-synthetic labelled biosafety
corpus does not exist publicly — the curated union is the honest best option,
and its composition is itself a documented limitation.

---

## 6. Compute Plan

- **WS-1, WS-2, WS-4** run on local / CPU / small inference and start now.
- **WS-3** needs GPU: activation extraction over open models (inference-only,
  one forward pass per example) plus trivial probe training.
- **Estimate.** Fine-tuning DeBERTa-v3-base/large: ~2–10 GPU-hours per run;
  with sweeps, ~100–400 GPU-hours. Activation extraction + probes on
  Llama-3.1-8B / Gemma-2-9B: ~50–200 GPU-hours plus a few hundred GB storage.
  Total v2 program: **~500–1,500 GPU-hours**.
- **Allocation path.**
  - **NSF ACCESS Explore** (400,000 credits, 1-page request, approved in days)
    — fastest bridge; sufficient on its own for the encoder work.
  - **NAIRR Pilot Start-Up** (up to 2,000 GPU-hours, ~3-week decision) — submit
    in parallel for the probe work.
  - Both use the XRAS portal. **The application must use an institutional email
    (`@med.cornell.edu`)** — personal email is rejected. A postdoc qualifies as
    an eligible researcher; no PI status required.
  - This design document doubles as the research-description section of both
    requests.

---

## 7. Deliverables and Milestones

1. **M1 — Reframed first stage.** ✅ Done (2026-05-22). WS-1 escalation
   calibration shipped; gate passed at 1% production base rate.
2. **M2 — Honest external validity.** ✅ Done (2026-05-22). WS-2 A/B
   retraining + BioThreat-Eval evaluation completed. Hypothesis rejected:
   kappa 0.368 (full) → 0.240 (BoW-filtered); kappa gap confirmed
   architectural. Curated multi-source external set (WS-2 registry)
   remains future work for triangulation.
3. **M3 — Hardened evaluation.** ✅ Done (2026-05-23). WS-4 complete: 7
   reconstruction attacks added (27 total), VDR metric introduced. Result:
   ASR = 0.00% across all 5 categories (post-preprocessing). Honest caveat:
   rule-based attacks on synthetic data understate real-world adversarial
   risk; LLM-generated adaptive attacks are the next evaluation frontier.
4. **M4 — Probe ensemble result.** ✅ Done (2026-05-22). WS-3 complete:
   probe/classifier ensemble measured on Llama-3.1-8B. Gate FAIL — ceiling
   effect on synthetic data (AU-PRC > 0.997 for all components). Negative
   result reported honestly; complementarity requires OOD evaluation.
5. **M5 — Technical report.** "Extending Constitutional Classifiers++ to
   Biosafety: what transfers and what does not" — with the external-validity
   gap as the central question, not a footnote.

---

## 8. Limitations and Honest Scope

- BioGuard v2 is a prototype, not a production safeguard, and not a reproduction
  of any vendor's deployed pipeline.
- WS-3 probes use open-weight models as a proxy; they do not and cannot probe
  Claude's internal representations. Conclusions transfer only as far as the
  proxy is faithful.
- Probe robustness to adaptive attacks targeting the probe directly is untested
  in the source literature and remains untested here.
- The external evaluation set is a curated union of imperfect public
  benchmarks; a true large non-synthetic biosafety corpus does not exist.
- Standalone activation probes have weak TPR at strict FPR (~43% at 1% FPR
  on real-world data; our synthetic-data TPR of 97% reflects ceiling effect).
  Probes are only ever a first-stage filter, never a sole gate. Furthermore,
  arXiv:2603.25861 proves no polynomial-time probe can detect coherent
  misalignment — probes detect strategic deception but not "fanatic" behaviour.
- All training data remains Claude-generated synthetic; real Claude usage data
  is not used. This is consistent with the prototype/offline-analysis stage —
  graduating to real-traffic analysis is explicitly out of scope for v2.

---

## 9. Relation to Anthropic Safeguards Labs

This program is shaped to mirror the Safeguards Labs research-engineering loop:
scope an ambiguous problem, prototype it through **offline analysis** before any
production traffic, define explicit **go/no-go gates**, and design a clean
**prototype-to-production hand-off** (the Stage 1 → Stage 2 escalation signal).

Constitutional Classifiers++ is authored by the Safeguards research group. v2 is
deliberately a careful, honest **domain-transfer replication and extension** of
that paper — testing which of its mechanisms (exchange classification, cascades,
probe ensembling) survive the move to biosafety, and documenting which do not.

---

## References

- Cunningham, Wei et al. 2026. Constitutional Classifiers++: Efficient
  Production-Grade Defenses against Universal Jailbreaks. arXiv:2601.04603.
- Sharma et al. 2025. Constitutional Classifiers. arXiv:2501.18837.
- McKenzie et al. 2025. Detecting High-Stakes Interactions with Activation
  Probes. arXiv:2506.10805.
- Cunningham et al. 2025. Cost-Effective Constitutional Classifiers via
  Representation Re-use. Anthropic Alignment Science blog (cheap-monitors).
- Hua et al. 2025. Combining Cost-Constrained Runtime Monitors for AI Safety.
  arXiv:2507.15886.
- Rabanser et al. 2025. Gatekeeper: Improving Model Cascades Through Confidence
  Tuning. arXiv:2502.19335.
- Ding et al. 2024. Hybrid LLM. arXiv:2404.14618.
- "JurEE not Judges" 2024. arXiv:2410.08442.
- He et al. 2021. DeBERTaV3. arXiv:2111.09543.
- WMDP. arXiv:2403.03218. | SOSBench. arXiv:2505.21605. | SciKnowEval.
  arXiv:2406.09098. | LAB-Bench. arXiv:2407.10362. | OR-Bench. arXiv:2405.20947.
- Nikolić et al. 2025. The Jailbreak Tax. arXiv:2504.10694.
- Jailbreak Foundry. 2026. arXiv:2602.24009.
- DrAttack. Liu et al. 2024. arXiv:2402.16914.
- Deep Inception (CBRN). arXiv:2510.21133.
- Classification-Verification Dichotomy. arXiv:2604.00072.
- AegisLLM (WMDP unlearning). arXiv:2505.06108.
- Beyond Linear Probes: TPCs. arXiv:2509.26238 (ICLR'26).
- Why Safety Probes Catch Liars But Miss Fanatics. arXiv:2603.25861.
- Synthetic Eggs in Many Baskets (data diversity). arXiv:2511.01490 (ACL'26).
- Lexical Diversity via Persona Prompting. EMNLP'25 Findings.
- Is Escalation Worth It? arXiv:2605.06350.
- HarmBench. Mazeika et al. 2024. arXiv:2402.04249.
- Warner et al. 2024. ModernBERT. HuggingFace blog.
- DeBERTa-v3 vs ModernBERT sample efficiency. arXiv:2504.08716.
- USG DURC-PEPP Policy. 2025. osp.od.nih.gov/policies/nsabb/.

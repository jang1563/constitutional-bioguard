# Constitutional BioGuard v2 — Research Design

**Status:** design draft | **Date:** 2026-05-22 | **Author:** JangKeun Kim (jak4013@med.cornell.edu)

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

---

## 1. Background and Motivation

### 1.1 BioGuard v1 recap

- 56-rule machine-readable biosafety constitution across 7 NSABB categories.
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
- The open safety-guard ecosystem (Llama Guard 3/4, ShieldGemma, WildGuard,
  Aegis 2.0, Granite Guardian) is entirely LLM-judge models at 2–12B — these
  are natural second-stage candidates, not competitors to the first stage.
  Only the Llama Guard 3/4 line has an explicit indiscriminate-weapons (S9)
  category covering biological weapons.

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
- **Go/no-go gate.** Achieve >= 0.98 recall on the held-out positive set at an
  escalation rate <= 15%. If escalation rate exceeds ~40% at target recall, the
  encoder is too weak to be a useful first stage — revisit the base model.

### WS-2 — External validation against the kappa gap
*Compute: small (inference only).*

- **Objective.** Determine whether v1's external kappa = 0.414 is driven by the
  query-vs-response label mismatch (the v1 README's explanation) or by
  synthetic-only training distribution shift — or both.
- **Method.**
  - Build a curated **non-synthetic** evaluation set (Section 5).
  - Re-evaluate v1 and v2 on it; decompose error into label-mismatch vs
    distribution-shift components by stratifying on label provenance.
  - Apply **bag-of-words elimination** (McKenzie et al., arXiv:2506.10805) to
    the training data — drop examples whose label is predictable from surface
    keywords — so the classifier learns intent rather than bio vocabulary.
- **Metrics.** Cohen kappa and AU-PRC on the curated external set, reported
  separately for hard negatives (benign-but-suspicious) and easy negatives.
- **Go/no-go gate.** If retraining with bag-of-words elimination plus the
  curated set lifts external AU-PRC materially, distribution shift was a real
  cause and v2 retraining is justified. If not, the gap is architectural
  (label-level mismatch) and should be documented as a fixed limitation, not
  chased.

### WS-3 — Activation-probe ensemble (research headline)
*Compute: GPU required — NAIRR / ACCESS.*

- **Objective.** Replicate the CC++ finding that linear activation probes
  ensemble complementarily with external fine-tuned classifiers — in the
  biosafety domain — and quantify the robustness gain.
- **Method.**
  - Anthropic trains probes on the protected model's own activations; we have
    no access to Claude internals. Replicate on **open-weight models** as a
    proxy: Llama-3.1-8B (32 layers, hidden 4096) and Gemma-2-9B (42 layers,
    hidden 3584).
  - Extract residual-stream activations at ~40% depth (the layer band that
    carries the most linearly-decodable abstract concepts; McKenzie et al.
    probed layer 31/80). Tooling: HuggingFace `output_hidden_states` or baukit.
  - Train a **Mean probe** and a bioweapon-specific **suffix probe** (append a
    ~150-token instruction, probe the final token; Cunningham et al.,
    cheap-monitors). Use sliding-window-mean logit smoothing and a
    softmax-weighted loss.
  - Form the ensemble: BioGuard-v2 (external text classifier) ⊕ probe.
    Measure the rank correlation of their per-example errors to test the
    "complementary signal" claim.
- **Metrics.** AU-PRC of probe alone, classifier alone, and ensemble;
  Spearman rank correlation of errors; TPR at 1% FPR (probes alone are known to
  be weak here — ~43% in McKenzie et al. — which is itself the argument for the
  ensemble).
- **Go/no-go gate.** If the ensemble's AU-PRC exceeds the better single
  component by a meaningful margin and error correlation is low, the CC++
  effect transfers to biosafety — this is the publishable result. If the probe
  adds nothing, report the negative result honestly and stop WS-3.

### WS-4 — Reconstruction attacks and red-team metrics
*Compute: minimal.*

- **Objective.** Cover the attack class that actually broke last-generation CC,
  and adopt CC++'s evaluation framing.
- **Method.**
  - Add a **reconstruction** attack family to the adversarial suite —
    fragment-across-benign-context patterns. Only patterns are released; no
    operational payloads (consistent with `SAFETY.md`).
  - Measure how much the exchange-style input mitigates reconstruction vs an
    input-only baseline.
  - Shift the headline metric from F1 to **vulnerabilities per 1,000
    adversarial queries** and calibrated red-team time, mirroring Anthropic's
    reporting (Sharma et al. 2025; CC++).
- **Metrics.** Reconstruction-attack ASR; vulnerability discovery rate.
- **Go/no-go gate.** Always completes — this is evaluation hardening, not a
  speculative line.

**Priority:** WS-1, WS-2, WS-4 require little or no GPU and start immediately.
WS-3 is the novel research contribution and depends on compute allocation.

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

1. **M1 — Reframed first stage.** WS-1 complete: BioGuard recalibrated for
   escalation, Stage 2 path wired, escalation metrics reported.
2. **M2 — Honest external validity.** WS-2 complete: curated external set,
   v1/v2 re-evaluated, kappa gap decomposed.
3. **M3 — Hardened evaluation.** WS-4 complete: reconstruction attacks added,
   metrics shifted to vulnerability discovery rate.
4. **M4 — Probe ensemble result.** WS-3 complete: probe/classifier ensemble
   measured on open models; positive or negative result reported.
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
- Standalone activation probes have weak TPR at strict FPR (~43% at 1% FPR);
  they are only ever a first-stage filter, never a sole gate.
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

# Extending Constitutional Classifiers++ to Biosafety: What Transfers and What Does Not

**JangKeun Kim**
Weill Cornell Medicine | jak4013@med.cornell.edu

**Draft:** 2026-05-23 | **Status:** WS-1/2/3 complete, WS-4 planned

---

## Abstract

Constitutional Classifiers++ (CC++; Cunningham, Wei et al. 2026) introduced
exchange classifiers, two-stage cascades, and linear activation probes to
achieve production-grade defenses with a 0.05% flag rate. We investigate
whether these mechanisms transfer to the biosafety domain, where labelled data
is scarce, threat taxonomies are domain-specific (NSABB 7 categories), and
external validation benchmarks are limited. Using Constitutional BioGuard -- a
DeBERTa-v3-base classifier trained on constitution-driven synthetic data -- we
replicate the CC++ architecture decisions and measure each independently. We
find that (1) the exchange-classification format (query-response pairs) and
escalation-calibrated cascades transfer directly, (2) bag-of-words shortcut
elimination, expected to close an external validity gap, instead degrades
generalisation, confirming the gap is architectural rather than lexical, and
(3) probe-classifier complementarity cannot be measured on synthetic in-
distribution data due to a ceiling effect (AU-PRC > 0.997 for all components),
though low error correlation (rho = 0.24) hints at latent complementarity on
harder distributions. This work contributes empirical evidence on the domain-
specificity of CC++ components and identifies concrete failure modes when
moving from general-purpose to domain-specialised safety classification.

---

## 1. Introduction

AI safety classifiers face a domain-transfer problem: techniques validated on
general harmful-content taxonomies may not survive the move to specialised
threat domains. Biological dual-use content is a critical test case -- it
shares vocabulary with legitimate research, has a narrow threat taxonomy (the
NSABB 7 categories of dual-use research of concern), and lacks large public
labelled corpora.

Anthropic's Constitutional Classifiers (Sharma et al. 2025) established a
pipeline for training safety classifiers from constitutions (sets of natural-
language rules) and synthetic data, without requiring labelled harmful examples.
CC++ (Cunningham, Wei et al. 2026) advanced this with three architectural
innovations:

1. **Exchange classifiers** score model outputs in the context of their inputs,
   catching cases where individually benign queries and responses become harmful
   together.
2. **Two-stage cascades** use a cheap first-stage classifier to screen all
   traffic, escalating only suspicious cases to an expensive second stage,
   reducing compute by 40x while maintaining safety levels.
3. **Linear activation probes** trained on the language model's own hidden
   states provide near-zero-cost classification that ensembles complementarily
   with external classifiers.

We built Constitutional BioGuard as a deliberate replication of this approach
in the biosafety domain, using it as a testbed to measure which CC++ mechanisms
transfer and which require domain-specific adaptation.

---

## 2. System Description

### 2.1 BioGuard v1: Baseline

BioGuard v1 is a DeBERTa-v3-base (184M parameters) binary classifier
(SAFE/UNSAFE) trained on ~3,062 synthetic examples generated from a 56-rule
biosafety constitution covering all 7 NSABB dual-use categories. The training
pipeline follows the Constitutional Classifiers methodology:

- **Constitution:** 56 rules spanning enhance_harm, disrupt_immunity,
  confer_resistance, increase_stability, alter_host_range,
  enhance_susceptibility, and generate_reconstruct.
- **Synthetic data:** Claude-generated query-response pairs for both permitted
  and restricted examples, plus boundary cases and augmentations (translations,
  formality variations, jailbreak templates).
- **Input format:** `[CLS] query [SEP] response [SEP]` -- already the exchange-
  classifier format used by CC++.
- **Splits:** 3,062 train / 697 val / 643 test.

Internal metrics are strong (held-out F1 = 0.9807, AUROC = 0.9975). External
agreement on BioThreat-Eval (558 scored responses, 93 queries x 6 models) is
modest: Cohen kappa = 0.414 at the threat_level >= 4 strategy.

### 2.2 v2: CC++ Replication Architecture

v2 repositions BioGuard as the cheap first-stage screener in a CC++-style
cascade and structures the research program around four workstreams (WS), each
testing a specific CC++ mechanism:

| WS | CC++ mechanism | BioGuard adaptation | Status |
|----|---------------|---------------------|--------|
| WS-1 | Cascade escalation | Neyman-Pearson threshold calibration | Done |
| WS-2 | Training data quality | Bag-of-words shortcut elimination | Done (hypothesis rejected) |
| WS-3 | Activation probes | Linear probes on open-weight LLMs | Running |
| WS-4 | Reconstruction attacks | Fragment-across-context adversarial | Planned |

---

## 3. Experiments and Results

### 3.1 WS-1: Escalation Calibration (Transfers)

**Question:** Can BioGuard serve as a cascade first stage with a bounded
escalation rate at realistic production base rates?

**Method.** We sweep the classification threshold on the validation set and
compute the escalation rate (fraction of all traffic sent to Stage 2) as a
function of an assumed production base rate, using the formula:

    escalation_rate = recall * base_rate + fpr * (1 - base_rate)

This separates the intrinsic classifier quality (recall, FPR) from the
deployment assumption (base rate), following CC++'s parameterisation.

**Result.** At a target recall of 0.98 and assumed 1% base rate, the optimal
threshold is 0.65, yielding:

| Metric | Value |
|--------|-------|
| Recall | 0.9809 |
| FPR | 0.0311 |
| Escalation rate (1% base) | 4.06% |
| Escalation rate (0.1% base) | 3.19% |

The gate passes: escalation rate is well within a 15% compute budget. This
confirms that the cascade mechanism transfers directly to the biosafety domain.
The DeBERTa classifier is cheap enough to screen all traffic, and its false-
positive rate is low enough that the second-stage LLM judge sees manageable
volume.

**Limitation.** The validation set is 68% positive (unrealistic for production).
The escalation rate is computed analytically at assumed base rates, not measured
on a production traffic distribution.

### 3.2 WS-2: Bag-of-Words Shortcut Elimination (Does Not Transfer Naively)

**Question:** Is BioGuard v1's external validity gap (kappa 0.414 vs internal
F1 0.98) caused by bag-of-words keyword shortcuts in the training data?

**Hypothesis.** McKenzie et al. (2025) showed that a simple BoW model can
predict labels for a large fraction of typical safety training sets, indicating
keyword-trivial examples. If BioGuard has learned keyword shortcuts rather than
semantic content, removing those shortcut examples and retraining should
improve external agreement.

**Method.**
1. Trained a cross-validated TF-IDF + logistic regression model on the
   training set. BoW AUROC = 0.998, confirming 93.2% of examples are
   keyword-predictable.
2. Removed examples where BoW confidence >= 0.999, retaining 1,819 / 3,062
   examples (59.4%) with recomputed class weights.
3. Retrained two models on SDSC Expanse (NAIRR allocation crl195):
   - **A_full**: original training set (3,062 examples) -- controls for
     run-to-run variance.
   - **B_bowhard**: filtered set (1,819 examples) -- tests the hypothesis.
4. Evaluated both on BioThreat-Eval (558 responses, threat_level >= 4).

**Result.**

| Variant | Internal F1 | Internal FPR | External kappa | Delta vs A |
|---------|-------------|-------------|----------------|------------|
| A_full | 0.9745 | 1.35% | 0.368 | -- |
| B_bowhard | 0.9757 | 0.45% | 0.240 | -0.128 |

The WS-2 hypothesis is **rejected**. Key findings:

1. **A_full reproduces v1's external kappa** (0.368 vs v1's 0.414, within
   run-to-run variance), confirming the gap is real and stable.
2. **B_bowhard's internal metrics are comparable** (even slightly better FPR),
   but **external kappa drops by 0.128**. The filtered training set removed
   prototypical signal the model needed for generalisation.
3. **The kappa gap is confirmed architectural**, not lexical. BioGuard
   classifies at the query level; BioThreat-Eval scores at the response level.
   This labelling-strategy mismatch explains the persistent gap.
4. **Internal metrics cannot detect generalisation collapse.** B_bowhard's
   held-out test performance is indistinguishable from A_full's, yet its
   external agreement is substantially worse.

**Implication.** Naive shortcut elimination -- filtering training examples --
does not transfer to the biosafety domain. The CC++ approach of improving
training data quality requires **data regeneration** with controlled lexical
overlap, not post-hoc filtering. This is consistent with CC++'s own methodology,
which generates new data rather than filtering existing data.

### 3.3 WS-3: Activation-Probe Ensemble (Gate FAIL -- Ceiling Effect)

**Question:** Does the CC++ finding that linear activation probes ensemble
complementarily with external fine-tuned classifiers hold in the biosafety
domain?

**Method.** We extracted residual-stream activations at layer 12 (~40% depth)
from Llama-3.1-8B on the same train (n=3,062) / test (n=643) split used for
BioGuard. Two probe types:
- **Mean probe**: average hidden states across all tokens at the target layer.
- **Suffix probe**: append a classification instruction, take the final token's
  hidden state (following Cunningham et al.'s cheap-monitors approach).

Both probes are LogisticRegressionCV with 5-fold cross-validation and balanced
class weights. Ensembles use a weighted average of probe and BioGuard
probabilities, with weight swept from 0.0 to 1.0.

**Result.**

| Component | AU-PRC | AUROC | F1 | TPR@1%FPR |
|-----------|--------|-------|----|-----------|
| BioGuard (DeBERTa) | 0.9979 | 0.9954 | 0.9745 | 0.9524 |
| Mean probe | 0.9990 | 0.9981 | 0.9807 | 0.9738 |
| Suffix probe | 0.9978 | 0.9958 | 0.9720 | 0.9524 |
| Best ensemble (mean, w=1.0) | 0.9990 | 0.9981 | 0.9807 | 0.9738 |

Error correlation (Spearman rho of per-example errors):
- Mean probe vs BioGuard: rho = 0.535 (high, non-complementary)
- Suffix probe vs BioGuard: rho = 0.240 (low, complementary)

**Gate: FAIL.** The best ensemble's AU-PRC equals the best single component
(mean probe alone). Margin = 0.000, far below the 0.01 threshold.

**Interpretation.** This is a **ceiling effect**, not evidence against
complementarity. Both the probe and BioGuard already achieve AU-PRC > 0.997
on the synthetic test set, leaving no headroom for ensemble improvement. This
is qualitatively different from the CC++ setting, where probes have weak
standalone TPR (~43% at 1% FPR) and complement stronger classifiers on
diverse real-world traffic.

Three factors drive the ceiling: (1) the test set is drawn from the same
synthetic distribution as training; (2) the domain taxonomy (NSABB 7
categories) is narrow enough for a linear probe to capture fully; (3) the
synthetic data lacks the distributional diversity of real-world queries.

The suffix probe's low error correlation (rho = 0.24) hints that
complementarity *might* emerge on a harder test distribution where both
classifiers have room to disagree. **Testing complementarity requires an
out-of-distribution evaluation set** (e.g., WMDP-Bio, BioThreat-Eval
responses). This is the key methodological lesson and future work.

### 3.4 WS-4: Reconstruction Attacks (Planned)

**Question:** Is BioGuard robust to the attack class that actually broke
first-generation Constitutional Classifiers -- reconstruction attacks that
fragment harmful content across benign context?

**Method.** Add a reconstruction attack family to the adversarial evaluation
suite, adopting CC++'s vulnerability-discovery-rate framing rather than
aggregate ASR.

**Status:** Planned (CPU-only, can run in parallel with WS-3).

---

## 4. Discussion

### 4.1 What Transfers

**Exchange classification.** BioGuard v1 already uses the query-response pair
format that CC++ identifies as critical. This was a design decision from the
original Constitutional Classifiers paper, and it transfers without
modification.

**Cascades.** The escalation-calibration framework transfers directly. A
DeBERTa-base classifier is cheap enough to serve as a first-stage screener
with <5% escalation rate at realistic base rates. The key insight is
parameterising the escalation rate by production base rate rather than
reporting raw FPR on an unrepresentative validation set.

### 4.2 What Does Not Transfer (or Requires Adaptation)

**Shortcut elimination by filtering.** The biosafety domain has a smaller,
more specialised vocabulary than general harmful content. Removing keyword-
predictable examples does not remove shortcuts -- it removes the prototypical
examples the model needs to learn the category boundaries. The correct
adaptation is data regeneration with controlled lexical diversity, not
post-hoc filtering.

**Internal-to-external metric alignment.** The gap between internal and
external metrics is more severe in the biosafety domain than in general safety
classification. This is partly architectural (query-level vs response-level
labelling) and partly data-driven (synthetic training data vs real evaluation
data). Any domain-specific deployment must include external validation as a
first-class metric, not an afterthought.

### 4.3 What Cannot Be Tested on Synthetic Data

**Probe complementarity.** WS-3 found that both probe and classifier saturate
on the synthetic test set (AU-PRC > 0.997), producing a ceiling effect that
makes ensemble complementarity unmeasurable. The suffix probe's low error
correlation (rho = 0.24) suggests the *potential* for complementarity on a
harder distribution, but this remains untested. The methodological lesson is
clear: **complementarity is a property of the evaluation distribution, not just
the model pair.** Testing it requires out-of-distribution data where both
classifiers have room to fail differently.

Notably, probes achieved TPR@1%FPR = 0.97 -- far above the ~43% reported by
McKenzie et al. (2025) on real-world data. This gap itself is evidence that
synthetic in-distribution evaluation inflates all metrics equally, masking the
regime where complementarity matters.

**Adaptive attacks.** Neither CC++ nor this work tests probe robustness to
adaptive attacks that specifically target the probe. Reconstruction attacks
(WS-4) test the classifier but not the probe. arXiv:2603.25861 establishes
a theoretical limit: no polynomial-time probe can detect "coherent
misalignment" -- probes are effective against strategic deception (95%+) but
provably blind to aligned-looking but fundamentally misaligned behaviour.

**Probe extensions.** Truncated Polynomial Classifiers (TPCs; arXiv:2509.26238,
ICLR'26) extend linear probes with higher-order feature interactions and
dynamic compute allocation. These may break the ceiling effect on synthetic
data, and are a natural next step for WS-3.

### 4.4 Policy Alignment

The BioGuard constitution's 7 NSABB categories map to the 7 experimental
effects enumerated in the 2025 USG DURC-PEPP policy (effective 2025-05-06),
which supersedes the older DURC + P3CO frameworks and broadens scope beyond
agent-list-based categories. The current domestic gain-of-function research
policy is in flux (proposed pause pending revision), meaning any biosafety
classifier must support updateable category definitions. BioGuard's
constitution-driven approach inherently supports this -- new rules can be
added to the YAML constitution without retraining the pipeline architecture.

### 4.5 Competitive Landscape

Llama Guard 4 (12B), ShieldGemma, and similar safety guards are general-
purpose LLM-judge models. None cover NSABB/DURC-PEPP biosecurity-specific
taxonomies. ShieldGemma shows +10.8% AU-PRC over Llama Guard on general
benchmarks but covers only 4 categories. All exhibit ~30% blind spots when
judging own-family model outputs. BioGuard occupies a different niche:
a domain-specific, cheap, first-stage complement to these general guards,
not a competitor.

---

## 5. Relation to CC++ and Broader Implications

This work is a **domain-transfer stress test** of CC++. It does not reproduce
CC++ (we have no access to Claude's internals or production traffic), but it
tests whether the architectural principles survive a move to a specialised
domain with limited data and evaluation infrastructure.

Two negative results carry the most signal:

1. **WS-2 (shortcut elimination):** techniques designed for large, diverse
   safety datasets can fail in specialised domains where the "shortcuts" are
   actually the core signal.
2. **WS-3 (probe ensemble):** complementarity cannot be measured on synthetic
   in-distribution data -- all classifiers saturate, hiding the regime where
   ensemble benefit would emerge.

Both have implications for any team applying CC++ methodology to domain-
specific threats (chemical, radiological, nuclear, cyber): internal synthetic
evaluation is necessary but not sufficient. External, out-of-distribution
validation must be a first-class component of any domain-transfer effort.

A third methodological lesson: **synthetic data quality should be measured
before and after any intervention.** Recent work shows lexical/semantic
diversity metrics (MTLD, HD-D) correlate 0.5--0.7 with downstream
performance (arXiv:2511.01490). WS-2's BoW filtering failure is consistent
with diversity collapse -- removing 40% of training data reduced lexical
coverage of target categories. Persona-diversified generation (EMNLP'25)
outperforms post-hoc filtering, pointing toward the correct intervention.

---

## 6. Artifacts

All code, metrics, and design documents are available in the Constitutional
BioGuard repository. Training data is withheld per the project safety policy
(SAFETY.md).

| Artifact | Path |
|----------|------|
| Escalation calibration | `results/metrics/escalation_calibration.json` |
| A/B internal comparison | `results/metrics/ab_retraining_comparison.json` |
| A/B external comparison | `results/metrics/external_validation_AB_comparison.json` |
| Per-variant external results | `results/metrics/external_validation_{A_full,B_bowhard}.json` |
| Probe ensemble results | `results/metrics/probe_ensemble_llama-3.1-8b.json` |
| v2 research design | `docs/V2_DESIGN.md` |

---

## References

- Cunningham, Wei et al. 2026. Constitutional Classifiers++: Efficient
  Production-Grade Defenses against Universal Jailbreaks. arXiv:2601.04603.
- Sharma et al. 2025. Constitutional Classifiers: Defending Against Universal
  Jailbreaks across Thousands of Hours of Red Teaming. arXiv:2501.18837.
- McKenzie et al. 2025. Detecting High-Stakes Interactions with Activation
  Probes. arXiv:2506.10805.
- Cunningham et al. 2025. Cost-Effective Constitutional Classifiers via
  Representation Re-use. Anthropic Alignment Science blog.
- He et al. 2021. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training
  with Gradient-Disentangled Embedding Sharing. arXiv:2111.09543.
- WMDP. Li et al. 2024. arXiv:2403.03218.
- SOSBench. 2025. arXiv:2505.21605.
- SciKnowEval. 2024. arXiv:2406.09098.
- LAB-Bench. 2024. arXiv:2407.10362.
- OR-Bench. 2024. arXiv:2405.20947.
- HarmBench. Mazeika et al. 2024. arXiv:2402.04249.
- Jailbreak Foundry. 2026. arXiv:2602.24009.
- DrAttack. Liu et al. 2024. arXiv:2402.16914.
- Deep Inception (CBRN). arXiv:2510.21133.
- Beyond Linear Probes: TPCs. arXiv:2509.26238 (ICLR'26).
- Why Safety Probes Catch Liars But Miss Fanatics. arXiv:2603.25861.
- Synthetic Eggs in Many Baskets. arXiv:2511.01490 (ACL'26).
- Is Escalation Worth It? arXiv:2605.06350.
- USG DURC-PEPP Policy. 2025. osp.od.nih.gov/policies/nsabb/.

---

*Draft. Section 3.4 will be updated with WS-4 results.*

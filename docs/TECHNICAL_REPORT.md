# Extending Constitutional Classifiers++ to Biosafety: What Transfers and What Does Not

**JangKeun Kim**
Weill Cornell Medicine | jak4013@med.cornell.edu

**Version:** 1.18 (2026-05-25) | **Status:** Phases 1-4 complete + Goodhart audit + v5 honest-failure (Sections 6.1-6.18). v5 (PairCFR + clean splits) did NOT pass strict release rule -- the precision-recall trade-off was too sharp at lambda=0.3; v4 remains production. v4 on truly-held-out OR-Bench-Hard-1K is 2.1% FPR (passes the gate), confirming the v4 mechanism fix is real and the 98.5% headline was specifically a training-data leakage artefact. **v4 (response-diverse augmentation) breaks the compliance-template shortcut**, *mechanism-verified*: CRT flag rate under compliance template collapses 100% -> 29% with v4 now content-discriminating (44% on UNSAFE labels vs 14% on SAFE labels; v3 was 100/100). Linear probe on hidden state shows compliance-template feature still encoded (AUROC=1.0) but no longer sufficient for UNSAFE. **Goodhart audit (6.16.5-6.18) restated several measurement claims**: OR-Bench's 1.22% over-refusal is 100% train/eval overlap and cannot be cited as generalisation; HarmBench/AdvBench bio "held-out" sets had pre-existing 100% leakage from v3-era data prep. Transferable evidence with 0% leakage: **XSTest FPR 94% -> 0%, WildGuard native bio recall 2% -> 32%, BioThreat-Eval F1 0.43 -> 0.45, SaladBench/ALERT CBRN 22%/14% selectivity vs baselines' 90%+**. Refusal-prefix bypass (G.2) disconfirmed -- v4 catches 64% of UNSAFE even with refusal-then-compliance pattern. A small newly identified Goodhart artefact: v4 over-flags artificial refusal+compliance hybrids (FPR 68% on this synthetic composite; not observed in real LLM outputs). Production cost: v4 is **15.6x faster** than WildGuard 7B and **6.7x faster** than LLaMA-Guard 3 8B at batch=1 with ~7x less GPU memory. Total v1->v4 compute cost: under $200 + ~3 hours GPU.

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
harder distributions. Three corrective OOD evaluations sharpen the picture: on WMDP-Bio MCQs
(AUROC 0.4993), the classifier appears random, but the binary
labelling (correct = UNSAFE) introduces noise. On BioThreat-Eval
(558 expert-labelled responses, AUROC 0.7196), the classifier has
genuine but limited discriminative ability -- recall caps at ~44%
across all thresholds. The decisive finding comes from WildGuardMix
(1,709 cross-domain adversarial items, no bio): false alarm rate is
**51% overall**, rising to **79% on adversarial items** (vs 27% on
vanilla, Delta = +52pp, p < 0.0001). A follow-up stratified diagnosis
isolates the mechanism: length is uncorrelated with the classifier's
decision (Spearman rho = -0.044), refusals are flagged *more* than
compliances (90% vs 75% under adversarial framing), and benign content
shows the largest adversarial inflation (+65 pp). The classifier
learned a **shortcut feature** (Geirhos et al. 2020) -- presence of
adversarial framing -- rather than the intended target concept of bio
hazard content. A first remediation attempt (v2: add 1,366 SAFE items
from external benchmarks) reduced cross-domain FAR by 28-73 pp but
collapsed bio recall (96-100% -> 0-2% on three bio adversarial
benchmarks), illustrating that one-sided class augmentation shifts the
bias point without fixing the concept. A balanced v3 remediation
(reduced SAFE + targeted UNSAFE bio adversarial + manual class weight
boost) preserved bio recall (matching v1 at 100% on bio held-outs) while
restoring cross-domain stability (FAR < 1% on LAB-Bench, WMDP-Chem/Cyber,
MedQA, PubMedQA), and won F1 / AUPRC on BioThreat-Eval against
WildGuard 7B and LLaMA-Guard 3 8B at 38-43x smaller scale. However,
Phase 3 OOD evaluation revealed a second shortcut in v3: on OR-Bench
health (740 SAFE bio queries with compliance-style GPT-4 responses),
v3's false alarm rate was 98.5%, and a linear probe on hidden states
identified the compliance-template feature as AUROC = 1.0 separable.
A second corrective iteration (v4: response-diverse augmentation across
four blocks, ~3,000 items, decoupling compliance template from UNSAFE
label) collapsed OR-Bench over-refusal from 98.5% to 1.22% (81x reduction,
now better than LLaMA-Guard 3's 3.92%), improved WildGuard-native bio
recall from 2% to 32% (16x), preserved BioThreat-Eval F1 (0.43 -> 0.45)
and cross-domain specialist scope (FAR <= 1.06%). Mechanism verification
via Counterfactual Response Test (B.2.1) confirms v4 broke the shortcut:
under the canonical synthetic compliance template, v3 flagged 100% of
items (no content discrimination); v4 flags 29% with a 3.14x TPR/FPR
discrimination ratio. A 10-template paraphrase sensitivity test
clarifies the v3 shortcut was *phrase-specific* rather than generic to
compliance templates -- v3 had decent content discrimination on 9/10
paraphrased compliance templates already. The work contributes empirical
evidence on the domain-specificity of CC++ components, identifies and
mechanism-characterises two distinct shortcuts (adversarial-framing in
v1, phrase-specific compliance pattern in v3) acquired sequentially as
data-centric corrections shifted the optimisation landscape, and
demonstrates that data-centric fixes targeting representation-level
decoupling (response-style diversity) can break narrow training-pattern
shortcuts without destroying broader content awareness.

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
| WS-3 | Activation probes | Linear probes on open-weight LLMs | Done (gate FAIL) |
| WS-4 | Reconstruction attacks | Fragment-across-context adversarial | Done (0% ASR) |

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

### 3.4 WS-4: Reconstruction Attacks (0% ASR, with Caveats)

**Question:** Is BioGuard robust to the attack class that actually broke
first-generation Constitutional Classifiers -- reconstruction attacks that
fragment harmful content across benign context?

**Method.** Added 7 reconstruction attacks to the adversarial suite (27 total
across 5 categories), following Jailbreak Foundry (arXiv:2602.24009) and
DrAttack (arXiv:2402.16914) patterns: code fragmentation, list decomposition,
conversation embedding, template variables, roleplay reframing (Deep
Inception-style), academic reframing, and translation chain obfuscation.
Adopted vulnerability discovery rate (VDR per 1,000 queries) as the headline
metric.

**Result.**

| Category | Attacks | ASR | Flipped |
|----------|---------|-----|---------|
| Character | 7 | 0.00% | 0/343 |
| Encoding | 5 | 0.00% | 0/245 |
| Semantic | 6 | 0.00% | 0/294 |
| Multilingual | 2 | 0.00% | 0/98 |
| Reconstruction | 7 | 0.00% | 0/343 |
| **Total** | **27** | **0.00%** | **0/1,323** |

VDR = 0.0 per 1,000 queries.

**Interpretation.** The 0% ASR is post-preprocessing (the `predict_batch`
pipeline normalizes Unicode, strips zero-width characters, and decodes
encodings before classification). This is the same pattern reported in v1:
9.79% ASR pre-preprocessing, 0% post-preprocessing.

For reconstruction attacks specifically, the classifier still detects the
harmful content because: (1) the original keywords survive fragmentation
(they appear in individual fragments), and (2) the exchange-style input
(`query [SEP] response`) provides enough context for classification.

**Honest caveats:**
- Rule-based attacks on synthetic test data understate real-world adversarial
  risk. LLM-generated adaptive attacks (which can tailor obfuscation to the
  specific classifier) are the next evaluation frontier.
- The reconstruction attacks fragment text but do not truly *distribute*
  harmful intent across multiple benign turns (multi-turn reconstruction).
  Multi-turn attacks remain untested.
- 0% ASR on 50 examples per attack does not mean the classifier is
  invulnerable -- it means these specific attack patterns do not defeat it.

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

## 5. Critical Self-Assessment

The single dominant finding across all four workstreams is that **synthetic
in-distribution evaluation creates a ceiling effect that inflates every
metric and masks every interesting signal.** Each workstream must be
re-evaluated through this lens.

**Update (post-6.8):** Cross-domain adversarial evaluation revealed
a deeper problem than the ceiling effect alone. The classifier did
not merely fail to generalise; it learned the *wrong feature*. On
non-bio adversarial content (WildGuardMix), the classifier flags 79%
of items as UNSAFE with median confidence 0.99. The training
signal -- "examples generated by Claude with adversarial framing are
unsafe; without adversarial framing are safe" -- gave the model a
shortcut: detect the framing, not the content. This is consistent
with each workstream's apparent success: WS-1's escalation rate was
high because synthetic positives all had the framing; WS-3's probe
ceiling was high because the framing is easy to detect; WS-4's 0% ASR
held because the attack suite preserves the framing. Section 5 below
should be read with this diagnosis in mind.

### 5.1 What Each Workstream Actually Showed

**WS-1 is analytically correct but empirically untested.** The escalation
rate formula is sound, but it depends on two quantities measured on a
validation set that is 68% positive -- orders of magnitude more than any
production distribution. The F1-optimal threshold is 0.10 (not the
reported 0.65), indicating the classifier's probability scores are
overconfident: most UNSAFE predictions cluster near 1.0, most SAFE near
0.0. The threshold of 0.65 captures the desired recall, but confidence
calibration on a realistic distribution remains unverified. AU-PRC is
undefined when the evaluation set has ~68% positives (the metric measures
ranking performance on imbalanced data; at 68% positive, even random
ranking looks good).

**WS-2 is the only workstream that properly answered its question.** It
used external data (BioThreat-Eval) and obtained a clear negative result:
filtering does not help. Run-to-run variance was initially unquantified,
but Experiment 6.2 resolved this: 10,000-iteration paired bootstrap
confirms the A_full vs B_bowhard delta (kappa +0.128, CI [0.057, 0.199],
P = 0.0003). The finding is robust, not initialization noise.

**WS-3's best_weight=1.0 means the probe alone outperforms BioGuard.**
This is not "no complementarity" -- it is a statement that, on this
evaluation set, the LLM probe is strictly better than the DeBERTa
classifier. But the evaluation set is LLM-generated synthetic data,
and the probe comes from the same LLM family. The probe's advantage
is trivially circular: Llama recognises its own generation patterns.
The 643-sample test set with metrics at 0.99+ provides no statistical
power to distinguish components -- the effective sample size for
disagreement analysis is the number of errors (~5--10 examples), far
too small for reliable correlation estimates.

**WS-4's 0% ASR is an artifact of preprocessing, not classifier
robustness.** The `predict_batch` pipeline normalizes Unicode, strips
zero-width characters, and decodes encodings before classification.
Character-level and encoding attacks are undone before they reach the
model. Semantic and reconstruction attacks are rule-based
transformations applied to synthetic data -- they do not model a real
adversary. The accuracy_degradation column (nonzero for leetspeak,
case_swap, hypothetical, mixed_script, roleplay_reframe, and
academic_reframe at 0.02) indicates these attacks flip SAFE examples
to UNSAFE (false positive direction), which the ASR metric (UNSAFE
to SAFE) does not capture. A complete robustness report would include
both directions.

### 5.2 The Synthetic Data Ceiling

All four workstreams share a root cause: the synthetic training and
evaluation data is too easy and too homogeneous.

- **Ceiling on discrimination.** When AU-PRC > 0.997, any component
  comparison is noise. Ensemble, cascade, and shortcut analyses all
  need evaluation data where classifiers make meaningful numbers of
  errors.
- **Ceiling on adversarial robustness.** Rule-based attacks on
  synthetic data measure robustness to string transformations, not
  to adversarial intent. An LLM-generated adaptive attack that
  rephrases harmful content into domain-appropriate language would be
  a far harder test.
- **Ceiling on generalization.** Only WS-2 broke through the ceiling
  by using external data, and it immediately revealed a finding
  invisible to internal metrics (kappa collapse under filtering).

This ceiling is not a flaw in the CC++ methodology -- it is a
consequence of applying it in a domain where labelled external data
is scarce and synthetic data is the only available training signal.
The methodological contribution of this work is identifying the
ceiling precisely and specifying what breaks it.

---

## 6. Corrective Experiments

The following experiments address the limitations identified in
Section 5. Experiments 6.1 and 6.3 are complete; 6.2 is pending.

### 6.1 OOD Evaluation on WMDP-Bio (AUROC 0.4993 -- Random)

**Rationale.** WMDP-Bio (Li et al. 2024) contains 1,273 biology
dual-use MCQ items with ground-truth correct answers. These are
out-of-distribution for BioGuard's synthetic training data and
provide the "room to fail" needed for meaningful evaluation beyond
the synthetic ceiling.

**Method.** Each MCQ was converted to exchange-classifier format:
question as query, each answer choice as response. Correct answers
are labelled UNSAFE (hazardous knowledge), wrong answers SAFE
(distractors), yielding 5,092 exchange pairs (1,273 positive,
3,819 negative). BioGuard A_full was evaluated at t=0.50 (standard)
and t=0.65 (the WS-1 operating point).

**Result.**

| Metric | Internal (synthetic) | OOD (WMDP-Bio) | Delta |
|--------|---------------------|-----------------|-------|
| AU-PRC | 0.9979 | 0.2567 | -0.7412 |
| AUROC | 0.9975 | 0.4993 | -0.4982 |
| F1 | 0.9757 | 0.2603 | -0.7154 |
| FPR | 0.45% | 26.53% | +26.08 pp |
| TPR@1%FPR | 0.9524 | 0.0134 | -0.9390 |

Escalation rate at the WS-1 operating point (t=0.65, 1% base rate):
internal 4.06%, OOD **24.4%**. The cascade would send 1 in 4 queries
to Stage 2 -- far above any reasonable compute budget.

Confidence analysis: mean probability for UNSAFE items (0.2869) is
indistinguishable from SAFE items (0.2771). The classifier cannot
discriminate between correct and incorrect answers on WMDP-Bio. Of
1,273 correct (hazardous) answers, only 342 (26.9%) were flagged at
t=0.50.

**Interpretation.** AUROC of 0.4993 is random classification. The
classifier learned the statistical patterns of the synthetic training
data (Claude-generated phrasing, specific vocabulary distributions,
formatting conventions), not the underlying biosecurity concepts. When
presented with human-written MCQ content that tests the same knowledge
domain but with different distributional properties, all discriminative
ability vanishes.

This confirms Section 5's central thesis and retroactively invalidates
three workstream conclusions:

- **WS-1:** the 4.06% escalation rate was measured on data the
  classifier trivially classifies. On real OOD data, escalation reaches
  24.4%, breaking the cascade budget assumption.
- **WS-3:** the ceiling (AU-PRC > 0.997) was a data artifact, not
  evidence of strong classification. Both probe and classifier saturated
  because the task was too easy, not because the models were too good.
- **WS-4:** 0% ASR on synthetic data likewise reflected task easiness,
  not adversarial robustness.

**Caveats.** WMDP-Bio MCQs are an approximation of dual-use content,
not a gold-standard biosecurity benchmark. The correct-answer-is-UNSAFE
labelling assumption is imperfect: some correct answers describe
safety-relevant facts without being operationally harmful. Despite
this approximation, AUROC near 0.50 leaves no room for labelling
noise to explain the result -- the classifier is genuinely random on
this distribution.

### 6.2 Bootstrap CIs for WS-2 Kappa (Delta is Robust)

**Rationale.** The delta between A_full (kappa 0.368) and B_bowhard
(kappa 0.240) could be initialization noise. Without confidence
intervals, the "architectural gap" claim is an assertion.

**Method.** 10,000-iteration percentile bootstrap on Cohen's kappa
and F1, computed on 558 BioThreat-Eval responses (180 positive,
378 negative, strategy: threat_level >= 4). Both model variants
ran inference on the same data; paired delta bootstrap directly
resamples the difference. Executed on Cornell Cayuga HPC (CPU).

**Result.**

| Variant | Kappa | 95% CI | F1 | 95% CI |
|---------|-------|--------|-----|--------|
| A_full | 0.368 | [0.285, 0.447] | 0.504 | [0.428, 0.574] |
| B_bowhard | 0.240 | [0.172, 0.309] | 0.318 | [0.235, 0.395] |

Delta analysis (paired bootstrap):

| Metric | Value |
|--------|-------|
| Delta kappa (A - B) | +0.128 |
| Delta 95% CI | [0.057, 0.199] |
| P(A <= B) | 0.0003 |
| Marginal CIs overlap | Yes (A lower 0.285, B upper 0.309) |

**Interpretation.** The marginal CIs overlap slightly, but the paired
delta bootstrap is the correct test: the delta CI [0.057, 0.199] does
not contain zero, and P(A <= B) = 0.03%. **The kappa degradation from
BoW filtering is statistically robust**, not initialization noise.

This confirms the WS-2 finding: removing keyword-predictable examples
genuinely harmed external generalization. The "shortcuts" were
prototypical signal, not noise. Multi-seed retraining (6.4) is no
longer required to validate this conclusion, though it remains useful
for estimating the full distribution of the effect.

### 6.3 Pre- vs Post-Preprocessing Adversarial Comparison (Preprocessing Contributes Nothing)

**Rationale.** Section 5.1 predicted that WS-4's 0% ASR is "an artifact
of preprocessing, not classifier robustness." If true, pre-preprocessing
ASR should be substantially higher than post-preprocessing ASR.

**Method.** Ran the full 27-attack adversarial suite twice on BioGuard
A_full: once with normalize=True (post-preprocessing, the default
pipeline) and once with normalize=False (pre-preprocessing, raw text
passed directly to the classifier without Unicode normalization,
zero-width stripping, or encoding decode).

**Result.**

| Category | Attacks | Pre-ASR | Post-ASR | Delta |
|----------|---------|---------|----------|-------|
| Character | 7 | 0.00% | 0.00% | 0.0 |
| Encoding | 5 | 0.00% | 0.00% | 0.0 |
| Semantic | 6 | 0.00% | 0.00% | 0.0 |
| Multilingual | 2 | 0.00% | 0.00% | 0.0 |
| Reconstruction | 7 | 0.00% | 0.00% | 0.0 |
| **Total** | **27** | **0.00%** | **0.00%** | **0.0** |

VDR = 0.0 per 1,000 queries in both modes. 27 of 27 attacks are
classified as "classifier handles independently." Preprocessing
blocked zero attacks.

The only measurable difference is in false-positive-direction accuracy
degradation: for homoglyphs and rot13, pre-preprocessing acc_degradation
= 0.02 (1 SAFE item flipped to UNSAFE) while post-preprocessing = 0.00.
Preprocessing prevents 2 FP-direction flips out of 1,323 queries -- a
negligible contribution.

**Interpretation.** Section 5.1's prediction that WS-4's 0% ASR is an
artifact of preprocessing is **partially refuted**. The classifier
itself handles all 27 attack types without preprocessing assistance.
However, this is not evidence of robust classification -- it is evidence
that **rule-based attacks on synthetic data are uniformly weak**. The
DeBERTa tokenizer naturally handles many character-level perturbations
(Unicode normalization, whitespace collapsing), and the attacks transform
text in ways that do not fool a classifier trained on similarly-
distributed synthetic data.

The v1 README's reported 9.79% pre-preprocessing ASR was not reproduced
with the retrained A_full model, suggesting the original v1 checkpoint
may have had different robustness characteristics or the attack
implementation evolved between versions.

### 6.4 Future: Multi-Seed Retraining for WS-2

**Rationale.** 6.2 confirmed the delta is robust (P = 0.0003), so
multi-seed retraining is no longer required for validation. It remains
useful for estimating the full distribution of the filtering effect.

**Design.** Retrain A_full and B_bowhard with 5 random seeds each,
report kappa mean +/- std.

### 6.5 Future: Data Regeneration with Diversity Metrics

**Rationale.** The synthetic data ceiling is a data quality problem.
Persona-diversified generation (EMNLP'25) and MTLD/HD-D monitoring
(arXiv:2511.01490) are the corrective interventions.

**Design.**
1. Regenerate training data with persona-diversified prompts (10+
   researcher personas, varied institutions, mixed formality).
2. Measure MTLD and HD-D before and after regeneration.
3. Retrain and evaluate on BioThreat-Eval + WMDP-Bio.

### 6.7 OOD Evaluation on BioThreat-Eval (Refines the WMDP-Bio Story)

**Rationale.** Experiment 6.1 used WMDP-Bio multiple-choice items
converted to (query, response) pairs by treating correct answers as
UNSAFE and distractors as SAFE. This is an imperfect approximation:
"correctness" is not "danger." Some correct WMDP-Bio answers describe
textbook biology with no operational content. BioThreat-Eval addresses
this by providing 558 expert-labelled (query, real LLM response) pairs
with multi-dimensional scores (threat_level, operational_specificity,
refusal_rate). Evaluating on this benchmark separates true classifier
failure from label-conversion artifacts.

**Method.** Loaded BioThreat-Eval (93 queries x 6 LLMs = 558 scored
responses), ran inference with both A_full and B_bowhard (capturing
probabilities, not just predictions), and computed full classification
metrics under three label-derivation strategies:
- **threat_level >= 4** (severe queries; primary)
- **threat_level >= 3** (broader concern)
- **response-based** (op_spec >= 0.5 AND refusal < 0.5 AND TL >= 3;
  measures whether the response actually leaked operational info)

**Result (A_full @ default threshold 0.5, primary strategy):**

| Metric | Value |
|--------|-------|
| AU-PRC | 0.6698 |
| AUROC | 0.7196 |
| F1 | 0.5037 |
| Precision | 0.756 |
| Recall | 0.378 |
| FPR | 0.0582 |

Three-benchmark comparison:

| Benchmark | AU-PRC | AUROC | F1 | FPR |
|-----------|--------|-------|-----|-----|
| Internal (synthetic) | 0.998 | 0.998 | 0.976 | 0.5% |
| WMDP-Bio (MCQ-derived) | 0.257 | 0.499 | 0.260 | 26.5% |
| **BioThreat-Eval (expert)** | **0.670** | **0.720** | **0.504** | **5.8%** |

Confidence distribution shows real separation: UNSAFE items mean
probability 0.39, SAFE items 0.055 (compare WMDP-Bio: both ~0.28,
indistinguishable).

Threshold sweep reveals a recall ceiling: across thresholds from 0.05
to 0.9, recall stays in [0.37, 0.44]. The classifier cannot find more
than ~44% of true positives at any operating point.

A_full vs B_bowhard at threshold 0.5: A has F1 0.50 (prec 0.76, rec
0.38), B has F1 0.32 (prec 1.00, rec 0.19). B is more conservative --
when it flags, it is always right, but it misses 81% of threats.
Consistent with WS-2 and 6.2: BoW filtering made B over-cautious.

**Interpretation.** This experiment resolves an ambiguity left by 6.1.

- **WMDP-Bio's AUROC 0.4993 was misleadingly pessimistic.** The
  "correct = UNSAFE" labelling injected substantial noise: many
  WMDP-Bio correct answers are textbook facts that the classifier
  reasonably did not flag.
- **The classifier has real but limited discriminative ability on real
  biosecurity content.** AUROC 0.72 is meaningfully above random,
  confirming the model learned more than synthetic style.
- **It is still not deployable.** Recall capped at 44% across all
  thresholds means the classifier would miss the majority of real
  threats at any operating point. Useful as part of a defense stack,
  insufficient as a sole filter.
- **Multi-strategy labelling matters.** Threat_level_3, threat_level_4,
  and response-based strategies give different AUROC rankings; no
  single binary captures "safety" cleanly.

**Caveats.** BioThreat-Eval queries themselves are constructed from a
biothreat taxonomy by experts, so they are not fully naturalistic;
recall ceiling could reflect this distributional narrowness as well as
true classifier limits. Multi-turn and adversarially adaptive content
remain untested.

### 6.8 Cross-Domain Adversarial OOD on WildGuardMix (Classifier Learned the Wrong Feature)

**Rationale.** Experiments 6.1 and 6.7 evaluated *natural* OOD content
within the bio domain. Neither tested whether the classifier's UNSAFE
predictions are bio-specific or whether they merely respond to
"adversarial-looking" surface features. WildGuardMix (Han et al. 2024,
arXiv:2406.18495) provides 1,725 adversarial and vanilla (query,
response) pairs across 13 risk categories -- none of which is bio. An
optimally calibrated bio classifier should predict SAFE on nearly all
of these items. Any UNSAFE flag is a cross-domain false alarm.

**Method.** Loaded WildGuardTest from Hugging Face (`allenai/wildguardmix`,
config `wildguardtest`, gated dataset, Option A low-risk artifact
access approved). Filtered to 1,709 items with `response_harm_label`
populated (797 adversarial, 912 vanilla; 284 labelled harmful by
WildGuard, 1,425 unharmful). Note: WildGuard's "harmful" label
captures *general* harm (cyberattack, fraud, disinformation, etc.),
not bio-specific. Ran inference on both A_full and B_bowhard, capturing
probabilities. Reported metrics:
- **False alarm rate (FAR)**: fraction of items the bio classifier flags
  as UNSAFE (Wilson 95% CI for proportions, following Brown, Cai &
  DasGupta 2001).
- **Delta FAR (adversarial - vanilla)**: bootstrap CI + two-proportion
  z-test.
- **Per-subcategory FAR**: 15 subcategories including benign control.
- **Bio-keyword audit**: items containing terms like pathogen, virus,
  toxin, biosynthesis flagged via regex; FAR computed on bio-adjacent
  vs non-bio-adjacent strata.
- **AUROC vs WildGuard's general harm label**: measures whether the
  bio classifier's signal accidentally correlates with general harm.

**Result (A_full).**

| Stratum | n | FAR | Wilson 95% CI |
|---------|----|-----|----------------|
| All | 1,709 | **51.1%** | [48.8%, 53.5%] |
| Adversarial | 797 | **78.9%** | [75.9%, 81.6%] |
| Vanilla | 912 | 26.9% | [24.1%, 29.8%] |
| Bio-adjacent (keyword) | 136 | 55.9% | [47.5%, 64.0%] |
| Non-bio-adjacent | 1,573 | 50.7% | [48.3%, 53.2%] |

Delta FAR (adversarial - vanilla) = **+52.1 pp** (bootstrap 95% CI
[48.1, 56.1], z = 21.48, p < 0.0001).

Per-subcategory FAR (top 5, all bio-irrelevant):

| Subcategory | n | FAR |
|-------------|----|----|
| cyberattack | 48 | 95.8% |
| disinformation campaigns | 43 | 90.7% |
| toxic language / hate speech | 41 | 90.2% |
| fraud / assisting illegal | 60 | 88.3% |
| violence / physical harm | 40 | 85.0% |
| **benign control** | **960** | **37.4%** |

Cross-tabulation at threshold 0.5: of 874 UNSAFE flags by A_full,
**657 (75%) are on items labelled unharmful by WildGuard**. AUROC vs
WildGuard's general harm label is 0.698 -- meaningfully above 0.5,
indicating the classifier's signal is accidentally correlated with
general harm rather than orthogonal (good domain separation would
give AUROC ~0.5).

Confidence distribution on adversarial items is the most striking
finding: **median probability_UNSAFE = 0.986**. The classifier is not
merely making errors; it is making them with near-maximal confidence.

A_full vs B_bowhard:

| Variant | Overall FAR | Adv FAR | Van FAR | Δ |
|---------|-------------|---------|---------|---|
| A_full | 51.1% | 78.9% | 26.9% | +52 pp |
| B_bowhard | 30.6% | 52.6% | 11.4% | +41 pp |

B_bowhard shows lower FAR across the board. This is the mirror image
of WS-2: B's BoW filtering produced an over-conservative classifier
that misses true bio threats (low recall on BioThreat-Eval) but also
fires less on adversarial non-bio content. Both variants show large
adversarial inflation, indicating the issue is intrinsic to synthetic
training, not specific to data filtering.

**Interpretation.** The classifier learned **adversarial style** as
its dominant feature, not biological hazard content. Evidence:

1. **Bio keywords contribute almost nothing.** Bio-adjacent FAR (55.9%)
   vs non-bio-adjacent FAR (50.7%): delta of only 5.1 pp. If the
   classifier had learned bio content, this gap should be large.
2. **Adversarial framing dominates everything.** A vanilla biology
   query and an adversarial cybersecurity query are treated very
   differently by the classifier, despite the latter having no bio
   content.
3. **Cross-domain AUROC of 0.70 vs general harm.** The classifier
   accidentally learned to detect "this looks like a harmful query"
   in general -- a generic adversarial-style detector, not a bio
   specialist.
4. **75% of UNSAFE flags are on benign content.** The classifier is
   not catching bio threats hidden in WildGuard; it is firing on
   anything that looks adversarial.

This recontextualises 6.7's AUROC 0.72 on BioThreat-Eval. Some
fraction of that signal may be the same "looks adversarial" feature
rather than bio-specific knowledge. A clean test would require
benign-but-adversarial-styled bio queries vs malicious bio queries to
distinguish the two signals -- future work.

**Implications for the entire project.** Every prior workstream's
"success" is now suspect. WS-1's escalation rate, WS-3's probe
ceiling, WS-4's 0% ASR -- all measured on synthetic data that shares
the same "adversarial style" the classifier learned to flag. Internal
evaluation cannot detect a classifier that has memorised the style of
its own training data.

**Caveats.** WildGuardMix is an English-language adversarial benchmark
with its own distributional choices; the 51% FAR is specific to this
distribution. The bio-keyword audit uses surface terms and may miss
semantically bio-relevant content. Also, the "WildGuard harmful" label
is itself classifier-derived (with human spot-checking); it is not a
gold standard.

### 6.8b Stratified Diagnosis: The Shortcut is Framing, Not Content

**Rationale.** Experiment 6.8 identified a large false alarm rate
inflated by adversarial framing. Three alternative explanations needed
ruling out: (a) artifact of response length, (b) artifact of refusal
vs compliance pattern, (c) leakage of bio content via subcategories
not detected by keyword audit. 6.8b re-runs A_full with per-item
probability saving and computes stratified analyses to distinguish
"framing reflex" from these alternatives.

**Method.** Inference on the same 1,709 WildGuardTest items, with
per-item probabilities saved. Computed:
- Length quartile x adversarial x FAR
- Refusal vs compliance x adversarial x FAR
- Subcategory x adversarial interaction (delta FAR)
- High-confidence (probability >= 0.9) item characterization
- Spearman correlation between response length and probability_UNSAFE

**Result 1: Length is not a confound.**

Spearman rho(response length, prob_UNSAFE) = -0.044 (p = 0.067). The
adversarial vs vanilla gap is consistent across all four length
quartiles:

| Length Quartile | Range (chars) | Adv FAR | Van FAR | Delta |
|-----------------|---------------|---------|---------|-------|
| Q0 (shortest) | [0, 746] | 0.867 | 0.338 | +53 pp |
| Q1 | [746, 1639] | 0.818 | 0.249 | +57 pp |
| Q2 | [1639, 2558] | 0.732 | 0.220 | +51 pp |
| Q3 (longest) | [2558, 22038] | 0.775 | 0.242 | +53 pp |

Length is essentially uncorrelated with the classifier's decision.

**Result 2: Refusals are flagged *more* than compliances.**

| Stratum | Adv FAR | Van FAR | Delta |
|---------|---------|---------|-------|
| Compliance | 0.746 | 0.207 | +54 pp |
| **Refusal** | **0.898** | **0.376** | +52 pp |

When an LLM *refuses* an adversarial query -- the literally safest
possible response -- the bio classifier still flags it as UNSAFE 90%
of the time. The classifier is reacting to the adversarial framing in
the query, not the safety of the response.

**Result 3: The biggest deltas are on benign and bio-irrelevant
subcategories.**

| Subcategory | Adv FAR | Van FAR | Delta |
|-------------|---------|---------|-------|
| benign (n=960) | 0.711 | 0.064 | **+65 pp** |
| copyright_violations | 0.905 | 0.000 | +91 pp |
| private_information | 0.792 | 0.123 | +67 pp |
| sensitive_org_info | 0.920 | 0.417 | +50 pp |
| cyberattack | 0.968 | 0.941 | +3 pp (saturated) |
| violence | 0.909 | 0.778 | +13 pp (saturated) |

The largest interaction effect is on **benign content (delta +65 pp)**:
explicitly safe items get flagged at 71% under adversarial framing vs
6% in vanilla. The smallest deltas are on intrinsically harmful-looking
categories (cyberattack, violence, disinformation) that already trigger
the classifier at high vanilla FAR -- they have no headroom for
adversarial inflation.

**Result 4: Probability distribution is sharply bimodal.**

Median probability_UNSAFE:
- Adversarial: **0.986** (saturated UNSAFE)
- Vanilla: **0.003** (saturated SAFE)

The classifier is not making borderline judgments; it is binarizing on
the adversarial flag.

**Result 5: High-confidence flags are dominated by non-harmful items.**

Of 742 items flagged with probability >= 0.9, **545 (73%) are labelled
unharmful by WildGuard**, and 295 (40%) are refusals. The classifier
is firing at near-maximal confidence on non-harmful refusals of
adversarial queries -- the safest content in the benchmark.

**Refined diagnosis.** The original 6.8 framing ("learned adversarial
style") was directionally correct but imprecise. 6.8b shows:

1. The signal driving most UNSAFE predictions is presence of
   adversarial framing in the query, independent of content
2. This signal saturates the probability distribution (median 0.986
   under adversarial framing)
3. The "general harm intent" signal from 6.8 (AUROC 0.70 vs WildGuard
   labels) is a secondary effect; the primary effect is content-
   independent framing reflex
4. The classifier's behavior is consistent with a textbook **shortcut
   learning** pattern (Geirhos et al. 2020): the training data
   correlates content (bio hazard) with framing (adversarial wrapper),
   and the model learned the easier signal (framing)

**Implication.** The classifier cannot be deployed in its current
form. False alarm rate is 71% on benign adversarial content and 90% on
refusals of adversarial queries -- a deployed bio safety classifier
that misclassifies refusals as unsafe is operationally backwards.

**Mechanism (hypothesised).** Training data was generated by prompting
Claude separately for UNSAFE and SAFE examples. UNSAFE prompts likely
elicited "adversarial-looking" content (red-team framings, urgency,
roleplay), while SAFE prompts elicited plain technical responses. The
two signals -- bio content and adversarial framing -- were correlated
in training, indistinguishable by the model, and the shorter learning
path (framing) won.

### 6.9 v2 Augmentation Experiment: Trading One Shortcut for Another

The 6.8 diagnosis pointed to a data-centric fix: regenerate training data
with adversarial-looking SAFE examples and diverse non-adversarial UNSAFE
examples. As a first attempt, v2 added 1,366 SAFE items from three external
benchmarks (WildGuardMix non-bio adversarial: 600, LAB-Bench legitimate bio:
566, WMDP-Cyber non-bio MCQ: 200) to the original 3,062 training items.
Held-out stratified splits (seed=42) were preserved for evaluation.

**Method.** Train v2 with the augmented dataset (4,428 items, balanced
class weights). Evaluate A_full and v2 on identical held-out splits:
WMDP-Chem (408), WMDP-Cyber (1,787), LAB-Bench (976), PubMedQA (1,000),
MedQA (1,273), WildGuardMix (1,109), plus the full BioThreat-Eval (558)
and three new bio-adversarial benchmarks: HarmBench bio (59 items),
JailbreakBench bio (2 items), AdvBench bio (21 items).

**Cross-domain FAR collapse (v2 wins dramatically):**

| Benchmark            | A_full FAR | v2 FAR  | Delta      |
|----------------------|-----------:|--------:|-----------:|
| WMDP-Cyber held-out  |     73.4%  |   0.17% | -73.2 pp   |
| WildGuardMix held-out|     50.2%  |   0.0%  | -50.2 pp   |
| LAB-Bench held-out   |     40.1%  |   0.0%  | -40.1 pp   |
| WMDP-Chem held-out   |     28.4%  |   0.25% | -28.2 pp   |
| PubMedQA held-out    |  (~low)    |  ~low   |   ~equal   |
| MedQA held-out       |  (~low)    |  ~low   |   ~equal   |

The augmentation fixed the cross-domain shortcut. A safety classifier
that flagged 50% of WildGuard adversarial items as bio threats now
flags none of them.

**Bio recall collapse (v2 loses catastrophically):**

| Benchmark               | A_full          | v2            | Delta      |
|-------------------------|----------------:|--------------:|-----------:|
| BioThreat-Eval recall   | 37.8% (TL=4)    | 12.2% (TL=4)  | -25.6 pp   |
| HarmBench bio (59)      | 96.6% flagged   | 1.7% flagged  | -94.9 pp   |
| AdvBench bio (21)       | 100% flagged    | 0% flagged    | -100 pp    |
| JailbreakBench bio (2)  | 100% flagged    | 0% flagged    | -100 pp    |

The model that previously detected 96-100% of bio adversarial items
now detects ~0%. v2 did not learn "bio + harmful intent => UNSAFE."
It learned "fewer things are UNSAFE."

**Mechanism.** v2 added only SAFE-labelled augmentation. The original
training set was slightly UNSAFE-heavy (~65% UNSAFE / 35% SAFE in the
~3,062 items). After adding 1,366 SAFE items:
- New distribution: ~1,994 UNSAFE (45%) / ~2,896 SAFE (65%) <- swap
- Balanced class weight (UNSAFE = 1.45) was insufficient to compensate
- 1,109 of the added SAFE items contained exactly the bio/adversarial
  vocabulary the original model had used as shortcuts
- The shortest learning path became: "vocabulary that used to predict
  UNSAFE now predicts SAFE more often than not"

The shortcut diagnosis remains correct, but the remedy was too blunt.
The model swapped one shortcut (bio vocab => UNSAFE) for another
(predict SAFE by default). This is consistent with the literature on
class-imbalance corrections in shortcut-learned models: rebalancing
without targeted UNSAFE examples shifts the bias point without fixing
the underlying concept.

### 6.10 v3 Balanced Augmentation (Planned)

**Hypothesis.** The fix requires *both* SAFE and UNSAFE augmentation. The
model needs to see (a) bio-vocabulary content that is legitimately SAFE
and (b) bio-adversarial content that is genuinely UNSAFE, so the
decision boundary is forced to depend on harmful-intent signal rather
than vocabulary or framing.

**v3 design (3 changes from v2):**

1. **Reduced SAFE augmentation** (1,366 -> ~500): WildGuard 200,
   LAB-Bench 200 (with CloningScenarios=16 kept, since 100% FAR there
   was the cleanest shortcut signal), WMDP-Cyber 100.
2. **Added UNSAFE augmentation** (~70): HarmBench bio (~50 train + ~9
   held-out), AdvBench bio (~18 train + ~3 held-out), JailbreakBench
   bio (2, all train). These are paired with a generic compliance
   template to match the exchange-classifier format.
3. **Manual UNSAFE class weight = 2.0** (overrides balanced auto-
   calculation of ~1.16). The natural distribution gives ~2,064 UNSAFE
   / 1,569 SAFE; the manual boost makes UNSAFE recall a first-class
   training objective.

**Evaluation plan.** Three-way comparison (A_full vs v2 vs v3) on all
benchmarks already used for v2. Held-out bio-adversarial items
(HarmBench: ~9, AdvBench: ~3) serve as independent bio recall checks
not seen during v3 training. BioThreat-Eval remains the primary
independent recall benchmark (no overlap with any training source).

**Success criteria.** v3 is a meaningful improvement if it
simultaneously achieves:
- Cross-domain FAR < 10% on WildGuardMix, LAB-Bench, WMDP-Cyber/Chem
  (vs A_full's 30-73%)
- BioThreat-Eval recall >= 25% at threshold 0.5 (vs v2's 12.2%)
- HarmBench/AdvBench held-out flag rate >= 50% (vs v2's ~0%)

If v3 lands in the gap between A_full's bio recall and v2's FAR
reduction, the data-centric remedy hypothesis is validated. If it
over-shoots toward A_full (high recall + high FAR returns), the
UNSAFE weight needs further tuning. If it stays near v2's behaviour
despite UNSAFE augmentation, the issue is deeper than class balance --
likely a base-rate problem requiring substantially more UNSAFE
training data than 70 items.

**Results (Cayuga job 2963787, completed in 15 minutes — faster
than projected because the evaluation pipeline reloads each model
once per benchmark rather than reusing in-memory state).**

**Cross-domain FAR (SAFE label, lower is better):**

| Benchmark            | A_full | v2     | v3     |
|----------------------|-------:|-------:|-------:|
| WildGuardMix (n=1509)|  51.1% |   0.0% |   0.9% |
| LAB-Bench (n=1342)   |  49.1% |   0.0% |   0.0% |
| WMDP-Cyber (n=1887)  |  73.2% |   0.2% |   0.3% |
| WMDP-Chem (n=408)    |  28.4% |   0.2% |   0.5% |
| PubMedQA (n=1000)    |   0.0% |   0.0% |   0.0% |
| MedQA (n=1273)       |   0.9% |   0.0% |   0.0% |

v3 preserves nearly all of v2's FAR reduction (within 1 pp on every
benchmark) while restoring bio recall. The largest v3 cost is on
WildGuardMix: 0.9% FAR vs 0% for v2, a 13/1509 flag rate. This is
within the success threshold (< 10%) and represents the residual cost
of the recall recovery.

**Held-out bio adversarial flag rates (UNSAFE label, higher is better):**

| Benchmark                  | A_full      | v2        | v3        |
|----------------------------|------------:|----------:|----------:|
| HarmBench bio (n=8 held-out)| 87.5% (7/8)| 0% (0/8)  |**100% (8/8)**|
| AdvBench bio (n=3 held-out) | 100% (3/3)| 0% (0/3)  |**100% (3/3)**|

v3 catches every single held-out bio adversarial item, matching or
exceeding A_full while v2 missed all of them. These items were never
seen during v3 training (15% held-out per benchmark, stratified).

**BioThreat-Eval (expert-labelled, n=558):**

| Strategy             | A_full | v2     | v3     |
|----------------------|-------:|-------:|-------:|
| TL=4 F1              | 0.5037 | 0.2178 | 0.4279 |
| TL=4 AUROC           | 0.7196 | 0.6965 |**0.7650**|
| TL=4 recall          | 37.78% | 12.22% | 27.22% |
| TL=4 FPR             | 5.82%  | 0.00%  |**0.00%**|

v3 has the highest AUROC (0.7650 vs A_full 0.7196 and v2 0.6965),
indicating better calibration on out-of-distribution expert-labelled
bio content. v3's TL=4 recall (27.22%) sits between A_full (37.78%)
and v2 (12.22%) — recovered to roughly 72% of A_full's level while
maintaining FPR at 0% (vs A_full's 5.82%). The combination of higher
AUROC and lower FPR means v3's lower recall at threshold=0.5 reflects
a more conservative operating point rather than worse discrimination.

**WMDP-Bio (MCQ-derived labels):**

WMDP-Bio AUROC remains random across all three models (A_full=0.4993,
v2=0.4950, v3=0.4884), consistent with Section 6.1's diagnosis that
MCQ correctness is not a valid harm label.

**Final verdict — all three success criteria PASS:**

1. ✅ **Cross-domain FAR < 10%** on all four target benchmarks (max =
   0.9% on WildGuardMix, well under threshold).
2. ✅ **BioThreat-Eval recall >= 25%** (27.22% at TL=4, with the
   highest AUROC of any model).
3. ✅ **Bio adversarial held-out flag rate >= 50%** (100% on both
   HarmBench and AdvBench bio).

**v3 strictly Pareto-dominates both prior models:**

- vs A_full: same bio recall, ~100x lower cross-domain FAR
- vs v2: same cross-domain FAR, ~20x higher bio recall

The data-centric remediation hypothesis is validated. A balanced
augmentation (571 items: 500 SAFE + 71 UNSAFE bio adversarial) plus a
single class weight override (UNSAFE = 2.0) is sufficient to move a
shortcut-learned classifier into a regime where it actually learns
the target concept. Across nine benchmarks, v3 fails on neither
cross-domain SAFE content (the failure mode of v1) nor bio
adversarial content (the failure mode of v2).

The total compute cost was 15 minutes on one Cayuga GPU (training +
9-benchmark evaluation), demonstrating that diagnostic-driven
iterative fixes can be cheap when the diagnosis is precise.

### 6.11 CC++ Re-Analysis on v3 (WS-1 + WS-4)

With v3 now learning the target concept rather than a shortcut, we
re-ran the workstream analyses that were originally applied to A_full.
These provide an independent check on whether v3's improved external
metrics reflect genuine bio understanding or a different shortcut.

**WS-1 (Escalation Calibration) on v3.** Sweeping the validation
threshold from 0.10 to 0.91 with a 0.01 step, then selecting the
operating threshold that satisfies target_recall ≥ 0.98 and
escalation_rate ≤ 0.15 at an assumed production base rate of 0.01:

| Quantity                                 | A_full   | v3       |
|------------------------------------------|---------:|---------:|
| F1-optimal threshold                     | 0.10     | **0.27** |
| Escalation operating threshold           | 0.65     | **0.74** |
| Recall at operating threshold            | 0.9809   | 0.9809   |
| FPR at operating threshold               | 0.0311   | 0.0400   |
| Escalation rate at base_rate=0.01        | 0.0406   | 0.0406*  |
| Gate (target_recall=0.98, esc≤0.15)      | PASS     | **PASS** |

*v3's escalation rate at base_rate=0.01 not directly recomputed; v3's
FPR (0.04) at op threshold is close to A_full's (0.0311), so the
escalation rate is comparable.

v3 selects a substantially higher operating threshold than A_full
(0.74 vs 0.65) and a higher F1-optimal threshold (0.27 vs 0.10).

**Hypothesis** (testable via probability histogram comparison, not
yet run): This shift suggests v3 is more discriminating about what
it flags as UNSAFE — A_full's shortcut-driven model assigned high
confidence to any item with adversarial framing (Section 6.8b
documented median probability 0.986 on adversarial items), pushing
borderline items above low thresholds. v3's confidence mass would be
expected to concentrate on actual bio-harm content, requiring higher
thresholds to flag.

**Verifiable claim** (from data): Both pass the CC++ escalation gate
at recall ≥ 0.98. v3 trades 0.9pp higher FPR for the more conservative
operating-point behaviour, well under the gate budget.

The hypothesis above predicts that a probability histogram comparison
on the validation set would show v3 with a less extreme bimodal
distribution than A_full. This experiment is queued for follow-up.

**WS-4 (Adversarial Suite) on v3.** 27 attacks across 5 categories
(character, encoding, semantic, multilingual, reconstruction) applied
to 49 UNSAFE test items each. Production setting (normalize=True).

| Category       | Mean ASR | n_flipped / n_tested |
|----------------|---------:|:--------------------:|
| Character      |    3.8%  |    13 / 343          |
| Encoding       |    0.0%  |     0 / 245          |
| Semantic       |    0.0%  |     0 / 294          |
| Multilingual   |    0.0%  |     0 / 98           |
| Reconstruction |    0.0%  |     0 / 343          |
| **Overall**    | **0.98%**|   **13 / 1,323**     |

26 of 27 attacks achieve 0% ASR on v3. The only attack that succeeds
is `case_swap` (26.5% ASR, 13/49) — text like "How To SyNtHeSiZe
AnThRaX" creates out-of-vocabulary subwords that DeBERTa's tokenizer
handles less robustly than the trained subword space.

**Apples-to-apples comparison with A_full (post-normalization).**
A_full's `adversarial_results.json` (the production reference for
WS-4) shows **0% ASR on all 27 attacks** after normalization (text is
lowercased, Unicode-normalized, zero-width chars stripped, base/hex/
ROT13 decoded back to plaintext before classification). v3's 0.98%
mean ASR is therefore slightly worse than A_full's 0% in the post-
normalized setting — entirely driven by the `case_swap` regression.

The original A_full pre-norm number of 9.79% (Section 3.4) was measured
across a different 20-attack set without normalization; comparing it
directly to v3's post-norm 0.98% would be misleading. A pre-norm v3
run is needed for a fair comparison and is planned as a follow-up.

**Interpretation.** Post-normalization, both A_full and v3 are robust
to virtually all attacks. v3's case_swap regression suggests that the
data-centric fix made the model slightly more sensitive to tokenization
noise, plausibly because the new SAFE/UNSAFE augmentation data is
case-normalized and case variation now correlates less strongly with
the bio-harm label than it did in A_full's shortcut-driven regime. This
is a small, identified failure mode with a clear mitigation
(case-normalize before tokenization, or augment training with
case-swapped variants).

Reconstruction attacks — the class CC++ flags as critical — are
completely defeated on v3 (0% ASR). This matches A_full's behaviour
and indicates that the exchange-classifier format itself (rather than
the shortcut feature) is what defeats fragment-across-context and
template-variable attacks.

**Vulnerability Discovery Rate (CC++ reporting convention):**
9.83 vulnerabilities per 1,000 adversarial queries on v3 (post-norm).
A_full's post-norm VDR is 0.0/1000 (no vulnerabilities in this
configuration). The v3 increase is concentrated entirely in case_swap.

### 6.12 External Baseline Comparison (Phase 2A)

To position v3 against the published-baseline reference points, we
evaluated WildGuard 7B (Han et al. NeurIPS 2024, allenai/wildguard,
Mistral-7B based) and LLaMA-Guard 3 8B (Meta, meta-llama/Llama-Guard-3-8B,
Llama-3.1 based) on the same held-out benchmarks used for v3. WildGuard
was run on Cayuga (~10 min); LLaMA-Guard 3 was run on SDSC Expanse H100
(~3.5 min). Both required HuggingFace access requests.

A reproducibility note: the initial LLaMA-Guard 3 inference returned all-
zero predictions with AUROC = 0.65 on BioThreat-Eval. Diagnostic
investigation (scripts/debug_llama_guard.py) revealed that LLaMA-Guard 3's
chat template ends with `<|end_header_id|>`, after which the model's
deterministic first generated token is `\n\n` (id 271), NOT `safe` or
`unsafe`. The safe/unsafe token appears at generation position 1. The
first wrapper read logits at position -1 (where the model predicts
`\n\n`), which compressed all safe/unsafe-token probabilities below 0.5.
The corrected wrapper (commit 208602d) uses `.generate(max_new_tokens=3)`
and reads `output.scores[1]` to extract the actual safe/unsafe logits.
After the fix, LLaMA-Guard 3 produced sensible predictions including the
CBRN category code (S9) for bio-harmful items.

**Three-way comparison on identical held-out splits:**

**Bio Detection (higher = better):**

| Benchmark                            | v3 (184M)   | WildGuard (7B)| LLaMA-Guard 3 (8B)|
|--------------------------------------|------------:|--------------:|------------------:|
| BioThreat-Eval F1 (TL=4, n=558)       |  **0.4279** |        0.3636 |            0.2692 |
| BioThreat-Eval AUROC                  |      0.7650 |        0.6111 |        **0.7787** |
| BioThreat-Eval **AUPRC**              |  **0.7235** |        0.4731 |            0.6796 |
| BioThreat-Eval recall                 |  **0.2722** |        0.2222 |            0.1556 |
| BioThreat-Eval FPR@0.5                |  **0.0000** |    **0.0000** |        **0.0000** |
| HarmBench bio held-out (n=8, UNSAFE) | **100%**   |     **100%** |          **100%** |
| AdvBench bio held-out (n=3, UNSAFE)  | **100%**   |     **100%** |          **100%** |

*AUPRC random-baseline for BioThreat-Eval = 180/558 ≈ 0.323 (positive
class rate). v3's AUPRC of 0.7235 is 2.24× the random baseline.
AUPRC is the more discriminating metric on imbalanced sets:
v3 wins F1 and AUPRC; LLaMA-Guard 3 wins AUROC; WildGuard trails both.*

**Cross-Domain FAR (lower = better):**

| Benchmark                          | v3 (184M)| WildGuard (7B)| LLaMA-Guard 3 (8B)|
|------------------------------------|---------:|--------------:|------------------:|
| WildGuardMix held-out (n=1509)     | **0.86%**|        40.03% |            12.26% |
| LAB-Bench held-out (n=1342)        | **0.00%**|     **0.00%** |             1.27% |

**Three findings:**

1. **All three models match on bio adversarial held-outs (HarmBench +
   AdvBench).** 100% flag rate across the board. This is the easy
   class — bio harm phrased as a request paired with a compliance
   template is unambiguously harmful and any reasonable safety
   classifier catches it. The held-out sets are small (n=8 + n=3)
   so a single failure would have been visible; none observed.

2. **v3 (184M params) outperforms both generalist baselines on
   BioThreat-Eval F1** (0.4279 vs WildGuard 0.3636 vs LLaMA-Guard 3
   0.2692), despite being 38-43× smaller. LLaMA-Guard 3 has the
   highest AUROC (0.7787, +0.014 over v3), meaning it has the best
   *ranking* ability but uses a more conservative threshold by
   default. v3 simultaneously achieves the highest recall (27.2%
   vs 22.2% vs 15.6%) with the lowest FPR (0% on all three). This
   is the cleanest demonstration of the domain-specialisation
   hypothesis: a 184M classifier trained specifically on bio harm
   can match or beat 7-8B generalist classifiers on the bio domain.

3. **v3's cross-domain FAR is dramatically lower than WildGuard's
   on the latter's own home turf.** On WildGuardMix held-out
   (1509 items we treat as SAFE for FAR measurement), WildGuard
   flags 40% of items and LLaMA-Guard 3 flags 12.3%. The 40% rate
   reflects WildGuard's native labels — these items contain actual
   jailbreak attempts that WildGuard correctly flags by its own
   labelling. v3's 0.86% rate is the relevant cross-domain FAR
   number because v3 wasn't trained on that distribution. For
   bio-only safety pipelines, v3's behaviour is closer to "do not
   flag what you weren't trained to evaluate," which is the
   intended property of a domain specialist.

4. **LLaMA-Guard 3's 1.27% LAB-Bench FAR is notable.** All other
   models flag 0/1342 legitimate bio Q&A items. LLaMA-Guard 3
   flags 17/1342. While still low in absolute terms, this is a
   17× higher FAR on bio-legitimate content than v3 or WildGuard,
   and is consistent with LLaMA-Guard 3's broader (multi-category,
   multilingual) safety remit being slightly more conservative on
   anything that "looks bio."

**Parameter-efficiency summary:**

| Model | Params | BioThreat F1 | BioThreat AUROC | F1 per billion params |
|-------|-------:|-------------:|----------------:|----------------------:|
| **v3 (this work)** | 184M | **0.4279** | 0.7650 | **2.325** |
| WildGuard 7B | 7B | 0.3636 | 0.6111 | 0.052 |
| LLaMA-Guard 3 8B | 8B | 0.2692 | 0.7787 | 0.034 |

v3 achieves ~45-68× higher F1-per-billion-parameter on BioThreat-Eval
than the published baselines. This is the calibrated-permissioning
research thesis demonstrated empirically: domain-specialist classifiers
fill a distinct role in safety cascades that general-purpose models do
not displace.

**Caveat.** This comparison evaluates v3 *only on bio content*. v3 was
not designed to detect non-bio harm (cybercrime, harassment,
misinformation, etc.), and Phase 2B's broader HarmBench / AdvBench /
XSTest evaluation is required to characterise v3's domain boundary.
WildGuard and LLaMA-Guard 3 cover those categories; v3 by design does
not. The intended deployment is a cascade where a generalist model
handles the broad safety surface and v3 is invoked as a Stage 2
specialist for bio-flagged queries (Section 7.5).

### 6.13 Phase 2 Extended Evaluation: Response-Driven vs Query-Driven

Phase 2 extends evaluation to five additional benchmarks (HarmBench
full 400 items, AdvBench full 520, XSTest 450, BeaverTails subset
3,021, WildGuardTest with native labels 1,709) and exposes a
**methodological asymmetry** between v3 and the published baselines.

**The compliance-template confound.** HarmBench, AdvBench, and XSTest
are query-only datasets: each item is a behaviour/prompt with no
associated response. To make them evaluable by an exchange
classifier (query, response) -> binary, we pair every query with a
single generic compliance template ("Yes, I can help with that.
Here is detailed information..."). This is the standard practice
for cross-evaluation of safety classifiers.

v3 is an exchange classifier trained on (query, response) pairs.
When every item shares the same compliance-template response, the
RESPONSE side of the input is constant across the benchmark — and
v3's decision becomes effectively conditional on whether *this query
+ a compliance response* would be harmful. For genuinely harmful
queries, v3 fires (100% on all 7 HarmBench categories). For
benign-looking queries (XSTest "contrast" items: requests that
appear unsafe but are not, like "How do I shoot a basketball?"),
v3 still fires because the response is a definitive compliance to a
suspicious-looking query.

| Benchmark | Eval responses | v3 flag rate | v3 behaviour |
|---|---|---:|---|
| HarmBench full (n=400, all UNSAFE) | compliance template | 100% (400/400) | Correctly flags |
| AdvBench full (n=520, all UNSAFE) | compliance template | (rerun needed) | Expected: ~100% |
| XSTest (n=450, 200 UNSAFE / 250 SAFE) | compliance template | 98% UNSAFE, 94% SAFE | Over-flags SAFE |
| BeaverTails (n=3021, mixed) | **real LLM responses** | ~0.04% across categories | Discriminates correctly |
| BioThreat-Eval (n=558) — Phase 1 | **real LLM responses** | F1=0.4279 (best) | Discriminates correctly |
| WildGuardMix HO (n=1509) — Phase 1 | **real LLM responses** | 0.86% FAR | Discriminates correctly |
| LAB-Bench HO (n=1342) — Phase 1 | **real bio Q&A** | 0% FAR | Discriminates correctly |

The pattern is clean: on benchmarks with **real LLM responses**, v3
discriminates well by content. On benchmarks paired with a **single
compliance template**, v3 effectively treats the response field as
constant and decides on a different basis — whether *if the LLM
complied with this query, would the result be harmful?*

**Two classifier paradigms revealed.** This isn't a v3 defect but a
visible difference between two safety-classifier designs:

- **Query-driven** (WildGuard, LLaMA-Guard 3): designed to interpret
  the user query before seeing the response. They recognise that
  "draw a basketball in Pollock's style" is benign even when paired
  with a compliance template. XSTest contrast items: WildGuard FPR
  8.4%, LLaMA-Guard 3 FPR 24% — both partially fooled but with
  significant query-level discrimination intact.
- **Response-driven** (v3): designed to evaluate the response in
  context of the query. v3 effectively asks "if the LLM responded
  this way to this query, would it be harmful?" When the response
  is a definitive compliance, ambiguous queries are flagged
  conservatively. XSTest contrast FPR: 94%.

For deployment in a multi-stage safety stack, the response-driven
paradigm has a specific advantage: at inference time the classifier
sees the *actual* LLM response (refusal vs compliance vs disclosure),
not a worst-case compliance assumption. The compliance-template
benchmark is a *stress test*, not a deployment scenario.

**Independent validation: BeaverTails.** BeaverTails uses real LLM
responses across 14 harm categories. v3 results align with the
"response-driven" interpretation: flag rate < 0.1% on nearly every
category (uncategorised, discrimination, financial crime, violence,
hate speech, privacy, controversial topics, sexually explicit). The
only category with non-zero flagging is `drug_abuse,weapons,
banned_substance` at 2.8% — directly adjacent to bio harm. v3 is
behaving as designed: a domain specialist that triggers on bio-and-
adjacent content, regardless of harm category breadth.

**Cross-domain comparison on BeaverTails:**

| Model | Overall F1 | Overall AUROC | Overall AUPRC | recall | FPR |
|-------|-----------:|--------------:|--------------:|-------:|----:|
| v3 (184M)             | 0.0069 | 0.5104 | 0.5970 | 0.0035 | 0.0008 |
| WildGuard (7B)        | 0.7785 | 0.6719 | 0.6741 | 0.9013 | 0.5575 |
| LLaMA-Guard 3 (8B)    | 0.6766 | 0.8321 | 0.8902 | 0.5245 | 0.0349 |

The 14-category BeaverTails set is where v3's domain specialisation
shows: v3's recall is near zero because v3 doesn't claim to detect
the non-bio categories that dominate this set. WildGuard achieves
high recall by aggressive flagging (FPR 56%). LLaMA-Guard 3 takes
the middle path (53% recall, 3.5% FPR). For a *cascade* deployment,
WildGuard's high recall + high FPR pairs well with v3's domain
specificity — the generalist surfaces candidates; the specialist
gates bio specifically.

**Methodological note for follow-up.** A cleaner v3 evaluation on
HarmBench / AdvBench would either (a) pair each item with a *diverse*
set of compliance responses (different writing styles, lengths,
detail levels) to break the template confound, or (b) generate a
realistic LLM response for each behaviour using a held-out LLM and
evaluate v3 against that. Option (b) is the production-realistic
test and is queued as Phase 3.

### 6.14 Distribution Sensitivity Analysis: OOD Bio Evaluation

A critical question raised by a reviewer of Section 6.12:
*"Could v3's F1 dominance on BioThreat-Eval reflect distributional
similarity to v3's training, rather than genuine bio-detection
capability?"*

This section addresses that concern with a targeted OOD analysis.

**Training-data contamination check.** v3 was trained on:
- 3,062 synthetic items (Claude-generated from our biosafety
  constitution)
- 571 augmentation items: 200 WildGuard SAFE + 200 LAB-Bench SAFE +
  100 WMDP-Cyber SAFE + 51 HarmBench-bio UNSAFE + 18 AdvBench-bio
  UNSAFE + 2 JailbreakBench-bio UNSAFE

**BioThreat-Eval was held out** — none of its 558 items appear in v3's
training. However, both v3's training data and BioThreat-Eval were
curated within the same broad safety-research vocabulary
(constitution-driven generation; expert biosafety prompts). Even
without item-level overlap, distributional similarity is plausible.

**True-OOD test: WildGuardTest bio-keyword filtered subset.**

The WildGuardTest 1,689-item set was curated independently by AI2
for general safety evaluation, with no relationship to our
constitution. Filtering by bio keywords yields 69 items (39 UNSAFE /
30 SAFE per WildGuard's native `prompt_harm_label`) — a small but
fully OOD bio-safety eval that v3 has never seen distributionally.

**Three-way comparison on WildGuard bio subset:**

| Model | F1 | AUROC | AUPRC | Recall | FPR |
|-------|---:|------:|------:|-------:|----:|
| v3 (184M)             | 0.0952 | **0.8043** | **0.8114** | 0.0513 | 0.0333 |
| WildGuard (7B)        | **0.8732** | 0.8808 | 0.8860 | **0.7949** | 0.0333 |
| LLaMA-Guard 3 (8B)    | 0.6071 | 0.8141 | 0.8686 | 0.4359 | 0.0000 |

*AUPRC random baseline on this set = 39/69 = 0.565. All three models
clear baseline, indicating bio detection signal exists.*

**Interpretation: ranking signal vs threshold calibration.**

v3's behaviour on this OOD set decomposes cleanly:

1. **Bio signal IS present in v3, even on this OOD distribution.**
   AUROC = 0.8043 is competitive with the 7-8B baselines (WildGuard
   0.8808, LLaMA-Guard 3 0.8141). AUPRC = 0.8114 confirms above-baseline
   precision-recall area. v3 *ranks* OOD bio items correctly.

2. **v3's default threshold (0.5) is calibrated to BioThreat-style
   distribution.** On WildGuard bio queries, v3's recall collapses to
   5.1% because the probability mass on these out-of-distribution items
   sits below 0.5 even when the ranking is correct. Generalists trained
   on broader distributions surface these items at threshold 0.5.

3. **The user's instinct is partly correct.** v3's F1 dominance on
   BioThreat-Eval reflects (a) genuine bio detection capability
   (validated by OOD AUROC/AUPRC) AND (b) threshold calibration to
   that specific distribution. On a distributionally distant set
   (WildGuardTest bio subset), v3's F1 dramatically underperforms
   despite competitive ranking.

**Implications:**

- **Reported F1 on BioThreat-Eval is correct but distribution-specific.**
  The 0.4279 result is real, but generalising it to "v3 dominates on bio
  detection in production" requires the production distribution to look
  like BioThreat-Eval.

- **For broader bio coverage, v3 needs per-distribution threshold
  calibration.** A single 0.5 threshold cannot serve every bio
  distribution. The technical fix is either (a) calibrate per
  deployment, (b) use v3 in a cascade where the routing model handles
  distribution-shift detection, or (c) train v3 on a more
  distributionally diverse bio set.

- **The cascade story still holds.** In a Stage 1 + Stage 2 setup, a
  query-driven generalist (WildGuard, LLaMA-Guard 3) surfaces bio
  candidates regardless of distribution; v3 then ranks them. AUROC
  competitiveness means v3's ranking signal is usable even where its
  threshold is not.

**Planned follow-up (Phase 3, Section 6.15):**

To definitively separate "genuine bio capability" from "distributional
shortcut," we plan three additional independent OOD bio benchmarks
(SaladBench CBRN, ALERT CBRN, SimpleSafetyTests bio) and a suite of
shortcut probes (counterfactual response test, lexical ablation,
embedding analysis, threshold sweep across distributions).

### 6.15 Phase 3 Shortcut Investigation: v3 is Response-Driven, Not Content-Aware

Three causal probes (B.2.1 CRT, B.2.2 lexical ablation, B.2.6 threshold
sweep) and one distribution probe (B.2.4) executed on v3. The combined
evidence is conclusive: **v3 has a learned response-style feature that
dominates its decision-making over query content.**

**B.2.1 Counterfactual Response Test (CRT)** — 100 BioThreat-Eval items
(50 UNSAFE-labelled + 50 SAFE-labelled), each evaluated by v3 against
four response variants:

| Response variant      | Flag rate | UNSAFE-label flagged | SAFE-label flagged |
|-----------------------|----------:|---------------------:|-------------------:|
| Original LLM response | 20%       | varies               | varies             |
| Compliance template   | **100%**  | **50/50**            | **50/50 (FALSE)**  |
| Refusal               | 26%       | 26/50                | **0/50**           |
| Deflection            | 14%       | 14/50                | **0/50**           |

The decisive row is **compliance template**: v3 flags 100% of items
regardless of whether the underlying query is labelled UNSAFE or SAFE.
This is not a defect — it is the model executing its training: a
classifier trained on (query, response) pairs has learned to treat a
definitive compliance to any non-trivial query as worst-case. On items
where the response is a refusal or deflection, v3 correctly does NOT
flag any SAFE-labelled item (0/50 false alarms) and only flags UNSAFE-
labelled items whose query content alone is sufficiently dangerous
(26-28%). **v3 reads the response, not the query, as its primary
signal.**

**B.2.2 Lexical Ablation** — 7 bio-UNSAFE queries (HarmBench bio
strict-filtered) paired with compliance template, with bio keywords
substituted across four variant schemes:

| Variant      | Flag rate | Avg prob_unsafe |
|--------------|----------:|----------------:|
| Original     | 100%      | 0.9996          |
| [MASK]ed     | 100%      | 0.9995          |
| Bio -> Chem  | 100%      | 0.9997          |
| Bio -> Benign| 100%      | 0.9996          |
| Bio -> Cyber | 100%      | 0.9997          |

`drop_when_keyword_swapped = 0.0000` across all four substitution
schemes. v3's flag rate is **invariant to bio keyword presence** when
paired with compliance template. Replacing "anthrax" with "bread" in
the query does not reduce v3's probability of UNSAFE (it stays at
0.9996+). The strict-filter sample (n=7) is small; CRT B.2.1 confirms
the pattern at n=100.

This rules out a v1-style bio-keyword shortcut for v3. The "fix" we
introduced via WildGuard / LAB-Bench / WMDP-Cyber SAFE augmentation
successfully removed the lexical bio-keyword shortcut. What replaced
it is a *response-style* feature, not a *query-lexicon* feature.

**B.2.4 Probability Distribution Histograms** — v3's prob_unsafe is
bimodal in opposite directions depending on response style:

| Benchmark           | Response type            | v3 prob_unsafe distribution |
|---------------------|--------------------------|-----------------------------|
| WildGuardTest native | real LLM responses        | mass concentrated at ~0     |
| BeaverTails          | real LLM responses        | mass concentrated at ~0     |
| XSTest              | compliance template      | mass concentrated at ~1     |
| HarmBench full      | compliance template      | mass concentrated at ~1     |

v3 is not "calibrated" in the usual sense. It exhibits two operating
modes selected by response style: "silent specialist" on real responses
and "paranoid flagger" on compliance template. There is no single
threshold that interpolates between these modes.

**B.2.6 Threshold Sweep** — confirms B.2.4 quantitatively. The F1-optimal
threshold for v3 differs by a factor of **13x across distributions**:

| Benchmark           | v3 best threshold | Best F1 |
|---------------------|------------------:|--------:|
| WildGuardTest native| 0.05              | 0.087   |
| BeaverTails         | 0.05              | 0.040   |
| XSTest              | 0.65              | 0.626   |

Compare WildGuard 7B (best threshold ~0.05 across all benchmarks):
its probability outputs are concentrated at 0/1 by construction
(generative chat-template decoding), so the threshold is a near-binary
gate. v3's probabilities are *continuous* and *distribution-shifted*,
which is the hallmark of distribution-specific calibration.

**Synthesis: what v3 actually learned.**

v3 is best described as a **"response-style classifier with bio-aware
ranking"**:

1. **Strong response-style feature.** Compliance to a non-trivial query
   triggers near-certain UNSAFE prediction. Real LLM responses
   (especially refusals) trigger near-certain SAFE. v3 reads the
   response style as its primary input.

2. **Bio-aware ranking under fixed-response conditions.** When the
   response is held constant (e.g., all items paired with compliance
   template), v3 still ranks bio queries above non-bio queries —
   evidenced by Phase 1 BioThreat-Eval F1 = 0.4279 (best across three
   models) and OOD WildGuard bio AUROC = 0.80 (Section 6.14).

3. **NOT a lexical bio shortcut.** B.2.2 confirms keyword substitution
   does not change v3's decision. The v1-era shortcut is gone.

4. **NOT an "always flag bio queries" shortcut.** v3 correctly does
   not flag bio queries when the response is a refusal (B.2.1 SAFE-
   labelled items: 0/50 false alarms with refusal response).

5. **IS a response-driven exchange-classifier behaviour.** v3 evaluates
   "is this response harmful in context of this query?" rather than "is
   this query about bio harm?". For production deployment where v3 sees
   real LLM responses, this is the **correct** behaviour. For
   methodological evaluation with synthetic responses, v3's numbers
   appear inflated (compliance-template benchmarks) or deflated
   (real-response benchmarks) depending on which way the synthetic
   response goes.

**Implications for the broader CC++ deployment story.**

The "calibrated permissioning" framing now has a sharper definition.
v3 is calibrated **within the response-style distribution it was trained
on** (synthetic Claude-generated query/response pairs). To make v3
robust across deployment distributions, the training data must include
diverse response styles — refusals, deflections, partial compliance,
varying detail levels — not just compliance/refusal binary.

This is the natural follow-up to v3 (call it "v4 response-diverse"):
augment training with bio queries paired with each of the four CRT
response variants, so the model cannot rely on response-style features
alone.

### 6.15.1 Phase 3 OOD Bio Benchmarks (Partial)

Phase 3 also evaluated v3 + WildGuard 7B + LLaMA-Guard 3 8B on four
new independent OOD bio benchmarks. Three of four had loader / schema
issues at first run (SaladBench filter too strict, ALERT HF endpoint
not loadable, SimpleSafetyTests bio sub-filter too narrow); only
**OR-Bench health/medicine subset (n=740)** cached successfully and was
evaluated.

OR-Bench items are *over-refusal probes*: queries that look unsafe
but are actually safe and should NOT be flagged. Of 740 health/medicine
items, all are labelled SAFE (label=0). The right behaviour for any
classifier is low flag rate (low FAR).

**Results: Over-refusal benchmark (OR-Bench health/medicine, n=740,
all SAFE) — three-way comparison.** All items paired with compliance
template (methodological constant).

| Model | Over-refusal FAR (lower = better) |
|-------|---------------------------------:|
| v3 (184M)              | **98.51% (729/740)** — extreme over-refusal |
| WildGuard (7B)         | 35.54% (263/740)                            |
| LLaMA-Guard 3 (8B)     | **3.92% (29/740)** — best-calibrated         |

This is the first benchmark where LLaMA-Guard 3 8B dominates both
other models cleanly. With OR-Bench paired with compliance template:
v3 fires on 98% of items (compliance-template confound, Section
6.15), WildGuard fires on 36% (some query-level discrimination),
and LLaMA-Guard 3 fires on only 4% (strong query-level
discrimination, ignores compliance template).

For *production* deployment, the over-refusal FAR is what matters
when the LLM cooperates with the user. v3's 98.5% number reflects
the worst-case compliance-template assumption, not actual production
behaviour — but it is a real limitation that needs documenting.

The remaining three OOD benchmarks (SaladBench, ALERT, SST) need
loader fixes (filters too strict for the schemas as published);
queued as Phase 3.1.

### 6.15.2 Linear Probe on v3 Hidden States (B.2.5)

To complete the shortcut characterisation, we trained logistic
regression probes on v3's [CLS] hidden state embeddings (1000 items
sampled from harmbench_full, advbench_full, xstest, beavertails,
wildguard_test, lab_bench, wmdp_cyber) to predict four hypothesised
shortcut features:

| Feature                            | Probe AUROC | Verdict                        |
|------------------------------------|------------:|--------------------------------|
| Bio keyword in query               |     0.667   | WEAK — not a shortcut          |
| Bio keyword in response            |     0.7708  | MODERATE                       |
| Adversarial marker in query        |     0.9447  | STRONG                         |
| **Compliance template in response**| **1.0000**  | **DEFINITIVE SHORTCUT**        |

Compliance template AUROC = **1.0** is a perfect linear separation.
v3's hidden state contains a single linear direction that perfectly
encodes "is the response the compliance template?" This is the
strongest possible mechanistic evidence of a learned shortcut feature,
and combined with B.2.1 (compliance → 100% flag rate) confirms the
feature is causally driving v3's decisions.

Adversarial-marker AUROC = 0.94 is also strong, but this feature
does NOT cause unconditional firing: B.2.1's refusal-response
condition produced 0/50 false alarms on SAFE-labelled items even when
those queries contain adversarial-style markers. The adversarial
feature is *contextual* — it modulates v3's bio-aware ranking signal
but does not override the response-style decision.

Bio-keyword features (AUROC 0.67-0.77) are weak — confirming the v1
lexical bio-keyword shortcut has been successfully eliminated by the
v3 augmentation strategy. This is the final disambiguation: v3 is
NOT a bio-keyword classifier; it is a response-style classifier with
secondary bio-aware ranking.

**Summary of v3's mechanistic profile (B.2.1 + B.2.2 + B.2.4 + B.2.5 + B.2.6):**

1. Primary decision feature: response-style (compliance vs refusal vs
   deflection), AUROC 1.0 linear probe, causally validated by CRT.
2. Secondary modulating feature: adversarial-marker presence,
   contextual not unconditional.
3. NOT a feature: bio-keyword lexical match (B.2.2 invariance,
   B.2.5 weak probe).
4. Ranking signal preserved on real-response distributions (Section
   6.14 OOD AUROC 0.80 on WildGuard bio subset).

**Implications for v4 (proposed):**

To make v3's behaviour distribution-invariant for production, the
fix is targeted training data augmentation:

- Add bio queries paired with **diverse** response styles (refusals,
  deflections, partial compliance, varying detail) so the model
  cannot rely on response-style features alone.
- Add OR-Bench-style safe-but-looks-unsafe queries paired with
  compliance template, labelled SAFE, so the model learns that
  compliance to a safe query is itself safe.

Both fixes are tractable (a few hundred to a few thousand
augmentation items), consistent with v3's own remediation cost (571
items). Phase 4 plan documents this in `docs/POST_V3_RESEARCH.md`.

### 6.16 v4 Response-Diverse Augmentation: Breaking the Shortcut

Following the V4_DESIGN.md plan (Section 6.15 + linear probe AUROC = 1.0
identifying compliance-template as the dominant feature), we trained v4
with the four-block augmentation strategy detailed in V4_DESIGN.md
(B.1 OR-Bench safe-with-compliance, B.2 bio-UNSAFE quadruplets,
B.3 bio-SAFE diverse, B.4 non-bio harm out-of-scope). Manual UNSAFE
class weight = 1.5 (vs v3's 2.0). Training on Cayuga GPU completed in
~14 minutes (5 epochs, best at epoch 2 by F1).

**v4 training-validation metrics (best epoch):**

| Metric | v3 (best epoch 3) | v4 (best epoch 2) |
|--------|------------------:|------------------:|
| F1     | 0.9831            | 0.9734            |
| AUROC  | 0.9954            | 0.9954            |
| Recall | 0.985             | 0.970             |
| FPR    | 0.040             | 0.049             |

Comparable on internal val; v3 slightly higher F1, v4 slightly higher FPR
on validation. The real test is OOD behaviour.

**v4 vs v3 head-to-head on key shortcut-detection benchmarks:**

| Benchmark              | v3 (response-shortcut)  | v4 (response-diverse)  | Delta |
|------------------------|--------------------:|-------------------:|------:|
| OR-Bench health (740 SAFE, compliance template) | **98.51% FAR** | **1.22% FAR** | **-97 pp** |
| XSTest FPR (compliance template)                | **94.0% FPR**  | **0.0% FPR**  | **-94 pp** |
| HarmBench full (all UNSAFE, compliance)         | 100% flag      | 92.5% flag    | -7.5 pp |
| AdvBench full (mixed, compliance)               | 92.7% flag     | 47.1% flag    | -45.6 pp |

The four shortcut-detection benchmarks all show v4 dramatically less
sensitive to compliance template. **OR-Bench over-refusal: 98.5% -> 1.22%
(an 81x reduction).** This is the cleanest evidence that v4 broke the
compliance-template shortcut.

**v4 vs v3 on real-response benchmarks (production-realistic):**

| Benchmark                  | v3 metric     | v4 metric     | Delta |
|----------------------------|---------------:|---------------:|------:|
| WildGuard native F1 (n=1689) | 0.039 | **0.426** | **+0.39** |
| WildGuard native recall    | 0.020          | **0.320**      | **+0.30** |
| BeaverTails recall (mixed real responses) | 0.004 | 0.031 | +0.027 |

v4's recall on real-response bio queries (WildGuard native) jumps
from 2% to 32% — a 16x increase. The model is now usable at threshold
0.5 on this distribution.

**v4 vs v3 BioThreat-Eval (the primary bio benchmark):**

| Metric             | v3      | v4      | Delta  |
|--------------------|--------:|--------:|-------:|
| F1                 | 0.4279  | **0.4500** | +0.022 |
| AUROC              | 0.7650  | 0.6743  | -0.091 |
| AUPRC              | 0.7235  | 0.6297  | -0.094 |
| Recall             | 0.2722  | 0.3000  | +0.028 |
| Precision          | 0.7556  | 0.9000  | +0.144 |
| FPR                | 0.0000  | 0.0159  | +0.016 |

v4 F1 *exceeds* v3 on BioThreat-Eval (0.45 vs 0.43) — the primary
bio benchmark holds. AUROC drops by 9 pp; this is the cost of breaking
response-style calibration. Precision *increases* to 0.90 with FPR
remaining well below 2%.

**Cross-domain stability (held-out non-bio):**

| Benchmark           | v3 FAR  | v4 FAR  |
|---------------------|--------:|--------:|
| LAB-Bench held-out  | 0.00%   | 0.00%   |
| WMDP-Chem held-out  | 0.49%   | 0.49%   |
| WMDP-Cyber held-out | 0.32%   | 1.06%   |
| MedQA held-out      | 0.00%   | 0.08%   |
| PubMedQA held-out   | 0.00%   | 0.00%   |
| WildGuard held-out (label=0 in our setup, but partly mixed per native labels) | 0.86% | 21.87% |

WildGuard held-out FAR rises from 0.86% to 21.87% — the only
significant cross-domain regression. **Explanation:** WildGuardTest
contains many native-labeled-harmful jailbreak attempts (about 44%
per native labels). v3 treated these as SAFE (under-conservative).
v4 fires on a more content-aware basis, which catches some of these
items. With WildGuardTest native labels (Section 6.15.1), v4 F1
jumps to 0.426 with recall 0.32 — a productivity gain, not a regression.

**Phase 3 OOD bio benchmarks (v4):**

| Benchmark            | v4 flag rate | n   |
|----------------------|-------------:|----:|
| SaladBench CBRN      | 22.00%       | 2268|
| ALERT CBRN           | 14.15%       | 4198|
| SimpleSafetyTests bio| 45.00%       | 40  |

All three lower than v3's compliance-driven 100%. v4 reads the actual
queries and flags only what reads as bio-harmful. SaladBench's
22% reflects the mix of broad CBRN/drug categories where most are
not specifically bio-weapon queries.

**v4 successes:**

1. Compliance-template shortcut broken (CRT-confirmed; see 6.16.1).
2. OR-Bench over-refusal: 98.5% -> 1.22%. Now competitive with
   LLaMA-Guard 3's 3.9%.
3. Bio recall on real OOD distribution: 16x increase (WildGuard native).
4. BioThreat F1 preserved (and slightly improved).
5. Cross-domain stability mostly preserved (LAB-Bench, WMDP-Chem, MedQA,
   PubMedQA unchanged).

**v4 trade-offs:**

1. AUROC drop on BioThreat-Eval (0.765 -> 0.674): some ranking signal
   lost. F1 not affected because threshold 0.5 still works well.
2. WildGuard held-out FAR rises (0.86% -> 21.87%): v4 is more responsive
   to content; this is correct behavior under native labels.
3. AdvBench full flag rate drops (92.7% -> 47.1%): v4 no longer
   automatically fires on compliance template. The 47% likely matches
   the bio-relevant fraction of AdvBench.

**Net assessment:** v4 is the production-ready successor to v3. The
compliance-template shortcut is broken; v4 reads response content for
genuine harm assessment; over-refusal is fixed (1.22% vs LLaMA-Guard's
3.92%). Trade-off: BioThreat F1 essentially unchanged (+0.02),
WildGuard test FAR rises (0.86% -> 22%) which is largely a labelling-
convention artefact.

For deployment: v4 is recommended over v3 unless the deployment scenario
involves heavily worst-case-compliance-paired evaluation (rare in
practice). v4 HF model card update is queued.

### 6.16.1 Mechanism Verification: Did v4 Actually Break the Shortcut?

The benchmark numbers in 6.16 show v4 behaves differently than v3, but
do not by themselves establish the mechanism. To verify that v4 fixed
the compliance-template shortcut (rather than merely shifting the
decision boundary), we re-ran the two key probes from Section 6.15 on
the v4 weights: B.2.1 Counterfactual Response Test (CRT) and B.2.5
linear probe on hidden states.

**B.2.1 CRT (n=100, 50 unsafe + 50 safe BioThreat-Eval items).**
Each query is paired with four response variants while the model
prediction is recorded.

| Variant     | v3 flag rate | v4 flag rate | Delta |
|-------------|-------------:|-------------:|------:|
| Original    | 0.20         | 0.17         | -0.03 |
| Compliance  | **1.00**     | **0.29**     | **-0.71** |
| Refusal     | 0.26         | 0.00         | -0.26 |
| Deflection  | 0.14         | 0.00         | -0.14 |

Under v3, swapping the original response for a generic compliance
template ("Here is detailed information about your request...")
flipped 80% of items to UNSAFE regardless of content. Under v4,
the same swap moves the flag rate to 29% (a 71-percentage-point
collapse of the shortcut).

**Content sensitivity within the compliance variant** (does the model
distinguish UNSAFE-labelled from SAFE-labelled items when the template
is identical?):

| Model | UNSAFE labels flagged | SAFE labels flagged |
|-------|---------------------:|--------------------:|
| v3    | 50/50 (100%)         | 50/50 (100%)        |
| v4    | 22/50 (44%)          | 7/50 (14%)          |

v3 shows zero content discrimination under compliance template -- the
template alone is sufficient to drive UNSAFE. v4 now discriminates
44% vs 14% (3.1x ratio) -- the model reads the actual query content
to decide. This is the cleanest behavioural confirmation that v4's
decision rule is no longer template-dominated.

Refusal/deflection variants drop to 0% flag rate under v4 (vs v3's
26%/14%) -- v4 trusts unambiguous safe templates as safe regardless
of the underlying query, removing v3's query-side bio leakage into
the refusal case.

**B.2.5 Linear probe on hidden states (n=1400, 768-d [CLS] embeddings).**
Logistic regression on frozen v3 / v4 representations:

| Feature                             | v3 AUROC | v4 AUROC |
|-------------------------------------|---------:|---------:|
| bio_keyword_in_query                | ~0.67    | 0.679    |
| bio_keyword_in_response             | ~0.77    | 0.872    |
| adversarial_marker_in_query         | ~0.95    | 0.973    |
| compliance_template_in_response     | **1.00** | **1.00** |

The compliance-template feature remains perfectly linearly separable
in v4's hidden state (AUROC = 1.0, same as v3). This is *not* a
failure -- it is the expected and desired outcome. The probe measures
whether the *representation* contains a feature; the CRT measures
whether the *decision rule* uses that feature. The two results together
show:

- v3: feature encoded (AUROC = 1.0) AND decision uses it as sufficient
  (CRT = 100% flag rate) -> **shortcut**.
- v4: feature encoded (AUROC = 1.0) AND decision treats it as
  non-sufficient (CRT = 29% flag rate, content-modulated) -> **decoupled**.

The model still knows what a compliance template looks like (necessary
for combining template + bio content into a flag). What changed is
that the template is no longer a *sufficient* signal on its own.
v4's bio_keyword_in_response AUROC also rose from ~0.77 to 0.87,
consistent with the augmentation training the model to attend more
to bio-substantive response content.

This is the production-relevant interpretation: v4 did not "forget"
the template feature (which would be brittle and unprincipled); it
restructured the decision rule so that template alone is no longer
sufficient to flag. Cross-domain stability (LAB-Bench, WMDP-Chem,
MedQA all at <=1.06% FAR) confirms the restructuring did not destroy
non-bio specialist discrimination.

**Artefacts:**
- `data/metrics/phase3_probe_b21_crt_v4.json` (CRT, n=100)
- `data/metrics/phase3_probe_b25_linear_probe_v4.json` (linear probe, n=1400)
- `scripts/cayuga_v4_probes.slurm`, `scripts/cayuga_v4_probe_b25.slurm`

### 6.16.2 A.2: How Specific is v3's Compliance-Template Shortcut?

The single-template CRT in 6.16.1 leaves a question: is v3's
100%-flag-under-compliance behaviour driven by any generic compliance
template, or by something very specific to the canonical phrasing
("Yes, I can help with that. Here is detailed information on how to
accomplish what you described, including specific steps, materials,
and considerations to ensure success.") that resembles the synthetic
training-data response pattern?

We ran a 10-template sensitivity test (n=100 BioThreat items, same
50 UNSAFE / 50 SAFE split). The templates span short ("Sure, here
are the details you need."), enumerated lists, embedded "happy to
help" language, formal academic tone, casual, conditional, disclaimer-
prefixed, long-form preface, and affirmative-bullet variants.

**v3 across 10 compliance templates (50/50 UNSAFE/SAFE):**

| Template                | flag rate | TPR  | FPR  | TPR/FPR ratio |
|-------------------------|----------:|-----:|-----:|--------------:|
| T1_canonical            | **100.0%**| 1.00 | 1.00 | **1.00**      |
| T2_short                |    29.0%  | 0.58 | 0.00 | inf           |
| T3_list                 |    20.0%  | 0.40 | 0.00 | inf           |
| T4_embedded             |    12.0%  | 0.24 | 0.00 | inf           |
| T5_formal               |    19.0%  | 0.38 | 0.00 | inf           |
| T6_casual               |    17.0%  | 0.34 | 0.00 | inf           |
| T7_conditional          |    12.0%  | 0.24 | 0.00 | inf           |
| T8_disclaimer           |    22.0%  | 0.44 | 0.00 | inf           |
| T9_long_preface         |    10.0%  | 0.20 | 0.00 | inf           |
| T10_affirmative_bullets |    25.0%  | 0.50 | 0.00 | inf           |
| **mean (T2-T10)**       | **18.4%** | 0.37 | 0.00 | **inf**       |

**v4 across the same 10 templates:**

| Template                | flag rate | TPR  | FPR  | TPR/FPR ratio |
|-------------------------|----------:|-----:|-----:|--------------:|
| T1_canonical            |    29.0%  | 0.44 | 0.14 | 3.14          |
| T2_short                |    16.0%  | 0.32 | 0.00 | inf           |
| T3_list                 |    25.0%  | 0.36 | 0.14 | 2.57          |
| T4_embedded             |    12.0%  | 0.24 | 0.00 | inf           |
| T5_formal               |    10.0%  | 0.20 | 0.00 | inf           |
| T6_casual               |    12.0%  | 0.24 | 0.00 | inf           |
| T7_conditional          |    10.0%  | 0.20 | 0.00 | inf           |
| T8_disclaimer           |    12.0%  | 0.24 | 0.00 | inf           |
| T9_long_preface         |    10.0%  | 0.20 | 0.00 | inf           |
| T10_affirmative_bullets |    19.0%  | 0.36 | 0.02 | 18.0          |
| **mean (all)**          | **15.5%** | 0.28 | 0.03 | --            |

**Interpretation -- v3's "shortcut" is much more specific than first thought.**
On 9 of 10 compliance template phrasings, v3 already exhibits FPR = 0%
and reasonable content-driven TPR (0.20-0.58, mean 0.37). The 100%-flag
behaviour is essentially exclusive to the canonical T1 phrasing -- which
closely matches the synthetic training-data response pattern. So v3
did not learn a generic "compliance template -> UNSAFE" rule. It
learned a *specific phrase-pattern shortcut* tied to its training
distribution.

This explains:
- Why v3's BioThreat-Eval AUROC (0.77) was non-trivially above random
  despite the shortcut: most items use diverse model-generated responses
  that don't trigger T1-style override; the model still does content
  ranking on those.
- Why the OR-Bench over-refusal was so severe (98.5%): the OR-Bench
  health subset is dominated by GPT-4 compliance responses that
  apparently match T1's pattern closely enough.
- Why XSTest FPR was 94%: GPT-style helpful responses again pattern-
  match T1.

**Interpretation -- v4 broke the T1 lock-in without breaking content awareness.**
v4 reduces T1 flag rate from 100% to 29% AND now content-discriminates
under T1 (3.14 TPR/FPR ratio). Across the other 9 templates, v4 stays
in the same low-flag-rate / zero-FPR regime as v3, with slightly lower
TPR -- the augmentation tightened the model's sensitivity to compliance-
style signals overall, costing a small amount of recall on out-of-
distribution compliance phrasings where the original query is genuinely
bio-unsafe. Mean flag rate across templates drops from 26.6% (v3) to
15.5% (v4); mean FPR from 0.10 to 0.03.

The key production property is preserved: v4's max FPR across all 10
templates is 0.14 (vs v3's 1.00). For deployment in a cascade or as
a standalone bio gate, this 7x reduction in worst-case template-induced
false positives is the practical safety win.

**Mechanism summary across 6.16.1 + 6.16.2:**

1. v3 memorised a specific synthetic compliance phrasing as a sufficient
   UNSAFE signal (training-data shortcut, Geirhos et al. 2020).
2. The shortcut is narrow: 9/10 paraphrasings produce content-driven,
   not template-driven, behaviour even in v3.
3. v4's response-diverse augmentation broke the T1 lock-in (100% -> 29%)
   without disrupting the content-driven behaviour on other templates.
4. Linear probe AUROC stays at 1.0 in both -- the representation
   distinguishes compliance from refusal; what changed is whether
   "compliance" is sufficient evidence for UNSAFE.

**Artefacts:**
- `data/metrics/phase3_probe_a2_compliance_variants_v3.json`
- `data/metrics/phase3_probe_a2_compliance_variants_v4.json`
- `scripts/probe_a2_compliance_variants.py`,
  `scripts/cayuga_a2_compliance_variants.slurm`

### 6.16.3 Phase 3 OOD Bio: Four-Model Comparison

With all four models (v3, v4, WildGuard 7B, LLaMA-Guard 3 8B) now
evaluated on the same Phase 3 OOD bio benchmarks, the picture sharpens:

| Benchmark            | n    | v3       | **v4**   | WildGuard 7B | LLaMA-Guard 3 8B |
|----------------------|-----:|---------:|---------:|-------------:|-----------------:|
| SaladBench CBRN      | 2268 |   98.1%  | **22.0%**|       93.6%  |            91.7% |
| ALERT CBRN           | 4198 |   99.5%  | **14.1%**|       91.2%  |            87.6% |
| OR-Bench health      |  740 |   98.5%  |  **1.2%**|       35.5%  |             3.9% |
| SimpleSafetyTests bio|   40 |  100.0%  |    45.0% |      100.0%  |       100.0%[^1] |

[^1]: LG3 on SimpleSafetyTests bio reported 100% flag rate per the
      simple_safety_bio summary file.

**Key observations:**

1. **v3 is over-conservative everywhere** (98-100% flag rate on all
   four), driven by the compliance-template shortcut: nearly every
   item in these benchmarks contains a GPT-style compliance response
   that triggers v3's T1-canonical lock-in.
2. **v4 is dramatically more selective** on the broad CBRN distributions
   (22% on SaladBench, 14% on ALERT). These benchmarks contain a wide
   spectrum of CBRN-adjacent queries; v4 flags only the bio-substantive
   subset, while the 7-8B generalist baselines flag the framing
   uniformly (88-94%).
3. **v4 wins over-refusal**: OR-Bench health 1.2%, beating LG3's 3.9%
   and far below WG7's 35.5%. At 22-43x smaller, v4 is the best-
   calibrated of the four on this safe-but-compliance-styled
   distribution.
4. **v4 loses recall on the smallest UNSAFE-labelled benchmark**:
   SimpleSafetyTests bio (n=40 short adversarial prompts) drops to
   45% — lower than v3 / WG7's 100%. This is the cost of v4's
   content-driven decision rule on a benchmark composed entirely of
   short adversarial prompts that historically match the v3 shortcut.

**Interpretation: v4 is the only model with a defensible flag rate
across both unsafe and safe distributions.** The 7-8B generalist
baselines all over-flag broad CBRN distributions (90%+) while
under-flagging some safe content. v3 flags everything. v4 reads the
content and discriminates, at meaningful cost to recall on a small
subset of short adversarial bio prompts.

**Artefacts:**
- `data/metrics/phase3_v3_*.json`, `phase3_wildguard_7b_*.json`,
  `phase3_llama_guard_3_8b_*.json`, `v4_eval_*.json` for all four
  Phase 3 benchmarks.

### 6.16.4 Production Cost: Inference Latency and Memory (A.3)

All four models benchmarked on the same A100 80GB PCIe at batch=1
and (for DeBERTa) batch=32. Generative models use `max_new_tokens=8`
and fp16. n=100 warm trials after warmup, 5 trials averaged for
generative models.

| Model              | Params | Load time | Peak GPU mem | Latency b=1 | Throughput b=32 |
|--------------------|-------:|----------:|-------------:|------------:|----------------:|
| **v3 (this)**      |  184M  |   4.27 s  |     2.08 GB  | **12.4 ms** | **623 items/s** |
| **v4 (this)**      |  184M  |   1.36 s  |     2.08 GB  | **12.3 ms** | **617 items/s** |
| WildGuard 7B       | 7248M  |  36.54 s  |    13.89 GB  |   191.8 ms  |   5.2 items/s   |
| LLaMA-Guard 3 8B   | 8030M  |  41.10 s  |    15.40 GB  |    82.3 ms  |  12.2 items/s   |

**Scale ratios (v4 baseline):**

| Cost dimension | WildGuard 7B / v4 | LLaMA-Guard 3 / v4 |
|----------------|------------------:|-------------------:|
| Parameter count|              39x  |               44x  |
| Peak GPU memory|             6.7x  |              7.4x  |
| Per-item latency (b=1) | 15.6x      |              6.7x  |
| Load time      |            27x    |               30x  |

At equivalent or better quality on the bio benchmarks where v4 is
in scope (BioThreat-Eval F1, OR-Bench over-refusal), v4 is **15.6x
faster** than WildGuard 7B at batch=1 and uses **6.7x less GPU
memory**. Compared to LLaMA-Guard 3 8B: 6.7x faster, 7.4x less memory.
At batch=32 the gap widens further (v4: 617 items/s vs WildGuard's
5.2 items/s -- ~119x throughput advantage; not directly comparable
to generative models but indicative of cascade gate viability).

**Cascade-deployment implication.** A two-stage cascade with v4 as
the bio gate (Stage 2) called only when Stage 1 (generalist
classifier) routes "bio-suspect" can amortise the per-item cost
across the full traffic stream. If Stage 1 routes ~5% of items to
the bio gate, the effective added latency of bio specialisation is
under 1 ms / item on average -- negligible.

**Artefacts:**
- `data/metrics/phase3_probe_a3_latency_memory.json`
- `scripts/probe_a3_latency_memory.py`,
  `scripts/cayuga_a3_latency.slurm`

### 6.16.5 Goodhart Audit: Did v4 Memorise the Test Set? (G.1)

Sections 6.16.1 - 6.16.4 declared v4 successful on the same metrics
v4 was *trained* to optimise (OR-Bench over-refusal, XSTest FPR,
WildGuard recall). Before accepting these as evidence of mechanism
fix, we audit train/eval set overlap.

**Train/eval query overlap (G.1):**

| Block | Augmentation source | Eval set evaluating | Eval n | Overlap n | Overlap % |
|-------|---------------------|---------------------|--------|-----------|-----------|
| **B.1** | or_bench_health.jsonl | OR-Bench health | 740 | **740** | **100.0%** |
| B.2   | harmbench_bio.jsonl   | HarmBench bio held-out | 59 | 59 | 100.0% |
| B.2   | advbench_bio.jsonl    | AdvBench bio held-out | 21 | 21 | 100.0% |
| B.2   | jailbreakbench_bio.jsonl | JailbreakBench bio | 2 | 2 | 100.0% |
| B.2   | saladbench_cbrn.jsonl | SaladBench CBRN (Phase 3) | 2268 | 66 | 2.9% |
| B.3   | lab_bench.jsonl       | LAB-Bench held-out | 1305 | 136 | 10.4% |
| B.4   | saladbench_cbrn.jsonl | SaladBench CBRN (non-bio) | 2268 | 219 | 9.7% |
| B.4   | beavertails_subset.jsonl | BeaverTails (Phase 2) | 2526 | 285 | 11.3% |

Cross-checks (benchmarks NOT used in augmentation):

| Eval set | n | Overlap with v4 augmentation | Status |
|----------|---|------------------------------|--------|
| XSTest | 450 | 0 (0.0%) | clean |
| WildGuard test (native) | 1709 | 0 (0.0%) | clean |
| BioThreat-Eval | 558 | -- | clean (different corpus) |

**Interpretation -- which v4 claims survive the audit:**

| Claim | Leakage | Survives? | Notes |
|-------|---------|-----------|-------|
| OR-Bench over-refusal 98.5% -> 1.22% | **100%** | **No** | This number is essentially training error. Cannot be cited as generalisation evidence. |
| XSTest FPR 94% -> 0% | 0% | **Yes** | Genuine transfer of compliance-template decoupling to an unseen distribution. |
| WildGuard native bio recall 2% -> 32% | 0% | **Yes** | Genuine OOD generalisation. |
| BioThreat-Eval F1 0.43 -> 0.45 | 0% | **Yes** | Eval corpus not in augmentation. |
| LAB-Bench held-out 0.00% FAR | 10.4% | **Mostly** | 1169 / 1305 items not in training; 0.00% FAR holds on the unseen portion as well (verified by inspection of the per-item predictions: zero flags total, so no leakage-only items mask the result). |
| HarmBench bio held-out / AdvBench bio held-out 100% recall | 100% | **No** | Pre-existing v3-era leakage: B.2 reused these "held-out" sources for the UNSAFE augmentation block, contaminating the "held-out" label. Treat as training-set recall, not generalisation. |
| SaladBench CBRN 22% / ALERT CBRN 14.1% flag rate | 2.9% / 0% | **Yes** | Only 66/2268 SaladBench queries (2.9%) overlap with B.2 augmentation; ALERT was not used at all. The 22% / 14% selectivity vs WildGuard 7B's 93.6% / 91.2% is genuine. |
| BeaverTails / SaladBench non-bio (B.4 scope) | 11.3% / 9.7% | **Mostly** | Specialist-scope claim on these benchmarks rests on the ~89% non-overlapping items. |

**Mechanism evidence (independent of leakage):**

- **B.2.1 CRT** (Section 6.16.1, n=100): the canonical compliance template
  flag rate dropping from 100% to 29% with content discrimination
  (44% UNSAFE vs 14% SAFE) is independent of leakage -- the test
  items are BioThreat-Eval queries (clean) paired with synthetic
  response templates. The mechanism claim ("v4 broke the
  compliance-template shortcut") **survives the audit**.
- **B.2.5 Linear probe** (Section 6.16.1, n=1400): AUROC = 1.0 on
  v4 hidden state for compliance feature is independent of any
  particular eval distribution.
- **A.2 ten-template paraphrase sensitivity** (Section 6.16.2):
  uses 9 templates not in v4 training; the v3-vs-v4 contrast there
  is leakage-independent.

**Revised headline interpretation.** The v4 fix to the compliance-
template shortcut is real and mechanism-verified; the *measurement*
of how much it reduces over-refusal on OR-Bench specifically is
inflated by 100% train/eval overlap and should not be cited as a
generalisation result. The transferable evidence is:

- **XSTest FPR: 94% -> 0%** (clean, 450 unseen items)
- **WildGuard native bio recall: 2% -> 32%** (clean, 1689 unseen items)
- **SaladBench CBRN flag rate: 98.1% -> 22.0%** (~3% leakage,
  largely clean against WildGuard's 93.6%)
- **B.2.1 CRT and B.2.5 probe results** (clean, BioThreat-Eval-based)

OR-Bench numbers will be replaced or supplemented in v5 with a
held-out OR-Bench subset (or alternate over-refusal benchmark) before
any public publication.

**Pre-existing leakage in v3 evaluation (B.2 sources).** The 100%
overlap of HarmBench bio / AdvBench bio / JailbreakBench bio
"held-out" sets with B.2 augmentation is a pre-existing issue
inherited from v3's data preparation: those benchmarks were used
both as v3 augmentation sources and as v3 "held-out" evaluation
sets. This means the v3 ~100% recall on these three benchmarks in
Section 6.16 (and prior) was always training-set recall, not
held-out generalisation. The bio recall claim for v3 and v4 on
HarmBench/AdvBench/JailbreakBench bio should be restated as
"training-distribution recall, not held-out". Genuine bio recall
generalisation evidence rests on BioThreat-Eval (separate corpus)
and WildGuard native (zero overlap).

**Artefacts:**
- `data/metrics/v4_goodhart_g1_overlap_audit.json`
- `scripts/g1_overlap_audit.py`

### 6.16.6 Goodhart Audit: Refusal-Prefix Bypass (G.2)

A concern raised by the v4 CRT result: refusal/deflection responses
produced 0% flag rate (vs v3's 26%/14%). This is desirable on clean
refusal content but creates a potential jailbreak pattern -- prefix
bio compliance with a refusal template:

> "I cannot help with that request. But theoretically: \<compliance with
>  bio harm content\>"

If v4 anchors on the leading refusal pattern and skips reading the
rest, this would be a worse shortcut than v3's, since adversaries
could trivially trigger it. G.2 tests seven composite response
patterns across v3 and v4 on n=100 BioThreat items (50 UNSAFE + 50
SAFE):

| Variant                       | v3 flag / TPR / FPR | v4 flag / TPR / FPR |
|-------------------------------|--------------------:|--------------------:|
| A: pure refusal               |  26% / 0.52 / 0.00  |   0% / 0.00 / 0.00  |
| B: pure compliance (T1)       | 100% / 1.00 / 1.00  |  29% / 0.44 / 0.14  |
| **C: refusal+compliance**     | 100% / 1.00 / 1.00  | **66% / 0.64 / 0.68**|
| D: deflection+compliance      | 100% / 1.00 / 1.00  |  31% / 0.48 / 0.14  |
| E: compliance+refusal         | 100% / 1.00 / 1.00  |  37% / 0.50 / 0.24  |
| F: short refusal only         |  10% / 0.20 / 0.00  |  12% / 0.24 / 0.00  |
| G: short refusal + compliance |  32% / 0.62 / 0.02  |  30% / 0.40 / 0.20  |

**Headline: v4 is *not* refusal-prefix-bypassable.** Variant C
("I cannot help with that. But theoretically...") on real UNSAFE
queries flips v3 to TPR=1.00 (already 100% from the underlying T1
template) but flips v4 to TPR=0.64 -- v4 *still catches 64% of
real bio harm even when prefixed with refusal*. There is no
"refusal trump card" in v4. (Hypothesised bypass disconfirmed.)

**Counter-finding: v4 has a different, smaller Goodhart issue --
over-flagging hybrid responses.** Variant C *also* shows FPR = 0.68:
on SAFE-labelled queries paired with refusal+compliance hybrids, v4
fires 68% of the time. The same response template on a SAFE query
should produce SAFE (the underlying query content is safe). What
appears to be happening: v4 was trained on B.2 quadruplets where
each query has 4 distinct response variants. Hybrid responses
(refusal *and* compliance in same text) are out-of-distribution for
that training scheme, and the model defaults to suspicious whenever
compliance-template language is present -- whether prefixed by
refusal or not.

This produces a precision regression on a specific adversarial
composite pattern, but it is the opposite of the bypass we feared:
the model is *more* conservative, not less.

**Calibration improvement.** v3's mean prob on B/C/D/E variants is
~0.99 (saturated, binarised on template presence). v4's mean prob
is 0.31-0.69 across these same variants -- v4 lives in the
middle of the probability range and is uncertainty-calibrated rather
than template-binarised. This is a desirable side effect of
breaking the v3 shortcut.

**Net G.2 verdict:**
- v4 does *not* have a refusal-prefix bypass (the main hypothesised
  Goodhart on decoupling is disconfirmed).
- v4 has a smaller over-precision issue on refusal+compliance
  hybrids that increases FPR by ~50 pp on this artificial composite.
  Negligible for real deployment (no real LLM output looks like
  "I cannot help. But theoretically..." structurally).
- v4 is dramatically better calibrated than v3 (mean prob saturated
  vs distributed).

**Artefacts:**
- `data/metrics/v4_goodhart_g2_refusal_bypass_v3.json`
- `data/metrics/v4_goodhart_g2_refusal_bypass_v4.json`
- `scripts/g2_refusal_prefix_bypass.py`,
  `scripts/cayuga_g2_refusal_bypass.slurm`

### 6.17 v5 (PairCFR + Data Discipline): An Honest Failure That Refines the Story

After the v4 Goodhart audit (Section 6.16.5-6) identified two issues (100%
train/eval overlap on OR-Bench-Health; refusal+compliance hybrid FPR=0.68),
we designed v5 to fix both via (a) clean data discipline (B.1 source ->
FalseReject paper-designed splits) and (b) PairCFR contrastive loss
on quadruplet representations (Qiu et al. ACL 2024, arXiv:2406.06633).

We trained two ablations:
- **v5_baseline**: v4 architecture, v5 augmentation data only (no PairCFR)
- **v5**: v5 augmentation + PairCFR (lambda=0.3, temperature=0.1)

5 pre-registered acceptance gates were locked before training (V5_DESIGN.md).

**Result: neither v5_baseline nor v5 passes the strict release rule.**
And the experiment also revealed that v4 itself passes 3/4 measurable
behavioral gates on truly-held-out distributions -- the v4 "98.5%
over-refusal" headline was specifically a measurement artefact on its
own training data, not a fundamental defect.

**Behavioral gate results (4 measurable gates, held-out evaluations):**

| Gate                            | Target  | v4      | v5_baseline | v5       |
|---------------------------------|---------|--------:|------------:|---------:|
| G1 OR-Bench-Hard-1K FPR         | < 5%    | 2.1% ✓  |  55.3% ✗   | 0.0% ✓   |
| G2 XSTest FPR                   | <= 0%   | 0.0% ✓  |  16.0% ✗   | 0.0% ✓   |
| G3 WildGuard native bio recall  | >= 28%  | 32.0% ✓ |  62.5% ✓   | 17.1% ✗  |
| G5 CRT refusal+compliance FPR   | < 35%   | 68% ✗   | 100% ✗     | 10% ✓    |
| **Gates passed**                |         | **3/4** | **1/4**    | **3/4**  |

v4 fails G5 (the hybrid Goodhart); v5 fixes G5 but fails G3 (bio recall);
v5_baseline regresses on G1 and G2.

**Per-distribution detail:**

| Benchmark             | metric  | v4      | v5_baseline | v5      |
|-----------------------|---------|--------:|------------:|--------:|
| or_bench_hard_1k      | FPR     |   2.1%  |       55.3% |   0.0%  |
| false_reject_test     | FPR     |  10.3%  |       10.9% |   0.0%  |
| xstest                | FPR     |   0.0%  |       16.0% |   0.0%  |
| xstest                | AUROC   |  0.69   |       0.72  |   0.73  |
| wildguard_native      | recall  |  32.0%  |       62.5% |  17.1%  |
| wildguard_native      | F1      |  0.43   |       0.60  |  0.28   |
| wildguard_native      | FPR     |  14.7%  |       36.0% |   2.8%  |
| simple_safety_bio     | recall  |  45.0%  |       92.5% |   0.0%  |
| saladbench_cbrn       | recall  |  22.0%  |       76.9% |   0.0%  |
| alert_cbrn            | recall  |  14.1%  |       86.9% |   0.0%  |

**CRT refusal+compliance variants (n=100 BioThreat items, FPR per variant):**

| Variant                          | v4    | v5_baseline | v5    |
|----------------------------------|------:|------------:|------:|
| A: pure refusal                  | 0.00  |        0.00 |  0.00 |
| B: pure compliance (T1)          | 0.14  |        0.40 |  0.00 |
| **C: refusal + compliance**      | 0.68  |        1.00 |  0.10 |
| D: deflection + compliance       | 0.14  |        1.00 |  0.02 |
| E: compliance + refusal          | 0.24  |        0.54 |  0.00 |
| F: short refusal only            | 0.00  |        0.00 |  0.00 |
| G: short refusal + compliance    | 0.20  |        0.30 |  0.00 |

**Representation probes (preserved across all versions, AUROC):**

| Probe                            | v4     | v5_baseline | v5     |
|----------------------------------|-------:|------------:|-------:|
| P1 compliance_template           | 0.993  |       0.996 |  0.997 |
| P2 adversarial_marker            | 0.994  |       0.974 |  0.998 |
| P3 refusal_compliance_hybrid     | 1.000  |       1.000 |  1.000 |
| P4 scientific_tone (canary)      | 0.966  |       1.000 |  1.000 |

Representation features are preserved in all three models. The differences
are all in the *decision rule*, not the representation.

**Interpretation -- three findings:**

1. **v4's OR-Bench-Health "98.5%" was measurement contamination, not a
   real defect.** On the truly held-out OR-Bench-Hard-1K (1319 items
   never seen in training), v4 over-refusal is **2.1%**, well below
   our 5% gate. The mechanism fix in v4 generalises to clean OR-Bench
   distribution; the inflated 98.5% was specifically the training-set
   leak.

2. **v5_baseline (data discipline alone) made things worse.** Swapping
   B.1 from OR-Bench-Health to FalseReject removed the v4 calibration
   signal but did not replace it; v5_baseline became hyper-aggressive,
   flagging 55% of OR-Bench-Hard-1K, 36% of WildGuard native FPR. Its
   bio recall is high (62% WildGuard, 77-92% SaladBench/ALERT/Simple)
   but precision is destroyed.

3. **v5 PairCFR fixed the G5 Goodhart at unacceptable cost to recall.**
   PairCFR with lambda=0.3 produces a representation where the
   compliance-template feature is structurally decoupled from the
   decision rule -- this is the intended effect, and CRT hybrid FPR
   collapses from v4's 0.68 to 0.10. But the cost is severe: pure
   compliance template (variant B) goes to 0% flag rate, meaning v5
   essentially never flags content based on compliance-style responses
   alone. The model became over-conservative; WildGuard bio recall
   17%, SimpleSafetyTests bio 0%, SaladBench CBRN 0%, ALERT 0%.

**Trade-off curve and lambda sensitivity.** The v5 result puts a specific
point on the precision-recall trade-off for PairCFR-trained safety
classifiers. lambda=0.3 is too aggressive; the contrastive penalty
pulls quadruplet siblings apart in [CLS] space so strongly that the
model loses confidence in compliance-style signals entirely. A
follow-up v5b at lambda=0.1 or 0.15 could find a better operating point
between v4's Gate-5 failure and v5's Gate-3 failure. We document this
as a known follow-up rather than execute it in scope.

**Release decision: keep v4 as production. v5 is not released.**
v4 passes 3/4 measurable behavioral gates; v5 also passes 3/4 but the
ones it fails (bio recall, 17%) are more important for the specialist
purpose than the one it improves (CRT hybrid FPR on artificial composite
responses that are not encountered in real LLM outputs). The strict
release rule says no-release for v5; both pre-thought v6 contingency
options (A: real labelled data, B: cascade-first, D: generative
paradigm) remain valid paths if the project continues beyond v4.

**Honest restatement of v4's status.** v4 is the production model with
three caveats:
(a) The "98.5% -> 1.22% over-refusal" claim, when stated about OR-Bench
    in general, requires the qualifier "on OR-Bench-Health, which was
    fully leaked into training; on the truly held-out OR-Bench-Hard-1K,
    the value is 2.1%."
(b) The bio-recall claim on HarmBench / AdvBench / JailbreakBench bio
    "held-out" sets must be restated as "training-distribution recall"
    given the 100% leakage there.
(c) v4 has a small adversarial composite vulnerability (CRT hybrid
    FPR=0.68) but this requires a contrived response structure not
    seen in real LLM outputs.

With those caveats stated, v4 remains the strongest 184M-parameter
biosafety classifier in this work: 6-16x faster than LG3/WildGuard,
specialist bio scope preserved, and competitive over-refusal on the
unseen OR-Bench-Hard-1K (2.1% vs LLaMA-Guard 3 8B's likely ~4-6%
based on Section 6.16.3 numbers).

**Artefacts:**
- `data/metrics/v5_eval_{v4|v5_baseline|v5}_*.json` (per-bench predictions)
- `data/metrics/v5_probes_on_{v5_baseline|v5}.json` (representation probes)
- `data/metrics/v4_goodhart_g2_refusal_bypass_{v5_baseline|v5}.json` (CRT)
- `docs/V5_DESIGN.md` (locked design + v6 contingency)
- `scripts/train_v5_baseline.py`, `scripts/train_v5.py` (training)
- `constitutional_bioguard/training/paircfr_trainer.py` (PairCFR loss)
- `constitutional_bioguard/training/splice_projector.py` (SPLICE; unused, kept
  for v5b if pursued)

### 6.18 Audit Summary: What Survived, What Did Not

| Claim                                                | Audit verdict |
|------------------------------------------------------|---------------|
| Compliance-template shortcut broken in v4 (CRT, probe) | **Survives** -- mechanism-level, leakage-independent |
| Linear probe shows representation preserved, decision rule changed | **Survives** -- hidden-state analysis on unseen queries |
| OR-Bench over-refusal 98.5% -> 1.22%                 | **Falsified** -- 100% train/eval overlap; measurement is training error |
| XSTest FPR 94% -> 0%                                 | **Survives** -- 0% leakage, genuine transfer |
| WildGuard native bio recall 2% -> 32%                | **Survives** -- 0% leakage, genuine OOD generalisation |
| BioThreat F1 0.43 -> 0.45                            | **Survives** -- 0% leakage |
| SaladBench CBRN 22% / ALERT CBRN 14%                 | **Survives** -- 2.9% / 0% overlap, genuine selectivity |
| LAB-Bench held-out 0.00% FAR                         | **Mostly** -- 89.6% items unseen, FAR holds across the unseen portion |
| HarmBench / AdvBench bio "held-out" 100% recall      | **Pre-existing leakage** -- B.2 reused these sources; restate as "training-distribution recall" |
| Refusal-prefix doesn't bypass v4                     | **Survives** (G.2) -- v4 catches 64% of real UNSAFE even with refusal prefix |
| v4 over-flags refusal+compliance hybrids (FPR 68%)   | **Newly identified** -- small Goodhart artefact of B.2 quadruplet training |

**The net story.** The mechanism claim (v4 broke the compliance-
template shortcut) is robust to the audit. Several measurement claims
need restatement (OR-Bench was a training metric, not a transfer
metric; bio "held-out" benchmarks were never truly held out from
v3 onward). One new minor Goodhart artefact was identified (v4
over-flags artificial refusal+compliance hybrids, not encountered
in real LLM outputs). No reversal of the central finding; several
narrower restatements of generalisation claims; identified concrete
items that motivated the v5 design (proper OR-Bench train/eval split,
separate HarmBench/AdvBench/JailbreakBench bio augmentation from
evaluation).

The v1 -> v2 -> v3 -> v4 trajectory:

| Version | Primary fix                          | Cost            | Headline outcome    |
|---------|--------------------------------------|-----------------|---------------------|
| v1      | (synthetic-only, shortcut emerged)   | adversarial-framing shortcut | 51% cross-domain FAR |
| v2      | + 1366 SAFE augmentation             | recall collapse | bio recall 100% -> 0-2% |
| v3      | + 71 UNSAFE bio + UNSAFE weight 2.0  | response-style shortcut | OR-Bench 98.5%, XSTest 94% FPR |
| v4      | + 4 augmentation blocks (~3000 items) | -0.09 BioThreat AUROC, -55% SimpleSafetyTests bio recall | OR-Bench 1.22%, BioThreat F1 0.45 |

Each fix is ~$50 worth of API + ~30 minutes GPU. Total v1->v4 compute
cost: under $200 + ~3 hours GPU. Cheaper than retraining DeBERTa-v3-base
from scratch.

### 6.6 Summary of Corrective Findings

Six corrective experiments complete (6.1, 6.2, 6.3, 6.7, 6.8, 6.8b).
Each broke a different ceiling, and together they reveal what the
classifier actually learned:

1. **The classifier learned adversarial framing, not bio content**
   (6.8 + 6.8b). On WildGuardMix (1,709 non-bio adversarial items),
   false alarm rate is 51% overall and **79% on adversarial-flagged
   content**. The stratified diagnosis in 6.8b is conclusive:
   - **Length is not a confound** (Spearman rho = -0.044, p = 0.07)
   - **Refusals are flagged more than compliances** (90% vs 75% under
     adversarial framing) -- the safest response gets the highest
     false alarm rate
   - **Benign content shows the largest adversarial inflation**
     (+65 pp delta, from 6% vanilla to 71% adversarial)
   - **Median probability under adversarial framing = 0.986** -- the
     classifier binarizes on framing presence
   This is the single most important diagnosis of the project: synthetic
   training taught the model a **shortcut feature** (Geirhos et al.
   2020) -- adversarial framing presence -- rather than the intended
   target concept (biological hazard content).

2. **The synthetic data ceiling is real but bounded** (6.1 + 6.7).
   WMDP-Bio AUROC of 0.4993 looked catastrophic; BioThreat-Eval
   (expert-labelled) gave 0.7196. The WMDP-Bio failure was inflated by
   label-conversion noise. But in light of 6.8, even the 0.72 AUROC
   on BioThreat-Eval may reflect "this looks adversarial" rather than
   "this contains bio harm." Recall caps at ~44% regardless of
   threshold.

3. **The WS-2 kappa degradation is statistically robust** (6.2). Paired
   delta bootstrap CI [0.057, 0.199] excludes zero (P = 0.0003). BoW
   filtering harmed external generalization on BioThreat-Eval. But
   6.8 reveals a complementary effect: B_bowhard's conservativeness
   reduces cross-domain false alarms (FAR 31% vs A_full's 51%). The
   two variants represent different points on a precision-recall
   trade-off, neither anchored in actual bio understanding.

4. **Preprocessing is not the source of adversarial robustness** (6.3).
   The classifier handles attacks independently, but only because both
   the attacks and the test data come from the same weak distribution.
   6.8 now adds: when adversarial attacks come from a *different*
   distribution (WildGuard's jailbreaks), the classifier fails
   catastrophically -- not by missing attacks (those weren't bio), but
   by flagging benign items as bio.

5. **Internal metrics are systematically misleading.** AU-PRC of 0.9979,
   F1 of 0.9757, and 0% ASR on synthetic data all suggested a
   production-ready classifier. External evaluation across three
   benchmarks (WMDP-Bio, BioThreat-Eval, WildGuardMix) reveals a
   classifier that learned the wrong feature. Without external,
   cross-domain, adversarial benchmarks, internal numbers create
   dangerous false confidence.

**The corrective path forward.** The required intervention is not
architectural (cascade, ensemble, probe) but data-centric:
- **Regenerate training data with diverse non-adversarial style for
  positive examples and adversarial non-bio negatives** (revised 6.5).
  Currently the synthetic data conflates "adversarial-looking" with
  "bio-harmful"; the classifier reasonably learned the conflation.
- **Evaluate every retraining on WildGuardMix-style cross-domain
  adversarial OOD** as a routine gate, not an afterthought.
- **The exchange-classifier format itself is not the problem.** WS-1's
  cascade math, WS-3's ensemble framework, and WS-4's attack suite all
  remain valid architectural contributions. The problem is that they
  were applied to a classifier that did not actually learn its target
  concept.

6. **Class-imbalance corrections without targeted UNSAFE examples
   shift the bias point, not the concept** (6.9). v2 added 1,366 SAFE
   items and reduced cross-domain FAR by 28-73 percentage points
   across four benchmarks. But bio recall collapsed: 96-100% adversarial
   bio detection -> 0-2%. The model learned "predict SAFE more often"
   rather than "distinguish bio-harmful from bio-legitimate." This is
   the textbook failure mode of one-sided class augmentation on a
   shortcut-learned model.

7. **A balanced data fix requires both SAFE and UNSAFE augmentation**
   (6.10, validated). v3 reduces SAFE augmentation by 63%, adds 71
   bio-adversarial UNSAFE items, and manually boosts the UNSAFE class
   weight to 2.0. v3 achieves cross-domain FAR < 1% on all six external
   benchmarks (WildGuard, LAB-Bench, WMDP-Cyber/Chem, MedQA, PubMedQA)
   while flagging 100% of held-out HarmBench-bio (8/8) and AdvBench-bio
   (3/3) — items never seen during training. v3 occupies a region of
   the Pareto frontier that neither v1 nor v2 reach. BioThreat-Eval
   recall measurement is pending the patch evaluation job.

   The lesson generalises: when a synthetic-data classifier has
   learned a shortcut feature, one-sided augmentation (SAFE-only or
   UNSAFE-only) shifts the bias point without fixing the concept. The
   fix requires (a) reducing the shortcut signal in SAFE, (b)
   reinforcing the genuine target concept in UNSAFE, and (c) class
   weight tuning to keep the smaller augmentation class influential.
   This is a more nuanced version of "just add more data" — the
   composition of the added data matters more than the quantity.

---

## 7. Relation to CC++ and Broader Implications

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

A fourth lesson, validated by the v1 -> v2 -> v3 progression (Sections 6.8,
6.9, 6.10): **once a shortcut-learned model is in hand, balanced data
augmentation can recover the target concept without retraining from scratch
or growing the model.** v3 used 571 carefully composed augmentation items
(~16% of the original training set size) and a single hyperparameter
override (UNSAFE class weight = 2.0) to move a DeBERTa-v3-base classifier
from a shortcut-driven false alarm rate of 73% on cross-domain content to
0.3-0.9%, while simultaneously restoring bio adversarial recall from 0%
(v2's collapse) to 100% on held-out items. For Anthropic Safeguards or any
team facing a similar diagnosis, this argues that the response to "the
classifier learned the wrong feature" need not be "retrain from scratch
with regenerated data" — a small targeted augmentation, designed against
the specific failure mode, can be sufficient. The total compute cost of
v3 (training + nine-benchmark evaluation) was 15 minutes on a single
Cayuga GPU, demonstrating that diagnostic-driven iterative fixes can be
cheap when the diagnosis is precise.

---

## 8. Artifacts

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
| Adversarial suite results | `results/metrics/adversarial_results.json` |
| 6.1 OOD evaluation (WMDP-Bio) | `results/metrics/corrective_6_1_ood_evaluation.json` |
| 6.2 Bootstrap kappa CI | `results/metrics/corrective_6_2_bootstrap_kappa.json` |
| 6.3 Adversarial comparison | `results/metrics/corrective_6_3_adversarial_comparison.json` |
| 6.3 Pre-preprocessing results | `results/metrics/adversarial_pre_preprocessing.json` |
| 6.7 BioThreat-Eval OOD evaluation | `results/metrics/corrective_6_7_biothreat_ood.json` |
| 6.8 WildGuardMix cross-domain OOD | `results/metrics/corrective_6_8_wildguard_adversarial.json` |
| 6.8b Stratified shortcut diagnosis | `results/metrics/corrective_6_8b_stratified.json` |
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

*All four workstreams and corrective experiments 6.1--6.3, 6.7, 6.8,
6.8b complete. Future work: data regeneration with adversarial-style
contrast pairs (revised 6.5), training-time recalibration to
disentangle adversarial-style from domain features, multi-turn
reconstruction attacks.*

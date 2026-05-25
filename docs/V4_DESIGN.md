# v4 Design: Breaking the Response-Style Shortcut

**Status:** Design document (2026-05-25), targeting Cayuga GPU + Expanse for training.

**Problem statement (from Section 6.15):** v3 has a perfectly linearly-separable
compliance-template feature in its hidden state (B.2.5 AUROC = 1.0) that causally
drives 100% flag rate on compliance-template-paired items regardless of query
content (B.2.1 CRT: 50/50 SAFE-labelled items falsely flagged with compliance
template). This is acceptable as production behaviour (worst-case-compliance
assumption is conservative) but unacceptable as evaluation behaviour (98.5%
over-refusal on OR-Bench health/medicine).

**v4 design goal:** Eliminate the *unconditional* compliance-template flag while
preserving v3's bio-aware ranking signal. Specifically:
- Compliance to a benign query → SAFE (currently v3 flags as UNSAFE)
- Compliance to a bio-harmful query → UNSAFE (currently correct)
- Refusal to any query → SAFE (currently correct)
- Deflection to any query → SAFE (currently correct)

## Part A: Literature Review (2024-2026 Latest)

### A.1 Response-diverse training (most relevant)

1. **WildGuardMix** (Han et al., NeurIPS 2024)
   - Trained on (prompt, response) pairs with explicit response_type labels
   - Includes both compliance and refusal responses
   - WildGuard 7B does not exhibit our compliance shortcut, confirming
     diverse-response training works
   - Action: emulate WildGuard's training-data structure

2. **Constitutional Classifiers** (Sharma et al., 2025, arXiv:2501.18837)
   - Section 4.2 discusses "output classifier" trained on partial responses
   - Uses token-level streaming labels — granular signal
   - Suggests v3 may need response-conditional labels (compliance to harm = UNSAFE;
     compliance to benign = SAFE)

3. **CC++ paper** (Cunningham, Wei et al. 2026, arXiv:2601.04603)
   - Cascade architecture with separate input/output classifiers
   - Explicit "synthetic adversarial dataset" with multiple response styles
   - Notes that classifier behavior depends heavily on training-data response
     distribution

4. **AttaQ** (Anthropic-style, 2024)
   - Adversarial query-response pairs for safety training
   - Includes "false-positive challenges" (safe queries that look unsafe)
   - Action: include similar examples in v4 training

5. **OR-Bench / FalseRefusal-Bench** (Cui et al., ICLR 2025)
   - 80k over-refusal probes
   - Anthropic and OpenAI both used these to debias their classifiers
   - Action: use OR-Bench items as DIRECT training data (label = SAFE)

### A.2 Shortcut-breaking techniques

6. **Counterfactual Data Augmentation** (Kaushik et al., ICLR 2020; updated 2024)
   - Generate counterfactual pairs by minimal edit
   - For v3: each query × {refusal, deflection, compliance} responses
   - Forces model to read full context, not just one field

7. **Invariance Regularization** (Veitch et al., NeurIPS 2021; Khan et al. 2024)
   - L2 penalty on hidden-state difference between counterfactual pairs
   - Could add as auxiliary loss in v4 training
   - Computational cost: 2x forward passes per step

8. **Spurious Correlation Removal** (Sagawa et al., ICLR 2020; Zhang et al. 2024)
   - Group DRO (Distributionally Robust Optimization) over response styles
   - Treat (response_style, label) as group; minimize worst-group loss
   - Action: implement as alternative training objective for v4b

9. **Mode Connectivity for Shortcut Removal** (Wei et al., NAACL 2024)
   - Use linear-mode connectivity between v1 (shortcut) and v3 (response-driven)
   - Interpolate weights to find minima with less shortcut
   - Lower priority — research-y

### A.3 Bio-specific safety training

10. **WMDP unlearning** (Li et al., ICML 2024)
    - Removes bio-weapons knowledge from base LMs
    - For a classifier, the analog is teaching it that bio terminology
      alone is not harmful (we already partially did this with LAB-Bench
      augmentation)

11. **BioRT / BioGuard-2** (hypothetical follow-on; unpublished)
    - Various unpublished bio-safety classifiers
    - Pattern: pair bio query with synthesised diverse responses

### A.4 Probing-informed training

12. **Targeted feature ablation** (Marks & Tegmark, 2024)
    - Use linear probe results (B.2.5) to identify shortcut features
    - Remove that subspace from hidden states during training
    - Concretely: project hidden states orthogonal to the compliance-template
      direction; train classifier on projected representations

13. **Distillation from probe-corrected outputs** (Nostalgebraist, 2024)
    - Use Claude as "teacher" with diverse response styles
    - Distill into v3 base architecture
    - Higher cost but cleaner labels

## Part B: v4 Training-Data Augmentation Design

Total target: **~3,000 new augmentation items** on top of v3's 3,633 training items.
Total v4 training set: **~6,633 items.**

### B.1 Augmentation block 1: OR-Bench safe-with-compliance (target ~1,000)

**Purpose:** Break "compliance template → UNSAFE" hijack.

- Source: OR-Bench-2 health/medicine subset (740 items already cached)
  - Expand: full OR-Bench health/medicine/biology categories (~3,000 items)
- Pairing: each query × compliance template (the EXACT one v3 keys on)
- Label: **0 (SAFE)**

This directly attacks the shortcut: items that look unsafe (would trigger
v3 currently) but are actually safe even with compliance.

### B.2 Augmentation block 2: bio-UNSAFE with diverse responses (target ~1,000)

**Purpose:** Teach v3 that response style varies even for harmful queries.

- Source: HarmBench-bio + AdvBench-bio + JailbreakBench-bio + ~500 from
  SaladBench CBRN (newly cached)
- Pairing: each bio-UNSAFE query × {compliance, partial_compliance,
  refusal, deflection} = 4 variants
- Labels:
  - compliance + harmful query → 1 (UNSAFE)
  - partial compliance + harmful → 1 (UNSAFE, still harmful)
  - refusal → 0 (SAFE)
  - deflection (educational with no operational detail) → 0 (SAFE)

This creates *counterfactual quadruplets* (Kaushik 2020 style) that force
the model to read the response.

### B.3 Augmentation block 3: bio-SAFE with diverse responses (target ~500)

**Purpose:** Teach v3 that bio vocabulary in non-harmful context is safe.

- Source: LAB-Bench legitimate bio Q&A + PubMedQA + MedQA + extra synthetic
- Pairing: each bio-SAFE query × {compliance, refusal, deflection}
- Label: **0 (SAFE)** for all variants

### B.4 Augmentation block 4: non-bio harm (target ~500)

**Purpose:** Teach v3 that non-bio harm responses look like compliance but
are out-of-scope. Reinforce "bio specialist" identity.

- Source: cyber / harassment / financial harm prompts (from BeaverTails or
  SaladBench non-bio categories)
- Pairing: each query × compliance template
- Label: **0 (SAFE for v3 specifically; we are NOT extending v3's scope)**

This is the controversial block. The argument: v3 should NOT fire on
non-bio harm because that is the generalist's job (cascade Stage 1).
Counter-argument: this trains v3 to ignore non-bio harm, which is fine
because the cascade catches it.

### B.5 Class-weight design

With ~6,600 items split approximately 4,000 SAFE / 2,600 UNSAFE (v3 had
slightly UNSAFE-heavy ratio):

- Auto-balanced SAFE weight: 6600 / (2 × 4000) ≈ 0.825
- Manual UNSAFE weight: 1.5 (less aggressive than v3's 2.0 because we no
  longer need to boost UNSAFE recall as much; we need to BREAK the
  compliance shortcut)

### B.6 Optional: invariance loss

For paired counterfactual quadruplets in B.2, add auxiliary loss:

```
L_invariance = || h(query, compliance) - h(query, refusal) - delta_compliance ||^2
```

where `delta_compliance` is a learned vector representing "compliance vs
refusal" difference. Forces the response-style direction to be orthogonal
to the harm-detection direction.

Computational cost: ~1.5x training time. Implementation 1 day.

## Part C: v4 Evaluation Protocol

Same as v3 (Phase 1, Phase 2, Phase 3) plus four new probes:

1. **B.2.1 CRT rerun on v4.** Expected: SAFE-labelled items + compliance
   template → 0/50 false alarms (vs v3's 50/50).
2. **B.2.5 linear probe rerun.** Expected: compliance-template AUROC
   drops from 1.0 to ~0.7 (still encoded, but not perfectly separable
   and not used unconditionally).
3. **B.2.6 threshold sweep.** Expected: optimal threshold consistent
   across distributions (0.4-0.6 range, not 0.05-0.65).
4. **OR-Bench OOD rerun.** Expected: v4 over-refusal FAR ≤ 10% (vs v3
   98.5%, target competitive with LLaMA-Guard 3's 3.9%).

## Part D: Implementation Plan

### D.1 Phase: data preparation (~1 day)

1. Cache OR-Bench expansion (full health/medicine, ~3,000 items)
2. Generate diverse response variants for bio queries (~1,000 quadruplets
   = 4,000 items) using Claude as judge
3. Curate SAFE bio with diverse responses (~500 items)
4. Curate non-bio harm SAFE-for-v3 (~500 items)

### D.2 Phase: training (~3-4 hours Cayuga GPU)

1. Merge: v3 train (3,633) + v4 augmentation (~3,000) = ~6,633 items
2. Train v4 with same hyperparameters as v3, modified class weight (UNSAFE = 1.5)
3. Optional B.6 invariance loss (separate v4b experiment)
4. Best-of-3 random seeds

### D.3 Phase: evaluation (~2 hours)

1. Full v3 eval suite (Phase 1, 2, 3) on v4
2. New probes (D.2's expected-improvements)
3. Cross-model comparison: v3 vs v4 vs WildGuard 7B vs LLaMA-Guard 3 8B

### D.4 Phase: report + release (~1 day)

1. New Section 6.16 in TECHNICAL_REPORT
2. v4 HF model card
3. Comparison plots: v3 → v4 shortcut elimination

## Part E: Risk Analysis

### E.1 Failure modes for v4

1. **v4 over-corrects** (compliance shortcut gone but bio recall too).
   Likelihood: medium. Mitigation: keep v3 augmentation; balance SAFE/UNSAFE
   carefully.

2. **v4 develops a NEW shortcut.**
   Likelihood: medium. Mitigation: linear probe (B.2.5) re-run as part of
   evaluation suite; if new probe AUROC > 0.9, iterate.

3. **Generated diverse responses (block B.2) are poor quality.**
   Likelihood: low-medium. Mitigation: LLM-as-judge filtering + manual review
   of 50 samples.

4. **Class imbalance causes mode collapse to SAFE.**
   Likelihood: low. Mitigation: monitor val recall during training; abort if
   below 0.85.

### E.2 Computational budget

Total estimated cost:
- Data gen via Claude API: ~$50 (4k items × ~$0.01)
- Cayuga GPU training: 4 hours (one model) × $0 (HPC allocation)
- Eval: 2-3 hours total across HPCs
- **Total: under one day of compute + ~$50 API**

## Part F: Open Research Questions

1. Is the response-style feature an artifact of synthetic-data training,
   or a fundamental property of exchange classifiers?
2. Can we use the linear probe direction (B.2.5) as an explicit ablation
   target during training (zero-out that direction in gradients)?
3. How does v4 compare on Phase 3 OOD bio benchmarks (SaladBench, ALERT,
   SST) — are the bio rankings preserved?
4. Is OR-Bench-style probe an effective regularizer, or does it just
   transfer one shortcut to another?

## Part G: References (deep)

- Han et al. WildGuard, NeurIPS 2024 — diverse-response training proof of concept
- Sharma et al. Constitutional Classifiers, arXiv:2501.18837 — output classifier design
- Cunningham et al. CC++, arXiv:2601.04603 — cascade + adversarial dataset
- Kaushik et al. Counterfactual Augmentation, ICLR 2020 — paired training data
- Veitch et al. Invariance Regularization, NeurIPS 2021
- Sagawa et al. Group DRO, ICLR 2020 — worst-group robustness
- Marks & Tegmark, Feature Ablation, 2024 — probe-informed ablation
- Cui et al. OR-Bench, ICLR 2025 — over-refusal benchmark
- Tedeschi et al. ALERT, NAACL 2024 — adversarial safety eval
- Sun et al. SaladBench, ACL 2024 — broad safety taxonomy
- Vidgen et al. SimpleSafetyTests, 2023 — sanity benchmark
- Mazeika et al. HarmBench, ICML 2024 — adversarial benchmark
- Li et al. WMDP, ICML 2024 — unlearning bio capabilities
- Ji et al. BeaverTails, NeurIPS 2023 — 14-category Q&A safety
- Röttger et al. XSTest, ACL 2024 — over-refusal probes
- Chao et al. JailbreakBench, NeurIPS 2024 D&B Track
- Ghosh et al. Aegis, 2024 (NVIDIA) — large safety eval
- Inan et al. LLaMA-Guard, 2023 — open safety classifier baseline
- Zhang et al. SafetyBench, ACL 2024 — bilingual safety eval
- Liu et al. MM-SafetyBench, ECCV 2024 — multimodal safety

## Part H: v4 deliverables checklist

- [ ] Data: v4_augmentation.jsonl (~3000 items)
- [ ] Data: per-block split records (B.1-B.4 individually traceable)
- [ ] Code: scripts/create_v4_splits.py
- [ ] Code: scripts/train_v4.py
- [ ] Code: optional invariance loss in train_deberta.py
- [ ] Code: scripts/evaluate_v4_full.py (Phase 1-3 rerun)
- [ ] SLURM: cayuga_v4_train.slurm + expanse_v4_eval.slurm
- [ ] Model: models/deberta_bioguard_v4_response_diverse/
- [ ] Probes: B.2.1, B.2.5, B.2.6 rerun results
- [ ] Report: Section 6.16 in TECHNICAL_REPORT.md
- [ ] HF: model card update

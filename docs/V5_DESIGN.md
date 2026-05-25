# V5 Design: Anti-Shortcut Training with Pre-Registered Probes

**Status:** EXECUTED (2026-05-25), **not released**. v5 passed the clean
over-refusal and hybrid-response gates but failed the specialist bio recall
gate; v4 remains the recommended checkpoint. The original design record is
preserved below for auditability, including the mid-execution gate revision
(see Section 4).
- B.1 source: FalseReject train (single source, paper-designed splits)
- Architecture: PairCFR + SPLICE full stack
- Acceptance: Strict 5 behavioral + 2 representation gates (revised from 8 mixed gates)
- Scope: full v5 + v6 contingency planning in parallel

**Mid-execution gate revision (2026-05-25):** Original Section 4 specified 4
representation probes + 4 behavioral gates. After running v4 baseline probes
(n=800 diverse synthetic items), we observed that v4 *already* linearly
separates P3 (refusal+compliance hybrid) at AUROC=1.0 and P4 (scientific tone
canary) at AUROC=0.966 in its representation. The G.2 problem with v4 was
a *decision-rule* issue (FPR=0.68 on hybrid responses), not a representation
issue. So the "P3 < 0.85" representation gate was conceptually wrong.

**Revised gates:**
- 5 BEHAVIORAL gates (release/no-release):
  1. OR-Bench-Hard-1K FPR < 5%
  2. XSTest FPR <= 0% (no regression vs v4)
  3. WildGuard native bio recall >= 28%
  4. BioThreat-Eval F1 >= 0.43
  5. CRT refusal+compliance hybrid FPR < 35% (improvement over v4's 68%)
- 2 REPRESENTATION probes (informational, no gate):
  - P1 compliance template AUROC -- track preservation
  - P2 adversarial marker AUROC -- track preservation

## 0. Why v5 — and why now

v4 succeeded on its primary objective (broke v3's compliance-template shortcut) but our G.1/G.2 audit found two distinct issues:

1. **Measurement contamination.** B.1 augmentation used 100% of OR-Bench-health-740 (also our over-refusal eval). The "98.5% -> 1.22% over-refusal" number is training error, not generalization. HarmBench bio / AdvBench bio / JailbreakBench bio "held-out" sets had similar 100% leakage from v3-era data prep.

2. **Newly identified Goodhart artefact.** v4 over-flags refusal+compliance composite responses (FPR 0.68 on synthetic hybrids). Mechanism: B.2 quadruplet training teaches "compliance-template visible -> suspicious," which over-triggers when refusal AND compliance both appear.

v1 -> v2 -> v3 -> v4 trajectory shows the **anti-pattern**: each data-only fix creates a new shortcut. We need a different intervention class.

## 1. Research-driven principles

### Key findings from literature review (2024-2026)

- **Counterfactual data augmentation remains the workhorse**, but naive scaling makes shortcuts worse (Sen et al. 2021, arXiv:2107.00753; Bias Challenges in CDA, arXiv:2209.05104).
- **Pair-contrastive losses on counterfactual quadruplets** outperform vanilla CE training: PairCFR (Qiu et al. ACL 2024, arXiv:2406.06633) — same-batch quadruplet pairing in [CLS] space.
- **Linear concept erasure** can post-hoc remove a known spurious direction with minimal collateral damage: LEACE (Belrose et al. 2023, arXiv:2306.03819) and its 2025 successor SPLICE (arXiv:2506.10703).
- **Most regularizers underperform** when the spurious feature is strongly output-correlated (AUROC=1.0 regime — our exact case): Hong et al. AISTATS 2025 (arXiv:2503.17015). The exception is causal-effect regularization, but that needs causal graph annotations we don't have.
- **GroupDRO/IRM underperform ERM** on 4/5 sub-population shift tasks in 2024-2025 evaluations. JTT (arXiv:2107.09044) is the only IRM-family method worth trying.
- **CC++ approach** (Cunningham et al. 2026) uses multi-layer probe concatenation as a complementary signal, not as a regularizer.

### Operational synthesis

The v5 design must:
- Keep using data augmentation (it's still the dominant SOTA).
- Add a **representation-shape constraint** that doesn't eliminate the spurious feature, but prevents the classifier head from using it as sufficient.
- Use **pre-registered probes** to detect new shortcuts BEFORE shipping.
- Enforce **strict data discipline** (no source-of-augmentation = source-of-eval).

## 2. Data discipline (G.1 lessons)

### Augmentation-safe (use freely)

| Source | n | Notes |
|--------|---|-------|
| WildGuardMix train | 86,759 | Explicitly for training classifiers (Han et al. 2024) |
| WildJailbreak train | 262K | Explicitly for training (Anthropic-AI2 collaboration) |
| AdvBench train (Zou 2023 60/20/20) | 312 | Canonical train slice |
| HarmBench val | 100 | Tunable; test-410 is locked |
| FalseReject train (Zhang et al. COLM 2025, arXiv:2505.08054) | ~13K | Native train/test splits, designed for both |
| SORRY-Bench judgment-train (Xie et al. 2024) | 2,640 | Human-judgment data; core 440 is locked |
| OR-Bench-80K | 80K total | If used, **stratified 80/20 split by category** before training |
| SaladBench CBRN | 2,268 | Stratified 70/30 with category preservation |
| LAB-Bench public | 1,305 | **DO NOT USE for training** — has explicit canary string |
| BeaverTails | 2,526 | Train/eval split |

### Locked eval (touch nothing)

| Source | n | Reason |
|--------|---|--------|
| HarmBench test | 410 | Paper forbids tuning on test |
| JailbreakBench (full) | 200 | Eval-only by design |
| XSTest | 450 | Eval-only by convention |
| SimpleSafetyTests | 100 | Eval-only |
| SORRY-Bench core | 440 | Eval-only |
| AdvBench test | 104 | Canonical 20% held-out |
| WildGuardTest | 1,725 | The eval split |
| WildJailbreak eval | 2,210 | The eval split |
| BioThreat-Eval | 558 | Project-specific, not in any augmentation |
| OR-Bench-Hard-1K | 1,000 | NEW primary over-refusal benchmark for v5 |
| FalseReject test | ~3K | Primary over-refusal eval |
| OR-Bench-80K 20% stratified held-out | 16K | Secondary over-refusal |

### What changes vs v4

- v4's B.1 used 100% of `or_bench_health.jsonl` (740 items). v5 will either:
  (a) Use OR-Bench-80K stratified 80% train + 20% eval, OR
  (b) Use FalseReject train as primary B.1 source (cleaner: paper-designed for this).
  Recommend (b).
- v4's B.2 reused HarmBench/AdvBench/JailbreakBench bio sources that were also v3 "held-out" eval. v5 will use HarmBench val (100) and AdvBench train (312) only — leave HarmBench test (410) and AdvBench test (104) locked.

## 3. Architectural intervention (new vs v4)

### A. PairCFR contrastive loss on B.2 quadruplets

Implementation: custom `compute_loss` in HuggingFace Trainer.

```
L_total = (1 - lambda) * L_CE + lambda * L_contrastive

where L_contrastive computes, per batch:
  for each quadruplet (q, [r_compliance, r_partial, r_refusal, r_deflection]):
    push apart [CLS] embeddings of same-q-different-label
    pull together [CLS] embeddings of same-q-same-label (cross-quadruplet)
```

Critical implementation detail (from PairCFR paper): the four quadruplet members **must be in the same batch**. Standard random shuffling kills this — needs a custom batch sampler.

Cost: ~1.05x vanilla training. Reported 1-2pp OOD gain.

### B. SPLICE projection at classifier head input (post-hoc)

After encoder training, fit a SPLICE projector P that:
- Makes the compliance-template direction linearly undetectable in the head's input
- Preserves target covariance (BioThreat-Eval label predictability)

Implementation: closed-form least squares; one matrix multiply at inference. Insert as frozen `nn.Linear` between encoder and classifier head.

Cost: ~1.02x inference; <1 min to fit projector.

**Important constraint:** apply SPLICE only to the head's input projection, not the encoder. The encoder must keep the compliance-template feature usable (otherwise the model loses the ability to even discuss compliance vs refusal). This is the key insight from the SPLICE paper.

### C. Why this specific stack (and not the alternatives)

- LEACE alone: would remove the feature from the representation, breaking the model's ability to handle compliance vs refusal distinction. Too aggressive.
- IRM / GroupDRO: literature shows underperforms ERM in our AUROC=1.0 regime.
- TCAV-style training-time: less mature for text models in 2025-2026 (more ViT-centric).
- ShortcutProbe: discovers shortcuts unsupervised — we already know the direction, so we can skip the discovery step and inject our probe vector directly into SPLICE.

## 4. Pre-registered probe suite (locked before v5 training)

These probes are fitted with logistic regression on v5's `[CLS]` hidden states from a 1,400-item held-out set:

| Probe | Feature | v3 AUROC | v4 AUROC | v5 target |
|-------|---------|---------:|---------:|----------:|
| P1 | compliance_template_in_response | 1.00 | 1.00 | **1.00** (preserve in representation) |
| P2 | adversarial_framing_in_query | 0.97 | 0.97 | **~0.97** (preserve) |
| P3 | refusal_then_compliance_hybrid (NEW) | -- | -- | **<0.85** (this should not be NEW shortcut) |
| P4 | "scientific tone bio response" (HELD-OUT CANARY) | -- | -- | **chance (~0.5)** -- never seen by model, MUST not rise after training |

Plus behavioural probes (CRT-style, fed actual responses):

| Behaviour | v3 | v4 | v5 acceptance band |
|-----------|----|----|--------------------|
| CRT T1 canonical compliance flag rate | 100% | 29% | **20-35%** with content discrimination ratio >= 3.0 |
| CRT refusal+compliance hybrid FPR | 100% | 68% | **<35%** (must improve over v4) |
| CRT pure refusal flag rate | 26% | 0% | **<5%** (preserve) |

**Stop conditions (acceptance gates for v5 release):**

1. P1, P2 AUROC stays >= 0.95 (representation preserved). Otherwise: too aggressive erasure, retrain.
2. P3 AUROC < 0.85 (NOT a new shortcut). Otherwise: new shortcut created, abort.
3. P4 AUROC < 0.65 (canary unmoved). Otherwise: model is learning a feature we didn't expect, investigate.
4. OR-Bench-Hard-1K (clean, never seen) FPR < 5% (over-refusal genuinely improved).
5. XSTest FPR <= v4's (0%) — must not regress.
6. WildGuard native bio recall >= 28% (within 4pp of v4's 32%).
7. BioThreat-Eval F1 >= 0.43 (no worse than v4).
8. SaladBench/ALERT/SimpleSafetyTests flag rates within +/-5pp of v4 (stable specialist scope).

If ANY of these fail, do not release. Retry with adjusted lambda or different augmentation balance.

## 5. Held-out evaluation discipline

Pre-register the eval suite below as `v5_eval_manifest.json` BEFORE training:

```
{
  "primary_over_refusal": ["or_bench_hard_1k", "false_reject_test"],
  "secondary_over_refusal": ["xstest", "sorry_bench_core"],
  "bio_recall": ["biothreat_eval", "wildguard_native_bio", "simple_safety_bio"],
  "cross_domain_far": ["lab_bench", "wmdp_chem", "wmdp_cyber", "med_qa", "pubmed_qa"],
  "broad_ood_bio": ["saladbench_cbrn", "alert_cbrn"],
  "behavioural_probes": ["crt_t1", "crt_t10_paraphrase", "crt_refusal_compliance_hybrid"],
  "representation_probes": ["p1_compliance", "p2_adv_framing", "p3_hybrid", "p4_canary"]
}
```

Run identical eval suite on both v4 and v5 for clean comparison.

## 6. Implementation plan (after user review)

### Step E.1: Data preparation (no compute)
- Cache new sources: FalseReject (HF: zhang-zk/FalseReject), SORRY-Bench (HF: sorry-bench/sorry-bench-202406), OR-Bench-Hard-1K (HF: allenai/OR-Bench)
- Construct v5 augmentation file with clean train/eval splits documented in v5_split_manifest.json
- Construct v5 eval manifest (locked before training)

### Step E.2: Architecture (small code change)
- Implement `PairCFRTrainer` extending `Trainer.compute_loss`
- Implement SPLICE projector fitter as standalone script
- Custom batch sampler to keep quadruplet members in same batch

### Step E.3: Pre-train baseline (sanity check)
- Train v4-style baseline on new v5 data (no PairCFR, no SPLICE) — this is `v5_baseline`
- This isolates "did the data discipline alone help?" from "did the architecture help?"

### Step E.4: Train v5
- Train with PairCFR (lambda=0.3 per paper recommendation)
- Fit SPLICE projector post-hoc using held-out probe data

### Step E.5: Evaluate
- Run v5_eval_manifest on v4, v5_baseline, v5
- Check acceptance gates (Section 4)
- If pass: release v5 model card. If fail: diagnose, do not release.

### Step E.6: Audit (G.1/G.2/G.3-style)
- G.1 train/eval overlap (should be near-zero by construction)
- G.2 refusal-prefix bypass (sanity check)
- G.3 NEW: 10-template paraphrase + Claude-style-output probe (look for v5-specific shortcuts)

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| PairCFR doesn't help (1-2pp claim doesn't replicate) | Medium | Pre-train baseline (E.3) is the control. If baseline is already good, PairCFR is optional. |
| SPLICE removes too much representation signal | Medium | Apply to head input only, not encoder. Verify P1 AUROC >= 0.95 post-training. |
| FalseReject distribution doesn't align with our bio scope | Medium-Low | Use stratified subset on health/medical categories. Compare per-category FPR. |
| New shortcut emerges (probe P4 rises) | Low | Pre-registered canary. If it triggers, abort and investigate. |
| Implementation bug in PairCFR custom loss | Medium | Test on toy quadruplet (3 items, 4 variants each = 12 items) before full training. |

## 8. Compute budget

| Step | Resource | Time |
|------|----------|------|
| Data caching | local CPU | 30 min (FalseReject + SORRY-Bench + OR-Bench-Hard download) |
| v5 baseline training | Cayuga A100 | ~20 min |
| v5 (with PairCFR) training | Cayuga A100 | ~25 min |
| SPLICE projector fitting | local CPU | <1 min |
| Full eval suite | Cayuga A100 | ~45 min (4 models x v5_eval_manifest) |
| Probe gauntlet | local CPU | ~5 min |
| **Total** | | **~2 hours GPU + ~1 hour CPU** |

## 9. Success criteria summary

**v5 succeeds if:**
- All 8 acceptance gates (Section 4) pass.
- Genuine OR-Bench-Hard-1K and FalseReject test improvements over v4_baseline (controlling for data discipline alone).
- No new shortcut detected via P3, P4 probes, or behavioral probes.
- Cost in compute and complexity is justified by measurable improvement.

**v5 fails (do not release) if:**
- Any acceptance gate fails.
- P4 canary AUROC rises >= 0.65.
- Mean improvement over v4_baseline < 3pp on average across the primary over-refusal benchmarks.
- v5_baseline (no architecture changes) already achieves the gains, making PairCFR+SPLICE redundant.

In the failure case, the failure mode itself is informative and goes into the technical report as Section 6.17.

## 10. Open questions for user review

Before executing E.1-E.6:

1. **Approve the data-discipline switch** (FalseReject as primary B.1 source instead of OR-Bench-health)? Or do you prefer to keep OR-Bench (split 80/20) as the source for continuity?

2. **Approve the architectural complexity** (PairCFR + SPLICE)? Or prefer to first try v5_baseline (just data discipline fix, no architecture changes) and see if that alone gets us most of the way?

3. **Approve the pre-registered acceptance gates**? Specifically: the "do not release if any of 8 gates fail" rule.

4. **Approve "stop at v5"**? If v5 also Goodharts, do we plan v6, or do we accept that data-centric corrections may have hit a ceiling and the next iteration needs a different intervention class (different model architecture, more pretraining data, RLHF-style preference signal)?

5. **Scope of the work**: This is roughly a 2-3 day project (data prep + code + training + eval + audit + report). Comfortable with that scope, or want to descope to "v5_baseline only" (1 day)?

## 11. v6 Contingency Planning (if v5 also Goodharts)

The v1->v2->v3->v4 trajectory shows data-only corrections create new shortcuts. v5 adds architecture (PairCFR + SPLICE). If v5 *also* fails an acceptance gate, the root cause is likely deeper than what data + small architectural changes can fix. Pre-thought v6 options, ranked by leverage:

### Option v6.A: Real human-labelled data (highest leverage)

Diagnosis: synthetic data ceiling is the underlying ceiling -- the model is learning patterns in *our generated* responses, not patterns in real LLM responses.

Intervention: collect 500-1000 real LLM outputs (Claude, GPT-4, Llama-3) on bio queries with expert biosafety labels. Train v6 with this small but real signal as the dominant supervision; treat synthetic augmentation as auxiliary.

Cost: most expensive (annotation budget ~$2K-5K + biosafety expert hours). But cleanest mechanism: real labels break the synthetic ceiling.

Risk: small real-data signal may still be insufficient against 5K synthetic items; need to weight real items 5-10x.

### Option v6.B: Cascade-first, not isolated-classifier-first

Diagnosis: trying to make v_specialist robust as a standalone is the wrong frame. The deployment target is Stage1 (generalist) + Stage2 (specialist) cascade.

Intervention: rebuild v6 specifically for the cascade input distribution. Stage1 (WildGuard or LG3) routes only "bio-suspect" items to v6. v6 sees a much narrower input distribution -- mostly bio compliance, mostly UNSAFE prior probability -- so the calibration regime is fundamentally different. Train v6 with calibrated routing signals as input features.

Cost: low (architectural redesign, not new data). Implementable on Cayuga in 1 day.

Risk: ties v6's identity to one specific Stage1, may not generalize.

### Option v6.C: Larger model (capacity ceiling test)

Diagnosis: 184M is too small to learn the causal feature vs spurious feature distinction; bigger model has the capacity for proper representations.

Intervention: replace DeBERTa-v3-base (184M) with DeBERTa-v3-large (304M) or RoBERTa-large (354M). Re-run v5 training pipeline on larger backbone.

Cost: medium (~5x training time, ~5x memory, but still fits on A100 80GB).

Risk: capacity may not be the bottleneck. If v5 already shows shortcut behaviour at 184M, scaling up may just amplify the same shortcut.

### Option v6.D: Different training paradigm (generative)

Diagnosis: classification-head fine-tunes are fundamentally susceptible to shortcuts in the single logit. Generative classifiers (LLaMA-Guard style) with reasoning produce more interpretable, more debuggable decisions.

Intervention: switch to LLaMA-Guard or WildGuard style: fine-tune a small (3B-7B) instruction-tuned model to generate "harmful" / "unharmful" tokens with optional category labels.

Cost: high (~10-20x compute), but well-trodden path.

Risk: loses the 6-16x inference cost advantage. Re-enters the regime of "184M v4 was specifically attractive because it's tiny."

### Decision criterion at v5 failure

If v5 acceptance gate failure mode is:
- P4 canary rises (new shortcut) -> try v6.A (real data breaks synthetic ceiling)
- OR-Bench-Hard FPR doesn't improve over v5_baseline -> try v6.B (cascade-first reframing)
- Probes P1-P2 over-erased (representation damaged) -> try v6.C (capacity ceiling)
- Refusal-prefix bypass returns -> try v6.A (architectural fix didn't help, root cause is data)

### v6 stop rule

If v6 also fails, the project conclusion is: **data-centric corrections in the 184M synthetic-data regime have hit a ceiling.** The honest move is to write that conclusion in the technical report (not v7), and recommend either cascade-with-existing-baselines or a real-data-annotation campaign as the path forward.

## References

- PairCFR: Qiu et al. ACL 2024, [arXiv:2406.06633](https://arxiv.org/abs/2406.06633)
- LEACE: Belrose et al. 2023, [arXiv:2306.03819](https://arxiv.org/abs/2306.03819)
- SPLICE: 2025, [arXiv:2506.10703](https://arxiv.org/abs/2506.10703)
- ShortcutProbe: Yang et al. IJCAI 2025, [arXiv:2505.13910](https://arxiv.org/abs/2505.13910)
- FalseReject: Zhang et al. COLM 2025, [arXiv:2505.08054](https://arxiv.org/abs/2505.08054)
- SORRY-Bench: Xie et al. 2024, [GitHub](https://github.com/SORRY-Bench/sorry-bench)
- Do Regularizers Work As Intended: Hong et al. AISTATS 2025, [arXiv:2503.17015](https://arxiv.org/abs/2503.17015)
- CAD ineffectiveness analysis: Sen et al. 2021, [arXiv:2107.00753](https://arxiv.org/abs/2107.00753)
- CC++: Cunningham et al. 2026
- Constitutional Classifiers v1: Sharma et al. 2025, [arXiv:2501.18837](https://arxiv.org/abs/2501.18837)

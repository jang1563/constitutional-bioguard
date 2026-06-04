# V6 Design v2: Cascade + Inference-Time Projection (No New Training)

> **⚠️ Historical design doc.** Forward-looking v6 plan; all three v6 interventions later FAILED their
> pre-registered acceptance gates (see `TECHNICAL_REPORT.md` §6.20). The v4-baseline framing here was
> superseded by the v7/v8 work; authoritative current card: [`MODEL_CARD.md`](MODEL_CARD.md).

**Status:** APPROVED for execution (2026-05-28). User decisions locked:
- Q1 = (a) Synthetic probe set responses
- Q2 = (c) Both Stage1 candidates (WildGuard 7B + LLaMA-Guard 3 8B), compare
- Q3 = (c) Maximal scope (SPLICE + Cascade + WildGuardMix final-layer retrain with encoder frozen). 5-6 days.
- Q4 = (a) Transparent negative-result publish if v6 fails acceptance gates

Execution order: F.0 (G.1 audit) → F.1 (WildGuard re-measure) → F.2 (SPLICE) → F.3 (Cascade calib) → F.4 (final-layer retrain) → F.5 (eval + report).

**Companion docs:**
- `V5_DESIGN.md` (locked design that failed, lessons applied here)
- `TECHNICAL_REPORT.md` Sections 6.16-6.18 (v4-v5 results and audit)
- This doc supersedes the v6 contingency planning in V5_DESIGN.md Section 11.

---

## 0. Why this design exists

v3 → v4 → v5 → v5b trajectory: every iteration of training-time data-centric correction has produced a new Goodhart. v5 (λ=0.3) and v5b (λ=0.1) both collapsed bio recall on real distributions. The pattern is clear: **PairCFR (and likely any aggressive representation-level training intervention) breaks the bio-selectivity v4 spent four iterations building.**

The lesson is not "try harder." It is: **v4 is genuinely strong; the right v6 is to preserve v4 and add a calibration layer above it, not modify v4.**

v6 is therefore inference-time only. v4 weights stay frozen.

---

## 1. v4's locked state (the baseline we will not regress)

**Genuine strengths (validated on held-out OOD, leakage-audited):**

| Property | v4 value | Significance |
|---|---|---|
| SaladBench-O39 bio recall (n=52) | 98.1% | matches LG3 8B (98.1%), WG7 (96.2%) at 38-43x smaller scale |
| SaladBench non-bio FPR (n=2216) | 20.2% | baselines flag 91-94% indiscriminately |
| **Bio selectivity ratio** | **4.85x** | **only model at any scale with proper bio domain scope** |
| ALERT bio recall (weapon_biological) | 47.9% | v4 has positive selectivity vs WG7/LG3 inverted 0.6x |
| ALERT non-bio FPR | 12.3% | baselines flag 89-93% |
| OR-Bench-Hard-1K FPR (clean held-out) | 2.1% | gate <5% passed |
| XSTest FPR (clean held-out) | 0% | perfect |
| BioThreat-Eval F1 | 0.45 | competitive with LG3 8B |
| LAB-Bench / WMDP-Chem / MedQA FAR | 0-0.5% | specialist scope preserved |
| Inference latency (batch=1) | 12.4 ms | 6.7-15.6x faster than 7-8B baselines |
| Peak GPU memory | 2.08 GB | 7x lower than 7-8B baselines |
| Parameter count | 184M | 38-43x smaller than baselines |

**Genuine remaining limitations (improvement targets for v6):**

| Property | v4 value | Notes |
|---|---|---|
| CRT refusal+compliance hybrid FPR | 68% | Artificial scenario; not seen in real LLM outputs |
| WildGuard native bio recall | 32% | **Measurement may be wrong** — using prompt_harm_label, should use response_harm_label |
| SimpleSafetyTests bio recall | 45% | Short adversarial bio prompts; real defect |
| ALERT bio recall | 47.9% | Real defect, but selectivity must be preserved |

---

## 2. Why "no new training" is the principled choice

Three independent research lines converge on this (deep research summaries in the conversation; key papers cited):

**(a) PairCFR family failures (Hong et al. AISTATS 2025, arXiv:2503.17015):**
> "L1/L2/EYE/causal/causal-effect regularizers all fail when the shortcut is strongly output-correlated (AUROC=1.0 regime)."
This is exactly our regime. v4's compliance-template feature has AUROC=1.0; any regularizer that pulls embeddings apart removes label-correlated signal. v5 verified this empirically.

**(b) SPLICE (Holstege, Ravfogel, Wouters; NeurIPS 2025, arXiv:2506.10703) is the formal fix:**
> "Oblique projection that exactly preserves covariance with the target label, removing the linearly decodable concept without destroying label-correlated variance."
SPLICE has a closed-form solution. There is no λ to tune wrong. The bound on recall preservation is *provable*, not empirical.

**(c) Industry consensus (Anthropic CC++ Jan 2026, OpenAI o3 system card 2025, ASL-3 May 2025):**
The deployed pattern is cascade with weighted logit fusion. CC++ achieves 40x cost reduction with this pattern. Standalone safety classifier work has shifted to cascade-component work. v4's bio-selectivity is the exact Stage-2 property the cascade pattern needs.

**Combined: v6 = (v4 frozen) + SPLICE + cascade calibration. No new training.**

---

## 3. v6 architecture

```
                 [Query, Response]
                        │
                        ▼
        ┌─────────────────────────────┐
        │  Stage 1: General Safety   │  
        │  WildGuard 7B  OR  LG3 8B   │  ── prob = p_s1
        └─────────────────────────────┘
                        │
        Bio keyword filter on Query (NSABB-aligned)
                        │
                  ┌─────┴─────┐
                  │ route?    │  if (p_s1 > τ_1) OR (bio_keyword): route
                  └─────┬─────┘
                        │ yes
                        ▼
        ┌─────────────────────────────┐
        │  Stage 2: v4 + SPLICE       │  ── prob = p_s2 (frozen v4 weights)
        │  bio specialist             │
        └─────────────────────────────┘
                        │
        Weighted logit fusion: z_final = w·logit(p_s1) + (1-w)·logit(p_s2)
                        │
                        ▼
                  Final UNSAFE prob
```

**Key design choices:**
- Stage1 is *unchanged* (use existing WildGuard 7B or LG3 8B; no training)
- Bio keyword filter is *rule-based*, NSABB-aligned (no training)
- Stage2 is *frozen v4* + SPLICE projector at [CLS] embedding (post-hoc fit, no training)
- Weighted logit fusion (à la CC++) instead of pure handoff
- All calibration knobs (τ_1, w) tuned on **held-out bio dev set**, not on benchmarks we will report

---

## 4. SPLICE projector specification

**Goal:** remove the "refusal+compliance hybrid response style" direction from v4's [CLS] representation, while preserving the bio-harm label direction.

**Inputs:**
- v4 [CLS] embeddings on a probe set (call it `D_probe`, never used in training, never used in final reporting)
- Concept labels per item: `c_i ∈ {0, 1}` for "is this a hybrid-style response?"
- Task labels per item: `y_i ∈ {0, 1}` for "is this a bio-harm pair?"

**Probe set construction:**
- Build `D_probe` of ~800-1200 items from clean held-out queries (OR-Bench-Hard-1K queries we don't report on, FalseReject test queries, BioThreat-Eval queries that are NOT in our v5 reporting)
- Pair each query with 5 synthetic response variants: refusal, deflection, compliance, refusal+compliance, scientific tone
- Concept label = 1 if response variant is refusal+compliance hybrid, else 0
- Task label = 1 if query is bio-harm (per existing BioThreat-Eval labels), else 0

**Fit procedure (closed-form):**
```
X = embeddings (n × d), centered
y_c = concept labels - mean
y_t = task labels - mean

w_c = (X^T X + λI)^-1 X^T y_c    # least-squares concept direction
w_t = (X^T X + λI)^-1 X^T y_t    # least-squares task direction

# Make w_c orthogonal to w_t (preserve task direction)
w_c_orth = w_c - (w_c · w_t / ||w_t||^2) · w_t
w_c_norm = w_c_orth / ||w_c_orth||

# Projector: I - w_c_norm @ w_c_norm.T
P = I - w_c_norm @ w_c_norm.T
```

**Validation before deployment:**
1. Concept probe AUROC on `D_probe` test split must drop from ~1.0 → ≤ 0.6 *after* projection (concept removed)
2. Task probe AUROC on `D_probe` test split must stay ≥ 0.85 *after* projection (task preserved)
3. v4-with-SPLICE bio recall on SaladBench-O39 must be ≥ 96% (do-no-harm)
4. v4-with-SPLICE CRT hybrid FPR (G.2) must be < 40% (improvement target)

If any of (1)-(4) fails, **discard SPLICE projector, ship v4 unmodified.**

---

## 5. Cascade calibration recipe

**Stage1 candidates (run all three for comparison):**
- WildGuard 7B
- LLaMA-Guard 3 8B
- (Optional) An ensemble of both

**Routing function (3 variants to compare):**
1. **`R_prob`**: route to Stage2 if `p_s1 ≥ τ_1`
2. **`R_keyword`**: route to Stage2 if query contains NSABB-aligned bio keyword
3. **`R_combined`**: route if `R_prob OR R_keyword`

**Fusion function:**
- `z_final = w·logit(p_s1) + (1-w)·logit(p_s2)`
- (CC++ uses 0.55/0.45 weights; we will calibrate on dev set)

**Calibration set construction:**
- Held-out bio dev set: 200-400 items, **never used in any reported eval**
- Half bio UNSAFE (from WildGuardMix bio-filtered held-out, AIR-Bench bio held-out)
- Half non-bio (mix of safe and non-bio harm)
- Compute Pareto curve over (τ_1, w) ∈ [0, 1]²
- Pick the operating point with **best bio F1 at ≤ baseline FPR**

**Calibration must NOT use:**
- BioThreat-Eval (locked eval)
- OR-Bench-Hard-1K (locked eval)
- SaladBench (locked eval)
- ALERT (locked eval)
- XSTest, FalseReject test, SimpleSafetyTests (all locked eval)

---

## 6. Pre-registered acceptance gates

**Do-no-harm constraints (strict, all 8 must pass for v6 release):**

| # | Metric | v4 baseline | v6 floor | Notes |
|---|---|---|---|---|
| 1 | SaladBench-O39 bio recall | 98.1% | ≥ 96.0% | -2.1 pp tolerance |
| 2 | SaladBench non-bio FPR | 20.2% | ≤ 25.0% | +4.8 pp tolerance |
| 3 | **Bio selectivity ratio** | **4.85x** | **≥ 4.0x** | core unique property |
| 4 | OR-Bench-Hard-1K FPR | 2.1% | ≤ 5.0% | clean held-out |
| 5 | XSTest FPR | 0% | ≤ 2.0% | clean held-out |
| 6 | BioThreat-Eval F1 | 0.45 | ≥ 0.43 | -0.02 tolerance |
| 7 | LAB-Bench / WMDP-Chem / MedQA FAR | 0-0.5% | ≤ 1.5% | specialist scope |
| 8 | v4 standalone bio recall (SimpleSafety, ALERT) | 45%, 47.9% | ≥ 40%, ≥ 42% | -5 pp tolerance |

**Improvement targets (≥ 1 must achieve for v6 to add value over v4):**

| # | Metric | v4 baseline | v6 target | Statistical bar |
|---|---|---|---|---|
| A | CRT refusal+compliance hybrid FPR | 68% | < 40% | -28 pp meaningful |
| B | Cascade Pareto: bio-F1 at fixed latency | n/a | strict Pareto improvement over LG3/WG7 alone | bootstrap CI separation |
| C | WildGuard native bio recall (using response_harm_label) | TBD (re-measure) | +5 pp over v4-with-correct-label | bootstrap CI |
| D | Calibration ECE on bio dev set | TBD | < 0.10 | absolute threshold |

**Failure handling:**
- If any of gates 1-8 fails → **discard v6 component (SPLICE or cascade), ship v4 unmodified**
- If gate 8 fails on SimpleSafety/ALERT specifically → flag in technical report, do not silently ship
- If no improvement target is met → write up as "v6 negative result," v4 remains production

---

## 7. Leakage discipline (G.1 audit lessons)

**Locked eval sets (never touch in any v6 training/calibration):**

```
BioThreat-Eval (558)
OR-Bench-Hard-1K (1319)
FalseReject test (1187)
XSTest (450)
SimpleSafetyTests bio (40)
SaladBench-CBRN (2268) - including bio O39 subset (52)
ALERT-CBRN (4198)
WildGuard test (1709)
LAB-Bench public (1305) - has canary string
HarmBench test (410)
AdvBench test (104)
JailbreakBench bio (2)
WMDP-Bio (1273)
WMDP-Chem (408)
WMDP-Cyber (1887)
MedQA test (1273)
PubMedQA labeled (1000)
```

**Permitted v6 calibration sources (must be paper-designated train splits or stratified held-out):**

```
WildGuardMix train (86K) - paired refusal/compliance schema
WildJailbreak train (262K) - if used, exclude WildGuardTest overlap
AIR-Bench 2024 (Stanford CRFM, CC-BY-4.0) - filter to CBRN-bio (~150-300 items)
FalseReject train (14,624) - paper-designated split
SaladBench-CBRN train slice (70% bio-filtered) - stratified
HarmBench val (100) - paper allows tuning
AdvBench train (Zou 2023 60/20/20 train slice) - canonical
BeaverTails train - paper split
```

**Pre-training G.1 audit script:**
```python
# scripts/g1_v6_overlap_audit.py
# Verifies 0% overlap between v6 calibration/probe sources and all locked eval sets
# Run BEFORE any SPLICE fit or cascade calibration
```

If audit finds > 1% overlap on any eval set, halt v6 and re-construct sources.

---

## 8. Experiments E.1-E.5

### E.1: Re-measure v4 baseline using correct WildGuard label (½ day)

**Hypothesis:** v4's 32% WildGuard native bio recall is measured with `prompt_harm_label`, which conflates "query is harmful" with "response is harmful." Re-evaluate using `response_harm_label`. v4 may already be passing this metric.

**Procedure:**
- Load WildGuard test split with all three label columns
- Filter to items where `response_harm_label` is annotated and `prompt_harm_label` is bio-related
- Compute v4 recall against `response_harm_label`
- Same on a bio-only subset

**If v4 already at ≥ 45% on `response_harm_label`:** WildGuard native is not a real defect, downgrade in v6 plan.

### E.2: SPLICE projector fit + validation (1 day)

**Procedure:**
1. Build `D_probe` (800-1200 items, leakage-audited)
2. Extract v4 [CLS] embeddings on `D_probe`
3. Fit SPLICE projector P (closed-form)
4. Validate:
   - Concept AUROC: ~1.0 → ≤ 0.6 after projection
   - Task AUROC: stays ≥ 0.85
5. Bolt P as frozen `nn.Linear` between encoder and classifier head
6. Re-evaluate on all locked eval sets
7. Check acceptance gates 1-8

**Deliverable:** `models/deberta_bioguard_v4_splice/` with projector + provenance

**Decision rule:** if all gates pass and CRT hybrid FPR drops to < 40%, keep SPLICE. Else discard.

### E.3: Cascade calibration (1 day)

**Procedure:**
1. Build calibration dev set (200-400 items, leakage-audited)
2. For each (Stage1 model) × (routing variant) × (τ_1, w) grid:
   - Compute cascade predictions on dev set
   - Score: bio F1, bio recall, non-bio FPR, latency
3. Pick Pareto-optimal point on dev set
4. Evaluate at that fixed point on all locked eval sets
5. Compare to: WildGuard alone, LG3 alone, v4 alone, v4-with-SPLICE alone

**Deliverable:** `configs/cascade_calibration_v6.json` with chosen (Stage1, route_fn, τ_1, w)

**Decision rule:** if cascade strictly Pareto-dominates v4 alone on (bio F1, latency), ship. Else, ship v4 alone (with SPLICE if E.2 succeeded).

### E.4: Final eval + audit (½ day)

**Run on:** v4, v4+SPLICE, cascade(v4+SPLICE), cascade(v4 alone), all baselines
**Eval suite:** the locked eval list from Section 7
**Audit:** rerun G.1 leakage audit to confirm no creep
**Probe gauntlet:** rerun P1-P4 representation probes from V5_DESIGN

### E.5: Documentation (½ day)

**If v6 succeeds:**
- Add Section 6.20 to technical report
- Update v4 model card with cascade deployment example
- (Optional) Write separate "cascade recipe" companion doc

**If v6 fails:**
- Add Section 6.20 documenting negative result
- v4 remains the recommended model
- Note specific failure mode (SPLICE didn't help, cascade didn't Pareto-dominate, etc.)

**Total v6 budget:** 3-4 days, ~4 hours GPU, no new training runs.

---

## 9. Why this design avoids each v5 failure mode

| v5 failure | v6 prevention |
|---|---|
| OR-Bench-Health 100% train/eval overlap (G.1) | v6 uses no OR-Bench in calibration; G.1 audit before E.2 and E.3 |
| Phantom problem fix broke real strengths | v6 explicitly preserves v4 weights; SPLICE is provably label-covariance-preserving |
| PairCFR λ=0.3 too aggressive | No PairCFR. No λ to tune wrong. Closed-form projector. |
| PairCFR λ=0.1 also failed | Same — no contrastive loss at all |
| Bio recall collapsed | Pre-registered gates 1-3 and 8 abort if recall drops |
| Acceptance gates not strict enough | 8 do-no-harm gates AND ≥ 1 improvement target required |
| v5_baseline (data only) regressed | v6 uses no new training data for v4 encoder; only SPLICE projector + cascade calibration |

---

## 10. Risks and mitigations (revised)

| Risk | Likelihood | Mitigation |
|---|---|---|
| SPLICE concept direction wrong (removes too little or too much) | Medium | Validation steps 1-2 in Section 4; if AUROC drop is wrong, refit with different probe set |
| Cascade calibration overfits dev set | Medium | Use 200-400 items dev set; report dev metrics separately from eval metrics |
| Stage1 (WildGuard/LG3) over-flags non-bio so cascade fires too often | Low | Bio keyword filter dampens this; can ablate to keyword-only routing if Stage1 fails |
| WildGuardMix bio subset has hidden leakage with WildGuardTest | Medium | E.2 G.1 audit explicitly checks; if overlap > 1%, exclude affected items |
| SPLICE breaks something we don't measure | Low-medium | Run full eval suite including v3-era benchmarks (HarmBench, AdvBench, BeaverTails, etc.) |
| v6 succeeds but cascade is too operationally complex | Low | Standalone v4+SPLICE is fallback; document both |
| Goodhart on cascade-level F1 (Stage1 useless alone) | Medium | Report each stage's standalone metrics; insist Stage1 has meaningful flag rate |

---

## 11. Comparison to v5 design (what's different)

| Aspect | v5 design | v6 design |
|---|---|---|
| New training | Yes (encoder fine-tune) | **No** |
| Loss function | CE + PairCFR contrastive (λ=0.3) | None (no training) |
| v4 weights | Modified | **Frozen** |
| Data discipline | Switched B.1 to FalseReject | **Same data; no augmentation** |
| Architecture intervention | PairCFR Trainer + custom sampler | **SPLICE projector + cascade fusion** |
| Acceptance gates | 5 gates (4 behavioral + G5) | **8 do-no-harm + ≥1 improvement** |
| Reversibility | Hard to undo (model weights changed) | **Trivial (discard projector / config)** |
| Compute | ~25 min GPU training | **~10 min total GPU (inference only)** |
| Failure cost | Wasted training + lost time | **Diagnostic value; trivial cleanup** |

---

## 12. Open questions for user review

Before executing E.1-E.5, please confirm:

**Q1: SPLICE probe set construction approval.**
The SPLICE concept direction comes from a probe set of ~1000 synthetic (query, response) pairs. These pairs use queries from clean held-out distributions (BioThreat-Eval, OR-Bench-Hard-1K) but the responses are synthetic.
- (a) Approve synthetic-response probe set as planned
- (b) Request real-LLM-generated responses instead (slower, costs API)
- (c) Request alternative probe construction

**Q2: Cascade Stage1 choice.**
Three viable Stage1 options:
- (a) WildGuard 7B (we have predictions cached; cheapest)
- (b) LLaMA-Guard 3 8B (we have predictions cached; also viable)
- (c) Both, compare (recommended for paper quality)

**Q3: Scope of v6.**
- (a) Minimal: SPLICE only (E.1, E.2, E.5). 2 days. Lowest risk.
- (b) Recommended: SPLICE + cascade (E.1-E.5). 3-4 days. Industry-aligned.
- (c) Maximal: SPLICE + cascade + WildGuardMix data refresh for v4's final classifier head (still no encoder training). 5-6 days.

**Q4: Pre-commitment to negative-result publication.**
If v6 fails to pass any improvement target:
- (a) Write up as documented negative result in technical report (transparency)
- (b) Silently abandon v6, keep v4 as released
- (c) Other framing

---

## 13. Decision tree

```
Start
  │
  ▼
Approve V6_DESIGN_v2.md (Sections 1-11)?
  │
  ├─ No ──→ Revise based on user feedback
  │
  └─ Yes ──→ Execute E.1 (re-measure WildGuard)
              │
              ▼
        v4 already passes on response_harm_label?
              │
              ├─ Yes ──→ Downgrade WildGuard target,
              │           proceed to E.2 (SPLICE) for hybrid FPR fix
              │
              └─ No ──→ E.2 (SPLICE) for both targets
                          │
                          ▼
                  All gates 1-8 pass + improvement target met?
                          │
                          ├─ Yes ──→ E.3 (cascade calibration)
                          │             │
                          │             ▼
                          │       Cascade Pareto-dominates v4+SPLICE alone?
                          │             │
                          │             ├─ Yes ──→ Ship cascade as v6
                          │             └─ No ──→ Ship v4+SPLICE as v6
                          │
                          └─ No ──→ Discard SPLICE; v4 remains production
                                    Document in Section 6.20
```

---

## 14. References

- **CC++**: Cunningham, Wei et al. 2026, arXiv:2601.04603 — cascade recipe (0.55/0.45 logit fusion, 40x cost reduction)
- **SPLICE**: Holstege, Ravfogel, Wouters NeurIPS 2025, arXiv:2506.10703 — formal label-covariance-preserving concept erasure
- **LEACE**: Belrose et al. 2023, arXiv:2306.03819 — SPLICE's predecessor
- **PairCFR (negative reference)**: Qiu et al. ACL 2024, arXiv:2406.06633 — what v5 used and broke
- **AlphaSteer**: Shen et al. ICLR 2026, arXiv:2506.07022 — null-space-constrained alternative if SPLICE insufficient
- **Hong et al. AISTATS 2025**: arXiv:2503.17015 — regularizers fail in AUROC=1.0 regime
- **WildGuardMix**: Han et al. NeurIPS 2024, arXiv:2406.18495 — paired refusal/compliance training source
- **AIR-Bench 2024**: Zeng et al., arXiv:2407.17436 — regulation-grounded CBRN data
- **OpenAI o3 system card**: 2025 — domain-router → safety-reasoner cascade in production
- **Anthropic ASL-3 Deployment Safeguards**: May 2025 — production cascade announcement

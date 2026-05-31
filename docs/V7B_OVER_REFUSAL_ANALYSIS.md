# V7.B Over-Refusal Analysis & Mitigation Plan

**Date**: 2026-05-28
**Status**: Post-training diagnosis
**Model**: v7.B = Qwen3-4B + QLoRA (rank 16, all-linear) + Nemotron-style CoT + BYO Policy
**Sibling reference**: docs/V7_DESIGN.md

---

## 1. Symptom Summary

v7.B training metrics excellent (train_loss 0.436, mean_token_accuracy 94.9%), but eval shows extreme over-refusal:

| Benchmark | N | Flag% | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| OR-Bench-Hard-1K (safe) | 1319 | **73.5%** ⚠️ | — | — | — | — |
| FalseReject (safe) | 1187 | **81.9%** ⚠️ | — | — | — | — |
| XSTest | 450 | 66.9% | 0.581 | 0.875 | 0.699 | **0.504** ⚠️ |
| WildGuard (response_harm) | 1709 | 12.0% | 0.587 | 0.426 | 0.494 | 0.060 |
| SimpleSafety-Bio (harmful) | 40 | 85.0% ✅ | — | — | — | — |
| SaladBench-CBRN (harmful) | 2268 | 89.9% ✅ | — | — | — | — |
| ALERT-CBRN | 4198 | pending | — | — | — | — |

**Verdict**: Production-grade unfit. Over-flags ~75% of benign content. Recall on real CBRN harms is strong, but the FPR makes it unusable as a standalone classifier.

---

## 2. Root Cause Analysis

### 2.1 Closest precedent comparison: Nemotron-CSR-4B (the model we copied)

Nemotron-Content-Safety-Reasoning-4B (NVIDIA, the direct inspiration for v7.B) achieves XSTest F1 = **0.922 (no_think) / 0.908 (think)** vs v7.B's 0.699. Same 4B decoder + CoT + BYO pattern.

| Recipe element | Nemotron CSR-4B | v7.B | Delta |
|---|---|---|---|
| Base model | Gemma-3-4B-**IT** (instruction-tuned) | Qwen3-4B-**base** | NCS is pre-aligned |
| Training data | ~36,000 items | 5,650 items | **6.4× more data** |
| Method | Full fine-tune | QLoRA rank 16 | Less aggressive update |
| Learning rate | 1e-6 | 2e-4 | **200× higher** |
| CoT default | `/no_think` (CoT hurts F1) | `/think` forced | NCS recognized CoT cost |
| Policy variants | Single + topic-following data | strict/permissive flip | **Our flips contaminate labels** |
| Label balance | Not disclosed but balanced | 60% harmful | Skewed prior |

### 2.2 Root cause ranking (deep research evidence)

| Rank | Cause | Confidence | Evidence |
|---|---|---|---|
| 1 | **Strict-variant label flipping contaminates safe class** (1077 bio-safe→harmful flips) | HIGH | arXiv:2507.04250 "Just Enough Shifts" exactly this pattern; asymmetric strict vs permissive |
| 2 | **60% harmful base imbalance** drives prior toward "harmful" on uncertain cases | HIGH | OR-Bench paper arXiv:2405.20947; no successful safety classifier uses >55% harmful |
| 3 | **Template-based CoT + QLoRA shortcut memorization** (3 templates × 5650 items) | HIGH | AdvChain arXiv:2509.24269: *"standard CoT safety training fails... surface-level refusal patterns"* |
| 4 | **lr 2e-4 vs precedent 1e-6** flips safety behavior fast | MED-HIGH | Qi et al. arXiv:2310.20624: lr 2e-4 LoRA can flip safety in tens of steps |
| 5 | **Qwen3-4B-base vs Gemma-3-4B-IT** lacks pre-aligned refusal calibration | MEDIUM | NCS uses Gemma-IT; Qwen3Guard uses Qwen3-Instruct |
| 6 | Pure scale (4B too small) | LOW | Qwen3Guard-4B-Gen achieves XSTest F1 92.7 — 4B is sufficient |

### 2.3 Comparable model recipes (all succeed where v7.B fails)

| Model | Size | Train N | Harm/Safe | CoT | XSTest F1 / FPR | Source |
|---|---|---|---|---|---|---|
| **v7.B** | 4B | 5,650 | 60/40 | Yes (forced) | F1 0.699 / FPR 0.504 | (us) |
| Nemotron-CSR-4B | 4B | ~36,000 | balanced | Toggle (no_think default) | F1 0.922 / — | HF model card |
| WildGuard 7B | 7B | 87,000 | ~50/50 | No | F1 0.947 / 0.4% refusal-on-benign | arXiv:2406.18495 |
| Llama Guard 3 | 8B | LG2 + multi | balanced | No | F1 ~0.884 / FPR 0.040 | Model card |
| Aegis 2.0 | 8B | 34,248 | 3-class + 5,200 explicit benign | No | F1 0.862 / — | arXiv:2501.09004 |
| ShieldGemma 9B | 9B | 100k+15k | mixed synthetic | No (Yes/No logit) | F1 0.828 on OpenAI Mod | arXiv:2407.21772 |

**Diagnostic verdict**: v7.B is the **only published-or-attempted 4B-class generative safety classifier using LoRA + multi-policy variants + <10k items**. Every successful comparable uses ≥30k data + balanced labels + (mostly) no-CoT default.

### 2.4 Empirical confirmation — /no_think diagnostic (2026-05-28)

Root cause #3 (forced CoT) was tested directly on the **existing** v7.B adapter
with **zero retraining**: re-ran eval with `/no_think` instead of `/think`. The
chain-of-thought was rationalizing the model into flagging benign content.

| Benchmark | /think | /no_think | unknown_rate |
|---|---|---|---|
| OR-Bench-Hard (benign) flag% | 73.5 | **24.7** | 0.4% |
| FalseReject (benign) flag% | 81.9 | **41.7** | 0.5% |
| XSTest FPR | 0.504 | **0.072** | 0% |
| XSTest F1 (prec/rec) | 0.699 (0.581/0.875) | **0.779** (0.885/0.695) | — |
| SimpleSafety-Bio (harmful) flag% | 85.0 | 75.0 | 0% |

**Interpretation**: Favorable asymmetry — FPR falls ~43pt while harmful recall
falls only 10–18pt, and XSTest **F1 rises** 0.699→0.779 (precision 0.581→0.885).
This is a genuine calibration gain, not a uniform threshold shift. `unknown_rate`
< 0.5% everywhere rules out the truncation confound (no_think did not simply cut
generation off before the label token). Empirically reproduces Nemotron-CSR's
`/no_think` default on our own model.

**Consequences**:
- v7.B runs in `/no_think` by default (cascade stage-2 + any standalone use).
- Phase 2 retrain: drop forced CoT; reasoning optional/off by default.
- FalseReject still 41.7% under no_think → hardest benign cases need Phase 2 data
  work (benign hard-negatives). Prompt change is necessary but not sufficient.

---

## 3. Three Phase Mitigation Plan

### Phase 1: Cascade integration v7.A → v7.B (THIS WEEK)

**Rationale**: Zero retraining. Joint FPR drops multiplicatively. Highest ROI immediate action.

**Architecture**:
```
input → v7.A (BioClinical ModernBERT-large, 396M, FPR 0.046)
         ├─ prob < 0.2 (confident safe) → PASS
         ├─ prob > 0.8 (confident harmful) → FLAG
         └─ 0.2 ≤ prob ≤ 0.8 (borderline) → v7.B (CoT reasoning)
                                              ├─ harmful → FLAG
                                              └─ unharmful → PASS
```

**Expected joint FPR**:
- v7.A passes ~5–15% of inputs to v7.B (calibrated on OR-Bench-Hard)
- Joint FPR on benign ≈ v7.A FPR × v7.B FPR on borderline ≈ 0.046 × 0.5 ≈ **0.023**
- CBRN recall preserved on confident negatives where v7.A correctly flags; elevated on borderline where v7.B's CoT helps

**Implementation**:
- `scripts/cascade_v7a_v7b.py` — router class wrapping both models
- Calibrate routing band on held-out OR-Bench-Hard subset (target: 5–15% escalation rate)
- Eval on full benchmark suite + report cascade vs individual model metrics
- Stage-2 runs v7.B in `/no_think` (flag `--v7b-no-think`, see §2.4); `/think` kept as a baseline variant for the contribution comparison
- Reference: Llama Stack PromptGuard → Llama Guard pattern (Red Hat 2026/05/04)

**Acceptance**:
- XSTest FPR ≤ 0.10 (vs current 0.504)
- OR-Bench-Hard flag rate ≤ 0.25 (vs current 0.735)
- SaladBench-CBRN flag rate ≥ 0.70 (preserve ≥80% of v7.B's recall)
- WildGuard F1 ≥ 0.55 (do no harm vs v7.A's 0.577)

### Phase 2: v7.B' retrain (NEXT WEEK)

**Rationale**: Even with cascade, v7.B is overfit. A properly trained v7.B' is the long-term need.

**Recipe changes** (each addresses a ranked root cause):

1. **Drop strict-variant label flipping** (addresses root cause #1)
   - Keep `permissive` variant (it's not the problem — it adds safe-leaning signal)
   - OR redesign strict variants to NOT flip labels; instead train the model to read the policy and act on it without changing ground truth

2. **Rebalance labels** (addresses root cause #2)
   - Target 45/55 harmful/safe (max 50/50)
   - Augment with FalseReject-style + Aegis-style **benign hard-negatives**
     - Target: 3,000+ "seemingly toxic but actually benign" bio prompts
     - e.g., vaccine development, biosafety protocols, dual-use educational content
   - Use entity-graph + adversarial multi-agent generation per FalseReject paper

3. **Reduce CoT template footprint** (addresses root cause #3)
   - Distill to **one-sentence reasoning traces** (Nemotron's exact approach)
   - Generate per-item reasoning via larger model (Claude/GPT-4o) instead of 3 fixed templates
   - Add PairCFR-style minimal-edit counterfactual pairs (bio-vocab kept, label flipped)

4. **Match NCS hyperparameters** (addresses root causes #4, #5)
   - Switch base to **Gemma-3-4B-it** (instruction-aligned)
   - lr **1e-6** (vs current 2e-4 — 200× reduction)
   - 5 epochs, batch 32
   - Full fine-tune if compute allows (Cayuga A100 80GB can handle 4B FT); fallback: LoRA rank 32 on Gemma-3-4B-it

5. **Target data scale: ~15,000–30,000 items**
   - 1× default policy: 10–20K items
   - Topic-following dataset (CantTalkAboutThis-style) for custom-policy generalization
   - No multi-variant label flipping

**Acceptance**:
- XSTest F1 ≥ 0.85 (vs current 0.699)
- XSTest FPR ≤ 0.10
- OR-Bench-Hard flag rate ≤ 0.15
- WildGuard F1 ≥ 0.60 (vs current 0.494)
- SaladBench-CBRN flag rate ≥ 0.75

### Phase 3: v7.C (Llama-3.1-8B QLoRA) — direct WildGuard 7B competitor

Per V7_DESIGN.md task #90. Proceeds in parallel with Phase 2. Different design philosophy from v7.B (simpler classifier, no CoT, larger base) — direct head-to-head with WildGuard 7B.

---

## 4. Fix Strategies Considered (and Rejected)

| Strategy | Verdict | Reason |
|---|---|---|
| Self-consistency (N=5 majority vote) | ❌ | Systematic bias amplifies under majority vote |
| Token-level multi-token threshold | ❌ | No literature support |
| DPO/Constitutional AI alignment | Deferred | Expensive; simpler retrain works (FalseReject precedent) |
| Logit bias suppression alone | Triage only | Doesn't fix root cause |
| Lower LoRA rank (rank 4/8) | Conditional | Try as ablation if v7.B' retrain insufficient |
| Discriminative head on top | ❌ | Overlaps with cascade option which is cheaper |

---

## 5. Calibration / Quick-Win Operations (TODAY)

While Phase 1 cascade is being implemented:

- **Temperature scaling** on the `harmful` token logit using held-out calibration set
- **Logit bias** of −5 to −10 on "harmful" token at decode time
- **Threshold tuning** with FPR weight 3–5× higher than FNR in the cost function

Reference: Shen et al. arXiv:2409.19817 (Adaptive Temperature Scaling for fine-tuned LLMs).

---

## 6. Key Papers / Evidence (most load-bearing)

- **FalseReject** (Zhang et al., COLM 2025) — arXiv:2505.08054 — direct match for v7.B pathology; **OR-Bench compliance gains 44–97%** with safety preserved
- **Nemotron-CSR-4B** — HF: nvidia/Nemotron-Content-Safety-Reasoning-4B — XSTest F1 0.922; uses `/no_think` default
- **Just Enough Shifts** (Wang et al. 2025) — arXiv:2507.04250 — labels imbalance → "overgeneralized refusal patterns"; ACTOR achieves OR-Bench-Hard 29.5%→76.3% in 4-minute fine-tune
- **WildGuard** (Han et al., NeurIPS 2024) — arXiv:2406.18495 — 87k balanced data; 0.4% refusal-on-benign
- **Aegis 2.0** (NVIDIA) — arXiv:2501.09004 — 5,200 explicit benign refusal-deflection samples
- **AdvChain** (Cao et al. 2025) — arXiv:2509.24269 — *"standard CoT safety training fails"*
- **LoRA undoes safety** (Qi et al. 2023) — arXiv:2310.20624 — lr 2e-4 LoRA can flip safety behavior

---

## 7. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-28 | v7.B as-is **rejected** for standalone use | XSTest FPR 50.4% incompatible with production |
| 2026-05-28 | Phase 1 cascade prioritized | Zero retraining, multiplicative FPR drop, deployable this week |
| 2026-05-28 | Phase 2 retrain planned with NCS recipe alignment | 5 of 6 root causes addressable by recipe; cheaper than DPO/CAI |
| 2026-05-28 | v7.C (Llama-3.1-8B) continues per V7_DESIGN.md | Different design hypothesis; parallel work |
| 2026-05-28 | `/no_think` confirmed as primary over-refusal lever (root cause #3) | XSTest FPR 0.504→0.072, F1 0.699→0.779, zero retraining; cascade stage-2 + Phase 2 adopt no_think default (§2.4) |
| 2026-05-30 | WildGuard /no_think gate #4 measured **definitively**: F1 **0.548** (narrow FAIL of ≥0.55 by 0.002) | Validated parser (unknown 0.8%); action-aware ceiling 0.549 proves the recall gap is real, not a parser artifact (§8) |
| 2026-05-30 | Phase-1 /no_think scorecard: **4/5 gates PASS**, WildGuard the lone narrow miss | XSTest FPR 0.072 ✓, OR-Bench 24.7% ✓, SaladBench 71.5% ✓, WildGuard F1 0.548 ✗(−0.002), ALERT 0.646 (secondary) (§8) |
| 2026-05-30 | **Validity audit**: 3 verified threats → eval + training task-spec must be fixed *before* a Phase-2 recall retrain | Template-CoT random categories; prompt_harm≡response_harm collapse; CBRN gates use a constant stub response. Leakage CLEAN. (§9) |
| 2026-05-30 | v7.B2 task-spec retrain (§9.3+§9.4) **regressed at ship `/no_think`** — WildGuard F1 0.548→0.292, XSTest FPR 0.072→0.436, SaladBench 0.715→0.551 | Single-artifact A/B, but the data file bundles 3 changes (dual-labels + grounded-CoT + 37% volume cut) → not attributable to §9.4 alone (§10) |
| 2026-05-30 | **Keep v7.B; v7.B2 = documented negative result** | (0,1) quadrant unlearnable from bio source (0/469); (1,0) trained (289) yet regressed; `/think` footer collapse (1/1709) moot since ship=`/no_think` (§10) |

---

## 8. Definitive /no_think Phase-1 Scorecard (2026-05-30)

All Phase-1 acceptance gates (§3 Phase 1) re-measured on the **existing v7.B adapter** in `/no_think`, with the colloquial-fallback parser fix applied (verdict_source tracking + raw_output saved). No retraining.

| Gate | Benchmark | Target | Measured | Status |
|---|---|---|---|---|
| #1 | XSTest FPR | ≤ 0.10 | **0.072** | ✅ PASS |
| #2 | OR-Bench-Hard flag% | ≤ 0.25 | **0.247** | ✅ PASS |
| #3 | SaladBench-CBRN flag% | ≥ 0.70 | **0.715** | ✅ PASS\* |
| #4 | WildGuard F1 | ≥ 0.55 | **0.548** | ❌ FAIL (−0.002) |
| (secondary) | ALERT-CBRN recall | ≥ 0.70 | **0.646**\* | near |

> \*Both CBRN gates are curation-inflated (§9.6). True CBRN-bio/chem is only 2.3% of the SaladBench-CBRN set (drugs + terrorism dominate); ALERT's honest CBRN-weapon recall is **0.480**, not 0.646. The rebuilt `alert_cbrn_strict` (n=590) is the honest replacement gate.

**WildGuard /no_think — definitive (n=1709)**: F1 **0.5479**, recall 0.493, precision 0.617, FPR 0.061, flag-rate 0.133 (227 flags). Parser validated: unknown 0.82% (14 items), colloquial-route 23.8% (406 items). Artifact: `results/metrics/v7b_nothink_eval_wildguard_native.json`.

The earlier "F1 0.552" reading was a **parser confound**, not a real number: ~26% of verdicts concluded colloquially ("Safe." rather than the trained "Response harm: unharmful") and the regex silently dropped them to unknown→0. After the colloquial fix the true F1 is 0.548. (0.552 was coincidentally close because most dropped colloquials were safe items.)

**Parser ceiling (offline re-parse, `scripts/reparse_ceiling_wildguard.py`)**: an action-aware reader that resolves every saved `raw_output` to a verdict yields ceiling F1 **0.549** (recall 0.495) — i.e. **no parser change crosses 0.55**. The gap is in the model's verdicts, not the reader: 108 clean *structured* "unharmful" misses dominate the false-negatives. → gate #4 failure is a **real recall gap**, not a code fix.

> A buggy `violat\w*` cue was tried and removed during this analysis: it matched the negated SAFE phrase "No … categories are violated" and fabricated 10 false positives. Documented in the script.

**ALERT-CBRN /no_think (n=4198, all label=1, `scripts/analyze_alert_misses.py`)**: headline recall **0.646** (1486 misses), parser clean (unknown 0.07%). Misses are confident structured "unharmful" (1484/1486), not parser drops. Per-category recall shows the headline is **curation-inflated**:

| alert_category | recall | mission relevance |
|---|---|---|
| weapon_radioactive | 0.292 | on-mission CBRN |
| weapon_firearm | 0.384 | off-mission (not bio/CBRN) |
| weapon_other | 0.508 | mixed |
| weapon_biological | 0.512 | **on-mission CBRN** |
| weapon_chemical | 0.588 | on-mission CBRN |
| crime_injury | 0.697 | off-mission (general harm) |
| substance_other | 0.699 | off-mission |
| substance_drug | 0.756 | off-mission |

Real **CBRN-weapon** recall (bio+chem+rad, n=590) is **0.480** — *lower* than the 0.646 headline, which is propped up by easy off-mission categories (drugs, general crime) that the curation filter admitted (§9.6).

**Benchmark rebuilt (2026-05-30):** `data/external/alert_cbrn_strict.jsonl` (n=590: bio 213 / chem 216 / rad 161; nuclear 0 in source) is the honest CBRN-weapon gate. `cache_alert_cbrn_strict()` keeps only `weapon_{biological,chemical,radioactive,nuclear}`, tags each row with `cbrn_class`, and derives deterministically from the full ALERT cache (no HF re-download). The polluted `cache_alert` builder is kept for reproducibility but deprecated in its docstring; the benchmark is registered as `alert_cbrn_strict` in `eval_v7b_qwen3_cot.py`. v7.B /no_think recall on the strict set is **0.480** — the number future evals and the post-retrain gate use. Responses remain the `COMPLIANCE_TEMPLATE` stub → treat as a **prompt-harm** gate (§9.5).

---

## 9. Validity Audit: Are the Phase-1 Gates Measuring the Right Thing? (2026-05-30)

Before any Phase-2 retrain spends Cayuga compute chasing the failing/near gates (#4 WildGuard, ALERT), this audit checks the gates themselves for leakage, benchmark-construction errors, and Goodhart traps. **Finding: the gates are partly mis-specified — so Phase 2 must fix the eval + training task-spec _before_ adding recall data, or it Goodharts a mis-specified target.**

### 9.1 Leakage — CLEAN ✓
`scripts/g1_v7_overlap_audit.py` (new; v7 had no prior leakage audit — g1 existed only for v5/v6). Exact-norm + prefix-64 overlap between training queries (`train.jsonl` + `wildguard_mix_train_bio.jsonl`) and every eval benchmark = **0**. The WildGuard split is disjoint (train_bio = 234 unique queries; 0 exact overlap with wildguard_test). Phase-1 gates are not contaminated by train/eval overlap.

### 9.2 No lexical / response-template shortcut in training — CLEAN ✓
v4_train (n=3062) has **1848 unique responses, 0 COMPLIANCE_TEMPLATE stubs**, max repeat 5. The model is not keying on a canned response string (the v3 failure mode, §6.16.2 of TECHNICAL_REPORT). The shortcut that broke v3 is not present here.

### 9.3 THREAT — synthetic template CoT with random categories (Goodhart)
`scripts/build_v7b_cot_data.py::make_cot()` builds the chain-of-thought by `RNG.choice(templates)` with the biosafety category `cat = RNG.choice(UNSAFE_CATS)` **drawn at random**, independent of the item. The model regurgitates these templates near-verbatim at eval (the SAFE template "…categories B1-B5 are not triggered. Safe." appears verbatim in /no_think output). → the "reasoning" is **decorative, not functional**: the B1-B5 category assignment carries no item-specific signal. This confirms root-cause #3 (§2.4) at the data-builder level.

### 9.4 THREAT — prompt_harm ≡ response_harm training collapse
`build_v7b_cot_data.py::format_output()` sets **both** output fields to the *same* item label, always:
```python
prompt_harm   = "harmful" if label == 1 else "unharmful"
response_harm = "harmful" if label == 1 else "unharmful"   # identical, by construction
```
The model never saw an example where prompt-harm and response-harm differ. But **WildGuard's response_harm task tests exactly that distinction** (harmful prompt + refusal = *safe* response). Gate #4 asks the model to make a separation it was structurally trained never to make → part of the 0.548 "recall gap" is **task-spec mismatch**, not capability. Fix: train on `wildguard_mix`'s native dual labels. → **Tested in v7.B2 (§10): regressed** — and the result cannot be pinned on this fix alone (bundled-change confound, §10).

### 9.5 THREAT — CBRN gates are OOD stub-response proxies
`extended_benchmarks.py::cache_alert` (and the SaladBench builder) set every item's response to a **constant** `COMPLIANCE_TEMPLATE` ("Yes, I can help with that. Here is detailed information … including specific steps, materials, and considerations…"), label=1. Training responses are real and diverse (§9.2). So SaladBench/ALERT measure *prompt-harm-via-constant-stub*, a distribution the model never trained on — not response-harm. The passing CBRN gate (#3) and the near ALERT gate are **OOD proxies**; their numbers should not be read as response-harm recall.

### 9.6 THREAT — ALERT curation pollution
`cache_alert`'s `bio_categories` filter admits `crime_injury`, `substance_drug`, `substance_other`, and `weapon_firearm` alongside true CBRN. Those off-mission categories carry the **highest** recall (0.70–0.76) and inflate the 0.646 headline above the true CBRN-weapon recall of **0.480** (§8). The ALERT gate, as built, is not a clean CBRN-recall measurement. **Fixed** → `alert_cbrn_strict` (§8).

**SaladBench has the same flaw (found 2026-05-30, eval-fix #107).** `cache_saladbench_cbrn` uses the same union with `cbrn_ids = {O35, O36, O4, O19, O54}`, but by `salad_category_3` the set is dominated by **non-CBRN**: O19 Illegal-Drugs (563), O54 Drug-related (351), O4 Terrorism (330), O35/O36 generic Weapon-gen/mgmt (462/322). The only unambiguously CBRN bucket, O39 Biological/Chemical, is **52/2268 (2.3%)**. So gate #3's "PASS 0.715" is *also* curation-inflated and is not a CBRN-bio measurement. A strict SaladBench rebuild (O39 + chem/rad + bio-keyword arm only) is tracked in #107; pending that, treat the SaladBench PASS as soft.

### 9.7 Implication for Phase 2 (analysis, not a decision)
Gate #4 (fail) and the ALERT gate (near) are each **partly an artifact of eval/task-spec construction**, not purely a model recall deficit. Recommended sequencing so a retrain does not optimize a mis-specified target:
1. **Fix eval first** — rebuild `alert_cbrn` to true CBRN categories; acquire a real-response CBRN benchmark; report CBRN-weapon recall separately from off-mission.
2. **Fix training task-spec** — separate prompt/response harm labels (use wildguard_mix dual labels); replace template CoT with per-item teacher CoT (Nemotron one-sentence style, per §3 Phase 2.3).
3. **Then** re-measure recall on the aligned task + distribution, and only then decide whether a recall-targeted data expansion is warranted.

These remain Phase-2 *options*; the go/no-go and ordering are open decisions, not settled here.

> → **v7.B2 executed step 2 of this plan (§10).** Outcome: regression at ship config, v7.B retained. The eval fixes (step 1, #106/#107) remain open.

---

## 10. v7.B2 — Empirical Test of the §9.3/§9.4 Task-Spec Fixes (2026-05-30)

§9.7 said to fix the training task-spec before any recall retrain. **v7.B2 is that test, run as a clean single-_artifact_ A/B**: `cayuga_v7b2_pipeline.slurm` changes *only* the training-data file vs the v7.B pipeline (`data/processed/v7b2_cot_train.jsonl`, n=3531); every hyperparameter is identical (epochs 3 · bs 4 · ga 4 · lr 2e-4 · r16/a32 · seq 2048). Ship-config eval is `/no_think`, same as v7.B.

> **Confound, stated up front.** The one changed artifact bundles **three** conceptual changes, so a regression cannot be pinned on any single one:
> 1. **§9.4 dual labels** — prompt-harm and response-harm supervised separately (289 divergent rows).
> 2. **§9.3 grounded CoT** — item-specific category phrase + B-code, not `RNG.choice` of fixed templates.
> 3. **~37% volume cut + policy-augmentation drop** — 5650 → 3531 rows, default-policy only (strict/permissive label-flip variants dropped).
>
> v7.B2 is decisive about *"does this data file beat v7.B"* (it does not) but **not** about *"is the §9.4 dual-label idea good in isolation."*

### 10.1 Result — comprehensive regression at ship config (`/no_think`)

From `results/metrics/v7b2_nothink_eval_*.json` vs `v7b_nothink_eval_*.json`:

| Gate | v7.B | v7.B2 | read |
|---|---|---|---|
| WildGuard F1 | **0.5479** | **0.2922** | precision 0.617→0.186, recall 0.493→0.683 |
| XSTest FPR (F1) | **0.072** (0.779) | **0.436** (0.459) | over-refusal 6× worse |
| OR-Bench-Hard flag% | 0.247 | 0.428 | +73% over-flag |
| SaladBench-CBRN flag% | 0.715 | 0.551 | CBRN recall *lost* |
| FalseReject flag% | 0.417 | 0.388 | apparent gain is an unknown→0 artifact (unk 15.1%) |

v7.B2 `/no_think` enters a **global over-flagging regime** (WildGuard flag 0.61 / FPR 0.60) while *also* losing CBRN recall (SaladBench 0.72→0.55) — strictly worse on both axes that matter. WildGuard recall rises (0.68) only because precision collapses (0.19): it flags ~everything. unknown_rate rises to 3–15% (was <1% for v7.B).

### 10.2 The `/think` anomaly — structured-output collapse (moot for ship, documented)

v7.B2 is `/think`-**trained**, so a `/think` eval was run (`cayuga_v7b2_think_eval.slurm`, wildguard first so the decisive gate survives a wall). It did not recover the model; it exposed an output-format collapse:

| `/think` gate | n | flag% | **unknown%** | F1 |
|---|---|---|---|---|
| wildguard_native | 1709 | 17.6 | **65.3** | 0.332 |
| xstest | 450 | 1.6 | **92.9** | 0.0 |
| simple_safety_bio | 40 | 2.5 | **95.0** | – |
| or_bench_hard_1k | 1319 | 6.1 | **85.6** | – |
| false_reject_test | 1187 | 6.7 | **88.3** | – |

Content-blind structural diagnostic on the saved predictions (`v7b2_think_eval_wildguard_native.json`): the trained `Response harm:` footer appears in only **1/1709** outputs; `</think>` closes in **97.4%**; the colloquial fallback recovers ~35%; the other **65%** yield no parseable verdict → default unharmful. The low flag% on the over-refusal gates is therefore an **unknown→0 artifact**, not an over-refusal win.

Two mechanistic hypotheses, **both falsified without GPU**:
- *Footer truncated in training* (long §9.3 CoT exceeding seq 2048): **0/3531** rows truncate the footer; templated token-len p99 952, max **1179** « 2048.
- *Data starvation*: 3531 rows is ample for a fixed output suffix, and v7.B learned it on a comparable distribution.

The collapse is `/think`-specific (`/no_think` footer mostly works, unk 3–15%) and most plausibly tied to the §9.3 CoT change, not the §9.4 labels. **Not pursued** — ship config is `/no_think`, and §10.1 already settles the A/B.

### 10.3 Data-distribution bound on what §9.4 can buy

Off-diagonal mass (prompt-harm ≠ response-harm) bounds the value of separating the labels (quadrant counts, content-blind):

| (prompt, response) | `wildguard_mix_train_bio` n=469 | `wildguard_test` n=1709 | v7.B2 train n=3531 |
|---|---|---|---|
| (harmful, harmful) | 143 | 272 | 2041 |
| **(1,0)** harmful→unharmful | **194** | **477 (28%)** | 289 |
| **(0,1)** unharmful→harmful | **0** | **10 (0.6%)** | **0** |
| (unharmful, unharmful) | 132 | 930 | 1201 |

- **(1,0)** — the "harmful prompt, safe/refused response" case §9.4 named — is **real and common** (28% of eval, 41% of the bio WG source). §9.4's premise holds.
- **(0,1)** — "benign prompt, harmful response" — is **absent from the bio training source (0/469)** and 0.6% of eval. That half of the response-harm task is **unlearnable from this data**; a "populate (0,1)" v7.B3 is impossible without a different source.
- v7.B2 *did* train on 289 (1,0) rows (194 native WildGuard + 95 v4 refusal-heuristic) and still regressed → adding the (1,0) supervision **as bundled here** did not close gate #4.

### 10.4 Decision

**Keep v7.B (`/no_think`, WildGuard F1 0.5479) as the v7 generative classifier.** v7.B2 is retained as a rigorous negative result, not shipped. If the §9.4 dual-label hypothesis is revisited it must be **isolated**: change *only* the labels (keep v7.B's CoT builder and the full 5650-row volume), source (1,0) response labels from WildGuard's **native** annotations (not the v4 refusal heuristic), and leave (0,1) out of scope until a bio source containing it exists.

Retrospective lesson: **task-spec "correctness" and training-data volume/format are not separable knobs.** Bundling a principled label fix with a CoT change and a volume cut produced a model worse than the "incorrectly" specified baseline on *every* shipping axis. The disciplined move was to kill the variant, not to keep tuning it.

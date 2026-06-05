# v2 Dataset: Key Findings Summary

**Dataset:** unified_overrefusal_taxonomy_v2.jsonl  
**Records:** 23,330 (+91% from v1)  
**Status:** ✅ Ready for classifier training with proper stratification

---

## Core Improvements Over v1

| Metric | v1 | v2 | Status |
|--------|----|----|--------|
| **Legitimate class** | 97.4% | 58.8% | ✅ **58.6 pp reduction** |
| **Negative class** | 0.4% | 32.8% | ✅ **82x increase** |
| **Legitimate:Negative ratio** | 243:1 | 1.79:1 | ✅ **Trainable** |
| **Total records** | 12,209 | 23,330 | ✅ **+11,121 new** |
| **Data sources** | 7 | 14 | ✅ **+7 sources** |

**Verdict:** v2 is fundamentally better balanced for binary classification.

---

## Critical Constraints for Training

### 1. **Domain Skew** (⚠️ CRITICAL)
- **Protein engineering: 50.7%** ← Dominates from FRT P2
- **CBRN safety: 27.7%** ← External benchmarks
- **Real biology: 11.9%** ← Virology, toxicology, pathogen, etc.

**Action:** Downsample FRT P2 to 5,000-6,000 records; oversample underrepresented domains 2-3x.

### 2. **Model Bias** (⚠️ HIGH)
- **Opus 4.7 via FRT API: 50.7%** ← Single model, may overfit
- **Benchmark models: 37.4%** ← No response text, binary labels
- **Real sessions: 3%** ← Limited coverage

**Action:** Train/test separately per model; create domain-specific submodels.

### 3. **Response Text Sparsity** (⚠️ MEDIUM)
- **Full response text: 45.4%** of records
- **No response: 54.6%** (FRT raw refusals, benchmarks, expert annotations)

**Action:** Use prompt features (length, keywords) for records without response text; don't force imputation.

### 4. **Session Context Availability** (ℹ️ FYI)
- **Context recovered: 531 records (2.3%)**
- **Full context with prior turns available**

**Action:** Use session blocks for Phase 2 context-aware classifier; separate evaluation task.

---

## Classification Scheme Recommendations

### Primary Task (Binary Classification)
```
LEGITIMATE (13,727 records, 58.8%):
  ├─ Select agent research (7,698) → Research using known select agent proteins
  ├─ Control proteins (3,895) → Should not block (clearly benign)
  ├─ Dual-use benign (353) → Legitimate despite dual-use potential
  └─ Research workflow (462) + other assists (1,319)

NEGATIVE (7,653 records, 32.8%):
  ├─ Benchmark correctly refused (7,007) → Expert-labeled harmful
  └─ Expected refusal (43) + refuse routes (603)
```

**Training set:** Select agent + control proteins + benchmark refused = **15,600 high-confidence**  
**Additional:** Add low-confidence with weight 0.5 = **+3,461** → **19,061 total (82%)**  
**Hold out:** Ambiguous + meta-artifacts = **4,269 (18%)**

### Secondary Tasks (Multi-class)
```
AMBIGUITY DETECTION:
  - Dual-use ambiguous (276)
  - Anonymous protein unclear (225)
  - Mixed/boundary cases (1,449)

META-REFUSAL DETECTION (separate eval):
  - Continuation commands (114)
  - Tool result context (103)
  - Session context inherited (82)
```

---

## Recommended Training Strategy

### Phase 1: Baseline Binary Classifier (Week 1)
```
Approach: Stratified 5-fold K-fold, high-confidence labels only
Data: 18,600 records (top 3 tiers)
Features: block_trigger, legitimacy_tier, content_domain, query length, model, stop_reason
Models: Logistic regression → SVM → BERT fine-tuning
Target F1: ≥0.85
```

### Phase 2: Domain Stratification (Week 2)
```
Validation: Per-domain F1 analysis
Oversample: Virology, toxicology, synthetic_biology, genomics
Ensemble: Separate heads per domain cluster
Target: All domains ≥0.70 F1
```

### Phase 3: Cross-Model Generalization (Week 3)
```
Test on: Claude Sonnet, Haiku, Gemini, GPT-4o subsets separately
Holdout: 10% session blocks (real user over-refusal)
Report: Per-model F1 with confidence intervals
```

---

## Data Quality Issues (Priority Order)

| Issue | Impact | Fix | Effort |
|-------|--------|-----|--------|
| FRT P2 repetition (11,330 similar queries) | HIGH | Downsample + augment with new domains | 1 day |
| Domain skew (50% protein engineering) | HIGH | Stratified sampling, oversample minorities | 1 day |
| No response text for 54% of records | MEDIUM | Use prompt features; don't impute | Already done |
| Meta-refusals mixed in | MEDIUM | Create separate evaluation task | 2 hours |
| Small session sample (701) | MEDIUM | Augment with new user logs (if available) | 1-2 days |
| Benchmark label variance | MEDIUM | Spot-check 50 records per benchmark | 1 day |

---

## Quick Validation Checklist

- [ ] Spot-check 20 protein_identity_select_agent records (verify protein ID in SELECT_AGENTS set)
- [ ] Spot-check 20 control_protein_clearly_benign records (verify protein ID in CONTROLS set)
- [ ] Verify 10 session_context_recovered records (check prior turns exist)
- [ ] Check 10 continuation_contamination records (verify CONTINUATION_RX regex correct)
- [ ] Sample 5 records each from: overrefusal_api, saladbench, wildguard, constitution_rules
- [ ] Confirm all 23,330 records parse as valid JSON, have required fields

---

## Estimated Classifier Performance

**Baseline Binary (block_trigger + legitimacy_tier only):**
- Expected F1: 0.82-0.85
- Reason: Triggers are ~100% predictive for protein-based records

**Improved (+ query text + model + domain):**
- Expected F1: 0.85-0.90
- Reason: Better coverage of edge cases, less overfitting to protein IDs

**With context (+ prior turns + project type):**
- Expected F1: 0.88-0.92
- Reason: Can distinguish continuation_contamination from legitimate continuation

**Cross-model (test on Sonnet/Haiku/Gemini):**
- Expected per-model F1: 0.80-0.87 (slight degradation from Opus)
- Reason: Models may have different safety boundaries

---

## Next Immediate Actions

1. **Validate dataset integrity (1-2 hours)**
   - Run spot-check script on 100 random records
   - Verify all 23,330 records are unique and valid JSON
   - Check for unexpected missing fields

2. **Build training/validation split (2 hours)**
   - Stratified K-fold by domain + class + source
   - Reserve AmbiguityCasebook (36) for boundary case eval
   - Reserve 10% session blocks for cross-validation

3. **Start baseline classifier training (1 day)**
   - Implement feature extraction pipeline
   - Train logistic regression baseline
   - Achieve ≥0.85 F1; document per-domain breakdown

4. **Parallel: Deep-dive on weak domains (1 day)**
   - Identify which domains have <0.70 F1
   - Augment with similar-domain records
   - Plan ensemble or transfer learning approach

---

**Generated:** 2026-05-28  
**Dataset Status:** ✅ Ready for training  
**Next milestone:** Baseline classifier (0.85+ F1) by end of week

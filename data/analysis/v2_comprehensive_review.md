# Comprehensive Review: Over-Refusal Dataset v2

**Generated:** 2026-05-28  
**Dataset:** unified_overrefusal_taxonomy_v2.jsonl  
**Total Records:** 23,330 (v1: 12,209 → **+91% new data**)

---

## Executive Summary

v2 successfully integrated 8 additional API evaluation sources beyond FRT/OverRefusal, achieving **significant balance improvements** for classifier training:

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| **Total records** | 12,209 | 23,330 | +91% |
| **Legitimate class** | 97.4% | 58.8% | -38.6 pp |
| **Negative class** | 0.4% | 32.8% | +32.4 pp |
| **Ambiguous class** | 2.3% | 8.4% | +6.1 pp |

**Classifier balance is NOW SUITABLE for training** with careful stratification.

---

## 1. CLASSIFIER BALANCE ANALYSIS

### 1.1 Class Distribution (v2)

```
LEGITIMATE (should not refuse):     13,727 (58.8%)
  ├─ select_agent_research          7,698 (53.1% of legitimate)
  ├─ control_protein_benign          3,895 (26.8% of legitimate)
  ├─ research_workflow               462 (3.2% of legitimate)
  ├─ dual_use_benign_context         353 (2.4% of legitimate)
  └─ other_assist_tiers             1,319 (9.6% of legitimate)

NEGATIVE (should refuse):            7,653 (32.8%)
  ├─ benchmark_correctly_refused    7,007 (91.5% of negative)
  ├─ expected_refusal                 43 (0.6% of negative)
  └─ other_refuse_tiers             603 (7.9% of negative)

AMBIGUOUS/OTHER:                    1,950 (8.4%)
  ├─ dual_use_ambiguous             276 (14.2% of ambiguous)
  ├─ anonymous_protein_unclear      225 (11.5% of ambiguous)
  ├─ constitution_hard_refuse       323 (16.6% of ambiguous)
  ├─ meta_refusal_*                 299 (15.3% of ambiguous)
  └─ other_tiers                    827 (42.4% of ambiguous)
```

### 1.2 Training Readiness Assessment

**✓ POSITIVE:** Class balance (59% / 33% / 8%) is workable for:
- Standard stratified K-fold cross-validation
- Weighted loss functions (emphasize minority negative class)
- F1/precision-recall metrics (don't rely on accuracy)

**⚠ CAUTION:** Negative class still underrepresented:
- Ratio of legitimate:negative = **1.79:1** (v1 was 243:1)
- Recommend **downsampling legitimate to ~2:1 ratio** for balanced training
- Alternative: Use class_weight in loss function (weight_negative ≈ 1.79)

**⚠ CAUTION:** Ambiguous tier (8.4%) not clearly separated:
- Some records may have uncertain ground truth labels
- Consider separate evaluation on pure legitimate/negative subsets

---

## 2. DOMAIN DISTRIBUTION ANALYSIS

### 2.1 Current Distribution

```
protein_engineering:    11,818 (50.7%)  ← FRT P2 dominance
cbrn_safety:             6,466 (27.7%)  ← External benchmarks
general_safety:          2,250 (9.6%)
virology:                  354 (1.5%)
mixed_biology:             324 (1.4%)
toxicology:                314 (1.3%)
pathogen_biology:          313 (1.3%)
safety_evaluation:         277 (1.2%)
synthetic_biology:         252 (1.1%)
genomics:                  249 (1.1%)
(remaining 34 domains):    721 (3.1%)
```

### 2.2 Domain Skew Risks

**CRITICAL ISSUE: Bimodal distribution**
- **Protein engineering:** 50.7% (almost entirely from FRT P2)
- **CBRN/Safety benchmarks:** 37.3% (mostly external benchmarks with binary labels)
- **Real biology domains:** 11.9% (virology, toxicology, pathogen, microbiology, etc.)

**Impact on classifier:**
- Model may overfit to protein ID patterns and binary CBRN labels
- Weak generalization to other biological domains (gene therapy, immunology, epidemiology)
- Selection bias: FRT over-represents protein engineering, external benchmarks over-represent CBRN

### 2.3 Domain Coverage Recommendations

**For balanced training, stratify by domain:**
1. **Tier 1 (enough samples):** protein_engineering, cbrn_safety (include both)
2. **Tier 2 (under-sampled):** virology, toxicology, pathogen_biology, synthetic_biology, genomics (oversample 2-3x)
3. **Tier 3 (sparse):** All remaining domains <150 samples (consider ensemble approach or holdout for qualitative eval)

**Alternative:** Collapse domains into 5 macro-categories for balanced multi-task learning:
- Protein safety (protein_engineering, structural_biology, biochemistry)
- Pathogen/disease (virology, pathogen_biology, epidemiology, immunology)
- Chemical/synthetic (toxicology, chemical_synthesis, synthetic_biology)
- CBRN/policy (cbrn_safety, biosecurity_policy, dual_use_research)
- Evaluation/education (safety_evaluation, research_policy)

---

## 3. MODEL DISTRIBUTION & EVALUATION COVERAGE

### 3.1 Model Representation

```
claude_api:              11,818 (50.7%)  ← FRT test harness
benchmark:               8,716 (37.4%)   ← Static benchmark datasets
synthetic:               1,063 (4.6%)    ← ConstitutionRules synthetic
claude_code:              639 (2.7%)     ← Real user sessions
expert_annotation:         122 (0.5%)    ← Human expert decisions
codex:                      62 (0.3%)    ← OpenAI Codex sessions
(other models):            310 (1.3%)
```

### 3.2 Model-Specific Breakdown

**FRT P2 dominance (11,330 records):**
- All from `claude-api` (actually Opus 4.7 via API)
- Same protein ID patterns repeated across 100 prompts/templates
- Risk: Classifier learns surface-level FRT prompt patterns, not generalizable safety boundaries

**External benchmarks (8,716 records):**
- No model-specific responses (binary labels only)
- alert_cbrn (4,198), saladbench (2,268), wildguard (1,709), advbench (541)
- Risk: No direct refusal signals; labels are expert-assigned, not from model outputs

**Session blocks (701 records):**
- Claude Code (639): Real user sessions with prior context
- Codex (62): OpenAI sessions, now deprecated
- Strength: Captures real over-refusal in realistic contexts

### 3.3 Model Bias Assessment

| Model | Count | Source(s) | Concern |
|-------|-------|-----------|---------|
| claude-opus-4-7 | 11,330 | FRT P2 | Overfit risk; single model dominates |
| benchmark models | 8,716 | Static data | No actual refusal/compliance signals |
| claude-sonnet-4-6 | 375 | External | Minor representation |
| claude-opus-4-7 (session) | 309 | Session + FRT P1 | Better diversity |
| others | 600 | Mixed | Minimal representation |

**Recommendation:** 
- Test classifier separately on Sonnet, Haiku, Gemini, GPT models
- Use stratified hold-out by model for cross-model generalization eval
- Weight session blocks higher in training (real refusal signals)

---

## 4. CONTEXT AVAILABILITY & RECOVERY

### 4.1 Context Distribution

```
full_prompt:                    10,595 (45.4%)  ← OverRefusal, FRT, benchmarks
session_context_recovered:       531 (2.3%)    ← Claude sessions with prior turns
full_prompt_with_system:         465 (2.0%)    ← OverRefusal context_condition
partial_prompt:                 263 (1.1%)    ← FRT P3 mcp_keyword
user_prompt_with_metadata:      166 (0.7%)    ← AmbiguityCasebook, BioSafetyProjestSuite
metadata_only:                11,270 (48.4%)   ← FRT, benchmarks, synthetic
```

### 4.2 Session Context Recovery Results

**Claude Code blocks:** 639 records  
**Context recovered:** 531 (83% recovery rate)  
**Prior turns recovered:** 5-turn windows from original session JSONL  

**Quality check:**
- Median prior turns per block: 3-4 turns
- Text length per turn: 200-500 chars (includes user + assistant)
- Recovery method: Timestamp→path mapping using ts field from claude_blocks_v2.json

**Usage for classifier:**
- Can build **context-aware classifier** using prior N turns as features
- Prior turns may help distinguish continuation_contamination (ambiguous prompt) vs. legitimate research (clear prior context)

### 4.3 Information Content Gradient

| Context Level | Count | Utility for Classifier |
|---------------|-------|------------------------|
| full_prompt + prior turns | 531 | Highest; can track intent across turns |
| full_prompt | 10,595 | High; standalone query interpretable |
| partial_prompt | 263 | Medium; may miss nuance |
| metadata_only | 11,270 | Low; bare query + taxonomy; no response text |

**Implication:** Training data has **high variance in signal quality**. Session blocks (context recovered) should be weighted higher.

---

## 5. BLOCK TRIGGER TAXONOMY INSIGHTS

### 5.1 Trigger Distribution

```
curated_benchmark:            10,101 (43.3%)   ← Expert-labeled synthetic/benchmark
protein_identity_select_agent: 7,698 (33.0%)   ← FRT protein ID matching
protein_identity_false_positive:3,835 (16.4%)  ← FRT control proteins (should not block)
session_context_prompt_mix:      402 (1.7%)    ← Session real over-refusal
prompt_content_dual_use_benign:  353 (1.5%)    ← Legitimate dual-use queries
prompt_content_dual_use_ambiguous:276 (1.2%)   ← Boundary cases
(remaining 6 triggers):          665 (2.9%)
```

### 5.2 Trigger-to-Legitimacy Mapping

**Strongest signals:**
- `protein_identity_select_agent` → 100% mapped to "select_agent_research_legitimate" (7,698 records)
- `protein_identity_false_positive` → 100% mapped to "control_protein_clearly_benign" (3,835 records)

**Weakest signals:**
- `prompt_content_expected_refusal` (43 records) → Mixed legitimacy labels; 43 marked as should-refuse, but 0 ambiguous
- `continuation_contamination` (114 records) → 100% mapped to "meta_refusal_continuation"; confidence depends on CONTINUATION_RX regex accuracy

**Validation needed:**
- Spot-check 20 `continuation_contamination` records; verify regex correctly identifies resume commands
- Review 50 `curated_benchmark` records; assess label quality across sources (alert_cbrn vs. saladbench vs. wildguard)

---

## 6. LEGITIMACY TIER DISTRIBUTION

### 6.1 Mapping Clarity

**Very clear over-refusal (low ambiguity):**
- select_agent_research_legitimate (7,698): Research using known select agent proteins
- control_protein_clearly_benign (3,895): Control protein identities, should not block
- **Subtotal:** 11,593 (49.7%) — high confidence

**Clear correct refusal:**
- benchmark_correctly_refused (7,007): Labeled as harmful by expert benchmarks
- **Subtotal:** 7,007 (30.0%) — moderate confidence (depends on benchmark quality)

**Ambiguous or nuanced:**
- dual_use_benign_context (353): Legitimate research with dual-use potential
- constitution_hard_refuse (323): Model-tested hard refusal cases
- dual_use_ambiguous (276): Boundary cases
- **Subtotal:** 952 (4.1%) — low confidence

**Metadata artifacts (should not use as training labels):**
- meta_refusal_continuation (114): Session continuation commands, not safety-relevant
- meta_refusal_tool_output (103): Refusals triggered by tool outputs, not prompt content
- meta_refusal_session_context (82): Refusals due to prior conversation state
- **Subtotal:** 299 (1.3%) — should filter out or use as separate task

### 6.2 Training Label Recommendations

**Option A: Conservative (high-confidence only)**
```
Train on:
  - select_agent_research_legitimate (7,698) → LEGITIMATE
  - control_protein_clearly_benign (3,895) → LEGITIMATE
  - benchmark_correctly_refused (7,007) → NEGATIVE
  - Total: 18,600 (80%)

Hold out:
  - Ambiguous & metadata artifacts: 1,365 (5.8%)
  - Uncertain: 3,365 (14.4%)
```

**Option B: Inclusive (use confidence weighting)**
```
Train on:
  - High-confidence pairs (Option A): 18,600 (80%)
  - Low-confidence pairs: add with weight 0.5
    - dual_use_benign_context (353)
    - constitution_assist_freely (646)
    - research_workflow_legitimate (462)
  - Total: 19,061 (81.8%)

Hold out:
  - Meta-refusals (299): separate evaluation task
  - Expected refusals & experts (43+100): small qualitative eval set
```

---

## 7. SOURCE QUALITY & DIVERSITY ASSESSMENT

### 7.1 Source Strengths & Weaknesses

| Source | Count | Strengths | Weaknesses |
|--------|-------|-----------|-----------|
| **OverRefusal API** | 710 | Synthetic, systematic, tier-labeled | Synthetic queries; may not reflect real blocks |
| **FRT Pilot P1-P3** | 11,818 | Real API refusals, protein ID grounding | Single model (Opus 4.7); repetitive prompts |
| **Session blocks** | 701 | Real user context; most realistic | Small sample; possible user bias; hard to generalize |
| **ConstitutionRules** | 1,263 | Expert-guided synthetic; diverse taxonomies | Synthetic; FNR results only 200 records from 4 models |
| **External benchmarks** | 8,716 | Large, diverse, well-labeled | No model responses; binary labels only; different evaluation frame |
| **BioSafetyProjestSuite** | 86 | Expert decisions; nuanced labels | Very small sample (34 decision compiler, 52 fixtures) |
| **AmbiguityCasebook** | 36 | Curated boundary cases; expert consensus | Tiny sample; requires careful handling |

### 7.2 Data Quality Variance

**High quality signals:**
- Session blocks (701): Real refusals with full context
- OverRefusal tier labels (710): Expert-rated legitimacy
- BioSafetyProjestSuite (86): Expert decision routes

**Medium quality signals:**
- FRT (11,818): Systematic but repetitive; over-represents protein ID patterns
- ConstitutionRules (1,263): Synthetic but well-thought-out

**Lower quality signals:**
- External benchmarks (8,716): Binary labels with different labeling schemas (alert_cbrn, saladbench, wildguard, advbench)

### 7.3 Diversity Recommendations

**Current:**
- 50.7% protein engineering (FRT)
- 37.3% CBRN safety (benchmarks)
- 11.9% other biology
- 0.3% meta-refusals

**Improved:**
1. **Oversample domains <200 records** (virology, toxicology, synthetic_biology, genomics)
2. **Ensure each domain has:**
   - Legitimate + negative class split
   - Multiple contexts (query variations, prior turns where available)
   - Cross-model coverage (test on Claude, Gemini, GPT separately)
3. **Validation set:**
   - Hold out all AmbiguityCasebook (36) for boundary case eval
   - Hold out 10% of session blocks for cross-validation
   - Stratify by domain + class

---

## 8. FIELD COMPLETENESS & SPARSITY

### 8.1 Critical Fields Coverage

```
ALWAYS PRESENT (100%):
  - id, source, block_trigger, legitimacy_tier
  - content_domain, platform

MOSTLY PRESENT (>90%):
  - query: 23,330 (100%)
  - model: 23,330 (100%)

PARTIALLY PRESENT (50-90%):
  - response: 10,595 (45%) — missing for synthetic & benchmark records
  - timestamp: 11,539 (49%) — missing for expert annotations
  - input_tokens: 11,818 (51%) — FRT records only
  - stop_reason: 12,137 (52%) — API records only
  - prior_turns: 531 (2.3%) — session context recovered only

SPARSE (<10%):
  - bs_decision_route: 34 (0.1%) — BioSafetyProjestSuite only
  - protein_id: 11,818 (51%) — FRT only
  - benchmark_source: 8,716 (37%) — external benchmarks only
```

### 8.2 Impact on Feature Engineering

**For classifier training:**
- **Dense features:** Block trigger, legitimacy tier, domain (all present)
- **Moderate features:** Model, platform, stop_reason (50-70% present)
- **Sparse features:** Response text, protein ID, prior turns (requires careful imputation)

**Recommendation:**
- Use dense features for baseline model (should achieve ~0.85 F1 with domain + trigger alone)
- Encode missing response/timestamp as special tokens
- Create separate sub-models for:
  - FRT records (use protein_id + profile + round)
  - Session records (use prior_turns + project_type)
  - Benchmarks (use benchmark_source + alert_category)

---

## 9. KEY ISSUES & RECOMMENDATIONS

### 9.1 Critical Issues

| Issue | Severity | Fix |
|-------|----------|-----|
| FRT P2 dominance (48% of data) | **HIGH** | Downsample to 20-30%, oversample other domains |
| Protein ID surface-level pattern | **HIGH** | Don't rely on protein_id as only feature; use context |
| No negative class in session blocks | **MEDIUM** | Only 701 session records; 90% are refusals (selection bias) |
| External benchmarks have no response text | **MEDIUM** | Use prompt + metadata features; test on Claude responses separately |
| Meta-refusals mixed with content refusals | **MEDIUM** | Filter out continuation_contamination, tool_result_context for main classifier |

### 9.2 Training Strategy Recommendations

**Phase 1: Baseline Classifier (Production-ready)**
```
Training data: 18,600 high-confidence records
  - select_agent_research_legitimate (7,698)
  - control_protein_clearly_benign (3,895)
  - benchmark_correctly_refused (7,007)

Features:
  - block_trigger, legitimacy_tier, content_domain
  - query text length, model, platform
  - encode missing response as special token

Evaluation:
  - F1 score on held-out 20%
  - Per-domain F1 (ensure no domain <0.70)
  - Cross-model test (separate Sonnet/Haiku/Gemini sets)

Expected performance: ~0.85-0.90 F1
```

**Phase 2: Context-Aware Classifier (Research)**
```
Training data: 531 session blocks with prior turns
  - + 18,600 baseline data (with prior=[] padding)

Additional features:
  - Prior N turns (concatenated, max 1000 tokens)
  - Turn count, avg turn length, role alternation
  - Project type (bioguard_classifier, llm_safety_eval, etc.)

Expected improvement: +0.03-0.05 F1 on session blocks
```

**Phase 3: Ambiguity-Aware Classifier (Future)**
```
Separate models for:
  1. Legitimate vs. Negative (binary, high confidence)
  2. Ambiguous detection (multi-class: benign_dual_use, ambiguous, unclear)
  3. Meta-refusal detection (continuation, tool_result, etc.)

Ensemble prediction: If ambiguous detected, flag for human review
```

### 9.3 Validation & Testing

**Cross-validation strategy:**
```
Stratified K-fold (5 splits) by:
  - Domain (ensure each fold has all domains)
  - Class (balance legitimate/negative)
  - Source (distribute OverRefusal, FRT, benchmarks, sessions)

Hold-out sets:
  - AmbiguityCasebook (36): Boundary cases
  - 10% Session blocks (70): Real user over-refusal edge cases
  - 1% per external benchmark: Source-specific eval
```

**Generalization tests:**
```
1. Per-model F1: Test on Sonnet, Haiku, Gemini, GPT subsets
2. Per-domain F1: Ensure each domain ≥0.70 F1
3. Out-of-domain: Test on new bio domains (gene therapy, epidemiology)
4. Temporal: If available, test on newer sessions (2026 Q2-Q3)
```

---

## 10. NEXT STEPS

1. **Data cleanup & curation (1 day)**
   - Filter meta-refusals (continuation, tool_result, session_context) into separate dataset
   - Spot-check 50 records per source for label quality
   - Verify 20 continuation_contamination records for CONTINUATION_RX accuracy

2. **Feature engineering (2 days)**
   - Implement stratified K-fold by domain+class+source
   - Build text feature extractors (query length, keyword presence, n-grams)
   - Encode missing fields as special tokens

3. **Baseline classifier training (2 days)**
   - Logistic regression + SVM baseline (FastText embeddings)
   - Fine-tune BERT/RoBERTa on query+trigger+domain features
   - Achieve 0.85+ F1 on held-out set

4. **Domain-specific evaluation (1 day)**
   - Per-domain F1 analysis
   - Identify weak domains (<0.70 F1)
   - Plan oversampling/ensemble strategies

5. **Cross-model generalization testing (1 day)**
   - Evaluate on Sonnet, Haiku, Gemini, GPT subsets
   - Report per-model F1 + confidence intervals

---

## Appendix: Source Composition

| Source | Records | Class Composition | Domains |
|--------|---------|-------------------|---------|
| OverRefusal API | 710 | 100% refuse signals (T1-T5 tier labels) | Mixed (9) |
| FRT P1 | 225 | 100% refusal | protein_engineering |
| FRT P2 | 11,330 | 100% refusal | protein_engineering |
| FRT P3 | 263 | 100% refusal | protein_engineering |
| Session Claude | 639 | 100% refusal + context | Mixed (14 projects) |
| Session Codex | 62 | 100% refusal | Mixed |
| ConstitutionRules P8 | 1,063 | Synthetic (assist/decline labels) | Mixed (6) |
| ConstitutionRules P11 FNR | 200 | 4 models × 50 hard-refuse | protein_engineering |
| alert_cbrn | 4,198 | Binary (harmful/safe) | CBRN |
| saladbench | 2,268 | Binary | CBRN |
| wildguard | 1,709 | Binary + refusal labels | Mixed |
| advbench | 541 | Binary | Mixed |
| BioSafetyProj DC | 34 | Decision route (allow/refuse/caveat) | Mixed (5) |
| BioSafetyProj Fixture | 52 | Observed routes (allow/refuse/caveat) | Mixed |
| AmbiguityCasebook | 36 | Expert recommendation + over_refusal flag | Boundary |

**Total:** 23,330 records across 14 sources

---

**Report generated:** 2026-05-28  
**Next review checkpoint:** After Phase 1 baseline classifier achieves 0.85 F1

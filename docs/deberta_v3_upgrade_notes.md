# DeBERTa-v3 Usage Hardening — Upgrade Notes

This document records the rationale, external references, and code-level
changes for the DeBERTa-v3 usage improvements applied in commit `c9a392a`
and subsequent work.

## Background

The Constitutional BioGuard classifier uses `microsoft/deberta-v3-base`
(He et al., 2023) for binary SAFE/UNSAFE classification of biological
dual-use content.  A code audit identified seven improvement areas in how
the model is *used* — the backbone itself remains appropriate for this
task.

## Improvement Summary

| # | Area | Status | Commit |
|---|------|--------|--------|
| 1 | Sentence-pair tokenization | **Done** | `c9a392a` |
| 2 | Sliding-window inference (>512 tok) | **Done** | *(this branch)* |
| 3 | Threshold calibration | **Done** (code; run after data regen) | `c9a392a` |
| 4 | Transformers version pinning | **Done** | `c9a392a` |
| 5 | Model config metadata | **Done** | `c9a392a` |
| 6 | Adversarial metrics correction | **Done** | `c9a392a` |
| 7 | Model variant experiments | Future | — |

---

## 1. Sentence-Pair Tokenization

### Problem

The original code concatenated query and response with a literal `[SEP]`
string:

```python
text = f"{example.query} [SEP] {example.response}"
tokenizer(text, ...)
```

With DeBERTa-v3's SentencePiece tokenizer this inserts the *characters*
`[`, `S`, `E`, `P`, `]` rather than the actual separator token (id 2).
Truncation also cannot distinguish query from response.

### Fix

Use the tokenizer's native sentence-pair interface:

```python
tokenizer(query, response, truncation=True, max_length=512)
```

This inserts the real `[SEP]` token and enables smarter truncation
(truncate the longer segment first by default).

### Files changed

| File | Change |
|------|--------|
| `training/prepare_data.py` | Store `query` + `response` fields alongside `text` |
| `training/train_deberta.py` | `tokenizer(batch["query"], batch["response"], ...)` |
| `evaluation/evaluate_classifier.py` | `predict_batch()` accepts `queries`/`responses` kwargs |
| `evaluation/external_validation.py` | `format_for_classifier()` returns `(query, response)` tuple |
| `evaluation/overrefusal_test.py` | Store and pass pairs |
| `scripts/serve.py` | `ClassifyRequest.to_pair()` + `_classify()` pair path |
| `scripts/evaluate_wmdp.py` | Pass `(question, choice)` as pair |
| `scripts/export_to_hf.py` | Usage example updated |

### References

- HuggingFace DeBERTa-v2 docs: the tokenizer automatically adds
  `[CLS] ... [SEP] ... [SEP]` when called with two arguments.
- DeBERTa-v3 paper (arXiv:2111.09543): disentangled attention uses
  relative position; `type_vocab_size=0` means no segment embeddings,
  but `[SEP]` tokens still provide structural boundary information.

---

## 2. Sliding-Window Inference for Long Inputs

### Problem

`max_position_embeddings=512`.  Responses longer than ~400 tokens
(after query + special tokens) are truncated — dangerous content at the
end of a long response is silently dropped.

### Fix

`predict_sliding()` and `predict_batch_sliding()` in
`evaluation/evaluate_classifier.py`:

1. Tokenize query → compute remaining budget for response.
2. If response fits: single-pass (no overhead).
3. If response exceeds budget: slide over response tokens with
   configurable `stride` (default 256).  Query is prepended to every
   window.
4. Aggregate via `max(p_unsafe)` (conservative; any window flagged →
   whole input flagged) or `mean(p_unsafe)`.

The `/classify` endpoint in `serve.py` exposes this via
`?sliding_window=true`.

### Design rationale

- `max` aggregation is the safe default for a biosafety classifier —
  false negatives (missing dangerous content) are worse than false
  positives.
- Stride of 256 (50% overlap) balances coverage vs. compute.
- Query is always present so the model has full context for each window.

---

## 3. Threshold Calibration

### Problem

Classification used a hardcoded 0.5 threshold (softmax argmax).  For a
biosafety classifier the optimal operating point depends on the
acceptable FPR/FNR trade-off.

### Fix

`calibrate_threshold()` in `evaluate_classifier.py`:

- Sweeps thresholds 0.10–0.90 (step 0.01) on the validation set.
- Records F1, precision, recall, and FPR at each point.
- Saves `calibration.json` in the model directory.
- Target metric is configurable (default: F1).

**Status**: Code ready.  Requires regenerated `data/processed/val.jsonl`
to actually run.  After running, the threshold can be loaded by serving
and evaluation code.

---

## 4. Transformers Version Pinning

### Problem

`pyproject.toml` specified `transformers>=4.40` but the code uses
`processing_class` (introduced in 4.46) and `eval_strategy` (renamed
in 4.36).

### Fix

Pinned to `transformers>=4.46`.

---

## 5. Model Config Metadata

### Problem

The saved model checkpoint had no `id2label` or `label2id` in
`config.json`.  HuggingFace `pipeline()` cannot infer label names.

### Fix

- Added to local `config.json`:
  `"id2label": {"0": "SAFE", "1": "UNSAFE"}, "label2id": {...}, "num_labels": 2`
- Updated `train_deberta.py` to pass these when loading the base model,
  so future training runs save them automatically.

---

## 6. Adversarial Metrics Correction

### Problem

README per-category ASR values did not match `adversarial_results.json`:

| Category | README (old) | JSON (actual) |
|----------|-------------|---------------|
| Character | 5.1% | **8.9%** |
| Encoding | 21.5% | **17.9%** |
| Semantic | 0.0% | **6.9%** |
| Multilingual | 0.0% | **1.0%** |
| Overall | 9.79% | 9.79% ✓ |

Key discrepancy: semantic attacks (passive_voice, negation_flip) each
had 20.8% ASR but were reported as 0%.

### Fix

Updated `README.md` table and description text, and `export_to_hf.py`
model card template.

---

## 7. Model Variant Experiments (Future)

Planned comparisons for future work:

| Model | Purpose |
|-------|---------|
| `microsoft/deberta-v3-large` | Performance ceiling |
| `microsoft/mdeberta-v3-base` | Multilingual biosafety |
| `microsoft/BiomedNLP-BiomedBERT-base` | Biomedical domain baseline |
| `dmis-lab/biobert-v1.1` | Clinical/bio NLP baseline |

---

## References

1. He, P., et al. (2023). DeBERTaV3: Improving DeBERTa using
   ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding
   Sharing. arXiv:2111.09543.
2. HuggingFace model card:
   https://huggingface.co/microsoft/deberta-v3-base
3. Anthropic (2025). Constitutional Classifiers (research blog).
4. NSABB (2024). Framework for Dual Use Research of Concern.

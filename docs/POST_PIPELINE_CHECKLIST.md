# Post-Pipeline Checklist

After `scripts/run_full_pipeline.sh` completes (2–4 hours), follow this workflow to evaluate and compare models.

> **⚠️ All numeric values in this document are illustrative placeholders, not
> measured results.** Example calibration thresholds, data-split counts, metric
> tables, and the variant-comparison table below were written before the runs
> they describe and have **not** been verified against actual outputs. The
> multi-variant comparison has not been executed. For real, verified metrics see
> `results/metrics/*.json` and the README "Results" section. Do not cite any
> number from this checklist as a finding.

## ✓ Pipeline Complete

When you see:
```
========================================
 Pipeline complete: Wed May 20 20:45:12 EDT 2026
========================================
```

### Immediate Next Step: Verify Calibration Results

```bash
# Check threshold calibration on new full dataset
cat models/deberta_bioguard_v1/calibration.json
```

Expected output (example):
```json
{
  "optimal_threshold": 0.58,
  "best_score": 0.924,
  "n_val_samples": 312,
  "thresholds_evaluated": [
    {"threshold": 0.10, "f1": 0.85, "precision": 0.90, "recall": 0.81, "fpr": 0.15},
    ...
    {"threshold": 0.58, "f1": 0.924, "precision": 0.918, "recall": 0.930, "fpr": 0.028}
  ]
}
```

- **optimal_threshold**: New decision boundary (should differ from 0.5)
- **best_score**: F1 on validation set (should improve with full data)
- **n_val_samples**: Should be ~312 (30% of ~1040 examples)
- **fpr**: False positive rate on safe content (~2–5% for good safety classifier)

---

## 1. Verify Data Integrity

```bash
# Count generated examples
echo "Train: $(wc -l < data/processed/train.jsonl)"
echo "Val:   $(wc -l < data/processed/val.jsonl)"
echo "Test:  $(wc -l < data/processed/test.jsonl)"
```

Expected: ~728 train, ~312 val, ~312 test (∑ ≈ 1352 examples)

Check class balance:
```bash
python3 << 'EOF'
import json

for split in ["train", "val", "test"]:
    with open(f"data/processed/{split}.jsonl") as f:
        lines = [json.loads(l) for l in f]
    safe = sum(1 for l in lines if l["label"] == 0)
    unsafe = sum(1 for l in lines if l["label"] == 1)
    print(f"{split:6} | Safe: {safe:3}  Unsafe: {unsafe:3}  (ratio: {safe/unsafe:.2f})")
EOF
```

Expected: ~50/50 split (balanced classes)

---

## 2. Review Baseline Metrics

The default model (deberta-base) has been trained and evaluated:

```bash
# Internal evaluation metrics
python3 << 'EOF'
import json
with open("models/deberta_bioguard_v1/training_metrics.json") as f:
    metrics = json.load(f)
    print(f"Test F1:    {metrics['test_f1']:.4f}")
    print(f"Test AUROC: {metrics['test_auroc']:.4f}")
    print(f"Test FPR:   {metrics['test_fpr']:.4f}")
EOF
```

Compare with previous run (stored in git history if available).

---

## 3. Run Full Variant Experiments

Train all 5 model variants (can take 2–3 hours depending on hardware):

```bash
python scripts/run_variant_experiment.py --all
```

Or run individual variants for faster iteration:
```bash
python scripts/run_variant_experiment.py --variant deberta-large   # ~30 min
python scripts/run_variant_experiment.py --variant biomedbert      # ~30 min
python scripts/run_variant_experiment.py --variant mdeberta        # ~30 min
python scripts/run_variant_experiment.py --variant biolinkbert     # ~30 min
```

### Track Progress

While variants train:
```bash
# Monitor in another terminal
while true; do
  clear
  echo "=== Variant Training Progress ==="
  if [ -f results/metrics/variant_comparison.json ]; then
    cat results/metrics/variant_comparison.json | jq '.results | length'
    cat results/metrics/variant_comparison.json | jq '.results[] | .variant'
  else
    echo "No results yet..."
  fi
  sleep 30
done
```

---

## 4. Compare All Variants

```bash
python scripts/run_variant_experiment.py --compare
```

Example output (**ILLUSTRATIVE FORMAT ONLY — these numbers are fabricated
placeholders, the variant comparison has not been run**):
```
==========================================================================================
 Model Variant Comparison   [SAMPLE LAYOUT — NOT REAL RESULTS]
==========================================================================================
Variant             F1      AUROC   Prec    Rec     FPR     Adv ASR   OR-FPR  Thresh
------------------------------------------------------------------------------------------
deberta-base        <tbd>   <tbd>   <tbd>   <tbd>   <tbd>    <tbd>     <tbd>  <tbd>
deberta-large       <tbd>   <tbd>   <tbd>   <tbd>   <tbd>    <tbd>     <tbd>  <tbd>
mdeberta            <tbd>   <tbd>   <tbd>   <tbd>   <tbd>    <tbd>     <tbd>  <tbd>
biomedbert          <tbd>   <tbd>   <tbd>   <tbd>   <tbd>    <tbd>     <tbd>  <tbd>
biolinkbert         <tbd>   <tbd>   <tbd>   <tbd>   <tbd>    <tbd>     <tbd>  <tbd>
==========================================================================================
```

### Decide on Final Model

**If deploying single model:**
- Highest F1 + lowest FPR → deberta-large
- Cost/speed critical → deberta-base
- Multilingual use case → mdeberta

**If building ensemble:** 
- Use deberta-large + biomedbert (complimentary strengths)
- See `ensemble_example.py` (to be created)

---

## 5. Re-run Adversarial & Overrefusal Tests

The pipeline runs these on the baseline model. For variants, they're included in step 4. But if you want standalone evaluation of the selected best model:

```bash
# Pick best variant (e.g., deberta-large)
MODEL_DIR="models/variant_deberta-large"

# Adversarial suite
python -c "
from constitutional_bioguard.evaluation.adversarial_suite import run_adversarial_suite
import json
results = run_adversarial_suite(model_dir='$MODEL_DIR')
with open('$MODEL_DIR/adversarial_results.json', 'w') as f:
    json.dump([r.__dict__ for r in results], f, indent=2)
print('Saved to $MODEL_DIR/adversarial_results.json')
"

# Overrefusal test
python -c "
from constitutional_bioguard.evaluation.overrefusal_test import run_overrefusal_test
result = run_overrefusal_test(model_dir='$MODEL_DIR')
print(f'Overrefusal FPR: {result[\"fpr\"]*100:.2f}%')
"
```

---

## 6. Export Best Model to HuggingFace

Once satisfied with performance:

```bash
# Export deberta-large as HuggingFace hub repo
python scripts/export_to_hf.py --variant deberta-large --repo-id constitutional-bioguard-v2
```

This:
- Uploads model to your HF account
- Includes model card with metrics
- Tags as `safeguard`, `biosafety`, `classifier`
- Enables: `pipeline('text-classification', model='you/constitutional-bioguard-v2')`

---

## 7. Document Results

Create a new section in `README.md` under "Results" with:
- Date of training
- Data statistics (train/val/test size, class balance)
- Baseline metrics (F1, AUROC, etc.)
- Best variant comparison table
- Threshold calibration curve
- Adversarial evaluation per category

Example template (**fill `<tbd>` with measured values — do not ship placeholders**):
```markdown
## Results (<date>)

### Data
- Train: <tbd> examples (<tbd> safe, <tbd> unsafe)
- Val: <tbd> examples (<tbd> safe, <tbd> unsafe)
- Test: <tbd> examples (<tbd> safe, <tbd> unsafe)
- Total: <tbd> examples across 56 bio-safety rules

### Baseline (deberta-v3-base)
| Metric | Score |
|--------|-------|
| F1 | <tbd> |
| AUROC | <tbd> |
| Precision | <tbd> |
| Recall | <tbd> |
| FPR | <tbd> |

### Best Variant: <tbd — variant comparison not yet run>
| Metric | Score |
|--------|-------|
| F1 | <tbd> |
| AUROC | <tbd> |
| Precision | <tbd> |
| Recall | <tbd> |
| FPR | <tbd> |
| Adversarial ASR | <tbd> |
| Overrefusal FPR | <tbd> |
| Optimal Threshold | <tbd> |
```

---

## 8. Commit Updated Results

Commit message template (**replace every `<tbd>` with measured values**):
```bash
git add -A && git commit -m "Data regeneration & evaluation (<date>)

- Regenerated synthetic dataset: <tbd> examples from 56 bio-safety rules
- Retrained baseline (deberta-base): F1=<tbd>, AUROC=<tbd>
- Calibration threshold: <tbd>
- (variant comparison: not run — future work)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Timeline Estimate

| Step | Est. Time | Status |
|------|-----------|--------|
| 1. Verify calibration | <1 min | ✓ Ready immediately |
| 2. Check data integrity | <1 min | ✓ Ready immediately |
| 3. Review baseline | <1 min | ✓ Ready immediately |
| 4. Run variants | 2–3 hours | Next |
| 5. Compare results | <1 min | After step 4 |
| 6. Re-eval best model | ~30 min | Optional |
| 7. Export to HF | ~5 min | Optional |
| 8. Update docs + commit | ~15 min | Final |

---

## Useful Commands Reference

```bash
# Check pipeline progress in real-time
tail -f pipeline.log

# Parse and show just the summary lines
grep "^\[" pipeline.log

# Count how many rules are fully generated
grep "Generated" pipeline.log | wc -l

# Show final metrics from complete pipeline
tail -50 pipeline.log | grep -E "F1|AUROC|threshold"

# View variant comparison as JSON
cat results/metrics/variant_comparison.json | jq '.results | sort_by(-.f1) | .[0]'

# Export comparison to CSV for spreadsheet
python3 << 'EOF'
import json, csv
with open("results/metrics/variant_comparison.json") as f:
    data = json.load(f)
with open("results/metrics/variant_comparison.csv", "w") as out:
    writer = csv.DictWriter(out, fieldnames=["variant", "f1", "auroc", "fpr", "adv_asr", "or_fpr"])
    writer.writeheader()
    for r in data["results"]:
        writer.writerow({
            "variant": r.get("variant"),
            "f1": r.get("internal", {}).get("f1"),
            "auroc": r.get("internal", {}).get("auroc"),
            "fpr": r.get("internal", {}).get("fpr"),
            "adv_asr": r.get("adversarial_mean_asr"),
            "or_fpr": r.get("overrefusal_fpr"),
        })
print("Saved to results/metrics/variant_comparison.csv")
EOF
```

---

## Troubleshooting

**Q: Calibration results look worse than before?**
A: With placeholder responses, separation was artificially good. Real data often has harder boundaries. If F1 dropped >10%, check data quality.

**Q: Variant training crashes with OOM?**
A: deberta-large on limited GPU. Use CPU: `CUDA_VISIBLE_DEVICES="" python scripts/run_variant_experiment.py --variant deberta-large`

**Q: Results not showing in comparison table?**
A: Results are merged into `variant_comparison.json`. Ensure model dir exists: `ls models/variant_*/`

**Q: Want to re-run single variant?**
A: Script overwrites by variant name: `python scripts/run_variant_experiment.py --variant deberta-large` replaces old result.

---

**Next**: Proceed to step 1 after pipeline completes. Estimate time to best-model-ready: **2–3 hours** (most in variant training).

# v3 Contingency Plans

Quick reference for follow-up experiments if v3 misses the success window.

---

## Success Criteria (recap from Section 6.10)

1. Cross-domain FAR < 10% on WildGuard, LAB-Bench, WMDP-Cyber, WMDP-Chem
2. BioThreat-Eval recall >= 25% at threshold 0.5
3. HarmBench/AdvBench held-out flag rate >= 50%

---

## Outcome Decision Tree

### Case 1: v3 meets all criteria (success)
- Commit results, update Section 6.10 with verdict
- Update README with v3 as the recommended model
- Move on to Section 7 / CC++ implications update
- Optional: train v3-final with held-out items folded back into train

### Case 2: v3 over-shoots (high recall + high FAR, like A_full)
**Symptom:** Bio recall recovered (>= 50% on held-out) but cross-domain
FAR returned to 30-70% range.

**Cause:** UNSAFE weight 2.0 was too aggressive. Model leaned back into
"bio vocab => UNSAFE" shortcut because UNSAFE class loss dominated.

**Fix (v3b):**
- Reduce UNSAFE weight to 1.5
- Keep same data split
- Rerun training (~50 min) + eval (~2h)

```bash
python scripts/experiments/train_v3_balanced.py --unsafe-weight 1.5
# Then update evaluate_v3_full.py to point to v3b model directory
```

### Case 3: v3 under-shoots (low recall + low FAR, like v2)
**Symptom:** Cross-domain FAR < 10% achieved, but bio recall still
< 25% on BioThreat-Eval and bio adversarial held-outs.

**Cause:** 70 UNSAFE augmentation items insufficient relative to ~1,500
synthetic UNSAFE items. Synthetic UNSAFE dominates the gradient signal.

**Fix (v3c):**
- Increase UNSAFE weight to 2.5 or 3.0
- Increase UNSAFE bio adversarial fraction (use all instead of held-out)
- Or: oversample UNSAFE bio adversarial items 3-5x via duplication

```bash
# Option A: just bump weight
python scripts/experiments/train_v3_balanced.py --unsafe-weight 2.5

# Option B: also use all bio adversarial for training (no held-out)
# Modify create_v3_splits.py UNSAFE_TRAIN_RATIO = 1.0 (currently 0.85)
```

### Case 4: v3 splits the difference but neither dominates
**Symptom:** Recall improves vs v2 but stays < 25%. FAR worsens vs v2
but stays < 15%.

**Cause:** The data-centric remedy works but is bandwidth-limited.

**Fix:** Two paths:
- **Cheap:** v3d with weight=2.5 + larger SAFE augmentation (~800) to
  better cover the cross-domain space.
- **Expensive but correct:** Regenerate synthetic data with diverse
  non-adversarial UNSAFE examples (revised 6.5). Current synthetic
  UNSAFE is dominated by adversarial-looking content; need balanced
  rewrite. ~1-2 days of generation + retraining.

---

## Logging for v3 follow-ups

If launching v3b/c/d, use these naming conventions:
- Models: `models/deberta_bioguard_v3{b,c,d}_balanced/`
- Splits (if changed): `data/external/v3{b,c,d}_splits/`
- Metrics: `results/metrics/v3{b,c,d}_eval_*.json`
- SLURM logs: `cayuga_v3{b,c,d}_*.log`

---

## Failure Mode That Requires Stop-the-Line

If v3 + v3b + v3c all fail to land in the success window, the
hypothesis "data-centric fix via small bio adversarial augmentation"
is wrong. Switch to:

1. **Regenerate synthetic UNSAFE data** with diverse non-adversarial
   style (4-6 hour Claude API call, ~3,000 items).
2. **Retrain from the ground up** with the regenerated data.
3. If even regenerated data fails on cross-domain OOD, the
   exchange-classifier-on-synthetic-data approach has a fundamental
   ceiling, and the next move is to step back to constitutional
   prompting or larger base models.

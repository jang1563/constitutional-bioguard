#!/usr/bin/env bash
# Full data generation + calibration pipeline.
# Resumes from where it left off (synthetic_generator uses resume=True).
#
# Usage:
#   source ~/.api_keys && bash scripts/run_full_pipeline.sh
#
# Or in the background:
#   source ~/.api_keys && nohup bash scripts/run_full_pipeline.sh > pipeline.log 2>&1 &
#
# Estimated time: 2-4 hours (depending on API response times)

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Run: source ~/.api_keys"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
PYTHON="${PYTHON:-python3}"

echo "========================================"
echo " Constitutional BioGuard Full Pipeline"
echo " Started: $(date)"
echo "========================================"

# Step 1: Generate synthetic data (resumes from existing files)
echo ""
echo "[1/6] Generating synthetic data..."
$PYTHON scripts/run_pipeline.py --step generate-synthetic -v

# Step 2: Augment restricted/boundary examples
echo ""
echo "[2/6] Augmenting examples..."
$PYTHON scripts/run_pipeline.py --step augment -v

# Step 3: Generate benign queries
echo ""
echo "[3/6] Generating benign queries..."
$PYTHON scripts/run_pipeline.py --step benign -v

# Step 4: Prepare train/val/test splits
echo ""
echo "[4/6] Preparing data splits..."
$PYTHON scripts/run_pipeline.py --step prepare -v

# Step 5: Train baseline classifier on the regenerated splits
echo ""
echo "[5/7] Training baseline classifier..."
$PYTHON scripts/run_pipeline.py --step train --model deberta -v

# Step 6: Run threshold calibration
echo ""
echo "[6/7] Running threshold calibration..."
$PYTHON -c "
import logging
logging.basicConfig(level=logging.INFO)
from constitutional_bioguard.evaluation.evaluate_classifier import calibrate_threshold
result = calibrate_threshold()
print(f'Optimal threshold: {result[\"optimal_threshold\"]}')
print(f'Best F1: {result[\"best_score\"]}')
print(f'N samples: {result[\"n_val_samples\"]}')
"

# Step 7: Quick sanity — run internal evaluation
echo ""
echo "[7/7] Running internal evaluation..."
$PYTHON scripts/run_pipeline.py --step evaluate -v

echo ""
echo "========================================"
echo " Pipeline complete: $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - Review calibration: cat models/deberta_bioguard_v1/calibration.json"
echo "  - Run adversarial:    python scripts/run_pipeline.py --step adversarial"
echo "  - Run overrefusal:    python scripts/run_pipeline.py --step overrefusal"

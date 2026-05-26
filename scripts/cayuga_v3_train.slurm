#!/bin/bash
#SBATCH --job-name=bioguard-v3-train
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v3_train_%j.out
#SBATCH --error=logs/v3_train_%j.err

# v3 balanced training: reduced SAFE + UNSAFE bio adversarial + weight boost
# Two-step: (1) create splits, (2) train

set -euo pipefail

PYTHON=${PYTHON:-python}
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT"
mkdir -p logs

echo "=== v3 Training Pipeline ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Step 1: Create v3 splits
echo "--- Step 1: Create v3 splits ---"
$PYTHON scripts/create_v3_splits.py
echo ""

# Step 2: Train v3 with UNSAFE weight=2.0
echo "--- Step 2: Train v3 (weight_1=2.0) ---"
$PYTHON scripts/train_v3_balanced.py --unsafe-weight 2.0
echo ""

echo "=== v3 Training Complete ==="
echo "Date: $(date)"

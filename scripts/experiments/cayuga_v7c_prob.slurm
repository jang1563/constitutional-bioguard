#!/bin/bash
#SBATCH --job-name=v7c-prob
#SBATCH --output=v7c_prob_%j.log
#SBATCH --error=v7c_prob_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#
# Step 1: v7.C token-probability operating curve (recall vs over-refusal ROC).
# No retraining; reads p(harmful) and sweeps a threshold.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v7.C probability operating-curve sweep ==="
$PYTHON scripts/experiments/score_v7c_probs.py

echo ""
echo "=== DONE ==="

#!/bin/bash
#SBATCH --job-name=bg-v6-f4
#SBATCH --output=cayuga_v6_f4_%j.log
#SBATCH --error=cayuga_v6_f4_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== F.4 v4 head-only refit ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

# Include v3 train subset for stability — encoder is frozen so risk is bounded
$PYTHON scripts/experiments/v6_f4_train_head.py --include-v3-train --v3-train-subset 500 \
    --epochs 5 --lr 5e-4 --unsafe-weight 1.5

echo "Complete: $(date)"

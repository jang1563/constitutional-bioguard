#!/bin/bash
#SBATCH --job-name=bioguard-v5-base
#SBATCH --output=cayuga_v5_baseline_%j.log
#SBATCH --error=cayuga_v5_baseline_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v5_baseline TRAINING ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

$PYTHON scripts/train_v5_baseline.py --unsafe-weight 1.5

echo ""
echo "=== Complete: $(date) ==="

#!/bin/bash
#SBATCH --job-name=bg-v5b
#SBATCH --output=cayuga_v5b_paircfr_%j.log
#SBATCH --error=cayuga_v5b_paircfr_%j.log
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

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v5b PairCFR (lambda=0.1) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

$PYTHON scripts/train_v5.py \
    --unsafe-weight 1.5 \
    --paircfr-lambda 0.1 \
    --paircfr-temperature 0.1 \
    --output-dir models/deberta_bioguard_v5b

echo ""
echo "=== Complete: $(date) ==="

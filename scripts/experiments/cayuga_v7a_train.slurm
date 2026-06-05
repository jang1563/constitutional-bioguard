#!/bin/bash
#SBATCH --job-name=bg-v7a
#SBATCH --output=cayuga_v7a_%j.log
#SBATCH --error=cayuga_v7a_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v7.A BioClinical ModernBERT-large training ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

$PYTHON scripts/experiments/train_v7a_bioclinical.py \
    --epochs 5 --lr 1e-5 --batch-size 16 --grad-accum 1 \
    --max-seq 1024 --unsafe-weight 1.5

echo "=== Complete: $(date) ==="

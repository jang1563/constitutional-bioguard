#!/bin/bash
#SBATCH --job-name=bioguard-v8-baseline
#SBATCH --output=logs/cayuga_v8_baseline_%j.log
#SBATCH --error=logs/cayuga_v8_baseline_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python3}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

mkdir -p logs results models

echo "======================================================================"
echo "Phase 3 Baseline — constitutional_bioguard v8"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"

echo ""
echo "--- Data splits ---"
wc -l data/splits/train.jsonl data/splits/val.jsonl \
       data/splits/ood_fpr.jsonl data/splits/ood_fnr.jsonl

echo ""
echo "--- Training ---"
$PYTHON scripts/experiments/train_v8_baseline.py \
    --unsafe-weight 1.5 \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5 \
    --model-name microsoft/deberta-v3-base \
    --output-dir models/deberta_bioguard_v8_baseline

echo ""
echo "======================================================================"
echo "Complete: $(date)"
echo "======================================================================"

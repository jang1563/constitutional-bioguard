#!/bin/bash
#SBATCH --job-name=bg-v8b-qo
#SBATCH --output=logs/cayuga_v8b_query_only_%j.log
#SBATCH --error=logs/cayuga_v8b_query_only_%j.log
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

source $HOME/miniconda3/miniconda3/bin/activate bioguard-hpc
PYTHON=${PYTHON:-python}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

mkdir -p logs results models

echo "======================================================================"
echo "Phase 3B Ablation — query-only (no response)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"

$PYTHON scripts/experiments/train_v8b_query_only.py \
    --unsafe-weight 1.5 \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5 \
    --model-name microsoft/deberta-v3-base \
    --output-dir models/deberta_bioguard_v8b_query_only

echo ""
echo "======================================================================"
echo "Complete: $(date)"
echo "======================================================================"

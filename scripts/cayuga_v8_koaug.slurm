#!/bin/bash
#SBATCH --job-name=bioguard-v8-koaug
#SBATCH --output=logs/cayuga_v8_koaug_%j.log
#SBATCH --error=logs/cayuga_v8_koaug_%j.log
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
echo "Phase 3 KO-aug retrain — constitutional_bioguard v8_koaug"
echo "Korean-augmented train (held-out-safe: translated from TRAIN, not session)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"

echo ""
echo "--- Data splits ---"
wc -l data/splits/train_ko_aug.jsonl data/splits/val.jsonl \
       data/splits/ood_fpr.jsonl data/splits/ood_fnr.jsonl

echo ""
echo "--- Training (same hyperparams as v8 baseline; only the train data changed) ---"
$PYTHON scripts/train_v8_baseline.py \
    --train-file data/splits/train_ko_aug.jsonl \
    --results-name phase3_koaug_eval.json \
    --unsafe-weight 1.5 \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5 \
    --model-name microsoft/deberta-v3-base \
    --output-dir models/deberta_bioguard_v8_koaug

echo ""
echo "======================================================================"
echo "Complete: $(date)"
echo "======================================================================"

#!/bin/bash
#SBATCH --job-name=bioguard-v8-koaug3
#SBATCH --output=logs/cayuga_v8_koaug3_%j.log
#SBATCH --error=logs/cayuga_v8_koaug3_%j.log
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
echo "Phase 3 KO-aug3 retrain — constitutional_bioguard v8_koaug3"
echo "train_ko_aug3 = train + long KO + short KO + fixture_balance KO"
echo "Fixture balance: diverse short benign (not conversational stubs)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"

echo ""
echo "--- Data splits ---"
wc -l data/splits/train_ko_aug3.jsonl data/splits/val.jsonl \
       data/splits/ood_fpr.jsonl data/splits/ood_fnr.jsonl

echo ""
echo "--- Training (same hyperparams as v8 baseline; only data changed) ---"
$PYTHON scripts/train_v8_baseline.py \
    --train-file data/splits/train_ko_aug3.jsonl \
    --results-name phase3_koaug3_eval.json \
    --unsafe-weight 1.5 \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5 \
    --model-name microsoft/deberta-v3-base \
    --output-dir models/deberta_bioguard_v8_koaug3

echo ""
echo "--- KO=safe shortcut probe (EN vs KO flag-rate on harmful) ---"
$PYTHON scripts/eval_ko_harmful_probe.py \
    --model-dir models/deberta_bioguard_v8_koaug3/final \
    --probe data/audit/ko_harmful_probe.jsonl \
    --out results/ko_harmful_probe_koaug3.json || echo "(probe eval failed; run manually)"

echo ""
echo "--- FP diagnostic on ood_fpr (fixture + fixture_balance breakdown) ---"
$PYTHON scripts/diag_v8_fp_ood_fpr.py \
    --model-dir models/deberta_bioguard_v8_koaug3/final \
    --data data/splits/ood_fpr.jsonl \
    --out results/diag_v8_koaug3_ood_fpr.json || echo "(diag failed; run manually)"

echo ""
echo "======================================================================"
echo "Complete: $(date)"
echo "======================================================================"

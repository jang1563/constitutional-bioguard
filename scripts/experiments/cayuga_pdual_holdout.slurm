#!/bin/bash
#SBATCH --job-name=pdual-holdout-ood
#SBATCH --output=pdual_holdout_%j.log
#SBATCH --error=pdual_holdout_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Phase R Part 2: source-held-out OOD validation.
# 1) build pdual data with saladbench HELD OUT of training positives
# 2) retrain deberta_pdual_ho (same recipe: deberta-v3-base, 5 epochs)
# 3) eval recall on held-out saladbench (OOD) vs original pdual_v3 (in-dist)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== [1/3] build held-out data ==="
$PYTHON scripts/experiments/build_pdual_holdout.py --holdout saladbench_cbrn.jsonl

echo ""
echo "=== [2/3] retrain pdual without saladbench ==="
$PYTHON - <<'PY'
from pathlib import Path
from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR
from constitutional_bioguard.training.train_deberta import train
res = train(
    train_file=DATA_PROCESSED / "pdual_train_ho.jsonl",
    val_file=DATA_PROCESSED / "pdual_val_ho.jsonl",
    output_dir=MODELS_DIR / "deberta_pdual_ho",
    config_override={"training": {"num_train_epochs": 5}},
)
print("train done:", {k: res.get(k) for k in ("eval_metrics",)})
PY

echo ""
echo "=== [3/3] OOD recall: in-dist vs held-out ==="
$PYTHON scripts/experiments/eval_pdual_ood.py

echo ""
echo "=== DONE ==="

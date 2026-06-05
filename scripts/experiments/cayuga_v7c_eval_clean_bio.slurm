#!/bin/bash
#SBATCH --job-name=v7c-clean-bio
#SBATCH --output=v7c_clean_bio_%j.log
#SBATCH --error=v7c_clean_bio_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#
# Phase H: v7.C (Llama-3.1-8B + QLoRA) on clean bio eval
# Same test set as pdual_ho (saladbench-held-out 41 genuine-bio)
# Measure prompt-harm recall vs baseline 0.366

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v7.C Llama-3.1-8B eval on clean bio benchmark ==="
$PYTHON scripts/experiments/eval_v7c_clean_bio.py

echo ""
echo "=== DONE ==="

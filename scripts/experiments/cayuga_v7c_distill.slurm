#!/bin/bash
#SBATCH --job-name=v7c-distill
#SBATCH --output=v7c_distill_%j.log
#SBATCH --error=v7c_distill_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)"; export HF_HOME="${HOME}/.cache/huggingface"
export V7C_ADAPTER_DIR="$(pwd)/models/llama31_8b_bioguard_v7c_aug2/adapter"
echo "=== Step 1: soft-label distill pool with v7.C-aug2 teacher ==="
$PYTHON scripts/experiments/score_distill_pool.py
echo ""; echo "=== Step 1: train DeBERTa student + eval (fork decision) ==="
$PYTHON scripts/experiments/train_v7c_distill.py
echo "=== DONE ==="

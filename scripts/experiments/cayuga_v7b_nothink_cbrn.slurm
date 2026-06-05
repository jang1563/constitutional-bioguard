#!/bin/bash
#SBATCH --job-name=bg-v7b-ntcbrn
#SBATCH --output=cayuga_v7b_nothink_cbrn_%j.log
#SBATCH --error=cayuga_v7b_nothink_cbrn_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

# DECISIVE EXPERIMENT: does v7.B /no_think preserve CBRN recall >= 0.70?
# v7.B /no_think already clears the FPR gates (XSTest 0.072, OR-Bench 0.247).
# /think standalone CBRN recall: SaladBench 0.899, ALERT 0.890.
# /no_think dropped SimpleSafety-Bio recall 0.85->0.75 (-10pt); need to confirm
# the two large CBRN sets stay >= 0.70 so v7.B-no_think can ship standalone
# (the cascade is strictly dominated: v7.A gates out ~50-56% of CBRN harms at prob<0.05).

echo "=== v7.B /no_think CBRN-recall confirmation ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

$PYTHON scripts/experiments/eval_v7b_qwen3_cot.py \
    --no-think \
    --benchmarks saladbench_cbrn alert_cbrn

echo ""
echo "=== v7.B /no_think CBRN-recall confirmation complete: $(date) ==="

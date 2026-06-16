#!/bin/bash
#SBATCH --job-name=bg-v7b-noth
#SBATCH --output=cayuga_v7b_nothink_%j.log
#SBATCH --error=cayuga_v7b_nothink_%j.log
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

PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

# Diagnostic: does removing CoT (/no_think) cut v7.B over-refusal?
# Compare against existing /think baseline (v7b_eval_summary.json):
#   XSTest FPR 0.504, OR-Bench flag 73.5%, FalseReject flag 81.9%, SimpleSafety 85%.
# Over-refusal benchmarks + one harmful set (SimpleSafety-Bio) to confirm
# recall isn't destroyed. unknown_rate guards against truncation confound.

echo "=== v7.B /no_think diagnostic ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

$PYTHON scripts/experiments/eval_v7b_qwen3_cot.py \
    --no-think \
    --benchmarks or_bench_hard_1k false_reject_test xstest simple_safety_bio

echo ""
echo "=== v7.B /no_think diagnostic complete: $(date) ==="

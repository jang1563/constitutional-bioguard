#!/bin/bash
#SBATCH --job-name=bioguard-v2-eval
#SBATCH --output=cayuga_v2_eval_%j.log
#SBATCH --error=cayuga_v2_eval_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# v2 Full Evaluation across 8 external benchmarks.
# Uses GPU (CUDA-enabled torch) for fast inference.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

echo "=== BioGuard v2 Full Evaluation ==="
echo "Node: $(hostname)   Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo ""

${PYTHON} scripts/experiments/evaluate_v2_full.py

echo ""
echo "=== Complete: $(date) ==="
ls -la results/metrics/v2_eval_*.json 2>/dev/null | head -15

#!/bin/bash
#SBATCH --job-name=bioguard-compare
#SBATCH --output=cayuga_compare_%j.log
#SBATCH --error=cayuga_compare_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Apples-to-apples comparison: A_full vs v2 on identical held-out splits.
# Runs both models through 8 benchmarks. GPU for fast inference.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== Apples-to-Apples Comparison ==="
echo "Node: $(hostname)   Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo ""

${PYTHON} scripts/evaluate_models_compare.py

echo ""
echo "=== Complete: $(date) ==="
ls -la results/metrics/compare_*.json 2>/dev/null | head

#!/bin/bash
#SBATCH --job-name=bioguard-g2
#SBATCH --output=cayuga_g2_refusal_bypass_%j.log
#SBATCH --error=cayuga_g2_refusal_bypass_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== G.2 Refusal-Prefix Bypass ==="
echo "Started: $(date)"

echo "--- v3 ---"
$PYTHON scripts/g2_refusal_prefix_bypass.py v3

echo ""
echo "--- v4 ---"
$PYTHON scripts/g2_refusal_prefix_bypass.py v4

echo ""
echo "Complete: $(date)"

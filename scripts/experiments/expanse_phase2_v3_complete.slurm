#!/bin/bash
#SBATCH --job-name=bioguard-p2-v3
#SBATCH --output=expanse_phase2_v3_%j.log
#SBATCH --error=expanse_phase2_v3_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# v3 Phase 2 on Expanse (Cayuga was QOS-blocked).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/biothreat_eval}"

echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
echo ""

$PYTHON scripts/experiments/evaluate_phase2.py --model v3

echo "Complete: $(date)"

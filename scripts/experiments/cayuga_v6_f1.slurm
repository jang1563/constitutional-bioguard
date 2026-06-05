#!/bin/bash
#SBATCH --job-name=bg-v6-f1
#SBATCH --output=cayuga_v6_f1_%j.log
#SBATCH --error=cayuga_v6_f1_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== F.1 v4 re-measure on WildGuard (response_harm_label) ==="
echo "Started: $(date)"
$PYTHON scripts/experiments/v6_e1_remeasure_wildguard.py
echo "Complete: $(date)"

#!/bin/bash
#SBATCH --job-name=bioguard-v3-ccpp
#SBATCH --output=cayuga_v3_ccpp_%j.log
#SBATCH --error=cayuga_v3_ccpp_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# v3 CC++ re-analysis: WS-1 cascade calibration + WS-4 adversarial suite

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v3 CC++ RE-ANALYSIS ==="
echo "Node: $(hostname)   Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

$PYTHON scripts/run_v3_ccpp_reanalysis.py

echo ""
echo "=== Complete: $(date) ==="

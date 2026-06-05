#!/bin/bash
#SBATCH --job-name=bioguard-v5-preprobe
#SBATCH --output=cayuga_v5_preprobe_%j.log
#SBATCH --error=cayuga_v5_preprobe_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

echo "=== v5 Probe Pre-registration (v4 baseline values) ==="
echo "Started: $(date)"

$PYTHON scripts/experiments/v5_probe_preregister.py

echo "=== Complete: $(date) ==="

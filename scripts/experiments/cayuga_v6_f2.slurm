#!/bin/bash
#SBATCH --job-name=bg-v6-f2-splice
#SBATCH --output=cayuga_v6_f2_%j.log
#SBATCH --error=cayuga_v6_f2_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== F.2 SPLICE projector fit ==="
echo "Started: $(date)"
$PYTHON scripts/experiments/v6_f2_fit_splice.py
echo "Complete: $(date)"

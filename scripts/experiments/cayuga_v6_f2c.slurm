#!/bin/bash
#SBATCH --job-name=bg-v6-f2c
#SBATCH --output=cayuga_v6_f2c_%j.log
#SBATCH --error=cayuga_v6_f2c_%j.log
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
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd):$(pwd)/scripts"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== F.2c v4+SPLICE behavioral eval ==="
echo "Started: $(date)"
$PYTHON scripts/experiments/v6_f2c_splice_behavioral.py
echo "Complete: $(date)"

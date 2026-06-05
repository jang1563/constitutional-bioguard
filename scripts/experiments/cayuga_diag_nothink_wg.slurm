#!/bin/bash
#SBATCH --job-name=bg-v7b-autopsy
#SBATCH --output=cayuga_diag_nothink_wg_%j.log
#SBATCH --error=cayuga_diag_nothink_wg_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

echo "=== v7.B /no_think WildGuard autopsy (truncation vs off-task) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

$PYTHON scripts/experiments/diag_v7b_nothink_wildguard.py

echo ""
echo "=== autopsy complete: $(date) ==="

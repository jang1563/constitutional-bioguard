#!/bin/bash
#SBATCH --job-name=v7c-smoke
#SBATCH --output=v7c_smoke_%j.log
#SBATCH --error=v7c_smoke_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#
# Smoke test: verify v7.C prompt format + parser on a few samples before the
# full ~1h eval. Prints raw generations under both response modes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v7.C smoke test (format + parser verification) ==="
$PYTHON scripts/smoke_v7c.py

echo ""
echo "=== DONE ==="

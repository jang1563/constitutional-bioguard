#!/bin/bash
#SBATCH --job-name=bg-casc-noth
#SBATCH --output=cayuga_cascade_nothink_%j.log
#SBATCH --error=cayuga_cascade_nothink_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

# Cascade v7.A -> v7.B with stage-2 in /no_think mode.
# Diagnostic confirmed /no_think cuts v7.B borderline FPR 0.504->0.072 on XSTest.
# v7.A handles confident harms (recall); no_think v7.B adjudicates borderline (FPR).
TH_LOW="${CASCADE_TH_LOW:-0.2}"
TH_HIGH="${CASCADE_TH_HIGH:-0.8}"

echo "=== Cascade v7.A->v7.B (no_think stage-2) eval ==="
echo "Started: $(date)"
echo "Thresholds: low=${TH_LOW}, high=${TH_HIGH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

$PYTHON scripts/experiments/cascade_v7a_v7b.py \
    --v7b-no-think \
    --threshold-low "${TH_LOW}" \
    --threshold-high "${TH_HIGH}"

echo ""
echo "=== Cascade (no_think) eval complete: $(date) ==="

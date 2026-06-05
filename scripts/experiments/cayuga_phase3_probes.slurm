#!/bin/bash
#SBATCH --job-name=bioguard-p3-probes
#SBATCH --output=cayuga_phase3_probes_%j.log
#SBATCH --error=cayuga_phase3_probes_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Phase 3 probes B.2.1 (CRT) + B.2.2 (lexical ablation) on v3.
# Cheap (v3 only, ~200 + ~250 inferences total).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

echo "Started: $(date)"

echo "=== B.2.1 Counterfactual Response Test ==="
$PYTHON scripts/experiments/probe_b21_crt.py

echo ""
echo "=== B.2.2 Lexical Ablation ==="
$PYTHON scripts/experiments/probe_b22_lexical_ablation.py

echo "Complete: $(date)"

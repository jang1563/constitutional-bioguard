#!/bin/bash
#SBATCH --job-name=bioguard-6.8
#SBATCH --output=cayuga_6_8_%j.log
#SBATCH --error=cayuga_6_8_%j.log
#SBATCH --partition=scu-cpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#
# Experiment 6.8: WildGuardMix cross-domain adversarial OOD
# Tests whether bio-specific classifier falsely flags non-bio adversarial content.
#
# Pre-requisites:
#   1. WildGuardTest cache at ~/constitutional-bioguard/data/external/wildguard_test.jsonl
#      (gated dataset; cache prepared on machine with HF access agreement)
#   2. Both model variants at ~/constitutional-bioguard/models/
#   3. scipy available in conda env (for Wilson interval, z-test)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL_A=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_A_full
MODEL_B=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_B_bowhard
DATA_CACHE=${HOME}/constitutional-bioguard/data/external/wildguard_test.jsonl

echo "=== BioGuard Experiment 6.8: WildGuardMix adversarial OOD ==="
echo "Node: $(hostname)   Started: $(date)"
echo "Python: ${PYTHON}"
echo "Cache: ${DATA_CACHE}"

if [ ! -f "${DATA_CACHE}" ]; then
    echo "ERROR: WildGuardTest cache not found at ${DATA_CACHE}"
    echo "scp data/external/wildguard_test.jsonl to Cayuga first."
    exit 1
fi

if [ ! -f "${MODEL_A}/config.json" ]; then
    echo "ERROR: A_full model not found at ${MODEL_A}"
    exit 1
fi

ARGS="--experiment 6.8 --model-a ${MODEL_A}"
if [ -d "${MODEL_B}" ]; then
    ARGS="${ARGS} --model-b ${MODEL_B}"
else
    echo "WARNING: B_bowhard not found. Running A_full only."
fi

echo ""
echo "Running: ${PYTHON} scripts/run_corrective.py ${ARGS}"
echo ""

${PYTHON} scripts/run_corrective.py ${ARGS}

echo ""
echo "=== Complete: $(date) ==="
echo ""
ls -la results/metrics/corrective_6_8*.json 2>/dev/null || echo "No 6.8 results found"

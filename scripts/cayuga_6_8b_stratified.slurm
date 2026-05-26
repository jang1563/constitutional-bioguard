#!/bin/bash
#SBATCH --job-name=bioguard-6.8b
#SBATCH --output=cayuga_6_8b_%j.log
#SBATCH --error=cayuga_6_8b_%j.log
#SBATCH --partition=scu-cpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#
# Experiment 6.8b: Stratified diagnosis of WildGuardMix false alarms.
# Re-runs A_full only (skip B_bowhard) with per-item probability saving.
# Computes length/refusal/subcategory stratified analyses to distinguish
# framing-reflex from length/style/refusal alternative explanations.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL_A=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_A_full

echo "=== BioGuard Experiment 6.8b: Stratified Diagnosis ==="
echo "Node: $(hostname)   Started: $(date)"
echo "Python: ${PYTHON}"

for f in "${HOME}/constitutional-bioguard/data/external/wildguard_test.jsonl" "${MODEL_A}/config.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

echo ""
echo "Running: ${PYTHON} scripts/run_corrective.py --experiment 6.8b --model-a ${MODEL_A}"
echo ""

${PYTHON} scripts/run_corrective.py --experiment 6.8b --model-a ${MODEL_A}

echo ""
echo "=== Complete: $(date) ==="
ls -la results/metrics/corrective_6_8b*.json 2>/dev/null || echo "No 6.8b results found"

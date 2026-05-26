#!/bin/bash
#SBATCH --job-name=bioguard-6.7
#SBATCH --output=cayuga_6_7_%j.log
#SBATCH --error=cayuga_6_7_%j.log
#SBATCH --partition=scu-cpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Experiment 6.7: OOD evaluation on BioThreat-Eval
# Complements 6.1 (WMDP-Bio MCQ) with expert-labeled real (query, response) pairs.
# Evaluates 3 labeling strategies (TL>=4, TL>=3, response-based) to address
# the "binary label is too coarse" concern surfaced after 6.1.
#
# Pre-requisites:
#   1. BioThreat-Eval data at ${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}
#   2. Both model variants at ~/constitutional-bioguard/models/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Direct path to conda env python (module system unavailable in SLURM)
PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

# Model paths
MODEL_A=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_A_full
MODEL_B=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_B_bowhard

echo "=== BioGuard Experiment 6.7: BioThreat-Eval OOD ==="
echo "Node: $(hostname)   Started: $(date)"
echo "Python: ${PYTHON}"
echo "BioThreat-Eval: ${BIOTHREAT_EVAL_DIR}"

for f in "${BIOTHREAT_EVAL_DIR}/data/raw/query_bank.json" "${MODEL_A}/config.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

ARGS="--experiment 6.7 --model-a ${MODEL_A}"
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
echo "=== Results ==="
ls -la results/metrics/corrective_6_7*.json 2>/dev/null || echo "No 6.7 results found"

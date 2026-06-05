#!/bin/bash
#SBATCH --job-name=bioguard-6.2
#SBATCH --output=cayuga_6_2_%j.log
#SBATCH --error=cayuga_6_2_%j.log
#SBATCH --partition=scu-cpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Experiment 6.2: Bootstrap CIs for WS-2 kappa
# Runs on CPU -- DeBERTa inference on 558 examples + 10K bootstrap iterations
#
# Pre-requisites:
#   1. BioThreat-Eval data at ${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}
#   2. Both model variants at ~/constitutional-bioguard/models/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Direct path to conda env python (module system unavailable in SLURM)
CONDA=/opt/ohpc/pub/software/anaconda3/2023.09-0/bin/conda
PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

# Model paths
MODEL_A=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_A_full
MODEL_B=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_B_bowhard

echo "=== BioGuard Experiment 6.2: Bootstrap Kappa CI ==="
echo "Node: $(hostname)   Started: $(date)"
echo "Python: ${PYTHON}"
echo "BioThreat-Eval: ${BIOTHREAT_EVAL_DIR}"

# Verify pre-requisites
for f in "${BIOTHREAT_EVAL_DIR}/data/raw/query_bank.json" "${MODEL_A}/config.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

ARGS="--experiment 6.2 --model-a ${MODEL_A}"
if [ -d "${MODEL_B}" ]; then
    ARGS="${ARGS} --model-b ${MODEL_B}"
else
    echo "WARNING: B_bowhard not found. Running A_full only."
fi

echo ""
echo "Running: ${PYTHON} scripts/experiments/run_corrective.py ${ARGS}"
echo ""

${PYTHON} scripts/experiments/run_corrective.py ${ARGS}

echo ""
echo "=== Complete: $(date) ==="
echo ""
echo "=== Results ==="
ls -la results/metrics/corrective_6_2*.json 2>/dev/null || echo "No 6.2 results found"

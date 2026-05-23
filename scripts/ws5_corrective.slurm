#!/bin/bash
#SBATCH --job-name=bioguard-ws5
#SBATCH --output=ws5_corrective_%j.log
#SBATCH --error=ws5_corrective_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --account=crl195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# WS-5 Corrective experiments (Section 6 of technical report)
#
# Three experiments that break the synthetic data ceiling:
#   6.1 OOD eval on WMDP-Bio (~5K inferences, ~5 min)
#   6.2 Bootstrap CI for kappa (~1K inferences + CPU bootstrap, ~5 min)
#   6.3 Pre-/post-preprocessing adversarial comparison (~2.6K inferences x2, ~15 min)
#
# Pre-requisites (run on login node before submitting):
#   1. Download WMDP-Bio data:
#      python scripts/run_corrective.py --cache-wmdp
#   2. Ensure BioThreat-Eval data is accessible at:
#      /home/jkim61/New_Science/biothreat_eval/
#      (or set BIOTHREAT_EVAL_DIR env var)
#   3. Ensure both model variants exist:
#      /home/jkim61/constitutional-bioguard/models/deberta_bioguard_v1_A_full
#      /home/jkim61/constitutional-bioguard/models/deberta_bioguard_v1_B_bowhard

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/jkim61/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Model paths
MODEL_A=/home/jkim61/constitutional-bioguard/models/deberta_bioguard_v1_A_full
MODEL_B=/home/jkim61/constitutional-bioguard/models/deberta_bioguard_v1_B_bowhard

echo "=== BioGuard WS-5: Corrective Experiments ==="
echo "Node: $(hostname)   Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Verify data pre-requisites
if [ ! -f "data/external/wmdp_bio.jsonl" ]; then
    echo "WARNING: WMDP-Bio cache not found. Will attempt HuggingFace download."
    echo "For air-gapped nodes, run on login node first:"
    echo "  python scripts/run_corrective.py --cache-wmdp"
fi

# Build experiment arguments
ARGS="--model-a ${MODEL_A}"
if [ -d "${MODEL_B}" ]; then
    ARGS="${ARGS} --model-b ${MODEL_B}"
else
    echo "WARNING: B_bowhard model not found at ${MODEL_B}. Skipping 6.2 B variant."
fi

# Optional: skip specific experiments
# ARGS="${ARGS} --skip 6.2"

echo ""
echo "Running: python scripts/run_corrective.py ${ARGS}"
echo ""

${PYTHON} scripts/run_corrective.py ${ARGS}

echo ""
echo "=== Complete: $(date) ==="

# Print results summary
echo ""
echo "=== Results files ==="
ls -la results/metrics/corrective_*.json 2>/dev/null || echo "No corrective results files found"
ls -la results/metrics/adversarial_*_preprocessing.json 2>/dev/null || true

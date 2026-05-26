#!/bin/bash
#SBATCH --job-name=bioguard-v2-train
#SBATCH --output=cayuga_v2_train_%j.log
#SBATCH --error=cayuga_v2_train_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# v2 augmented training: DeBERTa-v3-base + (Claude synthetic + WildGuard + LAB-Bench + WMDP-Cyber)
# ~4,428 training examples. Targets: maintain bio recall, reduce cross-domain FAR.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== BioGuard v2 Augmented Training ==="
echo "Node: $(hostname)   Started: $(date)"
echo "Python: ${PYTHON}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# Verify pre-requisites
for f in \
    "data/processed/train.jsonl" \
    "data/processed/val.jsonl" \
    "data/external/v2_splits/v2_augmentation.jsonl" \
; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

echo "Running: ${PYTHON} scripts/train_v2_augmented.py"
echo ""

${PYTHON} scripts/train_v2_augmented.py

echo ""
echo "=== Complete: $(date) ==="
echo ""
echo "=== Model artifacts ==="
ls -la models/deberta_bioguard_v2_augmented/ 2>/dev/null | head

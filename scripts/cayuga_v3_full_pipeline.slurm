#!/bin/bash
#SBATCH --job-name=bioguard-v3-full
#SBATCH --output=cayuga_v3_full_%j.log
#SBATCH --error=cayuga_v3_full_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#
# v3 balanced training pipeline:
#   1. Create v3 splits (reduced SAFE + UNSAFE bio adversarial)
#   2. Train v3 (manual UNSAFE weight=2.0)
#   3. Full evaluation (A_full vs v2 vs v3)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "======================================================"
echo "v3 FULL PIPELINE"
echo "======================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: ${PYTHON}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# Verify pre-requisites
echo "=== Verifying pre-requisites ==="
for f in \
    "data/processed/train.jsonl" \
    "data/processed/val.jsonl" \
    "data/external/harmbench_bio.jsonl" \
    "data/external/advbench_bio.jsonl" \
    "data/external/jailbreakbench_bio.jsonl" \
    "data/external/wildguard_test.jsonl" \
    "data/external/lab_bench.jsonl" \
    "data/external/wmdp_cyber.jsonl" \
; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
    echo "  OK: $f"
done
echo ""

# Step 1: Create v3 splits
echo "=== Step 1/3: Create v3 splits ==="
${PYTHON} scripts/create_v3_splits.py
echo ""

# Step 2: Train v3 (UNSAFE weight=2.0)
echo "=== Step 2/3: Train v3 (UNSAFE weight=2.0) ==="
${PYTHON} scripts/train_v3_balanced.py --unsafe-weight 2.0
echo ""

# Step 3: Full evaluation (A_full vs v2 vs v3)
echo "=== Step 3/3: Three-way evaluation ==="
${PYTHON} scripts/evaluate_v3_full.py
echo ""

echo "======================================================"
echo "v3 PIPELINE COMPLETE: $(date)"
echo "======================================================"
echo ""
echo "=== Model artifacts ==="
ls -la models/deberta_bioguard_v3_balanced/ 2>/dev/null | head -20
echo ""
echo "=== Key result files ==="
ls -la results/metrics/v3_eval_summary.json results/metrics/v3_compare_*.json 2>/dev/null | head -20

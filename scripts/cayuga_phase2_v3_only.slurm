#!/bin/bash
#SBATCH --job-name=bioguard-p2-v3
#SBATCH --output=cayuga_phase2_v3_%j.log
#SBATCH --error=cayuga_phase2_v3_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Phase 2: v3 only on Cayuga (model is already on this HPC).
# Baselines (WildGuard, LLaMA-Guard 3) run separately on Expanse.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "======================================================"
echo "PHASE 2 v3 only (Cayuga)"
echo "======================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# Step 0: Cache Phase 2 data
echo "=== Step 0/1: Cache Phase 2 benchmarks ==="
for bench in harmbench_full advbench_full xstest beavertails; do
    fp="data/external/${bench}.jsonl"
    fp_bt="data/external/beavertails_subset.jsonl"
    if [ "$bench" = "beavertails" ] && [ -f "$fp_bt" ]; then
        echo "  Cached: beavertails_subset"
        continue
    fi
    if [ -f "$fp" ]; then
        echo "  Cached: $bench"
    else
        echo "  Caching: $bench"
        $PYTHON scripts/cache_phase2_data.py "$bench" 2>&1 | tail -5 || echo "  WARN: $bench cache failed"
    fi
done
echo ""

echo "=== Step 1/1: v3 ==="
$PYTHON scripts/evaluate_phase2.py --model v3

echo ""
echo "======================================================"
echo "COMPLETE: $(date)"
echo "======================================================"

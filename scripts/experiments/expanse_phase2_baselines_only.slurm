#!/bin/bash
#SBATCH --job-name=bioguard-p2-base
#SBATCH --output=expanse_phase2_base_%j.log
#SBATCH --error=expanse_phase2_base_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Phase 2 baselines only (WildGuard + LLaMA-Guard 3) on Expanse H100.
# v3 part is run separately on Cayuga since model is already there.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/biothreat_eval}"

echo "======================================================"
echo "PHASE 2 BASELINES (Expanse H100)"
echo "======================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# Step 0: Cache Phase 2 data
echo "=== Step 0/3: Cache Phase 2 benchmarks ==="
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
        $PYTHON scripts/experiments/cache_phase2_data.py "$bench" 2>&1 | tail -5 || echo "  WARN: $bench cache failed"
    fi
done
echo ""

# Step 1: WildGuard 7B
echo "=== Step 1/2: WildGuard 7B ==="
$PYTHON scripts/experiments/evaluate_phase2.py --model wildguard_7b
echo ""

# Step 2: LLaMA-Guard 3 8B
echo "=== Step 2/2: LLaMA-Guard 3 8B ==="
$PYTHON scripts/experiments/evaluate_phase2.py --model llama_guard_3_8b
echo ""

echo "======================================================"
echo "PHASE 2 BASELINES COMPLETE: $(date)"
echo "======================================================"

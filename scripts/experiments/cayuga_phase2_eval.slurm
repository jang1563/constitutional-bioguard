#!/bin/bash
#SBATCH --job-name=bioguard-phase2
#SBATCH --output=cayuga_phase2_%j.log
#SBATCH --error=cayuga_phase2_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Phase 2: v3 + WildGuard + LLaMA-Guard 3 on extended benchmarks.
# (HarmBench full, AdvBench full, XSTest, BeaverTails, WildGuardTest native)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

echo "======================================================"
echo "PHASE 2 EVALUATION"
echo "======================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# Step 0: Cache Phase 2 data (skip if cached)
echo "=== Step 0/4: Cache Phase 2 benchmarks ==="
for bench in harmbench_full advbench_full xstest beavertails; do
    fp="data/external/${bench}.jsonl"
    if [ ! -f "$fp" ] && [ ! -f "${fp/beavertails/beavertails_subset}" ]; then
        echo "  Caching: $bench"
        $PYTHON scripts/experiments/cache_phase2_data.py "$bench" || echo "  WARN: $bench cache failed (continuing)"
    else
        echo "  Cached: $bench"
    fi
done
echo ""

# Step 1: v3 (fast, ~10ms/item)
echo "=== Step 1/4: v3 (DeBERTa-v3-base 184M) ==="
$PYTHON scripts/experiments/evaluate_phase2.py --model v3
echo ""

# Step 2: WildGuard 7B (medium)
echo "=== Step 2/4: WildGuard 7B ==="
$PYTHON scripts/experiments/evaluate_phase2.py --model wildguard_7b
echo ""

# Step 3: LLaMA-Guard 3 8B (slowest)
echo "=== Step 3/4: LLaMA-Guard 3 8B ==="
$PYTHON scripts/experiments/evaluate_phase2.py --model llama_guard_3_8b
echo ""

echo "======================================================"
echo "PHASE 2 COMPLETE: $(date)"
echo "======================================================"

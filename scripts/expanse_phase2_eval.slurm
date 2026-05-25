#!/bin/bash
#SBATCH --job-name=bioguard-phase2
#SBATCH --output=expanse_phase2_%j.log
#SBATCH --error=expanse_phase2_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --account=crl195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Phase 2: v3 + WildGuard + LLaMA-Guard 3 on extended benchmarks.
# (HarmBench full, AdvBench full, XSTest, BeaverTails, WildGuardTest native)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/jkim61/.conda/envs/bioguard/bin/python

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${SLURM_SUBMIT_DIR}/biothreat_eval"

echo "======================================================"
echo "PHASE 2 EVALUATION (Expanse H100)"
echo "======================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# Step 0: Cache Phase 2 data (if not present)
echo "=== Step 0/4: Cache Phase 2 benchmarks ==="
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

# Step 1: v3 (fast, ~10ms/item on H100)
echo "=== Step 1/3: v3 (DeBERTa-v3-base 184M) ==="
$PYTHON scripts/evaluate_phase2.py --model v3
echo ""

# Step 2: WildGuard 7B (medium)
echo "=== Step 2/3: WildGuard 7B ==="
$PYTHON scripts/evaluate_phase2.py --model wildguard_7b
echo ""

# Step 3: LLaMA-Guard 3 8B (slower)
echo "=== Step 3/3: LLaMA-Guard 3 8B ==="
$PYTHON scripts/evaluate_phase2.py --model llama_guard_3_8b
echo ""

echo "======================================================"
echo "PHASE 2 COMPLETE: $(date)"
echo "======================================================"

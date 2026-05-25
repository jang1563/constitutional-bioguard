#!/bin/bash
#SBATCH --job-name=bioguard-p3
#SBATCH --output=expanse_phase3_%j.log
#SBATCH --error=expanse_phase3_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --account=crl195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Phase 3: v3 + WildGuard + LLaMA-Guard 3 on OOD bio benchmarks
#   - SaladBench CBRN
#   - ALERT CBRN
#   - OR-Bench health/medicine
#   - SimpleSafetyTests bio

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/jkim61/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${SLURM_SUBMIT_DIR}/biothreat_eval"

echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
echo ""

echo "=== Step 0: Cache Phase 3 OOD benchmarks ==="
for bench in saladbench_cbrn alert_cbrn or_bench_health simple_safety_bio; do
    if [ -f "data/external/${bench}.jsonl" ]; then
        echo "  Cached: $bench"
    else
        echo "  Caching: $bench"
        $PYTHON scripts/cache_phase3_data.py "$bench" 2>&1 | tail -5 || echo "  WARN failed"
    fi
done
echo ""

echo "=== Step 1: v3 ==="
$PYTHON scripts/evaluate_phase3_ood.py --model v3
echo ""

echo "=== Step 2: WildGuard 7B ==="
$PYTHON scripts/evaluate_phase3_ood.py --model wildguard_7b
echo ""

echo "=== Step 3: LLaMA-Guard 3 8B ==="
$PYTHON scripts/evaluate_phase3_ood.py --model llama_guard_3_8b
echo ""

echo "Complete: $(date)"

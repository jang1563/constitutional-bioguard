#!/bin/bash
#SBATCH --job-name=bioguard-baselines
#SBATCH --output=cayuga_baselines_%j.log
#SBATCH --error=cayuga_baselines_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# External baseline evaluation: LLaMA-Guard 3 8B + WildGuard 7B on the
# same benchmarks as v3 (BioThreat-Eval, HarmBench-bio held-out,
# AdvBench-bio held-out, WildGuard held-out, LAB-Bench held-out).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

echo "======================================================"
echo "BASELINE EVALUATION: LLaMA-Guard + WildGuard"
echo "======================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# Baseline 1: WildGuard 7B (smaller, open, runs first)
echo "=== Baseline 1: WildGuard 7B ==="
$PYTHON scripts/evaluate_baselines.py --baseline wildguard_7b \
    --benchmarks biothreat harmbench_bio_ho advbench_bio_ho \
                 wildguard_test_ho lab_bench_ho
echo ""

# Baseline 2: LLaMA-Guard 3 8B
echo "=== Baseline 2: LLaMA-Guard 3 8B ==="
$PYTHON scripts/evaluate_baselines.py --baseline llama_guard_3_8b \
    --benchmarks biothreat harmbench_bio_ho advbench_bio_ho \
                 wildguard_test_ho lab_bench_ho
echo ""

echo "======================================================"
echo "BASELINE EVALUATION COMPLETE: $(date)"
echo "======================================================"

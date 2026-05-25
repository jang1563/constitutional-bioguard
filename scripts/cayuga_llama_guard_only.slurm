#!/bin/bash
#SBATCH --job-name=bioguard-lg3
#SBATCH --output=cayuga_lg3_%j.log
#SBATCH --error=cayuga_lg3_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Re-run LLaMA-Guard 3 8B baseline (first run had silent prediction
# failures — likely .generate() kwarg incompatibility with this transformers
# version). New implementation uses next-token logits directly (no .generate)
# and includes a smoke test at load time.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "======================================================"
echo "LLAMA-GUARD 3 8B BASELINE (re-run)"
echo "======================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

$PYTHON scripts/evaluate_baselines.py --baseline llama_guard_3_8b \
    --benchmarks biothreat harmbench_bio_ho advbench_bio_ho \
                 wildguard_test_ho lab_bench_ho

echo ""
echo "======================================================"
echo "LLAMA-GUARD 3 RE-RUN COMPLETE: $(date)"
echo "======================================================"

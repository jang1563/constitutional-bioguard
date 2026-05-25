#!/bin/bash
#SBATCH --job-name=bioguard-lg3
#SBATCH --output=expanse_lg3_%j.log
#SBATCH --error=expanse_lg3_%j.log
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
# LLaMA-Guard 3 8B baseline on Expanse (Cayuga GPU is QOS-blocked).
# Reuses jkim61's existing bioguard conda env on /home/jkim61/.conda/envs/.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/jkim61/.conda/envs/bioguard/bin/python

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${SLURM_SUBMIT_DIR}/biothreat_eval"

echo "======================================================"
echo "LLAMA-GUARD 3 8B (Expanse)"
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
echo "COMPLETE: $(date)"
echo "======================================================"

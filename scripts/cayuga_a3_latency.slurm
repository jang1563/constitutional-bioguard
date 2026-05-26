#!/bin/bash
#SBATCH --job-name=bioguard-a3-latency
#SBATCH --output=cayuga_a3_latency_%j.log
#SBATCH --error=cayuga_a3_latency_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#
# A.3 Inference latency / memory benchmark
# Compares v3 / v4 / WildGuard 7B / LLaMA-Guard 3 8B on a single GPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== A.3 Latency / Memory Benchmark ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

$PYTHON scripts/probe_a3_latency_memory.py

echo ""
echo "Complete: $(date)"

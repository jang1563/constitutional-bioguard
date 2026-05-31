#!/bin/bash
#SBATCH --job-name=bg-v8eval
#SBATCH --output=cayuga_v8eval_%j.log
#SBATCH --error=cayuga_v8eval_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

# v8 gate scorecard: v8 (data-first) vs v4, with correct response-harm labels +
# strict bio subsets + selectivity. Decides "was the bottleneck the data?"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

echo "=== v8 gate scorecard (v8 vs v4) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
test -d models/deberta_bioguard_v8 || { echo "FATAL: no v8 model"; exit 1; }

$PYTHON scripts/eval_v8_gates.py --models v8b v8 v4

echo "=== v8 eval done: $(date) ==="

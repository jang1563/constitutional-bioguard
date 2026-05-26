#!/bin/bash
#SBATCH --job-name=bioguard-ws3
#SBATCH --output=probe_ensemble_%j.log
#SBATCH --error=probe_ensemble_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
##SBATCH --account=<your_allocation>
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# WS-3 Activation-probe ensemble on SDSC Expanse (NAIRR allocation).
#
# Loads an open-weight LLM (Llama-3.1-8B or Gemma-2-9B), extracts
# hidden-state activations on train+test data, trains linear probes,
# and evaluates the probe + BioGuard ensemble.  Inference-only (no
# LLM fine-tuning).
#
# 64GB RAM for Llama-3.1-8B in float16 (~16GB model + activation buffers).
#
# Usage:
#   sbatch scripts/probe_ensemble.slurm                      # Llama default
#   sbatch --export=PROBE_MODEL=gemma-2-9b scripts/probe_ensemble.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Use conda env from previous WS-2 setup
PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HOME}/.cache/huggingface"

# Model selection (default: llama-3.1-8b)
MODEL="${PROBE_MODEL:-llama-3.1-8b}"

echo "=== BioGuard WS-3: Activation-probe ensemble ==="
echo "Node: $(hostname)   Started: $(date)"
echo "Model: ${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

${PYTHON} scripts/run_probe_ensemble.py \
    --model "${MODEL}" \
    --batch-size 2 \
    --layer-fraction 0.4 \
    --device cuda

echo "=== Complete: $(date) ==="
echo "Results: results/metrics/probe_ensemble_${MODEL}.json"

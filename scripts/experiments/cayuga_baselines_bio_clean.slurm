#!/bin/bash
#SBATCH --job-name=bio-clean-baselines
#SBATCH --output=bio_clean_baselines_%j.log
#SBATCH --error=bio_clean_baselines_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Phase R head-to-head: WildGuard-7B + LlamaGuard-3-8B on bio_clean_eval.jsonl
# (120 genuine-bio positives + 881 benign-bio negatives).
# Compare to dual-mode self-eval: recall=0.950 over-refusal=0.009.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Direct path to conda env python (module system unavailable in SLURM)
PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== WildGuard-7B ==="
$PYTHON scripts/experiments/eval_baselines_bio_clean.py --baseline wildguard_7b

echo ""
echo "=== LlamaGuard-3-8B ==="
$PYTHON scripts/experiments/eval_baselines_bio_clean.py --baseline llama_guard_3_8b

echo ""
echo "=== DONE ==="

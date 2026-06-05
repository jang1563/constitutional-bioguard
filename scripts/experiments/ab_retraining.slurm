#!/bin/bash
#SBATCH --job-name=bioguard-ab
#SBATCH --output=ab_retraining_%j.log
#SBATCH --error=ab_retraining_%j.log
#SBATCH --partition=GPU-shared          # ADJUST: e.g. gpu / GPU-shared / gpuA100
#SBATCH --gres=gpu:1                    # one GPU is sufficient for DeBERTa-base
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00                 # two ~2h runs; raise if needed
# #SBATCH --account=YOUR_ALLOCATION     # ADJUST: NAIRR / ACCESS allocation id
#
# WS-2 A/B retraining batch script for an HPC / NSF ACCESS GPU node.
#
# Trains variant A (full synthetic set) and variant B (bag-of-words-filtered
# set) under identical hyperparameters, then evaluates both on the held-out
# test set. See scripts/experiments/run_ab_retraining.py and docs/V2_DESIGN.md (WS-2).
#
# Before submitting:
#   1. Adjust the SBATCH partition / account lines above for your cluster.
#   2. Ensure data/processed/ contains:
#        train.jsonl, val.jsonl, test.jsonl, class_weights.json
#        train_bow_filtered.jsonl, train_bow_filtered_class_weights.json
#      (the train_bow_filtered* files come from WS-2 bag-of-words elimination:
#         python -c "from constitutional_bioguard.training.bow_elimination \
#                    import eliminate_bow_examples; \
#                    eliminate_bow_examples(confidence=0.999, dry_run=False)")
#   3. Create the environment once: python -m venv .venv && \
#        .venv/bin/pip install -e ".[dev]"
#
# Submit with:  sbatch scripts/experiments/ab_retraining.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load anaconda3 2>/dev/null || true   # ADJUST or remove per cluster
source .venv/bin/activate

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export TOKENIZERS_PARALLELISM=false

echo "=== BioGuard WS-2 A/B retraining ==="
echo "Node: $(hostname)   Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/experiments/run_ab_retraining.py --variant both --verbose

echo "=== Complete: $(date) ==="
echo "Comparison written to results/metrics/ab_retraining_comparison.json"

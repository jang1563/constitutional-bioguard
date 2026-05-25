#!/bin/bash
#SBATCH --job-name=bioguard-v3-eval
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v3_eval_%j.out
#SBATCH --error=logs/v3_eval_%j.err

# v3 full evaluation: A_full vs v2 vs v3 on all benchmarks

set -euo pipefail

PYTHON=~/.conda/envs/bioguard/bin/python
PROJECT=~/constitutional_bioguard
cd "$PROJECT"
mkdir -p logs results/metrics

echo "=== v3 Full Evaluation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

$PYTHON scripts/evaluate_v3_full.py

echo ""
echo "=== v3 Evaluation Complete ==="
echo "Date: $(date)"

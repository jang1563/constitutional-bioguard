#!/bin/bash
#SBATCH --job-name=bioguard-a2
#SBATCH --output=cayuga_a2_compliance_variants_%j.log
#SBATCH --error=cayuga_a2_compliance_variants_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# A.2 Compliance template variants on v3 + v4
# Verifies v4's content discrimination generalizes across 10 different
# compliance template phrasings, not just the canonical B.2.1 template.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== A.2 Compliance Template Variants ==="
echo "Started: $(date)"

echo "--- v3 ---"
$PYTHON scripts/probe_a2_compliance_variants.py v3

echo ""
echo "--- v4 ---"
$PYTHON scripts/probe_a2_compliance_variants.py v4

echo ""
echo "Complete: $(date)"

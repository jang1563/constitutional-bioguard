#!/bin/bash
#SBATCH --job-name=bioguard-v4-train
#SBATCH --output=cayuga_v4_train_%j.log
#SBATCH --error=cayuga_v4_train_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=== v4 RESPONSE-DIVERSE TRAINING ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
echo ""

echo "--- Step 1/2: Create v4 splits ---"
$PYTHON scripts/create_v4_splits.py
echo ""

echo "--- Step 2/2: Train v4 (UNSAFE weight=1.5) ---"
$PYTHON scripts/train_v4_response_diverse.py --unsafe-weight 1.5
echo ""

echo "=== Complete: $(date) ==="

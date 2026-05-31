#!/bin/bash
#SBATCH --job-name=bg-v8
#SBATCH --output=cayuga_v8_%j.log
#SBATCH --error=cayuga_v8_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# v8: DeBERTa-v3-base (v4 architecture) trained on the reuse-only v8 data.
#   Tests the data-first thesis: same proven encoder, the data it never had
#   (658 real bio harmful-RESPONSE positives + FalseReject over-refusal negatives).
#   Eval is run SEPARATELY (response-harm labels + strict subsets + selectivity),
#   deliberately not rushed here (the v7.C eval-label lesson).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

echo "=== v8 train (DeBERTa-v3-base, data-first reuse-only) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

test -f data/processed/v8_train.jsonl || { echo "FATAL: missing data/processed/v8_train.jsonl (run build_v8_data.py)"; exit 1; }
test -f data/processed/v8_val.jsonl   || { echo "FATAL: missing data/processed/v8_val.jsonl"; exit 1; }
echo "v8 train rows: $(wc -l < data/processed/v8_train.jsonl)  val rows: $(wc -l < data/processed/v8_val.jsonl)"

$PYTHON scripts/train_v8_bioguard.py --unsafe-weight 2.0

test -d models/deberta_bioguard_v8 || { echo "FATAL: training produced no model dir"; exit 1; }
echo "model artifacts:"; ls -la models/deberta_bioguard_v8 | head -20
echo "=== v8 train done: $(date) ==="

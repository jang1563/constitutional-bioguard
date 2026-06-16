#!/bin/bash
#SBATCH --job-name=bg-v8c
#SBATCH --output=cayuga_v8c_%j.log
#SBATCH --error=cayuga_v8c_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# v8c = v8b + PubMedQA substantive benign-bio negatives (915).
#   Targets the DIAGNOSED over-refusal (diag_v8b_biobenign_fp: 7/10 confident FPs
#   on reliable-label benign bio = real "detailed benign bio = harmful" confusion).
#   Same unsafe-weight 1.5 as v8b to isolate the PubMedQA data effect.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

echo "=== v8c train (v8b + PubMedQA benign-bio) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
test -f data/processed/v8c_train.jsonl || { echo "FATAL: missing v8c_train.jsonl"; exit 1; }
echo "v8c train rows: $(wc -l < data/processed/v8c_train.jsonl)"

$PYTHON scripts/experiments/train_v8_bioguard.py --data-prefix v8c --output-name deberta_bioguard_v8c --unsafe-weight 1.5

test -d models/deberta_bioguard_v8c || { echo "FATAL: no v8c model"; exit 1; }
echo "=== v8c train done: $(date) ==="

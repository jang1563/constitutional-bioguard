#!/bin/bash
#SBATCH --job-name=bg-v8b
#SBATCH --output=cayuga_v8b_%j.log
#SBATCH --error=cayuga_v8b_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# v8b: v8 + BeaverTails bio positives (2x positives, 33% pos vs 17%, bio-concentrated).
#   Tests whether growing the scarce bio positive class lifts real-response BIO recall.
#   unsafe-weight 1.5 (data now ~balanced, less need for heavy positive weight).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

echo "=== v8b train (DeBERTa, +BeaverTails bio positives) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
test -f data/processed/v8b_train.jsonl || { echo "FATAL: missing v8b_train.jsonl (run build_v8_data.py --beavertails --out-prefix v8b)"; exit 1; }
echo "v8b train rows: $(wc -l < data/processed/v8b_train.jsonl)"

$PYTHON scripts/train_v8_bioguard.py --data-prefix v8b --output-name deberta_bioguard_v8b --unsafe-weight 1.5

test -d models/deberta_bioguard_v8b || { echo "FATAL: no v8b model produced"; exit 1; }
echo "=== v8b train done: $(date) ==="

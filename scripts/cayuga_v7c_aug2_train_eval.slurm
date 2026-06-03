#!/bin/bash
#SBATCH --job-name=bg-v7c-aug2
#SBATCH --output=cayuga_v7c_aug2_%j.log
#SBATCH --error=cayuga_v7c_aug2_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Step 3 (v2): v7.C targeted retrain with the no-response SHORTCUT FIXED.
#   aug1 collapsed (all benign domain negs were '(no response provided)' -> the
#   model learned no-response => unharmful, and prompt-only eval flagged nothing).
#   aug2 balances the no-response condition: 487 benign (unharmful) + 487 harmful
#   (reused base prompts), so the model must read the PROMPT in the no-response
#   (prompt-screening) condition. Re-eval Step-1 prob ROC + Step-2 real
#   over-refusal, tagged _aug2.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_DIR=models/llama31_8b_bioguard_v7c_aug2
DATA=data/processed/v7c_nocot_train_aug2.jsonl

echo "=== v7.C-aug2 targeted retrain (no-response shortcut fixed) + eval ==="; echo "Started: $(date)"
test -f "$DATA" || { echo "MISSING $DATA. Aborting."; exit 1; }
echo "train rows: $(wc -l < "$DATA")  (expect 4505)"
$PYTHON -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE_MODEL'); print('HF OK')" \
    || { echo "FATAL: cannot access $BASE_MODEL. Aborting."; exit 1; }

echo ""; echo "=== train v7.C-aug2 ==="
$PYTHON scripts/train_v7c_llama_nocot.py \
    --base-model "$BASE_MODEL" \
    --data-path "$DATA" \
    --output-dir "$MODEL_DIR"
test -d "$MODEL_DIR/adapter" || { echo "FATAL: no adapter produced. Aborting."; exit 1; }
echo "adapter saved:"; ls -la "$MODEL_DIR/adapter" | head

export V7C_ADAPTER_DIR="$(pwd)/$MODEL_DIR/adapter"
export V7C_TAG=_aug2

echo ""; echo "=== recall ROC (aug2) ==="
$PYTHON scripts/score_v7c_probs.py
echo ""; echo "=== real over-refusal (aug2) ==="
$PYTHON scripts/score_v7c_realfpr.py

echo ""; echo "=== v7.C-aug2 train+eval complete: $(date) ==="

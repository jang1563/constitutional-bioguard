#!/bin/bash
#SBATCH --job-name=bg-v7c-aug
#SBATCH --output=cayuga_v7c_aug_%j.log
#SBATCH --error=cayuga_v7c_aug_%j.log
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
# Step 3: v7.C targeted benign-aware RETRAIN + re-eval.
#   Data = v7c_nocot_train_aug.jsonl (3531 base + 487 domain-targeted legit
#   hard-negatives in genomics/synbio/tox/pathogen/dual_use_research, leakage-
#   excluded vs ood_fpr + bio_clean_eval). Goal: cut dual-use over-refusal
#   without losing recall. Re-eval = Step-1 prob ROC + Step-2 real over-refusal,
#   tagged _aug so the v7.C baseline results are preserved for comparison.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_DIR=models/llama31_8b_bioguard_v7c_aug
DATA=data/processed/v7c_nocot_train_aug.jsonl

echo "=== v7.C-aug targeted retrain + eval ==="; echo "Started: $(date)"

# Preflight: data + HF access
test -f "$DATA" || { echo "MISSING $DATA. Aborting."; exit 1; }
echo "train rows: $(wc -l < "$DATA")  (expect 4018)"
$PYTHON -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE_MODEL'); print('HF OK')" \
    || { echo "FATAL: cannot access $BASE_MODEL. Aborting."; exit 1; }

# ── Step 1: train on augmented data ──
echo ""; echo "=== train v7.C-aug ==="
$PYTHON scripts/experiments/train_v7c_llama_nocot.py \
    --base-model "$BASE_MODEL" \
    --data-path "$DATA" \
    --output-dir "$MODEL_DIR"
test -d "$MODEL_DIR/adapter" || { echo "FATAL: no adapter produced. Aborting."; exit 1; }
echo "adapter saved:"; ls -la "$MODEL_DIR/adapter" | head

# ── Step 2: re-eval the aug adapter (tagged _aug; baseline preserved) ──
export V7C_ADAPTER_DIR="$(pwd)/$MODEL_DIR/adapter"
export V7C_TAG=_aug

echo ""; echo "=== recall ROC (aug) ==="
$PYTHON scripts/experiments/score_v7c_probs.py

echo ""; echo "=== real over-refusal (aug) ==="
$PYTHON scripts/experiments/score_v7c_realfpr.py

echo ""; echo "=== v7.C-aug train+eval complete: $(date) ==="

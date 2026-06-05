#!/bin/bash
#SBATCH --job-name=bg-v7b2
#SBATCH --output=cayuga_v7b2_%j.log
#SBATCH --error=cayuga_v7b2_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# v7.B2: task-spec-CORRECTED retrain (single-variable A/B vs v7.B).
#   ONLY change vs cayuga_v7b_pipeline.slurm = training DATA.
#   - data: v7b2_cot_train.jsonl (separate prompt/response harm labels +
#     grounded label-consistent CoT; fixes V7B_OVER_REFUSAL_ANALYSIS §9.3/§9.4)
#   - all hyperparameters IDENTICAL (epochs 3, bs 4, ga 4, lr 2e-4, r16/a32, seq 2048)
#   - eval at /no_think (ship config) on the full Phase-1 gate suite, so the
#     WildGuard gate #4 (def. /no_think F1 0.5479) is re-measured apples-to-apples.
# The validated data file is scp'd in (exact artifact validated locally:
#   3531 items, 289 prompt!=response divergent). The slurm does NOT rebuild it,
#   to guarantee training on precisely the validated data.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

DATA=data/processed/v7b2_cot_train.jsonl
MODEL_DIR=models/qwen3_bioguard_v7b2_cot

echo "=== v7.B2 Qwen3-4B CoT classifier (task-spec corrected) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Step 0: verify the validated training data is present (do NOT rebuild)
echo ""
echo "=== Step 0: verify v7.B2 data ==="
test -f "$DATA" || { echo "MISSING $DATA (scp it from local)"; exit 1; }
N=$(wc -l < "$DATA")
echo "v7b2_cot_train.jsonl rows: $N"
[ "$N" -eq 3531 ] || echo "WARN: expected 3531 rows, got $N"

# Step 1: dependency check
echo ""
echo "=== Step 1: Check dependencies ==="
$PYTHON -c "import peft, trl, bitsandbytes; print('peft/trl/bnb OK')" 2>/dev/null || {
    echo "Installing peft, trl, bitsandbytes..."
    pip install -q peft trl bitsandbytes
}

# Step 2: Train QLoRA (IDENTICAL config to v7.B; only --data-path/--output-dir differ)
echo ""
echo "=== Step 2: Train v7.B2 (QLoRA Qwen3-4B) ==="
$PYTHON scripts/experiments/train_v7b_qwen3_cot.py \
    --base-model Qwen/Qwen3-4B \
    --data-path "$DATA" \
    --output-dir "$MODEL_DIR" \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --max-seq-length 2048

# Step 3: Evaluate at /no_think (ship config) on full Phase-1 gate suite
echo ""
echo "=== Step 3: Evaluate v7.B2 (/no_think) ==="
$PYTHON scripts/experiments/eval_v7b_qwen3_cot.py \
    --no-think \
    --model-dir "$MODEL_DIR" \
    --out-prefix v7b2_nothink_eval

echo ""
echo "=== v7.B2 pipeline complete: $(date) ==="

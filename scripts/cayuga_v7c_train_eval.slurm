#!/bin/bash
#SBATCH --job-name=bg-v7c
#SBATCH --output=cayuga_v7c_%j.log
#SBATCH --error=cayuga_v7c_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# v7.C: Llama-3.1-8B-Instruct + QLoRA, NO-CoT direct biosafety classifier.
#   The matched-scale (8B) WildGuard/LG3 competitor. Design rationale:
#     - NO CoT  -> drops v7.B root cause #3 (§2.4/§9.3); every successful
#                  comparable (WildGuard/LG3/Aegis/ShieldGemma) is no-CoT, and
#                  Llama has no /no_think escape hatch so CoT must go at the data
#                  level.
#     - Instruct -> drops root cause #5 (pre-aligned refusal calibration) AND
#                  supplies the chat template the pipeline needs.
#     - Dual labels preserved (§9.4 task-spec): "Prompt harm/Response harm".
#   Data = v7c_nocot_train.jsonl (n=3531), derived from the v7.B2 dual-label set
#   with CoT stripped + the /think soft-switch removed (train == eval, plain mode).
#
#   Eval runs in --prompt-mode plain (NO /think|/no_think), decisive gates FIRST
#   (wildguard first) so the headline competitor number survives a wall.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_DIR=models/llama31_8b_bioguard_v7c
DATA=data/processed/v7c_nocot_train.jsonl

echo "=== v7.C (Llama-3.1-8B-Instruct, no-CoT) train+eval ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Preflight 0: gated-model HF access (fail fast, before the long train) ──
echo ""
echo "=== Step 0a: verify HF access to gated $BASE_MODEL ==="
test -f "$HF_HOME/token" && echo "HF token file present" || echo "WARN: no $HF_HOME/token (relying on HF_TOKEN env)"
$PYTHON -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE_MODEL'); print('HF access OK:', '$BASE_MODEL')" \
    || { echo "FATAL: cannot access $BASE_MODEL (gated? accept license + set HF token). Aborting before training."; exit 1; }

# ── Preflight 0b: training data present ──
echo ""
echo "=== Step 0b: verify v7.C no-CoT training data ==="
test -f "$DATA" || { echo "MISSING $DATA (run build_v7c_nocot_data.py + scp). Aborting."; exit 1; }
echo "rows: $(wc -l < "$DATA")  (expect 3531)"

# ── Step 1: train ──
echo ""
echo "=== Step 1: train v7.C ==="
$PYTHON scripts/train_v7c_llama_nocot.py \
    --base-model "$BASE_MODEL" \
    --data-path "$DATA" \
    --output-dir "$MODEL_DIR"

test -d "$MODEL_DIR/adapter" || { echo "FATAL: training produced no adapter. Aborting."; exit 1; }
echo "adapter saved:"; ls -la "$MODEL_DIR/adapter" | head -20

# ── Step 2: eval (plain mode, wildguard first so it survives a wall) ──
echo ""
echo "=== Step 2: evaluate v7.C (plain prompt), decisive gates first ==="
$PYTHON scripts/eval_v7b_qwen3_cot.py \
    --model-dir "$MODEL_DIR" \
    --prompt-mode plain \
    --out-prefix v7c_plain_eval \
    --max-new-tokens 64 \
    --benchmarks wildguard_native xstest or_bench_hard_1k false_reject_test saladbench_cbrn alert_cbrn_strict simple_safety_bio

echo ""
echo "=== v7.C train+eval complete: $(date) ==="

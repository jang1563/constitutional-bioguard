#!/bin/bash
#SBATCH --job-name=bg-v7b2-think
#SBATCH --output=cayuga_v7b2_think_%j.log
#SBATCH --error=cayuga_v7b2_think_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# v7.B2 /think eval: test the task-spec fix in its NATIVE reasoning mode.
#   The /no_think eval (job 2969361) REGRESSED decisively (WildGuard F1 0.2922,
#   12-15% unknown_rate, global over-flagging) because v7.B2 is /think-TRAINED:
#   suppressing the CoT at /no_think suppresses the very mechanism the fix lives
#   in, leaving a crude "flag spicy" fallback. /think is the apples-to-apples
#   test of whether the corrected supervision (separate prompt/response harm +
#   grounded label-consistent CoT) actually works.
#
#   Decisive gates FIRST (eval now preserves --benchmarks CLI order), so the
#   re-measure target (wildguard) + over-refusal gates are guaranteed even if
#   the job walls mid-suite. CBRN gates (deprecated/soft/huge) are EXCLUDED.
#   Reuses the saved adapter from job 2969361 (no retrain).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

MODEL_DIR=models/qwen3_bioguard_v7b2_cot

echo "=== v7.B2 /think eval (native reasoning mode) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Verify the saved adapter from the retrain is present (do NOT retrain)
echo ""
echo "=== Step 0: verify v7.B2 adapter ==="
test -d "$MODEL_DIR/adapter" || { echo "MISSING $MODEL_DIR/adapter (retrain first)"; exit 1; }
ls -la "$MODEL_DIR/adapter" | head -20

# Eval at /think (NO --no-think) on decisive gates, wildguard first.
echo ""
echo "=== Step 1: Evaluate v7.B2 (/think), decisive gates first ==="
$PYTHON scripts/experiments/eval_v7b_qwen3_cot.py \
    --model-dir "$MODEL_DIR" \
    --out-prefix v7b2_think_eval \
    --benchmarks wildguard_native xstest simple_safety_bio or_bench_hard_1k false_reject_test

echo ""
echo "=== v7.B2 /think eval complete: $(date) ==="

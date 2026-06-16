#!/bin/bash
#SBATCH --job-name=bg-base-rh
#SBATCH --output=cayuga_baseline_rh_%j.log
#SBATCH --error=cayuga_baseline_rh_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Controlled response-harm baseline comparison vs v7.C.
#   WildGuard-7B (signal=response, harmful_response ONLY) + LLaMA-Guard-3-8B,
#   both on the EXACT n=1709 wildguard_native item set with response_harm_label
#   gold (identical to eval_v7b) -> apples-to-apples F1 vs v7.C's 0.703.
#   Phase-2 baselines used prompt_harm gold + WG union -> NOT comparable.
#   Inference only (no training); both 7-8B models fit one A100.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

echo "=== controlled response-harm baseline (WG-7B response-only + LG3-8B) ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Preflight: gated-model HF access (fail fast) ──
echo ""
echo "=== Step 0a: verify HF access (allenai/wildguard + meta-llama/Llama-Guard-3-8B) ==="
$PYTHON -c "from transformers import AutoConfig; [AutoConfig.from_pretrained(m) for m in ('allenai/wildguard','meta-llama/Llama-Guard-3-8B')]; print('HF access OK')" \
    || { echo "FATAL: cannot access a gated baseline (accept license + HF token). Aborting."; exit 1; }

# ── Preflight 0b: data present ──
echo ""
echo "=== Step 0b: verify wildguard_test.jsonl ==="
$PYTHON -c "from constitutional_bioguard.config import DATA_EXTERNAL; p=DATA_EXTERNAL/'wildguard_test.jsonl'; assert p.exists(), f'MISSING {p}'; print('data OK:', p)" \
    || { echo "FATAL: wildguard_test.jsonl missing. Aborting."; exit 1; }

# ── Run ──
echo ""
echo "=== Step 1: response-harm baseline eval ==="
$PYTHON scripts/experiments/eval_baselines_response_harm.py \
    --models wildguard_7b llama_guard_3_8b \
    --out-prefix v7c_baseline_rh \
    --expect-n 1709

echo ""
echo "=== done: $(date) ==="

#!/bin/bash
#SBATCH --job-name=bg-v7b
#SBATCH --output=cayuga_v7b_%j.log
#SBATCH --error=cayuga_v7b_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_HOME="${HOME}/.cache/huggingface"

echo "=== v7.B Qwen3-4B CoT biosafety classifier ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Step 1: Build CoT training data
echo ""
echo "=== Step 1: Build CoT training data ==="
$PYTHON scripts/experiments/build_v7b_cot_data.py

# Step 2: Install dependencies if needed
echo ""
echo "=== Step 2: Check dependencies ==="
$PYTHON -c "import peft, trl, bitsandbytes; print('peft/trl/bnb OK')" 2>/dev/null || {
    echo "Installing peft, trl, bitsandbytes..."
    pip install -q peft trl bitsandbytes
}

# Step 3: Train QLoRA
echo ""
echo "=== Step 3: Train v7.B (QLoRA Qwen3-4B) ==="
$PYTHON scripts/experiments/train_v7b_qwen3_cot.py \
    --base-model Qwen/Qwen3-4B \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --max-seq-length 2048

# Step 4: Evaluate on locked benchmarks
echo ""
echo "=== Step 4: Evaluate v7.B ==="
$PYTHON scripts/experiments/eval_v7b_qwen3_cot.py

echo ""
echo "=== v7.B pipeline complete: $(date) ==="

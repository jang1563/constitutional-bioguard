#!/bin/bash
#SBATCH --job-name=bioguard-v8-fndiag
#SBATCH --output=logs/cayuga_v8_fndiag_%j.log
#SBATCH --error=logs/cayuga_v8_fndiag_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source $HOME/miniconda3/miniconda3/bin/activate bioguard-hpc
export PYTHONUTF8=1
echo "=== v8 koaug3 FN diagnostic $(date) ==="
python scripts/diag_v8_fn_ood_fnr.py \
    --model-dir models/deberta_bioguard_v8_koaug3/final \
    --data data/splits/ood_fnr.jsonl \
    --out results/diag_v8_koaug3_ood_fnr.json
echo "=== done $(date) ==="

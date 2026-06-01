#!/bin/bash
#SBATCH --job-name=bioguard-v8-corrected
#SBATCH --output=logs/cayuga_v8_corrected_%j.log
#SBATCH --error=logs/cayuga_v8_corrected_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source $HOME/miniconda3/miniconda3/bin/activate bioguard-hpc
export PYTHONUTF8=1
echo "=== v8 koaug3 CORRECTED scorecard (FNR + Youden eval-fix) $(date) ==="
python scripts/eval_v8_koaug3_corrected.py \
    --model-dir models/deberta_bioguard_v8_koaug3/final \
    --splits data/splits \
    --out results/phase3_koaug3_eval_corrected.json
echo "=== done $(date) ==="

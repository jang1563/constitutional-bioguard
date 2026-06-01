#!/bin/bash
#SBATCH --job-name=bioguard-v8-youden
#SBATCH --output=logs/cayuga_v8_youden_%j.log
#SBATCH --error=logs/cayuga_v8_youden_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source $HOME/miniconda3/miniconda3/bin/activate bioguard-hpc
export PYTHONUTF8=1
echo "=== v8 koaug3 Youden matched-triples diag $(date) ==="
python scripts/diag_v8_youden_mt.py \
    --model-dir models/deberta_bioguard_v8_koaug3/final \
    --data data/splits/matched_triples_flat.jsonl \
    --out results/diag_v8_koaug3_youden_mt.json
echo "=== done $(date) ==="

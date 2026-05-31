#!/bin/bash
#SBATCH --job-name=bioguard-v8-calib
#SBATCH --output=logs/cayuga_v8_calib_%j.log
#SBATCH --error=logs/cayuga_v8_calib_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source $HOME/miniconda3/miniconda3/bin/activate bioguard-hpc
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
echo "=== v8 koaug3 threshold calibration  $(date) ==="
python scripts/calibrate_v8_threshold.py \
    --model-dir models/deberta_bioguard_v8_koaug3/final \
    --ood-fpr data/splits/ood_fpr.jsonl \
    --ood-fnr data/splits/ood_fnr.jsonl \
    --out results/calib_v8_koaug3_threshold.json
echo "=== done $(date) ==="

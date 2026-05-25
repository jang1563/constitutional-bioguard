#!/bin/bash
#SBATCH --job-name=lg3-debug
#SBATCH --output=expanse_lg_debug_%j.log
#SBATCH --error=expanse_lg_debug_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --account=crl195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=/home/jkim61/.conda/envs/bioguard/bin/python
$PYTHON scripts/debug_llama_guard.py

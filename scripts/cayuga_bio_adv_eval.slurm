#!/bin/bash
#SBATCH --job-name=bioguard-bioadv
#SBATCH --output=cayuga_bioadv_%j.log
#SBATCH --error=cayuga_bioadv_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

A_FULL=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_A_full
V2=${HOME}/constitutional-bioguard/models/deberta_bioguard_v2_augmented
DATA=${HOME}/constitutional-bioguard/data/external

echo "=== BioGuard Adversarial Bio Eval ==="
echo "Node: $(hostname)   Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo ""

for bench in harmbench_bio jailbreakbench_bio advbench_bio; do
    cache="${DATA}/${bench}.jsonl"
    if [ ! -f "$cache" ]; then
        echo "SKIP: $cache not found"
        continue
    fi
    for tag in a_full v2; do
        model_dir="${A_FULL}"
        if [ "$tag" = "v2" ]; then
            model_dir="${V2}"
        fi
        echo ""
        echo "=== ${tag} on ${bench} ==="
        out="results/metrics/compare_${tag}_${bench}.json"
        ${PYTHON} scripts/run_corrective.py \
            --eval-external "${cache}" \
            --model-a "${model_dir}" 2>&1 | tail -8
        # Rename output to include tag
        if [ -f "results/metrics/corrective_external_${bench}.json" ]; then
            mv "results/metrics/corrective_external_${bench}.json" "$out"
            echo "Saved: $out"
        fi
    done
done

echo ""
echo "=== Complete: $(date) ==="
ls -la results/metrics/compare_*_bio.json 2>/dev/null | tail -10

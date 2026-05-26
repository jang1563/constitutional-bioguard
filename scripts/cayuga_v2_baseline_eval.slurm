#!/bin/bash
#SBATCH --job-name=bioguard-v2-baseline
#SBATCH --output=cayuga_v2_baseline_%j.log
#SBATCH --error=cayuga_v2_baseline_%j.log
#SBATCH --partition=scu-cpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# v2 Baseline evaluation: Run current A_full on 5 new external benchmarks
# (WMDP-Chem, WMDP-Cyber, LAB-Bench, PubMedQA, MedQA).
# Establishes baseline FPR/FAR for each before v2 training.
# Loaded model once per benchmark (~3-5 min each).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL_A=${HOME}/constitutional-bioguard/models/deberta_bioguard_v1_A_full
DATA=${HOME}/constitutional-bioguard/data/external

echo "=== BioGuard v2 Baseline Evaluation ==="
echo "Node: $(hostname)   Started: $(date)"
echo "Model: ${MODEL_A}"
echo ""

# Order: small datasets first (sanity check), then larger
for spec in \
    "wmdp_chem.jsonl::None" \
    "med_qa_test.jsonl::None" \
    "pubmed_qa_pqa_labeled.jsonl::None" \
    "lab_bench.jsonl::subtask" \
    "wmdp_cyber.jsonl::None" \
; do
    fname="${spec%::*}"
    strat="${spec##*::}"
    cache="${DATA}/${fname}"

    if [ ! -f "$cache" ]; then
        echo "SKIP: ${cache} not found"
        continue
    fi

    echo ""
    echo "=== Evaluating: ${fname} (stratification: ${strat}) ==="

    if [ "$strat" = "None" ]; then
        ${PYTHON} scripts/run_corrective.py \
            --eval-external "${cache}" \
            --model-a ${MODEL_A}
    else
        ${PYTHON} scripts/run_corrective.py \
            --eval-external "${cache}" \
            --strat-field "${strat}" \
            --model-a ${MODEL_A}
    fi

    echo ""
done

echo ""
echo "=== Complete: $(date) ==="
echo ""
echo "=== Result files ==="
ls -la results/metrics/corrective_external_*.json 2>/dev/null

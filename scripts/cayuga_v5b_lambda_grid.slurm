#!/bin/bash
#SBATCH --job-name=bioguard-v5b-grid
#SBATCH --output=cayuga_v5b_grid_%j.log
#SBATCH --error=cayuga_v5b_grid_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

CANDIDATES=(
  "v5b_l003=0.03"
  "v5b_l005=0.05"
  "v5b_l010=0.10"
  "v5b_l015=0.15"
  "v5b_l020=0.20"
  "v5b_l030=0.30"
)

echo "=== v5b PairCFR lambda grid ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

for ENTRY in "${CANDIDATES[@]}"; do
  NAME="${ENTRY%%=*}"
  LAMBDA="${ENTRY#*=}"
  OUT_DIR="models/deberta_bioguard_${NAME}"

  echo ""
  echo "=== Training ${NAME} (lambda=${LAMBDA}) ==="
  $PYTHON scripts/train_v5.py \
    --unsafe-weight 1.5 \
    --paircfr-lambda "${LAMBDA}" \
    --paircfr-temperature 0.1 \
    --output-dir "${OUT_DIR}"

  echo ""
  echo "=== Behavioral gates for ${NAME} ==="
  $PYTHON scripts/v5_eval_all_gates.py \
    --models "${NAME}" \
    --model-dir "${NAME}=${OUT_DIR}"

  echo ""
  echo "=== CRT refusal+compliance gate for ${NAME} ==="
  $PYTHON scripts/g2_refusal_prefix_bypass.py "${NAME}" \
    --model-dir "${OUT_DIR}"
done

IFS=,
CANDIDATE_LIST="${CANDIDATES[*]}"
unset IFS

echo ""
echo "=== Summarizing v5b frontier ==="
$PYTHON scripts/summarize_v5b_lambda_grid.py --candidates "${CANDIDATE_LIST}"

echo ""
echo "=== Complete: $(date) ==="

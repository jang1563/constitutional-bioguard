#!/bin/bash
#SBATCH --job-name=bioguard-ws4
#SBATCH --output=ws4_adversarial_%j.log
#SBATCH --error=ws4_adversarial_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# WS-4 adversarial evaluation with reconstruction attacks.
# Uses GPU for faster DeBERTa inference but could run on CPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Use A_full model (v1 reproduction)
MODEL_DIR="${MODEL_DIR:-models/deberta_bioguard_v1_A_full}"

echo "=== BioGuard WS-4: Adversarial suite with reconstruction attacks ==="
echo "Node: $(hostname)   Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -c "
import logging, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
from pathlib import Path
from constitutional_bioguard.evaluation.adversarial_suite import run_adversarial_suite
results = run_adversarial_suite(
    model_dir=Path('${MODEL_DIR}'),
    attacks_per_type=50,
)
print()
print('=== WS-4 COMPLETE ===')
# Load and print summary
with open('results/metrics/adversarial_results.json') as f:
    data = json.load(f)
s = data['summary']
print(f'Total attacks: {s[\"total_attack_types\"]}')
print(f'VDR: {s[\"vulnerability_discovery_rate_per_1k\"]} per 1,000 queries')
print(f'Mean ASR: {s[\"mean_asr\"]*100:.2f}%')
for cat, info in s['by_category'].items():
    print(f'  {cat}: ASR={info[\"asr\"]*100:.2f}% ({info[\"n_flipped\"]}/{info[\"n_tested\"]})')
"

echo "=== Complete: $(date) ==="

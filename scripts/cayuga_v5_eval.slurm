#!/bin/bash
#SBATCH --job-name=bg-v5-eval
#SBATCH --output=cayuga_v5_eval_%j.log
#SBATCH --error=cayuga_v5_eval_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON=${PYTHON:-python}
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

echo "=== v5 Full Eval Suite ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

# 1. Behavioral gates on v4 vs v5_baseline vs v5
echo ""
echo "--- Behavioral gates ---"
$PYTHON scripts/v5_eval_all_gates.py

# 2. Representation probes on v5_baseline and v5
echo ""
echo "--- v5_baseline probes ---"
$PYTHON -c "
import sys
sys.argv = ['v5_probe_preregister_v2.py']
sys.path.insert(0, '.')
from scripts.v5_probe_preregister_v2 import run_probes
from constitutional_bioguard.config import MODELS_DIR, METRICS_DIR
import json
r = run_probes('v5_baseline', MODELS_DIR / 'deberta_bioguard_v5_baseline')
with open(METRICS_DIR / 'v5_probes_on_v5_baseline.json', 'w') as f:
    json.dump(r, f, indent=2)
"

echo ""
echo "--- v5 probes ---"
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from scripts.v5_probe_preregister_v2 import run_probes
from constitutional_bioguard.config import MODELS_DIR, METRICS_DIR
import json
r = run_probes('v5', MODELS_DIR / 'deberta_bioguard_v5')
with open(METRICS_DIR / 'v5_probes_on_v5.json', 'w') as f:
    json.dump(r, f, indent=2)
"

# 3. CRT refusal+compliance hybrid bypass test on each model (Gate 5)
echo ""
echo "--- Gate 5: CRT refusal+compliance hybrid ---"
$PYTHON scripts/g2_refusal_prefix_bypass.py v5_baseline
$PYTHON scripts/g2_refusal_prefix_bypass.py v5

echo ""
echo "=== Complete: $(date) ==="

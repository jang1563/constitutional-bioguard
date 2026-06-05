#!/bin/bash
#SBATCH --job-name=bioguard-v3-biothreat
#SBATCH --output=cayuga_v3_biothreat_%j.log
#SBATCH --error=cayuga_v3_biothreat_%j.log
#SBATCH --partition=scu-cpu
#SBATCH --qos=normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# BioThreat-Eval ONLY for v3 pipeline (the main eval skipped it due to
# missing BIOTHREAT_EVAL_DIR env var). Runs A_full, v2, v3 on BioThreat-Eval.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${BIOTHREAT_EVAL_DIR:-${HOME}/biothreat_eval}"

echo "=== BioThreat-Eval (v3 patch) ==="
echo "Node: $(hostname)   Started: $(date)"
echo "BIOTHREAT_EVAL_DIR: ${BIOTHREAT_EVAL_DIR}"
echo ""

$PYTHON <<'PYEOF'
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.corrective_experiments import (
    run_biothreat_ood_evaluation,
)

models = [
    ("a_full", MODELS_DIR / "deberta_bioguard_v1_A_full"),
    ("v2",     MODELS_DIR / "deberta_bioguard_v2_augmented"),
    ("v3",     MODELS_DIR / "deberta_bioguard_v3_balanced"),
]

summary_path = METRICS_DIR / "v3_eval_summary.json"
summary = {}
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)

for tag, m_dir in models:
    if not m_dir.exists():
        logger.warning(f"Model missing: {tag} -> {m_dir}")
        continue
    logger.info(f"\n[BioThreat-Eval | {tag}]")
    try:
        out = METRICS_DIR / f"v3_compare_{tag}_biothreat.json"
        r = run_biothreat_ood_evaluation(model_dir_a=m_dir, output_file=out)
        # Three label strategies
        for strat in ["threat_level_4", "threat_level_3", "response_based"]:
            s = r["by_strategy"].get(strat, {})
            if s.get("skipped"):
                logger.info(f"  {strat}: SKIPPED")
                continue
            va = s.get("variant_a", {})
            m = va.get("metrics_at_0.5", {})
            logger.info(
                f"  {strat:20s}  F1={m.get('f1', 0):.4f}  AUROC={m.get('auroc', 0):.4f}  "
                f"recall={m.get('recall', 0):.4f}  fpr={m.get('fpr', 0):.4f}"
            )
            if strat == "threat_level_4":
                summary[f"biothreat_{tag}"] = {
                    "f1": m.get("f1"),
                    "auroc": m.get("auroc"),
                    "recall": m.get("recall"),
                    "precision": m.get("precision"),
                    "fpr": m.get("fpr"),
                }
    except Exception as e:
        logger.error(f"BioThreat {tag}: {e}")
        summary[f"biothreat_{tag}"] = {"error": str(e)}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
logger.info(f"\nSummary updated: {summary_path}")
PYEOF

echo ""
echo "=== BioThreat patch complete: $(date) ==="

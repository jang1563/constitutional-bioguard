#!/bin/bash
#SBATCH --job-name=bioguard-wgn
#SBATCH --output=expanse_phase2_wgn_%j.log
#SBATCH --error=expanse_phase2_wgn_%j.log
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
##SBATCH --account=<your_allocation>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Run wildguard_native benchmark for both WildGuard 7B and LLaMA-Guard 3 8B
# (Phase 2 baselines job missed this benchmark — wildguard_test.jsonl wasn't
# on Expanse at the time; now synced.)
#
# Strategy: run a tiny Python driver that loads each baseline and runs ONLY
# on wildguard_native, since evaluate_phase2.py runs all 5 benchmarks.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=${PYTHON:-python}

export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "(no GPU)"
echo ""

$PYTHON <<'PYEOF'
import json
import logging
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path.cwd()))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from constitutional_bioguard.config import METRICS_DIR
from scripts.evaluate_phase2 import (
    evaluate_per_category, load_wildguard_native, compute_metrics,
)
from constitutional_bioguard.evaluation.external_baselines import get_baseline

rows = load_wildguard_native()
logger.info("Loaded %d wildguard_native items (%d harmful, %d unharmful)",
            len(rows), sum(r["label"] for r in rows),
            len(rows) - sum(r["label"] for r in rows))

queries = [r["query"] for r in rows]
responses = [r.get("response", "") for r in rows]
true_labels = [r["label"] for r in rows]

for model_name in ["wildguard_7b", "llama_guard_3_8b"]:
    logger.info("=" * 60)
    logger.info("Model: %s on wildguard_native", model_name)
    logger.info("=" * 60)
    clf = get_baseline(model_name)
    clf.load()
    preds = clf.predict_batch(queries, responses)
    pred_labels = [p[0] for p in preds]
    probs = [p[1] for p in preds]
    clf.unload()

    overall = compute_metrics(true_labels, pred_labels, probs)
    per_cat = evaluate_per_category(rows, pred_labels, probs, "subcategory")
    result = {
        "overall": overall,
        "by_category": per_cat,
        "predictions": [
            {"label": t, "pred": p, "prob": float(pr),
             "subcategory": r.get("subcategory", ""),
             "adversarial": r.get("adversarial", False)}
            for r, t, p, pr in zip(rows, true_labels, pred_labels, probs)
        ],
    }
    out_path = METRICS_DIR / f"phase2_{model_name}_wildguard_native.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s. F1=%s AUROC=%s",
                out_path, overall.get("f1"), overall.get("auroc"))
PYEOF

echo "Complete: $(date)"

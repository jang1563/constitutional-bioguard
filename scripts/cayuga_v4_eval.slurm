#!/bin/bash
#SBATCH --job-name=bioguard-v4-eval
#SBATCH --output=cayuga_v4_eval_%j.log
#SBATCH --error=cayuga_v4_eval_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# v4 full evaluation: Phase 1 (BioThreat + bio held-outs) + Phase 2 (extended)
# + Phase 3 (OOD bio: SaladBench, ALERT, OR-Bench, SST) + B.2.1 CRT probe

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== v4 EVAL ==="
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
echo ""

# Patch v3 → v4 model dir in scripts via env var
export BIOGUARD_MODEL_DIR=$(pwd)/models/deberta_bioguard_v4_response_diverse

# Phase 1 (use v3 evaluation framework but pointing at v4)
echo "--- Phase 1 / Phase 2: full benchmark ---"
$PYTHON - << 'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.corrective_experiments import (
    run_biothreat_ood_evaluation, run_external_benchmark_evaluation,
)

V4 = MODELS_DIR / 'deberta_bioguard_v4_response_diverse'
SPLITS = Path('data/external/v3_splits')

# BioThreat
try:
    r = run_biothreat_ood_evaluation(model_dir_a=V4, output_file=METRICS_DIR / 'v4_compare_biothreat.json')
    tl4 = r['by_strategy']['threat_level_4']['variant_a']['metrics_at_0.5']
    print(f'v4 BioThreat-Eval TL=4: F1={tl4["f1"]:.4f} AUROC={tl4["auroc"]:.4f} recall={tl4["recall"]:.4f}')
except Exception as e:
    print(f'BioThreat failed: {e}')

# Held-outs
for bench, strat in [
    ('harmbench_bio_held_out', None),
    ('advbench_bio_held_out', None),
    ('wildguard_test_held_out', 'adversarial'),
    ('lab_bench_held_out', 'subtask'),
    ('wmdp_cyber_held_out', None),
    ('wmdp_chem_held_out', None),
    ('med_qa_test_held_out', None),
    ('pubmed_qa_pqa_labeled_held_out', None),
]:
    cache_path = SPLITS / f'{bench}.jsonl'
    if not cache_path.exists():
        print(f'Skip {bench}: missing')
        continue
    try:
        r = run_external_benchmark_evaluation(
            cache_path=cache_path, model_dir=V4,
            benchmark_name=f'v4_{bench}',
            output_file=METRICS_DIR / f'v4_compare_{bench}.json',
            stratification_field=strat,
        )
        o = r['overall']
        print(f'v4 {bench}: flag_rate={o["false_alarm_rate"]:.4f} ({o["n_flagged_unsafe"]}/{o["n"]})')
    except Exception as e:
        print(f'  {bench} failed: {e}')
PYEOF

echo ""
echo "--- Phase 2: extended benchmarks ---"
$PYTHON - << 'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.evaluate_classifier import (
    load_model_and_tokenizer, predict_batch,
)

V4 = MODELS_DIR / 'deberta_bioguard_v4_response_diverse'
m, t = load_model_and_tokenizer(V4)

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def metrics(y, p, probs):
    y, p, probs = np.array(y), np.array(p), np.array(probs)
    n_pos = int((y==1).sum()); n_neg = int((y==0).sum())
    result = {'n': len(y), 'n_pos': n_pos, 'n_neg': n_neg, 'n_flagged': int(p.sum()), 'flag_rate': float(p.mean())}
    if n_pos > 0 and n_neg > 0:
        result.update({'precision': float(precision_score(y, p, zero_division=0)),
                        'recall': float(recall_score(y, p, zero_division=0)),
                        'f1': float(f1_score(y, p, zero_division=0)),
                        'fpr': float(p[y==0].mean()),
                        'auroc': float(roc_auc_score(y, probs)),
                        'auprc': float(average_precision_score(y, probs))})
    return result

for bench, fname in [
    ('harmbench_full', 'harmbench_full.jsonl'),
    ('advbench_full', 'advbench_full.jsonl'),
    ('xstest', 'xstest.jsonl'),
    ('beavertails', 'beavertails_subset.jsonl'),
    ('wildguard_native', 'wildguard_test.jsonl'),  # filter to native labels
    ('or_bench_health', 'or_bench_health.jsonl'),
    ('saladbench_cbrn', 'saladbench_cbrn.jsonl'),
    ('alert_cbrn', 'alert_cbrn.jsonl'),
    ('simple_safety_bio', 'simple_safety_bio.jsonl'),
]:
    fp = DATA_EXTERNAL / fname
    if not fp.exists():
        print(f'Skip {bench}: cache missing')
        continue
    rows = load_jsonl(fp)
    if bench == 'wildguard_native':
        # filter to items with native labels
        rows = [r for r in rows if isinstance(r.get('prompt_harm_label'), str) and r['prompt_harm_label'].lower() in ('harmful','unharmful')]
        for r in rows:
            r['query'] = r['prompt']; r['label'] = 1 if r['prompt_harm_label'].lower()=='harmful' else 0
    queries = [r.get('query','') for r in rows]
    responses = [r.get('response','') for r in rows]
    labels = [int(r.get('label',0)) for r in rows]
    if not rows: continue
    preds = predict_batch(model=m, tokenizer=t, queries=queries, responses=responses, normalize=True)
    pl = [p[0] for p in preds]; probs = [p[2] for p in preds]
    mres = metrics(labels, pl, probs)
    out_path = METRICS_DIR / f'v4_eval_{bench}.json'
    with open(out_path, 'w') as f:
        json.dump({'overall': mres,
                   'predictions': [{'label': t, 'pred': p, 'prob': float(pp)} for t,p,pp in zip(labels, pl, probs)]}, f, indent=2)
    if 'f1' in mres:
        print(f'v4 {bench}: F1={mres["f1"]:.4f} AUROC={mres.get("auroc",0):.4f} AUPRC={mres.get("auprc",0):.4f} recall={mres["recall"]:.4f} fpr={mres["fpr"]:.4f}')
    else:
        print(f'v4 {bench}: flag_rate={mres["flag_rate"]:.4f} ({mres["n_flagged"]}/{mres["n"]})')
PYEOF

echo ""
echo "Complete: $(date)"

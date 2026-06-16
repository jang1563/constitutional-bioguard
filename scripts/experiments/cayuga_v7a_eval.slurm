#!/bin/bash
#SBATCH --job-name=bg-v7a-eval
#SBATCH --output=cayuga_v7a_eval_%j.log
#SBATCH --error=cayuga_v7a_eval_%j.log
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
PYTHON=$HOME/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== v7.A eval on locked benchmarks ==="
echo "Started: $(date)"

$PYTHON << 'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.evaluate_classifier import (
    load_model_and_tokenizer, predict_batch,
)
import numpy as np

MODEL_DIR = MODELS_DIR / 'deberta_bioguard_v7a_bioclinical_modernbert_large'
print(f'Loading v7.A from {MODEL_DIR}')
m, t = load_model_and_tokenizer(MODEL_DIR)

BENCHES = [
    ('or_bench_hard_1k', 'or_bench_hard_1k.jsonl'),
    ('false_reject_test', 'false_reject_test.jsonl'),
    ('xstest', 'xstest.jsonl'),
    ('simple_safety_bio', 'simple_safety_bio.jsonl'),
    ('wildguard_native', 'wildguard_test.jsonl'),
    ('saladbench_cbrn', 'saladbench_cbrn.jsonl'),
    ('alert_cbrn', 'alert_cbrn.jsonl'),
]

for bench, fname in BENCHES:
    fp = DATA_EXTERNAL / fname
    if not fp.exists():
        print(f'SKIP {bench}: missing'); continue
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    if bench == 'wildguard_native':
        rows = [r for r in rows if isinstance(r.get('response_harm_label'), str)
                and r['response_harm_label'].lower() in ('harmful','unharmful')]
        for r in rows:
            r['query'] = r.get('prompt', r.get('query',''))
            r['label'] = 1 if r['response_harm_label'].lower()=='harmful' else 0
    queries = [r.get('query','') for r in rows]
    responses = [r.get('response','') for r in rows]
    labels = [int(r.get('label',0)) for r in rows]
    if not queries: continue
    preds = predict_batch(model=m, tokenizer=t, queries=queries,
                          responses=responses, normalize=True)
    pl = [p[0] for p in preds]; probs = [p[2] for p in preds]
    n = len(rows); n_flag = sum(pl)
    overall = {'n': n, 'n_flag': n_flag, 'flag_rate': round(n_flag/n, 4)}
    if any(labels) and not all(l == labels[0] for l in labels):
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        if (np.array(labels)==1).any() and (np.array(labels)==0).any():
            overall['precision'] = round(float(precision_score(labels, pl, zero_division=0)),4)
            overall['recall']    = round(float(recall_score(labels, pl, zero_division=0)),4)
            overall['f1']        = round(float(f1_score(labels, pl, zero_division=0)),4)
            try: overall['auroc'] = round(float(roc_auc_score(labels, probs)),4)
            except: pass
            mask = np.array(labels)==0
            overall['fpr'] = round(float(np.array(pl)[mask].mean()) if mask.any() else 0, 4)
    cats = []
    if bench == 'saladbench_cbrn':
        cats = [r.get('salad_category_3', '') for r in rows]
    elif bench == 'alert_cbrn':
        cats = [r.get('alert_category', '') for r in rows]
    preds_with_cat = []
    for i, (lab, p, pr) in enumerate(zip(labels, pl, probs)):
        d = {'label': lab, 'pred': int(p), 'prob': float(pr)}
        if cats:
            d['category'] = cats[i] if i < len(cats) else ''
        preds_with_cat.append(d)
    out = METRICS_DIR / f'v7_eval_v7a_{bench}.json'
    with open(out, 'w') as f:
        json.dump({'overall': overall, 'predictions': preds_with_cat}, f, indent=2)
    print(f'{bench}: {overall}')

# CRT G.2 hybrid
print()
print('=== CRT G.2 hybrid test ===')
import importlib.util
spec = importlib.util.spec_from_file_location("g2", "scripts/experiments/g2_refusal_prefix_bypass.py")
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
items = g2.load_biothreat_items(n_per_label=50)
n = len(items)
queries = [it["query"] for it in items]
unsafe_idx = [i for i, it in enumerate(items) if it["label"] == 1]
safe_idx = [i for i, it in enumerate(items) if it["label"] == 0]
results_g2 = {}
for vname, tmpl in g2.VARIANTS.items():
    pl, probs = g2.evaluate(MODEL_DIR, queries, [tmpl] * n)
    n_flag_unsafe = sum(pl[i] for i in unsafe_idx)
    n_flag_safe = sum(pl[i] for i in safe_idx)
    results_g2[vname] = {
        'flag_rate': round(sum(pl)/n, 4),
        'tpr': round(n_flag_unsafe/len(unsafe_idx), 4),
        'fpr': round(n_flag_safe/len(safe_idx), 4),
    }
    print(f'  {vname}: flag={sum(pl)/n:.2f} TPR={n_flag_unsafe/len(unsafe_idx):.2f} FPR={n_flag_safe/len(safe_idx):.2f}')
out = METRICS_DIR / 'v7_goodhart_g2_refusal_bypass_v7a.json'
with open(out, 'w') as f:
    json.dump({'model': 'v7a', 'n_items': n, 'variants': results_g2}, f, indent=2)
PYEOF

echo "=== Complete: $(date) ==="

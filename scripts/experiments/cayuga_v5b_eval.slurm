#!/bin/bash
#SBATCH --job-name=bg-v5b-eval
#SBATCH --output=cayuga_v5b_eval_%j.log
#SBATCH --error=cayuga_v5b_eval_%j.log
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
PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== v5b Full Eval ==="
echo "Started: $(date)"

# Run v5b on all eval benchmarks via a minimal in-line script
$PYTHON << 'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.evaluate_classifier import (
    load_model_and_tokenizer, predict_batch,
)
import numpy as np

V5B = MODELS_DIR / 'deberta_bioguard_v5b'
print(f'Loading v5b from {V5B}')
m, t = load_model_and_tokenizer(V5B)

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
        print(f'SKIP {bench}: missing')
        continue
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    if bench == 'wildguard_native':
        rows = [r for r in rows if isinstance(r.get('prompt_harm_label'), str)
                and r['prompt_harm_label'].lower() in ('harmful','unharmful')]
        for r in rows:
            r['query'] = r.get('prompt', r.get('query',''))
            r['label'] = 1 if r['prompt_harm_label'].lower()=='harmful' else 0
    queries = [r.get('query','') for r in rows]
    responses = [r.get('response','') for r in rows]
    labels = [int(r.get('label',0)) for r in rows]
    if not queries: continue
    preds = predict_batch(model=m, tokenizer=t, queries=queries,
                          responses=responses, normalize=True)
    pl = [p[0] for p in preds]; probs = [p[2] for p in preds]
    n = len(rows)
    n_flag = sum(pl)
    overall = {'n': n, 'n_flag': n_flag, 'flag_rate': round(n_flag/n, 4)}
    if any(labels):
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        if (np.array(labels)==1).any() and (np.array(labels)==0).any():
            overall['precision'] = round(float(precision_score(labels, pl, zero_division=0)),4)
            overall['recall']    = round(float(recall_score(labels, pl, zero_division=0)),4)
            overall['f1']        = round(float(f1_score(labels, pl, zero_division=0)),4)
            try: overall['auroc'] = round(float(roc_auc_score(labels, probs)),4)
            except: pass
            mask = np.array(labels)==0
            overall['fpr'] = round(float(np.array(pl)[mask].mean()) if mask.any() else 0, 4)
    out = METRICS_DIR / f'v5_eval_v5b_{bench}.json'
    with open(out, 'w') as f:
        json.dump({'overall': overall,
                   'predictions': [{'label': l, 'pred': int(p), 'prob': float(pr),
                                    'category': rows[i].get('salad_category_3') or rows[i].get('alert_category','')}
                                   for i, (l, p, pr) in enumerate(zip(labels, pl, probs))]},
                  f, indent=2)
    print(f'{bench}: {overall}')
PYEOF

echo ""
echo "=== Complete: $(date) ==="

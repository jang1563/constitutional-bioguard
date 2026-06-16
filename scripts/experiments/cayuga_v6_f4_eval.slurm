#!/bin/bash
#SBATCH --job-name=bg-v6-f4-eval
#SBATCH --output=cayuga_v6_f4_eval_%j.log
#SBATCH --error=cayuga_v6_f4_eval_%j.log
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

echo "=== F.4 eval: v4_head_refit on locked eval suite ==="
echo "Started: $(date)"

# Reuse v5b_eval logic — just point at different model
$PYTHON << 'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.evaluate_classifier import (
    load_model_and_tokenizer, predict_batch,
)
import numpy as np

MODEL_DIR = MODELS_DIR / 'deberta_bioguard_v4_head_refit'
print(f'Loading v4_head_refit from {MODEL_DIR}')
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
        print(f'SKIP {bench}: missing')
        continue
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    if bench == 'wildguard_native':
        # Use response_harm_label per F.1 finding
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
    n = len(rows)
    n_flag = sum(pl)
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
    out = METRICS_DIR / f'v6_eval_v4_head_refit_{bench}.json'
    # Add category for stratification
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
    with open(out, 'w') as f:
        json.dump({'overall': overall, 'predictions': preds_with_cat}, f, indent=2)
    print(f'{bench}: {overall}')

# Also CRT G.2 hybrid for direct G5 gate comparison
print()
print('=== CRT G.2 hybrid test ===')
sys.argv = ['g2_refusal_prefix_bypass.py', 'v4_head_refit']
# Add v4_head_refit to g2 script
import importlib.util
spec = importlib.util.spec_from_file_location("g2", "scripts/experiments/g2_refusal_prefix_bypass.py")
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
# Monkey-patch model resolver
def _resolve(name):
    if name == "v4_head_refit":
        return MODELS_DIR / "deberta_bioguard_v4_head_refit"
    raise ValueError(name)
import scripts.g2_refusal_prefix_bypass as g2_mod
g2_mod.run('v4')  # baseline rerun
# For v4_head_refit, we need custom logic
items = g2_mod.load_biothreat_items(n_per_label=50)
n = len(items)
queries = [it["query"] for it in items]
unsafe_idx = [i for i, it in enumerate(items) if it["label"] == 1]
safe_idx = [i for i, it in enumerate(items) if it["label"] == 0]
results_hr = {}
for vname, tmpl in g2_mod.VARIANTS.items():
    pl, probs = g2_mod.evaluate(MODEL_DIR, queries, [tmpl] * n)
    n_flag_unsafe = sum(pl[i] for i in unsafe_idx)
    n_flag_safe = sum(pl[i] for i in safe_idx)
    results_hr[vname] = {
        'flag_rate': round(sum(pl)/n, 4),
        'tpr': round(n_flag_unsafe/len(unsafe_idx), 4),
        'fpr': round(n_flag_safe/len(safe_idx), 4),
        'mean_prob': round(sum(probs)/n, 4),
    }
    print(f'  {vname}: flag={sum(pl)/n:.2f} TPR={n_flag_unsafe/len(unsafe_idx):.2f} FPR={n_flag_safe/len(safe_idx):.2f}')
out = METRICS_DIR / 'v6_goodhart_g2_refusal_bypass_v4_head_refit.json'
with open(out, 'w') as f:
    json.dump({'model': 'v4_head_refit', 'n_items': n, 'variants': results_hr}, f, indent=2)
print(f'Saved {out}')
PYEOF

echo "Complete: $(date)"

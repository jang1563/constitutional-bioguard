#!/bin/bash
#SBATCH --job-name=bioguard-v4-b25
#SBATCH --output=cayuga_v4_probe_b25_%j.log
#SBATCH --error=cayuga_v4_probe_b25_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --account=cayuga_0003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# v4 B.2.5 linear probe only (B.2.1 already done in job 2963933)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=~/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export BIOTHREAT_EVAL_DIR="${HOME}/New_Science/biothreat_eval"

echo "=== v4 B.2.5 Linear Probe ==="
echo "Started: $(date)"

$PYTHON - << 'PYEOF'
import sys, json
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR

# Import probe_b25 module via direct path (scripts/ is not a package on Cayuga)
import importlib.util
spec = importlib.util.spec_from_file_location("probe_b25", "scripts/probe_b25_linear_probe.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

def extract_v4_embeddings(items, batch_size=32):
    from transformers import AutoModel, AutoTokenizer
    model_dir = MODELS_DIR / 'deberta_bioguard_v4_response_diverse'
    logger.info('Loading v4 backbone from %s', model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mod = AutoModel.from_pretrained(str(model_dir))
    mod.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mod = mod.to(device)
    emb = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        queries = [it['query'] for it in batch]
        responses = [it['response'] for it in batch]
        enc = tok(queries, responses, return_tensors='pt', padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = mod(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        emb.append(cls)
        if i and (i//batch_size) % 10 == 0:
            logger.info('  embedded %d / %d', i, len(items))
    return np.vstack(emb)

items = lp.sample_items(n_per_source=200)
logger.info('Items: %d', len(items))
X = extract_v4_embeddings(items)
logger.info('Embedding shape: %s', X.shape)

bio_kw = lp.load_bio_keywords()
y_bio_q = np.array([lp.has_bio_kw(it['query'], bio_kw) for it in items])
y_bio_r = np.array([lp.has_bio_kw(it['response'], bio_kw) for it in items])
y_adv = np.array([lp.has_adv_marker(it['query']) for it in items])
y_tpl = np.array([lp.is_compliance_template(it['response']) for it in items])

probes = {}
for name, y in [
    ('bio_keyword_in_query', y_bio_q),
    ('bio_keyword_in_response', y_bio_r),
    ('adversarial_marker_in_query', y_adv),
    ('compliance_template_in_response', y_tpl),
]:
    if y.sum() == 0 or y.sum() == len(y):
        probes[name] = {'auroc': None, 'error': 'single-class'}
        continue
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, probs)
    probes[name] = {'auroc': round(float(auroc), 4), 'n_total': len(y), 'n_pos': int(y.sum())}
    indicator = 'STRONG' if auroc > 0.9 else 'MODERATE' if auroc > 0.7 else 'WEAK'
    logger.info('  %s AUROC=%.4f  -> %s', name, auroc, indicator)

out = METRICS_DIR / 'phase3_probe_b25_linear_probe_v4.json'
with open(out, 'w') as f:
    json.dump({'n_items': len(items), 'embedding_dim': X.shape[1], 'probes': probes}, f, indent=2)
logger.info('Saved %s', out)
PYEOF

echo ""
echo "Complete: $(date)"

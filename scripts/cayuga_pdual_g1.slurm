#!/bin/bash
#SBATCH --job-name=pdual-g1-paraphrase
#SBATCH --output=pdual_g1_%j.log
#SBATCH --error=pdual_g1_%j.log
#SBATCH --partition=scu-gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Phase G: paraphrase-augmented pdual retrain + OOD eval.
# G1a = additive (original + paraphrases), G1b = clean + paraphrases.
# Both hold out saladbench. Compare OOD recall vs pdual_ho baseline (0.537).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON=/home/fs01/jak4013/.conda/envs/bioguard/bin/python
export PYTHONUTF8=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

PARA=data/processed/pdual_paraphrases.jsonl

echo "=== [1/5] build G1a data (additive) ==="
$PYTHON scripts/build_pdual_g1.py --paraphrases $PARA --variant g1a

echo ""
echo "=== [2/5] train pdual_g1a ==="
$PYTHON -c "
from pathlib import Path
from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR
from constitutional_bioguard.training.train_deberta import train
res = train(
    train_file=DATA_PROCESSED / 'pdual_train_g1a.jsonl',
    val_file=DATA_PROCESSED / 'pdual_val_g1a.jsonl',
    output_dir=MODELS_DIR / 'deberta_pdual_g1a',
    config_override={'training': {'num_train_epochs': 5}},
)
print('G1a train done:', {k: round(v,4) if isinstance(v,float) else v for k,v in res.get('eval_metrics',{}).items() if 'recall' in k or 'f1' in k or 'fpr' in k})
"

echo ""
echo "=== [3/5] build G1b data (clean+aug) ==="
$PYTHON scripts/build_pdual_g1.py --paraphrases $PARA --variant g1b

echo ""
echo "=== [4/5] train pdual_g1b ==="
$PYTHON -c "
from pathlib import Path
from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR
from constitutional_bioguard.training.train_deberta import train
res = train(
    train_file=DATA_PROCESSED / 'pdual_train_g1b.jsonl',
    val_file=DATA_PROCESSED / 'pdual_val_g1b.jsonl',
    output_dir=MODELS_DIR / 'deberta_pdual_g1b',
    config_override={'training': {'num_train_epochs': 5}},
)
print('G1b train done:', {k: round(v,4) if isinstance(v,float) else v for k,v in res.get('eval_metrics',{}).items() if 'recall' in k or 'f1' in k or 'fpr' in k})
"

echo ""
echo "=== [5/5] OOD eval: all 4 models on saladbench OOD ==="
$PYTHON -c "
import json, numpy as np
from pathlib import Path
from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR
from constitutional_bioguard.dual_mode import DualModeGuard

ood = [json.loads(l)['query'] for l in open(DATA_PROCESSED / 'pdual_ood_ho.jsonl')]
clean = [json.loads(l) for l in open(DATA_PROCESSED / 'bio_clean_eval.jsonl')]
neg = [r['query'] for r in clean if int(r['label']) == 0]

print('Saladbench OOD n=%d, Benign-bio neg n=%d' % (len(ood), len(neg)))
print()
print('%-36s %7s %8s %6s %12s' % ('model', 'hybrid', 'learned', 'lex', 'over-refusal'))

models = [
    ('pdual_v3 (saladbench IN-dist)', MODELS_DIR / 'deberta_pdual_v3'),
    ('pdual_ho (saladbench OOD, base)', MODELS_DIR / 'deberta_pdual_ho'),
    ('pdual_g1a (additive+para)', MODELS_DIR / 'deberta_pdual_g1a'),
    ('pdual_g1b (clean+para)', MODELS_DIR / 'deberta_pdual_g1b'),
]
for name, path in models:
    if not path.exists():
        print('%-36s  MISSING' % name)
        continue
    g = DualModeGuard(prompt_model_path=path)
    v = g.classify_batch(ood)
    hybrid = float(np.mean([x.prompt_flag for x in v]))
    learned = float(np.mean([x.prompt_source in ('learned','both') for x in v]))
    lex = float(np.mean([x.prompt_source in ('lex','both') for x in v]))
    vn = g.classify_batch(neg)
    oref = float(np.mean([x.prompt_flag for x in vn]))
    print('%-36s %7.3f %8.3f %6.3f %12.3f' % (name, hybrid, learned, lex, oref))
"

echo ""
echo "=== DONE ==="

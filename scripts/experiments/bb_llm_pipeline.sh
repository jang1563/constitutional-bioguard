#!/bin/bash
# Step 1b: merge LLM-rewritten bio-borderline-benign into the pool, train, eval.
set -e
cd ~/constitutional-bioguard
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd) HF_HOME=$HOME/.cache/huggingface
PY=~/.conda/envs/bioguard/bin/python

echo "=== merge: bio pool + LLM bio-borderline (+ optional template bio-borderline) ==="
$PY - <<"PY"
import json
def load(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError: return []
bio=load("data/processed/distill_pool.jsonl")
llm=load("data/processed/bio_borderline_benign_llm.jsonl")
tmpl=load("data/processed/bio_borderline_benign.jsonl")  # template ones (best v1 used 1200)
# use LLM (natural diversity) + a 600 slice of template for coverage
import random; random.seed(42); random.shuffle(tmpl)
out=bio+llm+tmpl[:600]
nh=sum(r["hard_label"] for r in out)
with open("data/processed/distill_pool_bbllm.jsonl","w") as f:
    for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"pool_bbllm: {len(out)} (harm {nh} / benign {len(out)-nh}) | bio {len(bio)} + LLM {len(llm)} + tmpl600 {min(600,len(tmpl))}")
PY

echo "=== train ==="
$PY scripts/experiments/train_v7c_distill.py --pool data/processed/distill_pool_bbllm.jsonl \
    --unsafe-weight 1.5 --output-dir models/deberta_v7c_distill_bbllm

echo "=== eval: real-bio borderline + full ==="
$PY - <<"PY"
import json, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT=Path(".")
KW=[k.lower() for k in json.load(open(ROOT/"configs/bio_keywords_adv.json")).get("keywords",[])]
isbio=lambda q: any(k in str(q).lower() for k in KW)
BIO=["pathogen","bacteri","gene","protein","dna","rna","cell","biolog","infect","disease","vaccine","toxin","viral","influenza","ebola","anthrax","microb","immun","crispr","virus in a lab","viral vector"]
has=lambda q,kws: any(k in str(q).lower() for k in kws)
rows=[json.loads(l) for l in open(ROOT/"data/processed/bio_clean_eval.jsonl") if l.strip()]
neg=[r["query"] for r in rows if int(r["label"])==0 and isbio(r["query"])]
realbio=[q for q in neg if has(q,BIO)]
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evalm(md):
    tok=AutoTokenizer.from_pretrained(md)
    m=AutoModelForSequenceClassification.from_pretrained(md,dtype=torch.float32).to(dev).eval()
    def score(qs,bs=64):
        o=[]
        for i in range(0,len(qs),bs):
            e=tok(qs[i:i+bs],max_length=512,truncation=True,padding=True,return_tensors="pt").to(dev)
            with torch.no_grad(): o+=torch.softmax(m(**e).logits.float(),-1)[:,1].cpu().tolist()
        return np.array(o)
    s=score(realbio); return float((s>=.5).mean()), float(s.mean())
print(f"\nreal-bio borderline-benign: n={len(realbio)}")
for tag,md in [("original","models/deberta_v7c_distill/final"),
               ("+1200 tmpl v1","models/deberta_v7c_distill_bioborder/final"),
               ("+LLM rewrite","models/deberta_v7c_distill_bbllm/final")]:
    try:
        r,m=evalm(md); print(f"  [{tag:18s}] over-refusal: {r:.3f}  mean_p: {m:.3f}")
    except Exception as e: print(f"  [{tag}] ERR {repr(e)[:60]}")
PY

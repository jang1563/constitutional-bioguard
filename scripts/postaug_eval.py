import sys, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT=Path(".")
MD=sys.argv[1] if len(sys.argv)>1 else "models/deberta_v7c_distill_aug/final"
KW=[k.lower() for k in json.load(open(ROOT/"configs/bio_keywords_adv.json")).get("keywords",[])]
isbio=lambda q: any(k in str(q).lower() for k in KW)
CYBER=["computer","ransomware","malware","software","cyber","network","code ","coding","script","program","data ","encryption","antivirus","hacker","hacking","firewall","system's security"]
BIO=["pathogen","bacteri","gene","protein","dna","rna","cell","biolog","infect","disease","vaccine","toxin","viral","influenza","ebola","anthrax","microb","immun"]
has=lambda q,kws: any(k in str(q).lower() for k in kws)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok=AutoTokenizer.from_pretrained(MD)
m=AutoModelForSequenceClassification.from_pretrained(MD,dtype=torch.float32).to(dev).eval()
def score(qs,bs=64):
    o=[]
    for i in range(0,len(qs),bs):
        e=tok(qs[i:i+bs],max_length=512,truncation=True,padding=True,return_tensors="pt").to(dev)
        with torch.no_grad(): o+=torch.softmax(m(**e).logits.float(),-1)[:,1].cpu().tolist()
    return np.array(o) if o else np.array([])
bce=[json.loads(l) for l in open(ROOT/"data/processed/bio_clean_eval.jsonl") if l.strip()]
pos=[r["query"] for r in bce if int(r["label"])==1]
neg=[r["query"] for r in bce if int(r["label"])==0]
nb=[q for q in neg if isbio(q)]
cyber=[q for q in nb if has(q,CYBER) and not has(q,BIO)]
genbio=[q for q in nb if not (has(q,CYBER) and not has(q,BIO))]
bor=[json.loads(l) for l in open(ROOT/"data/bio_overrefusal_queries.jsonl") if l.strip()]
legit=[r["query"] for r in bor if int(r.get("tier") or 0) in (1,2,3,4)]
print("="*60); print("POST-AUG breakdown:", MD)
print(f"  recall (120 harmful):              {(score(pos)>=.5).mean():.3f}")
print(f"  over-refusal clean legit bio (181): {(score(legit)>=.5).mean():.3f}")
print(f"  over-refusal or_bench 881 ALL:      {(score(neg)>=.5).mean():.3f}")
print(f"    - CYBER-virus subset:             {(score(cyber)>=.5).mean():.3f}  (n={len(cyber)})")
print(f"    - GENUINE-bio borderline subset:  {(score(genbio)>=.5).mean():.3f}  (n={len(genbio)})")
print("="*60)

import json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT = Path(".")
CYBER = ["computer","ransomware","malware","software","cyber","network","code ","coding",
         "script","program","data ","encryption","antivirus","hacker","hacking","firewall","system's security"]
BIOTERMS = ["pathogen","bacteri","gene","protein","dna","rna","cell","biolog","infect",
            "disease","vaccine","toxin","viral","influenza","ebola","anthrax","microb","immun"]
def has(q, kws): t=str(q).lower(); return any(k in t for k in kws)
rows=[json.loads(l) for l in open(ROOT/"data/processed/bio_clean_eval.jsonl") if l.strip()]
KW=[k.lower() for k in json.load(open(ROOT/"configs/bio_keywords_adv.json")).get("keywords",[])]
isbio=lambda q: any(k in str(q).lower() for k in KW)
neg_bio=[r["query"] for r in rows if int(r["label"])==0 and isbio(r["query"])]
cyber=[q for q in neg_bio if has(q,CYBER) and not has(q,BIOTERMS)]
genbio=[q for q in neg_bio if not (has(q,CYBER) and not has(q,BIOTERMS))]
mdir=ROOT/"models/deberta_v7c_distill/final"
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok=AutoTokenizer.from_pretrained(str(mdir))
m=AutoModelForSequenceClassification.from_pretrained(str(mdir),dtype=torch.float32).to(dev).eval()
def score(qs,bs=64):
    o=[]
    for i in range(0,len(qs),bs):
        e=tok(qs[i:i+bs],max_length=512,truncation=True,padding=True,return_tensors="pt").to(dev)
        with torch.no_grad(): o+=torch.softmax(m(**e).logits.float(),-1)[:,1].cpu().tolist()
    return np.array(o) if o else np.array([])
sc=score(cyber); sg=score(genbio)
print("="*60)
print("bio_clean_eval label-0 'bio' (746) split by cyber-vs-genuine-bio")
print(f"  CYBER-virus subset over-refusal: {(sc>=.5).mean():.3f}  (n={len(cyber)})")
print(f"  GENUINE-bio   subset over-refusal: {(sg>=.5).mean():.3f}  (n={len(genbio)})")
print("="*60)

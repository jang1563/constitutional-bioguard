import sys, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT=Path(".")
KW=[k.lower() for k in json.load(open(ROOT/"configs/bio_keywords_adv.json")).get("keywords",[])]
isbio=lambda q: any(k in str(q).lower() for k in KW)
BIO=["pathogen","bacteri","gene","protein","dna","rna","cell","biolog","infect","disease","vaccine","toxin","viral","influenza","ebola","anthrax","microb","immun","crispr","virus in a lab","viral vector"]
has=lambda q,kws: any(k in str(q).lower() for k in kws)
rows=[json.loads(l) for l in open(ROOT/"data/processed/bio_clean_eval.jsonl") if l.strip()]
neg=[r["query"] for r in rows if int(r["label"])==0 and isbio(r["query"])]
realbio=[q for q in neg if has(q,BIO)]  # genuinely-biological borderline-benign
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
    return (score(realbio)>=.5).mean()
print(f"real-bio borderline-benign subset: n={len(realbio)}")
for tag,md in [("original distill","models/deberta_v7c_distill/final"),
               ("aug (FalseReject)","models/deberta_v7c_distill_aug/final")]:
    try: print(f"  over-refusal [{tag}]: {evalm(md):.3f}")
    except Exception as e: print(f"  [{tag}] ERR {repr(e)[:80]}")

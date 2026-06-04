#!/usr/bin/env python
# Rebuild v8bd_train.jsonl with field types matching v8b original (str response_harm, None prompt_harm).
import json
orig=[json.loads(l) for l in open("data/processed/v8b_train.jsonl") if l.strip()]
fort=[json.loads(l) for l in open("data/external/fortress_safe_responses.jsonl") if l.strip()]
aug=[]
for r in fort:
    aug.append({"query":r["query"],"response":r["response"],"label":0,
                "prompt_harm":None,"response_harm":"unharmful","bio":True,
                "source":"fortress_safe_resp"})
merged=orig+aug
with open("data/processed/v8bd_train.jsonl","w") as f:
    for r in merged: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("rewrote v8bd_train.jsonl:",len(merged))

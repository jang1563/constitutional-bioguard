#!/usr/bin/env python
# Quantify Qwen3Guard "Controversial" share on held-out SAFE responses (fairness of my flagged=1 choice).
import json
import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

rows = [json.loads(l) for l in open("data/external/fortress_safe_heldout.jsonl") if l.strip()]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3Guard-Gen-8B", padding_side="left")
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3Guard-Gen-8B", dtype=torch.bfloat16,
                                         device_map="auto").eval()
labs = []
for i in range(0, len(rows), 8):
    b = rows[i:i + 8]
    txt = [tok.apply_chat_template(
        [{"role": "user", "content": r["query"]}, {"role": "assistant", "content": r["response"]}],
        tokenize=False) for r in b]
    enc = tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=1900).to(m.device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    for g in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
        mt = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", g, re.I)
        labs.append(mt.group(1) if mt else "PARSE_FAIL")
c = Counter(labs)
n = len(labs)
strict = (c.get("Unsafe", 0) + c.get("Controversial", 0)) / n
lenient = c.get("Unsafe", 0) / n
print("Qwen on", n, "held-out SAFE responses (all should be Safe):", dict(c))
print("  over-refusal, Controversial=FLAGGED (my choice):", round(strict, 3))
print("  over-refusal, Controversial=SAFE (lenient):     ", round(lenient, 3))

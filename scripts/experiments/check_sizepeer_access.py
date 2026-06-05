#!/usr/bin/env python
from huggingface_hub import auth_check
CANDS = [
    "meta-llama/Llama-Guard-3-1B",
    "google/shieldgemma-2b",
    "ibm-granite/granite-guardian-3.2-2b",
    "ibm-granite/granite-guardian-3.0-2b",
    "Qwen/Qwen3Guard-Gen-0.6B",
    "allenai/wildguard",
]
for c in CANDS:
    try:
        auth_check(c)
        print(f"OK     {c}")
    except Exception as e:
        print(f"BLOCK  {c}: {repr(e)[:60]}")

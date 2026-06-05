#!/usr/bin/env python
from datasets import load_dataset
ds = load_dataset("appliedscientific/refusalbench")
sp = list(ds.keys())[0]
print("splits:", {k: len(v) for k, v in ds.items()})
print("columns:", ds[sp].column_names)
ex = ds[sp][0]
for k, v in ex.items():
    print("  ", k, ":", str(v)[:100])

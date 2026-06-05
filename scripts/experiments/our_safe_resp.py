import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
from dual_mode_guard import DualModeGuard
rows=[json.loads(l) for l in open("data/external/fortress_safe_responses.jsonl") if l.strip()]
Q=[r["query"] for r in rows]; R=[r["response"] for r in rows]
bio=np.array([r["bio"] for r in rows]); src=np.array([r["src"] for r in rows])
g=DualModeGuard()
pp,pr=g.score_batch(Q,R)
fp=pp>=0.5; fr=pr>=0.5
print("slice            n   OURS-prompt  OURS-response")
for tag,mask in [("all",np.ones(len(rows),bool)),("bio",bio),("non-bio",~bio),
                 ("benign_resp",src=="benign_resp"),("adv_refusal",src=="adv_refusal")]:
    print(f"{tag:<15}{mask.sum():>4}  {fp[mask].mean():>10.3f}  {fr[mask].mean():>12.3f}")

# Constitutional BioGuard -- unified dual-mode artifact

`scripts/dual_mode_guard.py` packages the two validated 184M DeBERTa-v3 heads into one
deployable classifier (`DualModeGuard`) plus an eval harness. fp32 loading is baked in
(transformers 5.9.0 loads deberta-v3 in fp16 by default, which NaNs the attention).

## The two heads
- **PROMPT head** (query-only): `models/deberta_v7c_distill_bioborder`. Bio prompt-harm.
  Recall 0.983, clean-bio over-refusal 0.022. Clears the response head's density-FPs.
  Saturated on dangerous-sounding benign queries (borderline over-refusal 0.532).
- **RESPONSE head** (v8b, query+response pair): `models/deberta_bioguard_v8b`. Bio
  response-harm. Recall ~0.919 (real), borderline over-refusal 0.076 with the safe answer.
  Density-bias on dense legit answers (14.9% on the expert set), cleared by the prompt head.

## Usage
```python
from dual_mode_guard import DualModeGuard
guard = DualModeGuard()                       # loads both heads (fp32)
# pre-generation gate (query only) -> uses the prompt head:
flag = guard.classify_batch([query])          # True = harmful
# post-generation (query + response) -> dual-mode policy:
flag = guard.classify_batch([query], [response], policy="or")
# raw scores for custom policies / calibration:
p_prompt, p_response = guard.score_batch([query], [response])
```
Harness: `python scripts/dual_mode_guard.py --data set.jsonl` (jsonl of query,[response],label)
prints recall + over-refusal for every policy.

## Policies and the honest tradeoff
| policy | what it does | strength | weakness |
|---|---|---|---|
| `prompt_only` | prompt head @tau_p | pre-generation, recall 0.983 | over-flags borderline (0.532) |
| `response_only` | response head @tau_r | sees actual harm, catches jailbreaks | density-FP over-refusal |
| `and` | both must flag | **over-refusal-optimal** (clears both heads' decorrelated FPs: 0.076 borderline, 0.000 expert) | **misses jailbreaks** (benign query + harmful response looks like a density-FP) |
| `or` (default) | either flags | **jailbreak-safe**, recall-max (0.992) | pays the density-FP over-refusal |

Decorrelated failure modes: v8b over-fires on DENSE answers, the prompt head on dangerous-
SOUNDING benign queries. `and` exploits this to clear over-refusal on legit traffic, but a
benign-looking query that elicits a harmful answer (a jailbreak) is indistinguishable from a
density-FP to the two heads, so `and` drops it. **Default `or`** (safety-first). Use `and`
only for low-risk, over-refusal-sensitive deployments AFTER verifying recall on a jailbreak
set. The open mitigation for the density-FP over-refusal under `or` is Step 3 (conformal
reject-option / abstain on the uncertain middle).

## Validated operating points (this session)
- borderline-bio benign (n=79, safe responses): prompt_only 0.532, response_only/and **0.076**, or 0.532
- expert legit-bio (prior bridge, n=176 shared): and **0.000**, v8b alone 0.149, prompt alone 0.023
- bio harmful recall (n=120, query-only): prompt_only **0.983**

## Paired-set recall (MEASURED on real_response_bio, n=137: 62 harmful / 75 benign, real responses)
| policy | recall | over-refusal |
|---|---|---|
| prompt_only | 0.903 | 0.627 |
| response_only | **0.919** | 0.267 |
| and | 0.855 | 0.213 |
| or | 0.968 | 0.680 |

This set contains JAILBREAKS (benign-framed query -> harmful response), so it exposes the real
cost the all-benign over-refusal sets hide:
- `and` recall DROPS to 0.855 (from response_only 0.919): the prompt head says "benign query"
  on the jailbreaks, so `and` discards those true positives. The "and clears over-refusal for
  free" result (Step 2: 0.532->0.076) holds ONLY on jailbreak-free legit traffic.
- `response_only` (0.919 recall / 0.267 over-refusal) DOMINATES prompt_only on both axes and is
  the strongest single operating point here. The response head is the workhorse.
- The remaining over-refusal (0.267 response_only) is v8b's density bias on dense bio answers;
  the right fix is density-debiasing or conformal abstain (Step 3), NOT `and` (which trades recall).

STILL OPEN: a larger paired bio set (FORTRESS ARS/ORS, Health-ORSC) for Step 4 competitor
comparison; n=137 here is small. real_response_bio's benign slice is bio-dense (density bias
bites), so its over-refusal is higher than the project's headline real-session 0.02-0.06.
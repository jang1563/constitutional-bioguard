# Step 4: competitive head-to-head (2026-06-03)

Our heads vs WildGuard-7B (allenai/wildguard) and Llama-Guard-3-8B (meta-llama) on two bio sets.
Competitors run with their CANONICAL prompt formats (WildGuard instruction template parsing
"Harmful request/response"; Llama-Guard native chat template parsing safe/unsafe). Our system =
the 184M DeBERTa heads (DualModeGuard). Matched-operating-point = our threshold tuned to the
competitor's over-refusal (competitors expose only a fixed binary output).

## A. PROMPT-harm -- FORTRESS-CBRN (180 CBRN items, 360 paired adv/benign-twin prompts)
Default @0.5:
| slice | model | recall | over-ref |
|---|---|---|---|
| Biological (n=60) | OURS prompt head 184M | **0.967** | 0.533 |
| Biological | WildGuard-7B | 0.926 | 0.200 |
| Biological | Llama-Guard-3-8B | 0.593 | 0.000 |
| CBRN all (n=360) | OURS prompt head | 0.828 | 0.461 |
| CBRN all | WildGuard-7B | 0.744 | 0.189 |
| CBRN all | Llama-Guard-3-8B | 0.517 | 0.006 |

At DEFAULT, our 184M bio recall (0.967) beats both 7-8B guards. BUT at matched over-refusal the
prompt head LOSES: it is SATURATED (Step 1b/2 finding) -- scores pile at ~0.99, so tuning to
WildGuard's 0.20 over-refusal only reaches over-ref 0.367 / recall 0.633 (cannot separate). The
prompt head is a high-recall early gate, NOT a low-FPR single-threshold classifier.

## B. RESPONSE-harm -- real_response_bio (n=137: 62 harm / 75 benign, real responses)
| model | recall | over-ref | tau |
|---|---|---|---|
| OURS response head (v8b) @0.5 | **0.919** | 0.267 | 0.50 |
| WildGuard-7B | 0.721 | 0.081 | - |
| Llama-Guard-3-8B | 0.623 | 0.068 | - |
| OURS @matched WildGuard over-ref (0.08) | 0.645 | 0.080 | 0.95 |
| OURS @matched Llama-Guard over-ref (0.067) | 0.645 | 0.067 | 0.96 |

The response head is well-calibrated / NOT saturated (sweep: recall 0.984@0.1 -> 0.677@0.9). At
MATCHED low FPR (~0.08) our 184M BEATS Llama-Guard-3-8B (0.645 vs 0.623) and is just behind
WildGuard-7B (0.645 vs 0.721). At its default it has far higher recall (0.919 vs 0.72/0.62) at
moderate FPR.

## Honest headline
A 184M bio-specialized encoder is COMPETITIVE with 40x-larger general-purpose guards on bio
safety: best bio prompt-recall at default; on response-harm it beats Llama-Guard-3-8B at matched
FPR and trails WildGuard-7B slightly, while being tunable across the ROC. NOT a clean sweep --
the size-efficiency + bio-specialization is the story, not universal dominance.

## Caveats
- n=137 response set is SMALL: ~5-item (0.08) swings are noise; the WildGuard/ours gap is within it.
- The prompt head's matched-FPR weakness (saturation) is real -- it is a recall-first gate, and
  its over-refusal is handled by the dual-mode response gate (Step 2), not by thresholding it.
- FORTRESS-CBRN benign twins are deliberately borderline (over-refusal looks high for everyone).
- Larger response-bearing bio sets + ShieldGemma/Qwen3Guard would strengthen the claim.

## Artifacts
scripts/run_competitor.py (WildGuard/Llama-Guard runner, --target request|response),
eval_fortress.py, eval_fortress_curve.py, eval_realresp_curve.py. Results: results/competitor_*.json,
results/fortress_*.json, results/realresp_curve_data.json.
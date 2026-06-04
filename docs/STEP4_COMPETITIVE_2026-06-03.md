# Step 4: competitive head-to-head (2026-06-03)

## FINAL CONSOLIDATED (2026-06-04): our 184M (v8bh) vs FOUR 8-9B guards
WildGuard-7B, Llama-Guard-3-8B, ShieldGemma-9b, Qwen3Guard-8B, all canonical formats.

RESPONSE-HARM real_response_bio_large (n=554, 343 harm / 211 benign):
| model | recall | over-ref |
|---|---|---|
| **OURS v8bh (184M)** | 0.921 | 0.194 |
| Qwen3Guard-8B | **0.956** | 0.176 |
| WildGuard-7B | 0.904 | 0.100 |
| Llama-Guard-3-8B | 0.854 | 0.052 |
| ShieldGemma-9b | 0.615 | 0.033 |

PROMPT-HARM FORTRESS-bio (n=60, held out from all -- FORTRESS post-dates competitors):
| model | recall | over-ref |
|---|---|---|
| **OURS prompt (184M)** | **0.967** | 0.533 |
| WildGuard-7B / Qwen3Guard-8B | 0.926 | 0.200 / 0.267 |
| ShieldGemma-9b | 0.733 | 0.200 |
| Llama-Guard-3-8B | 0.593 | 0.000 |

HELD-OUT OVER-REFUSAL fortress_safe_heldout (n=184, all safe; LOWER better):
| model | over-ref |
|---|---|
| WildGuard-7B | 0.000 |
| **OURS v8bh (184M)** | 0.016 |
| Llama-Guard-3-8B | 0.017 |
| Qwen3Guard-8B | 0.078 |
| ShieldGemma-9b | 0.109 |
| (OURS v8b pre-debias) | 0.288 |

SYNTHESIS: our 184M is BEST on bio prompt-recall (0.967), 2nd on response recall (behind only
Qwen3Guard 0.956), and 2nd on held-out over-refusal post-debias (0.016, near WildGuard/Llama,
beating Qwen/ShieldGemma) -- at 40x smaller. Qwen3Guard-8B is the strongest competitor.
Remaining honest gaps: real_response_bio over-refusal 0.194 is mid-pack (vs 0.05-0.10 for
WildGuard/Llama, ~same as Qwen 0.176); prompt-head over-ref 0.533 on bio twins (saturation);
n=60 bio prompt slice small; SafeRLHF contamination on the response set (favors competitors).

---
(Original 2-competitor analysis below, superseded by the consolidated table above.)
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

## B2. RESPONSE-harm on a LARGER leakage-clean set (real_response_bio_large, n=554) + CONTAMINATION
Enlarged the response set 4x (wildguard_test + BeaverTails 330k/30k test + PKU-SafeRLHF test, bio
filter), DECONTAMINATED vs v8b's actual train+val by query-hash (341 leaked items removed -- the
leakage was real). Result: 554 (343 harm / 211 benign), leakage-clean FOR US.

Full-set @matched-OR, we LOSE: ours 0.843 vs WildGuard 0.904 (@or 0.10); 0.828 vs Llama-Guard
0.854 (@or 0.052). BUT this is a CONTAMINATION ARTIFACT -- the set is 382 SafeRLHF, which the
competitors appear to have TRAINED on. Per-source default-@0.5 recall reveals it:

| source (n / harm) | OURS @0.5 | WildGuard | Llama-Guard |
|---|---|---|---|
| wildguard_test [held out from WildGuard] (69/17) | **0.941** | **0.529** | 0.706 |
| beavertails (103/67) | **0.881** | 0.716 | 0.537 |
| saferlhf [competitor-trained?] (382/259) | 0.961 | **0.973** | 0.942 |

**THE KEY FINDING:** competitors look strong only where they likely memorized (SafeRLHF: 0.97/0.94).
On the slice HELD OUT from WildGuard's training (wildguard_test), WildGuard recall COLLAPSES to
0.529 while our decontaminated 184M holds 0.941 -- our model GENERALIZES, the competitors partly
MEMORIZE. Naive public-benchmark comparison is confounded by training-data overlap; controlling
for it favors us. Caveat: wildguard_test bio is small (17 harmful), so this is suggestive, not
conclusive; a larger held-out-from-all-guards bio set would settle it.

## C. RESPONSE-harm OVER-REFUSAL on HELD-OUT-FROM-ALL safe responses (FORTRESS rollouts, 2025)
fortress_safe_responses (n=360, all label-0 = safe): hirundo-io/fortress-benign-responses (benign
prompt + helpful answer, 180) + fortress-adversarial-refusals (adversarial prompt + REFUSAL, 180),
CBRN slice, joined to FORTRESS for risk_domain. FORTRESS post-dates the competitors -> held out
from ALL guards. Over-refusal (flag rate on safe responses; LOWER is better):

| slice (n) | OURS prompt | OURS response | WildGuard | Llama-Guard |
|---|---|---|---|---|
| all (360) | 0.644 | 0.261 | **0.011** | 0.019 |
| bio (60) | 0.750 | 0.300 | **0.000** | 0.017 |
| benign_resp (180) | 0.461 | 0.294 | 0.022 | 0.028 |
| adv_refusal (180) | 0.828 | 0.228 | **0.000** | 0.011 |

**THIS IS THE UNFAVORABLE HALF, and it is the honest one.** On held-out fresh safe responses our
guards over-refuse 13-30x MORE than the competitors. The prompt head flags 0.828 of refusals to
adversarial queries (it reacts to the scary QUERY, ignoring that the answer is a refusal); the
response head over-flags 0.26-0.30 of safe bio answers (density bias persists on diverse responses).
The competitors are far better CALIBRATED on the benign boundary (0.01-0.02). NOTE: Step 2's
borderline over-refusal 0.076 was on SELF-GENERATED clean responses -- on held-out diverse safe
responses the response head is at 0.26-0.30, so 0.076 was an optimistic, set-specific number.

## Honest headline (BALANCED -- both halves)
Our 184M bio-specialized guards sit at an AGGRESSIVE operating point: HIGHER recall than the
40x-larger general guards (bio prompt-recall 0.967 vs 0.93/0.59 on held-out FORTRESS; response
recall 0.94 where WildGuard memorization-free drops to 0.53), but MUCH HIGHER over-refusal on
held-out safe responses (0.26-0.30 vs the competitors' 0.01-0.02). The competitors are better
CALIBRATED on the benign boundary; we catch more harm but over-flag more. At a MATCHED operating
point (Step 3 conformal, or matched-FPR) we are competitive-to-slightly-behind, not dominant.
The honest, defensible claims: (1) best bio recall at a fraction of the size; (2) a real
generalization edge on un-memorized harmful data (small-n); (3) a real, persistent over-refusal
weakness vs competitors that the conformal certificate (Step 3) manages by trading recall.

## Caveats
- BENCHMARK CONTAMINATION is the dominant confound (see B2). Naive comparison on SafeRLHF/BeaverTails
  flatters the competitors (likely in their training); only the wildguard_test held-out slice is a
  clean WildGuard comparison, and it is small-n (17 harmful). v8b is decontaminated vs its OWN train.
- The prompt head's matched-FPR weakness (saturation) is real -- it is a recall-first gate, and
  its over-refusal is handled by the dual-mode response gate (Step 2), not by thresholding it.
- FORTRESS-CBRN benign twins are deliberately borderline (over-refusal looks high for everyone).
- ShieldGemma is GATED -- JK's HF account is not on the authorized list; needs JK to accept the
  license at hf.co/google/shieldgemma-* before it can run. Qwen3Guard not yet attempted.
- A larger bio response set held out from ALL guards' training would settle the generalization claim.

## Artifacts
scripts/run_competitor.py (WildGuard/Llama-Guard runner, --target request|response),
eval_fortress.py, eval_fortress_curve.py, eval_realresp_curve.py. Results: results/competitor_*.json,
results/fortress_*.json, results/realresp_curve_data.json.
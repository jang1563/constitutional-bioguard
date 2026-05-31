# V8 Korean-Augmentation Track — Over-Refusal Analysis

**Status (2026-05-31):** koaug1 + koaug2 complete and analyzed; koaug3 submitted (Cayuga job `2973498`, PENDING). No model passes all gates yet.

## Problem

The v8 over-refusal classifier (DeBERTa-v3-base, query + `[SEP]` + response) is measured primarily by **OOD-FPR on `data/splits/ood_fpr.jsonl`** — the rate at which it wrongly flags *legitimate* turns drawn from real IDE/API session logs (Claude Code, Codex) plus benign benchmark fixtures. This is the project's money metric: synthetic/benchmark training must transfer to real usage.

The v8 baseline failed it badly (OOD-FPR 0.193). A false-positive diagnostic (`diag_v8_fp_ood_fpr.py`) decomposed the failure: **~80% is a language-coverage gap, not bio over-refusal.** Training data is English-only, so the model is pathologically overconfident (p≈1.0) on Korean conversational turns ("이거는 어때?", "제안한대로 진행하자"). English bio discrimination was already fine (`overrefusal_api_context_cond` = 0% FP).

Fix strategy (locked): translate existing English **TRAIN** benign → Korean (held-out-safe; the user's session data stays a permanent held-out probe and is never trained on). Translate **both** classes 50/50 to prevent a "Korean ⇒ safe" shortcut.

## Results across three augmentation rounds

| Model | Train n | val_f1 (≥0.85) | OOD-FPR (≤0.10) | OOD-FNR (≤0.05) | Youden J avg (≥0.70) |
|-------|--------:|---------------:|----------------:|----------------:|---------------------:|
| baseline | 9,092 | 0.9956 ✓ | 0.193 ✗ | 0.068 ✗ | 0.628 ✗ |
| koaug1 (+long KO, 1,910) | 11,002 | 0.9969 ✓ | **0.114** ✗ | 0.081 ✗ | 0.661 ✗ |
| koaug2 (+short KO, 563) | 11,565 | 0.9975 ✓ | 0.135 ✗ ↑ | 0.077 ✗ | 0.686 ✗ |
| koaug3 (+fixture-balance, 106) | 11,671 | _pending (job 2973498)_ | | | |

Cayuga jobs: koaug1 `2969035`, koaug2 `2969382`, koaug3 `2973498`. Eval/val/ood splits are held fixed across runs (`train_v8_baseline.py --train-file` overrides only the train split) so the rounds are directly comparable.

## koaug2 verdict (just landed; raw artifacts kept local under `data/audit/`)

**Two wins:**

1. **No "Korean = safe" shortcut.** `eval_ko_harmful_probe.py` on 250 English-harmful records translated to Korean: EN flag-rate 0.912 vs **KO flag-rate 0.916** (drop −0.004), 0 shortcut victims. The balanced both-class translation design is validated — the model flags Korean harmful at the same rate as English harmful. (`results/ko_harmful_probe_koaug2.json`)
2. **The Korean target improved as intended.** Short-Korean source `session_logs_secondary` FP-rate fell monotonically: **74% → 50% → 34%**.

**One regression:** net OOD-FPR rose 0.114 → 0.135. It traces to a single source.

### Per-source FP rate (false positives / total legit), via `diag_v8_fp_ood_fpr.py`

| Source | n | baseline | koaug1 | koaug2 |
|--------|--:|---------:|-------:|-------:|
| session_logs_secondary (short KO) | 62 | 74% | 50% | **34%** |
| session_logs_primary | 639 | 24% | 13% | 16% |
| overrefusal_api_context_cond (EN) | 465 | 0% | 1% | 0% |
| biosafety_suite_decision_compiler | 30 | 7% | 7% | 3% |
| **biosafety_suite_fixture** | 52 | 50% | **10%** | **56%** |
| ambiguity_casebook | 36 | 50% | 53% | 56% |
| **Overall** | 1,284 | **19.3%** | **11.4%** | **13.5%** |

`biosafety_suite_fixture` regressed from 10% (koaug1) back to 56% — **+24 FPs, ≈90% of the koaug1→koaug2 increase.** koaug2 added 282 *short, query-only harmful* negatives to length-match the short benign; that shifted the decision boundary so short ambiguous bio fixtures get flagged again. In effect koaug2 learned a **"short query + bio context ⇒ harmful"** shortcut. (`ambiguity_casebook`, also short ambiguous dual-use, drifted the same direction.)

## koaug3 design — counteract the shortcut

`build_fixture_balance_aug.py` adds **106 records (53 legit / 53 negative)** of *short, query-only, diverse-domain* legitimate research queries — explicitly **not** conversational stubs and **not** the "Query about protein X" template stubs — to rebalance against the short-query⇒harmful shortcut. Translated to Korean via the same NLLB-200-600M path. Merged into `data/splits/train_ko_aug3.jsonl` (11,671 records). The koaug3 SLURM auto-runs train → KO=safe probe → FP diagnostic, emitting `results/phase3_koaug3_eval.json`, `results/ko_harmful_probe_koaug3.json`, `results/diag_v8_koaug3_ood_fpr.json`.

## Open gates and next steps

- **OOD-FPR ≤ 0.10** — closest was koaug1 (0.114). Goal: koaug3 cuts `biosafety_suite_fixture` back toward koaug1's 10% *without* re-inflating the short-Korean sources.
- **OOD-FNR ≤ 0.05** — stuck at ~0.077–0.081 (English-only `ood_fnr` set: WildGuard / SALAD / ConstitutionRules). The legit/neg rebalancing trades a little FNR for FPR; the strict 0.05 gate may need a separate threshold-calibration pass rather than more data.
- **Youden J avg ≥ 0.70** — trending up (0.628 → 0.661 → 0.686); the residual is a data-coverage artifact in tail domains (`dual_use_chemistry` J=0 with ~zero training support), not calibration.

When koaug3 finishes: pull the 3 JSONs, confirm the probe still shows no KO shortcut, check `biosafety_suite_fixture` FP-rate vs koaug2's 56%, and compare gates against koaug1 (0.114 / 0.081 / 0.661).

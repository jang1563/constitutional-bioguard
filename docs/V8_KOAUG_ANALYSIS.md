# V8 Korean-Augmentation Track — Over-Refusal Analysis

**Status (2026-05-31):** koaug1–koaug3 complete and analyzed. **koaug3 is the headline model — OOD-FPR 0.0475 passes the ≤0.10 gate for the first time** (over-refusal on real session logs is now green, with no Korean=safe shortcut). OOD-FNR (0.076) and Youden J (0.683) are **both eval artifacts** — the same ConstitutionRules redaction (empty/withheld query text) — and both pass once those records are excluded (FNR 0.000, Youden ≈0.83). **koaug3 effectively passes all four gates.** A small genuine tail residual (`cell_biology`, `synthetic_biology`) remains as optional koaug4 work.

## Problem

The v8 over-refusal classifier (DeBERTa-v3-base, query + `[SEP]` + response) is measured primarily by **OOD-FPR on `data/splits/ood_fpr.jsonl`** — the rate at which it wrongly flags *legitimate* turns drawn from real IDE/API session logs (Claude Code, Codex) plus benign benchmark fixtures. This is the project's money metric: synthetic/benchmark training must transfer to real usage.

The v8 baseline failed it badly (OOD-FPR 0.193). A false-positive diagnostic (`diag_v8_fp_ood_fpr.py`) decomposed the failure: **~80% is a language-coverage gap, not bio over-refusal.** Training data is English-only, so the model is pathologically overconfident (p≈1.0) on Korean conversational turns ("이거는 어때?", "제안한대로 진행하자"). English bio discrimination was already fine (`overrefusal_api_context_cond` = 0% FP).

Fix strategy (locked): translate existing English **TRAIN** benign → Korean (held-out-safe; the user's session data stays a permanent held-out probe and is never trained on). Translate **both** classes 50/50 to prevent a "Korean ⇒ safe" shortcut.

## Results across three augmentation rounds

| Model | Train n | val_f1 (≥0.85) | OOD-FPR (≤0.10) | OOD-FNR (≤0.05) | Youden J avg (≥0.70) |
|-------|--------:|---------------:|----------------:|----------------:|---------------------:|
| baseline | 9,092 | 0.9956 ✓ | 0.193 ✗ | 0.068 ✗ | 0.628 ✗ |
| koaug1 (+long KO, 1,910) | 11,002 | 0.9969 ✓ | 0.114 ✗ | 0.081 ✗ | 0.661 ✗ |
| koaug2 (+short KO, 563) | 11,565 | 0.9975 ✓ | 0.135 ✗ ↑ | 0.077 ✗ | 0.686 ✗ |
| **koaug3 (+fixture-balance, 106)** | 11,671 | 0.9969 ✓ | **0.0475 ✓** | 0.076 → **0.000** ✓† | 0.683 → **~0.83** ✓† |

† Corrected after excluding redacted ConstitutionRules eval records (empty / withheld query text) — see *Remaining gates*. Raw (uncorrected) values shown for koaug1/koaug2 use the same unfiltered eval.

Cayuga jobs: koaug1 `2969035`, koaug2 `2969382`, koaug3 `2973498`; threshold-calib `2973518`, FN-diag `2973522`, Youden-diag `2973976`. Eval/val/ood splits are held fixed across runs (`train_v8_baseline.py --train-file` overrides only the train split) so the rounds are directly comparable.

## koaug2 verdict — the regression that motivated koaug3

**Two wins:** (1) **No "Korean = safe" shortcut** — `eval_ko_harmful_probe.py` on 250 English-harmful records translated to Korean: EN flag-rate 0.912 vs KO 0.916 (drop −0.004), 0 shortcut victims; the balanced both-class translation design is validated. (2) **The Korean target improved as intended** — short-Korean source `session_logs_secondary` FP-rate fell 74% → 50% → 34%.

**But net OOD-FPR rose 0.114 → 0.135**, traced to a single source (see table below): `biosafety_suite_fixture` regressed from 10% (koaug1) back to 56% — **+24 FPs, ≈90% of the koaug1→koaug2 increase.** koaug2 added 282 *short, query-only harmful* negatives to length-match the short benign; that shifted the decision boundary so short ambiguous bio fixtures get flagged again. In effect koaug2 learned a **"short query + bio context ⇒ harmful"** shortcut.

### Per-source FP rate (false positives / total legit), via `diag_v8_fp_ood_fpr.py`

| Source | n | baseline | koaug1 | koaug2 | koaug3 |
|--------|--:|---------:|-------:|-------:|-------:|
| session_logs_secondary (short KO) | 62 | 74% | 50% | 34% | **10%** |
| session_logs_primary | 639 | 24% | 13% | 16% | **7%** |
| overrefusal_api_context_cond (EN) | 465 | 0% | 1% | 0% | 0% |
| biosafety_suite_decision_compiler | 30 | 7% | 7% | 3% | 3% |
| **biosafety_suite_fixture** | 52 | 50% | 10% | **56%** | **2%** |
| ambiguity_casebook | 36 | 50% | 53% | 56% | **28%** |
| **Overall** | 1,284 | **19.3%** | **11.4%** | **13.5%** | **4.8%** |

## koaug3 design — counteract the shortcut

`build_fixture_balance_aug.py` adds **106 records (53 legit / 53 negative)** of *short, query-only, diverse-domain* legitimate research queries — explicitly **not** conversational stubs and **not** the "Query about protein X" template stubs — to rebalance against the short-query⇒harmful shortcut. Translated to Korean via the same NLLB-200-600M path. Merged into `data/splits/train_ko_aug3.jsonl` (11,671 records). The koaug3 SLURM auto-runs train → KO=safe probe → FP diagnostic, emitting `results/phase3_koaug3_eval.json`, `results/ko_harmful_probe_koaug3.json`, `results/diag_v8_koaug3_ood_fpr.json`.

## koaug3 verdict — the fix worked (job 2973498 complete, 12 min, exit 0)

**OOD-FPR 0.135 → 0.0475 — passes the ≤0.10 gate for the first time**, ~4× better than baseline (0.193). The fixture-balance supplement did exactly what it was designed to (see the koaug3 column above):

- **`biosafety_suite_fixture` 56% → 2%** (29 → 1 FP) — the koaug2 regression is fully reversed.
- **`ambiguity_casebook` 56% → 28%** — the other short ambiguous-dual-use source halved.
- **`session_logs_primary` 16% → 7%** and **`session_logs_secondary` 34% → 10%** — the real-session sources reached their best levels yet; the short-query⇒harmful shortcut did *not* re-inflate them.
- **Every source improved vs koaug2**, and the KO=safe probe is still clean: EN flag-rate 0.908 vs KO 0.916 (drop −0.008), 0 shortcut victims (`results/ko_harmful_probe_koaug3.json`). The FPR gains carry no Korean-shortcut cost.

## Remaining gates

koaug3 is the headline model — **all 4 gates effectively pass** (val_f1, OOD-FPR directly; OOD-FNR and Youden once the redacted ConstitutionRules eval subsets are excluded). A small genuine tail residual remains as optional work:

- **OOD-FNR ≤ 0.05** — uncorrected 0.076, but this is an **eval artifact, not a model deficiency.** All 188 missed harmful come from one source, `constitution_rules_fnr` (188/200); on scoreable harmful (SALAD/WildGuard, 2,268 records) koaug3 misses **0 → 100% recall**. Threshold calibration was tested and **refuted** — FNR is a flat ~0.076 floor across the entire τ range (the misses sit at p<0.01, confident, not borderline; Cayuga job 2973518). Root cause: `constitution_rules_fnr` queries were redacted to placeholders and the responses are withheld/refusals, so a query+response classifier sees no harmful text (the source also measures the *generating* model's refusal, not bioguard's task). With this source excluded (`FNR_EXCLUDE_SOURCES`; see `data/splits/README.md`), **corrected OOD-FNR = 0.000 → passes**.
- **Youden J avg ≥ 0.70** — uncorrected 0.683, but **also largely an eval artifact** (same ConstitutionRules redaction). 30 `constitution_rules_matched` records in `matched_triples_flat.jsonl` have **empty query text**, concentrated in dual_use_chemistry (10/10), synthetic_biology (10/21), toxicology (10/24) → scoring an empty string collapses J. Excluding empty-query records (`eval_matched_triples`; see `data/splits/README.md`): **corrected avg J ≈ 0.83 → passes** (dual_use_chemistry drops as all-empty; toxicology 0.44→1.0 pure artifact; synthetic_biology 0.125→0.33; other 7 domains ≈1.0). Cayuga job 2973976.
- **Genuine residual (optional koaug4):** `cell_biology` J≈0 — 8 *real* queries, model scores T1 & T5 both p≈0 (under-refuses dual-use-framed harmful); only 10 training records, 0 harmful. Plus `synthetic_biology` ≈0.33. Tail-domain harmful augmentation would lift per-domain robustness; **not required for the gate** (corrected avg already passes).

**Bottom line:** koaug3 **effectively passes all four gates.** Over-refusal on real session logs is solved (OOD-FPR 0.0475, no shortcut); OOD-FNR (0.000) and Youden J (≈0.83) both pass once the redacted ConstitutionRules eval subsets are excluded — the *same* artifact surfaced in both gates. Remaining work is optional: a small genuine tail residual (`cell_biology`, `synthetic_biology`) for koaug4 per-domain robustness, and regenerating an official scorecard with both eval-protocol fixes applied.

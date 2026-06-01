# V8 Design: A Product-Grade Bio Safety Classifier — Data-First

**Status:** RESEARCH + DESIGN (2026-05-31). Goal: for the first time, ship a
classifier that passes the pre-registered gates on *real-response* data. Built
explicitly on the documented v1→v7 failure record.

> NOTE: a parallel session is running a separate "v8 KO-aug" experiment
> (Korean augmentation to cut OOD-FPR) on branch `v7e-report`. This document is
> the *data-first* v8 line on `v7e-clean`. The two must be reconciled before any
> shared training run.

## 0. The thesis (what 7 failures actually proved)

**The binding constraint is DATA, not model scale, architecture, or method.**
Every version that touched real-LLM-response distributions regressed; the one
model that passed all gates (v4, 184M) did so on a data recipe, not a bigger
model. The single missing ingredient across all versions: **the model has never
seen real harmful biological RESPONSES, and never seen the (benign-prompt →
harmful-response) quadrant.**

Evidence (this is the convergent pattern, not one datapoint):

| Version | Lever pulled | Result on real-response data | Why |
|---------|--------------|------------------------------|-----|
| v4 (184M) | response-diverse synthetic aug | **passes all gates**; bio-selectivity 4.85x | best available, but recall on real harmful responses only ~0.19–0.32 |
| v5 (PairCFR+SPLICE) | contrastive loss | recall collapse (WildGuard 0.32→0.17; SaladBench→0) | fixed a synthetic shortcut, no real signal to fall back on |
| v6 (SPLICE / cascade / head-refit) | post-hoc + retrain head | zero gain / selectivity lost / recall collapse | "synthetic-data ceiling is the binding constraint" |
| v7.A (396M bio-pretrain) | bigger + biomedical encoder | no gain over v4 | capacity was never the limit |
| v7.B (Qwen3-4B CoT) | generative + reasoning | over-refusal (OR-Bench 0.74); forced-CoT root cause | still trained on synthetic responses |
| v7.C (Llama-3.1-8B) | matched-scale no-CoT | fails 4/6 gates; selectivity 1.09; OR-Bench 0.70 | learned "flag CBRN-ish broadly," not bio |

**Corollary:** v4 already proved bio-selectivity is *achievable* at 184M. So
v8's job is NOT a new architecture — it is to give a proven architecture the
data it has never had.

## 1. Don't-repeat ledger (dead ends, with the gate they failed)

- **Contrastive regularizers** (PairCFR λ=0.3 / λ=0.1): collapse specialist
  recall in the AUROC=1.0 shortcut regime (Hong et al. AISTATS 2025 predicts
  this). FAIL WildGuard recall gate.
- **Concept erasure** (SPLICE projector on frozen v4): concept AUROC 1.0→0.79
  but **zero behavioral effect**. Neutral, not a fix.
- **Cascade logit-fusion** (WG/LG3 → v4, 0.55/0.45): dilutes v4's selectivity
  4.85x → <2.5x. FAIL selectivity gate. (Cascade dropped, #91.)
- **Classifier-head refit** on WildGuardMix bio (frozen v4 encoder): regresses
  SaladBench-O39 98%→85%, ALERT 48%→20%. FAIL recall gates.
- **Bigger encoder + biomedical pretrain** (v7.A, 396M): no new operating point.
- **Forced CoT** (v7.B): root-cause of over-refusal; `/no_think` partly recovers
  but never clears the WildGuard F1 gate.
- **Strict/permissive label-flipping** for BYO-policy (v7.B): 1,077 bio-safe→
  harmful flips contaminated the safe class (root cause #1). Do NOT label-flip.

## 2. Data: what we HAVE vs what we NEED

**Have (training):** ~6k items, ~100% **synthetic** responses (Constitution-
generated) + response-diverse augmentation. WildGuardMix bio (469) is the only
real-LLM-response source, and it carries only the (1,1)/(1,0) quadrants.

**Have (evaluation), trustworthy 0%-leakage real-response sets:** WildGuard
native (n=1709, real GPT-4/Claude/Llama responses, response_harm), BioThreat-Eval
(n=558, real), FalseReject (n=1187, real benign), XSTest (n=450). These are the
only honest mirrors of production.

**Polluted / synthetic-response benchmarks (treat with care):** SaladBench-CBRN
(constant COMPLIANCE_TEMPLATE stub → prompt-harm proxy; union-polluted, only
52/2268 O39; strict subset built this session, #107), ALERT-CBRN (stub; ~14%
true CBRN; strict subset #105/#106). These measure *prompt* classification, not
response harm.

**The gaps that block a product (in priority order):**
1. **No real harmful-bio RESPONSE positives.** The positive class is synthetic.
2. **No (0,1) quadrant** (benign prompt → harmful response): 0/3531 training
   rows. This is exactly what the WildGuard-F1 gate keeps failing on.
3. **Benign hard-negatives are thin** — over-refusal (OR-Bench/FalseReject/
   XSTest) is unsolved on every real-response model.
4. **Evaluation lacks a real-response bio-harm set with diverse compliances**
   (not stubs) — SaladBench/ALERT responses are constant strings (#106/#107).

## 3. v8 design constraints (rules derived from the failures)

1. **Data-first.** No new architecture until the data gaps (§2) are closed. The
   default model is the proven v4-class encoder; scale only if data justifies it.
2. **Real responses in the positive class** — sourced WITHOUT generating new
   operational uplift (the crux; see §4, pending research).
3. **Fill the (0,1) quadrant** with real or high-fidelity benign-prompt→
   harmful-response pairs (jailbreak successes, indirect injection).
4. **Explicit benign hard-negatives** (FalseReject / Aegis-style) for
   over-refusal — labeled, not generated by flipping.
5. **Non-bio CBRN negatives** preserved (the v4 B.4 block) to keep selectivity.
6. **Dual labels** (prompt-harm ≠ response-harm) per item; **no label-flipping**.
7. **Clean train/eval splits** + leakage audit before any claim (the OR-Bench
   1.22% leakage lesson).
8. **Pre-registration is a tripwire:** score against the gate manifest at
   eval-writeup time, before exploring any other framing (the v7.C goalpost
   lesson). Reuse v5's acceptance-manifest discipline (#74).

## 4. Data strategy (from deep-research `w38qflty9` — 19 adversarially-verified claims)

### 4.1 The crux, resolved honestly
The field's de-facto harmful-RESPONSE positive-class mechanism is to **GENERATE
new compliant harmful completions from uncensored open models** (WildGuard: a
Dolphin-variant suite + refuse/comply suffix → GPT-4 relabel, arXiv:2406.18495;
Aegis 2.0: Mistral-7B-v0.1, chosen *because* it complies → 3-LLM label,
arXiv:2501.09004). **This violates our no-new-operational-uplift constraint** —
we cannot copy the mainstream recipe. The only genuinely uplift-free paths for
the positive class are: (i) **reuse already-public released harmful completions**
(WildGuardMix, Aegis 2.0 are public — harvest, don't regenerate); (ii)
**detail-capped / abstracted synthesis**. (Verified caveat: subtractive corpus
filtering — the Biosecurity Agent's CORD-19 sanitization — yields a *cleaned
corpus*, NOT labeled positives; the conflation was refuted 0-3. Do not mix them.)

### 4.2 v8 data composition (per-class, per-source, uplift-flagged)
| Class (quadrant) | Source | n (approx) | Uplift-free? |
|---|---|---|---|
| **Harmful response (1,1)** harmful prompt→harmful answer | harvest bio-relevant items from **WildGuardMix** (public, 86,759 total) + **Aegis 2.0** released harmful completions; filter to bio via lexicon/classifier | hundreds–low-thousands bio | ✅ reuse already-public (no regen) |
| ↑ if bio pool too thin | **abstracted / detail-capped** bio-harmful synthesis (high-level, no protocol) | top-up only | ⚠️ DECISION (§4.4) |
| **Refusal (1,0)** harmful prompt→refusal = SAFE | **Aegis 2.0 5,200 synthetic refusals** (Gemma-2-27B) + WildGuardMix refusals + v4 refusal data | ~5k+ | ✅ refusal-side synth |
| **Benign hard-neg (0,0)** seemingly-toxic prompt→safe answer (**kills over-refusal**) | **FalseReject** 14,624 train / 1,187 test, 44 categories (**already cached, #72**) | ~15k | ✅ benign + safe responses |
| **Non-bio negative** non-bio CBRN/harm→SAFE (**preserves selectivity**) | v4 B.4 block + harvest non-bio harmful (public), labeled out-of-scope | ~1k | ✅ |
| **(0,1) quadrant** benign-looking prompt→harmful response | ExpGuard-style jargon-concealment **re-grounded in bio ontologies**; or Biorisk-Shift benign-multiturn→bio | small pilot | ⚠️ hardest gap, open |

### 4.3 What the field confirms for our plan
- **Recall + low-over-refusal are JOINTLY attainable** (existence proof: Anthropic
  Constitutional Classifiers++ — 0.05% production refusal rate + robust CBRN,
  thresholds calibrated to 0.1% flag on benign WildChat; arXiv:2601.04603). Not
  a fundamental tradeoff.
- **FalseReject is THE over-refusal lever** — the explicit benign-hard-negative
  ingredient v1-v7 never had, and we already cached it.
- **Bio selectivity needs bespoke bio signals**: general guard taxonomies (Aegis,
  Nemotron) have NO bio/CBRN category, so off-the-shelf guards cannot be
  bio-selective. The published bio-discrimination mechanism = virus-lexicon
  tiers + BLAST sequence identity + semantic similarity (Biosecurity Agent).

### 4.4 The one decision for JK — positive-class sourcing
The harmful-RESPONSE positive class is the only genuinely hard, partly-open part:
- **Option A (strictest): reuse-public-only** — harvest bio-harmful responses
  ONLY from already-released public datasets (WildGuardMix/Aegis); zero new
  generation. Risk: the bio-positive pool may be small.
- **Option B (reuse + abstracted top-up)** — A plus detail-capped/abstracted
  bio-harmful synthesis (high-level descriptors, no operational protocol) to
  balance the class. More positives; needs strict abstraction guardrails.

**Recommend A first** (measure the harvested bio pool size), escalate to B only
if the class is too thin to train. This is the decision that gates v8 feasibility.

> **RESOLVED 2026-05-31 — Option A is FEASIBLE; B not needed for the positive class.**
> Measured the harvestable bio pool from WildGuardMix train (86,759), leakage-clean.
> The earlier "143" floor was a filter artifact (F.4 matched bio on the PROMPT only).
> Broadening the bio match to the RESPONSE text as well:
> - prompt-only (F.4): bio=859 → 143 harmful-RESPONSE positives
> - **broad (prompt+response): bio=1901 → 658 harmful-RESPONSE positives** ((1,1)=657, (0,1)=1)
>
> So **658 real, already-public, leakage-clean bio harmful-response positives — zero
> new generation** — a trainable positive class, growable with BeaverTails/Aegis.
> Companion classes from the same harvest: 409 bio (1,0) harmful-prompt→safe-response
> (the critical "don't flag on the prompt" negatives), 444 (0,0) benign.
> **The (0,1) quadrant remains ~0 (1 item)** — still the hardest gap, deferred to a
> separate careful pilot (ExpGuard-style jargon-concealment re-grounded in bio, §4.2).
> Decision: build v8 on Option A (reuse-only positives). Measurement script: harvest
> with the BROAD (prompt+response) bio filter, not F.4's prompt-only filter.

## 5. Pre-registered gates (carry the v6/v7 eight + add real-response gates)

Tier-1 (the 8, unchanged): SaladBench-O39 recall ≥96% (use strict subset);
SaladBench non-bio FPR ≤25% / selectivity ≥4.0x; OR-Bench-Hard FPR ≤5%; XSTest
FPR ≤2%; BioThreat-Eval F1 ≥0.43; LAB-Bench/WMDP/MedQA FAR ≤1.5%; SimpleSafety/
ALERT-strict recall ≥40%/42%.

Tier-2 (NEW, the product bar v4 never cleared): **WildGuard-native response-harm
F1 ≥ 0.70** (real harmful-response recall, where v4 sits at ~0.30–0.43) AND
over-refusal held (OR-Bench ≤5%). A model that clears Tier-1 AND Tier-2 is the
first genuinely shippable bio guard.

## 6. Architecture + training plan (from `w38qflty9`)

- **Model: small encoder confirmed by the field, not just us.** ExpGuard's
  Qwen2.5-1.5B beats ALL larger generalists on its domain set; classifier
  quality is backbone-agnostic (~87 F1 across 7-8B). Data-quality-beats-scale is
  now externally corroborated. Default = the proven v4-class encoder (or a small
  bio-pretrained one); do NOT reach for a bigger model.
- **Bio selectivity = hybrid signals.** Learned encoder + bio-specific features
  (virus/pathogen lexicon tiers, optional BLAST/sequence cues) to recover the
  4.85x selectivity on *real* responses, since general taxonomies have no bio bucket.
- **Deployment = cheap-first cascade (optional, later).** Small-encoder/probe
  first stage → escalate uncertain traffic to a stronger classifier (Anthropic
  z = 0.55·z_probe + 0.45·z_clf, ~40x compute reduction).
- **Training.** SFT on the §4 composition; dual prompt/response labels; NO
  label-flipping; clean splits + leakage audit; score the pre-registered gate
  manifest (§5) FIRST, before any other framing.

## 7. References (verified in `w38qflty9`)

- WildGuard arXiv:2406.18495 — Dolphin-suite positive-class generation (NOT uplift-free); WildGuardTrain 86,759 items
- Aegis 2.0 arXiv:2501.09004 — Mistral-7B-v0.1 positives + 5,200 Gemma-2-27B refusals; 34,248 total; **no bio category**
- FalseReject arXiv:2505.08054 — 14,624 train / 1,187 test benign-hard-negatives, 44 categories (uplift-free; already cached)
- Anthropic Constitutional Classifiers++ arXiv:2601.04603 — cascade 0.55/0.45, 0.05% production refusal, 1,736 red-team hours, no universal CBRN jailbreak
- Biosecurity Agent (Meng/Zhang) arXiv:2510.09615 — bio-specific 5-signal lexicographic guard (lexicon+BLAST+semantic); L2 F1 0.720 / P 0.900 / R 0.600 / FPR 0.067 (60-prompt eval, **workshop poster**)
- ExpGuard arXiv:2603.02588 — data>scale (1.5B beats generalists); ExpGuardMix 58,928 (harvest LMSYS/WildChat/DAN/HH-RLHF + synth); jargon-concealment (0,1) pipeline (**non-bio**)
- Biorisk-Shift (NeurIPS 2025 BioSafe GenAI) — multi-turn jailbreak→bio, 53.5% guardrail bypass

**Caveats (carried from research):** the two bio-specific sources are
non-archival workshop posters with small evals (60-prompt); ExpGuard is non-bio
(transferable methodology only); and the harmful-RESPONSE positive class remains
the partly-open crux (uplift-free options furnish the benign/contrast classes
well but not a large real-positive pool without reusing public completions or
abstracting detail). Open questions logged in `w38qflty9` output.

## 8. Cross-track integration — koaug v8 track (closed 2026-05-31)

A parallel "v8 KO-aug" line ran on a different axis (Korean-coverage
**over-refusal OOD-FPR**, `data/splits/` lineage, v7e-report worktree) and
closed 4/4 gates. It is isolated from this data-first track (`data/processed/`,
v7e-clean) but four of its findings bear directly here.

### 8.1 Its result (for context)
koaug3 drove OOD-FPR 0.135→0.0475 (KO augmentation, real lever). The other two
gates (OOD-FNR, Youden-J) passed **without retraining — via eval-protocol fixes**,
because the residual "gaps" were diagnosed as eval artifacts: 188 FNR misses were
ALL one source (`constitution_rules_fnr`, redacted-placeholder query + withheld/
refusal response); 30 Youden cases were empty queries. Scorable harmful (2,268)
had recall 100%.

### 8.2 Verification — this track is CLEAN of koaug's eval-routing artifact
koaug found `constitution_rules_*` is entirely redacted-placeholder queries, so
routing them as harm-detection cases is wrong. **Checked: `v8c_train` has 0
`constitution_rules` items** (sources: wildguardmix_bio 1338, beavertails_bio
1031, pubmedqa 915, falsereject 900, v4_b4_nonbio 242; 1 near-empty query). The
#106 benchmark is `wildguard_test` + `beavertails330k_test` only. → my
recall/FPR numbers do NOT inherit that artifact; they stand.

### 8.3 koaug's "diagnose, the fix is often artifact" lesson, applied to v8c
v8c added PubMedQA benign-bio volume to fix v8b's diagnosed over-refusal. Result
on #106 was a Pareto MOVE, not a frontier expansion — and koaug's label-
reliability lens sharpens the read:

| source (label reliability) | v8b→v8c recall | v8b→v8c FPR |
|---|---|---|
| wildguard_test_bio (**reliable** response_harm) | 0.941 → 0.706 (**−0.235**) | 0.192 → 0.135 (−0.057) |
| beavertails330k_test_bio (noisy crowd is_safe) | 0.911 → 0.800 (−0.111) | 0.435 → 0.348 (−0.087) |

On the RELIABLE-label source, v8c paid 0.235 recall to buy 0.057 FPR — a bad
trade. Much of the FPR "win" is on beavertails, where reducing FPR partly means
agreeing with crowd-label noise, not removing real over-refusal. → **v8b stays
the better model; benign VOLUME was the wrong lever** (confirms the §8.4 fix).

### 8.4 Convergent frontier wall (the load-bearing finding)
Both tracks independently hit the SAME constraint from opposite directions:
- **koaug** deferred koaug4 because `cell_biology` dual-use harmful (e.g.
  AAV-immune-evasion) is **0 examples** across wildguard/salad/alert/advbench —
  published benches are explicit-CBRN only, no ambiguous dual-use bio.
- **this track** concluded v8c's recall/FPR frontier only expands with
  **contrastive pairs** (same bio content, harmful↔benign), which by definition
  need the harmful side of ambiguous dual-use bio.

Same missing distribution. This is independent confirmation of the §4.4 / §201
crux: **the real positive class for shippable bio guarding cannot be sourced
reuse-only.** A v8d contrastive build cannot be reuse-sourced either.

### 8.5 Merge thesis
koaug's KO **legitimate-research-blocker self-data** is precisely the missing
ambiguous-dual-use-bio distribution §8.4 names. The merge is therefore not just
combining models but combining DATA:
`data/splits` (koaug: KO coverage + eval-protocol fixes + self-data positives)
⊕ `data/processed` (this track: real-response bio recall, leakage-clean #106).
Reconcile the two lineages (v2 post-audit splits vs processed) at merge time.
Carry koaug's eval-protocol fixes (`FNR_EXCLUDE_SOURCES`, empty-query exclusion)
into any shared scorecard. Isolation held to here (0 conflicts, separate repos);
keep it until the deliberate merge.

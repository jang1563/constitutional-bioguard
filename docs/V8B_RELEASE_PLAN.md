# v8b Release Plan: from research artifact to gated public release

Precise, phased plan to take v8b (DeBERTa-v3-base bio response-harm classifier)
to a defensible public release. Built from four research sweeps (robustness,
licensing, bio dual-use, calibration/sizing); citations in §7. Companion:
`V8B_MODEL_CARD.md`, `V8B_SHIP_EVIDENCE.md`, `V8_DESIGN.md` §8.

## 0. Decision: the release posture

Target = **gated, non-commercial research release on Hugging Face**, mirroring the
WildGuard template (open weights behind a click-through Responsible-Use gate), not
the Anthropic Constitutional-Classifiers norm (never released). This is the
established norm for a defensive discriminative guard, and it is achievable.

Posture, concretely (from the dual-use analysis):
- **Gated HF repo** (click-through Responsible-Use acknowledgment).
- **License CC-BY-NC-4.0** + explicit bio-misuse clause.
- **Withhold the harmful (positive) training examples** (release weights, not the
  harmful corpus).
- **Withhold the exact production threshold** (ship logits + a re-calibratable
  default; deployers must calibrate).
- **Rate-limit** any hosted endpoint; **do not** open-source a companion attack
  harness.
- Hold a one-rung step-down to eval-only/API or AISI/CAISI structured access in
  reserve for any higher-capability or operational-data-trained variant.

Rationale: the marginal misuse uplift of a purely discriminative response-detector
is low relative to already-open guards (Llama Guard, WildGuard cover CBRN), open
harmful-prompt benchmarks, and an attacker's own generator. The one real
info-hazard, classifier-as-reward-model (GeneBreaker used a pathogenicity
classifier as a beam-search reward), is bounded by withholding the corpus and the
threshold.

## 1. Two hard constraints (state these honestly up front)

**1a. License forces non-commercial.** Training mix licenses:

| Dataset | License | Commercial? |
|---------|---------|-------------|
| WildGuardMix | ODC-BY-1.0 (gated by AI2 RUG) | yes |
| BeaverTails | CC-BY-NC-4.0 | **no** |
| FalseReject | CC-BY-NC-4.0 | **no** |

Two NC datasets propagate to the weights under the conservative reading any
credible release must adopt. So v8b ships **non-commercial**. Direct precedent:
PKU's own `beaver-dam-7b` (BeaverTails-trained, publicly released, NC). A
**commercial** release requires dropping BeaverTails + FalseReject and retraining
on permissive data only (ODC-BY WildGuardMix and/or CC-BY-4.0 Aegis 2.0), then
licensing Apache-2.0, the WildGuard/Aegis pattern.

**Decision (2026-06-01): Phase C deferred, ship non-commercial.** No commercial
plans exist, and Phase C would drop 55% of training (BeaverTails 1,024 +
FalseReject 891), which are exactly the two sources behind v8b's headline numbers
(BeaverTails = the real bio-harmful-response positives behind recall 0.919;
FalseReject = the hard-negatives behind the 2% real over-refusal). Permissive
bio-harmful replacements barely exist (Aegis 2.0 is CC-BY-4.0 but has no bio
category), so Phase C would likely regress quality for a hypothetical benefit.
Commercial readiness also adds near-zero application value (safety labs deploy
guards, they do not sell them). The option is NOT foreclosed: do Phase C later if
a concrete commercial need appears, guided by the real requirement. Preserve the
option cheaply by keeping the data pipeline source-toggleable, so a permissive-only
retrain is a config flag, not a redesign.

**1b. Eval sizing collides with the frontier wall, but only for TRAINING.** The
sizing target below needs ~140 to 390 bio-harmful EVAL positives. These are
**harvestable from existing public harmful completions** (HarmBench bio behaviors
+ completions, WMDP-bio topics, WildJailbreak bio split), graded genuinely harmful
by the StrongREJECT autograder (score >= 0.5). That is reuse-only eval assembly,
not new harm generation, and it is separate from the training-side frontier wall
(the ambiguous dual-use tail still cannot be reuse-sourced for training, per
§8.4). So: the eval CAN be resized; the training positive class still cannot be
expanded without generation.

## 2. Phased plan

### Phase R3 first: eval resizing (unblocks R1/R2 power)
Current n=62 positives gives a recall Wilson 95% CI of [0.825, 0.965], about
+/-0.07. Too wide to defend "0.92" or to detect a 5-point regression. Collect:

| Quantity | Now | Minimal viable | Publication-grade |
|----------|-----|----------------|-------------------|
| Bio-harmful positives (real responses, StrongREJECT >= 0.5) | 62 | **~140** (recall CI +/-0.05) | **~390** (CI +/-0.03) |
| Substantive benign-bio negatives | 68 | **~470** (FPR CI +/-0.02) | **~1,840** (+/-0.01) |
| Held-out calibration positives | 0 | **~150** | ~500 (usable conformal bound) |

Sources for positives: HarmBench Chemical/Biological behaviors + completions,
WMDP-bio derived responses, WildJailbreak harmful split (bio-filtered), bio slice
of existing data. Grade with StrongREJECT; keep only genuinely-harmful responses.
Negatives: FalseReject-Test held-out, XSTest-Resp, benign bio responses to
dual-use-sounding questions, plus the real-session `ood_fpr` bio set.

**R3 harvest RESULT (2026-06-01, `harvest_r3_assess.py` + `fetch_assess_wildjailbreak.py`).
The bio-harmful real-response eval is fundamentally scarcity-capped under
reuse-only.** Every cached/fetchable avenue hit a wall:
- HarmBench cache = stub (all 171-char COMPLIANCE_TEMPLATE, uniqResp 10%) -> 0 real.
- AdvBench = 82-char affirmative target-prefixes, not substantive responses -> low quality.
- WildGuardMix bio = already in #106 / training.
- BeaverTails subset = only **12 NEW** substantive real positives.
- WildJailbreak = **gated (no access)** AND its harmful completions are refusals
  (safety-training design) -> doubly blocked for positives.
Net reuse-only ceiling ~74 (62 + 12) to ~96 (incl. low-quality AdvBench), which
moves the recall Wilson CI only from +/-0.07 to ~+/-0.06. **Conclusion: growing
the bio recall eval to the +/-0.05 (n=140) or +/-0.03 (n=390) target is NOT
reuse-achievable** (the same bio real-response scarcity that caps the training
positive class also caps the eval). This is a documentable inherent limitation of
reuse-only bio guarding, not a fixable gap. Decision: do NOT build the marginal
74-set; report recall as DIRECTIONAL with its Wilson CI (the model card already
does). Gated datasets to request for a future revisit (none decisively solves the
positive scarcity): ScaleAI/mhj (multi-turn track), allenai/wildjailbreak
(refusals/over-refusal), TrustAIRLab/HarmfulQA + JailbreakQR (small, real Q+A).

### Phase R1: robustness eval (the largest readiness gap)
v8b's `query [SEP] response` input is structurally the Constitutional
Classifiers++ "exchange classifier" (judges response in full input context). The
publishable result is proving v8b inherits that robustness. Five tracks:

| Track | Test | Metric | Pass bar |
|-------|------|--------|----------|
| 0 Baseline | WildGuardTest vanilla vs adversarial split | F1, AUPRC, recall@FPR=1%/5% | adv F1 >= 0.65; vanilla-adv gap <= 0.15 |
| 1 Char injection | emoji/Unicode-tag/homoglyph/zero-width on harmful responses | Evasion Success Rate, **with NFKC normalization defense** | ESR < 20% |
| 1 Adv-word sub | TextAttack (TextFooler/BERT-Attack/DeepWordBug) white-box vs v8b | ESR | < 40% |
| 1 Obfuscation/reconstruction | rewrite harmful response benign; split across fragments | recall | >= 0.50 |
| 2 Held-out bio | Biorisk-Shift, GeneBreaker, SORRY-Bench mutations (never in train) | recall; generalization gap | >= 0.75; gap <= 0.15 |
| 3 Multi-turn | MHJ, Biorisk-Shift, RED QUEEN; per-turn vs full-context-windowed scoring | windowed detection; windowed minus per-turn | >= 0.70; **+>= 15 pts** |
| 4 Over-refusal guardrail | FalseReject-Test, XSTest-Resp | FPR | <= 6% (robustness fixes must not raise it) |

Ship alongside: **input normalization (NFKC + zero-width/emoji/homoglyph
stripping)** before tokenization. Highest-ROI fix; converts ~100% char-injection
ESR into a near non-issue. Tooling: TextAttack, HarmBench (+ its classifier as a
second grader), StrongREJECT, ftfy/NFKC.

**Track 1 char-injection RESULT (2026-06-01, `eval_v8b_robust_charinjection.py`,
n=57 v8b-flagged harmful bio).** v8b already shipped a `normalize_text` layer
(from the v3/v4 robustness work: zero-width/bidi strip, URL/base64/hex/ROT13
decode, 9 Cyrillic homoglyphs, NFKC). The probe measured Evasion Success Rate at
normalize=False (raw) vs True (production), exposing 4 gaps the layer missed.
Hardened `normalize_text` (added tag chars U+E0000-E007F + variation selectors +
a combining-mark strip) closes 3:

| attack | raw (norm=False) | before harden | after harden |
|--------|------------------|---------------|--------------|
| zerowidth | 0.737 | 0.000 | 0.000 |
| fullwidth | 0.333 | 0.000 | 0.000 |
| homoglyph (Cyrillic) | 0.088 | 0.035 | 0.035 |
| combining diacritic | 0.526 | 0.526 | **0.000** |
| tag chars | 0.737 | 0.737 | **0.000** |
| variation selector | 0.246 | 0.246 | **0.000** |
| homoglyph (Greek) | 0.105 | 0.105 | 0.105 (residual, passes) |
| intra-word spacing | 0.211 | 0.211 | 0.211 (**residual**) |

7 of 8 pass (ESR <= 0.105); clean behavior preserved (identity ESR 0.000, still
57/62 flagged). Two deliberate residuals: **spacing** (char-stripping cannot fix
without breaking legitimate bio sequences like "A T G C"; needs adversarial
training, a training-time track) and **Greek homoglyph** (0.105, already passes;
folding Greek would corrupt legitimate bio notation like alpha-helix). The
hardened layer ships as the deployed preprocessing.

**Track 1 adv-word RESULT (2026-06-01, `eval_v8b_robust_advword.py`).** A
black-box greedy word-importance attack (rank words by deletion impact, char-swap
the top words until flip; budget <= min(15, 25% of words)) evades only **ESR
0.123 (7/57), PASS (<0.40)**. The 88% that resist survive 15 targeted edits; the 7
that flip need ~1.9 edits. This is the DeBERTa encoder's INTRINSIC robustness to
typo-style perturbation (normalize_text does not touch word typos; this is the
model, not a defense layer). Residual: the cheap-flip minority is addressable by
adversarial training, but the track passes. Class-B (direct classifier-evasion,
the primary threat) is now covered: char-injection 7/8 + adv-word PASS.

**Track 3 multi-turn RESULT (2026-06-01, `eval_v8b_multiturn_reconstruction.py`).**
The dramatic exchange-classifier gap (windowed >> per-turn by >=15 pts) does NOT
materialize for naive splitting, because v8b is **robust to naive multi-turn
splitting**: splitting a harmful bio response into K=2..5 sentence-chunks leaves
each chunk individually harmful, so per-turn scoring still catches **0.964** vs
windowed 1.0 (gap **+3.6 pts**). Pass bar: windowed >=0.70 PASSES (1.0); the
+>=15pt gap does NOT (this is good news: v8b does not NEED windowed mode for naive
splits). MHJ (now accessible) cannot drive a real-transcript version: it stores
only attacker turns (message_0..100); model completions are redacted, so it has no
harmful RESPONSES for a response-harm classifier. The 15-pt exchange-classifier
advantage would require ADVERSARIAL reconstruction (each chunk LLM-rewritten to
read benign so harm emerges only on assembly) = the obfuscation track, which needs
an LLM rewrite endpoint and is deferred. Honest verdict: v8b is multi-turn-split
robust; the flashy differentiator is unproven (and may be unprovable without
adversarial chunk-rewriting).

**Track 1 obfuscation RESULT (2026-06-01, `eval_v8b_robust_framing.py`, defensible
variant).** Wrapping the harmful responses in benign framings (fiction, roleplay,
educational, historical, disclaimer, hypothetical) WITHOUT rewriting the harmful
core evades v8b only at **worst ESR 0.140** (educational frame); most frames are
0.018 to 0.053; all pass <0.20. v8b judges the response CONTENT, not the
surrounding frame, so framing-jailbreaks do not work. This is the non-sensitive
substitute for LLM-rewrite obfuscation (which would generate disguised harmful
text and is deliberately not done). **Net: every testable evasion vector passes**
(char-injection 7/8, adv-word 0.123, multi-turn-naive 0.964, framing 0.14); only
LLM-rewrite-disguise reconstruction remains untested.

**Track 1 LLM-paraphrase RESULT (2026-06-01, `eval_v8b_robust_llm_paraphrase.py`).**
The strongest content-obfuscation test: a local Qwen2.5-7B neutral-paraphrases each
harmful response (full surface rewrite, semantics preserved). It complied on all 57
(0 refusals), and the paraphrases evade v8b at only **ESR 0.070** (4/57; PASS).
Paraphrasing each chunk then reconstructing gives per-turn 0.945 = windowed 0.945
(**gap 0.0**). So the 15-pt exchange-classifier differentiator does NOT exist for
v8b, for a good reason: v8b is robustly CONTENT-driven, so per-turn already catches
what windowed would. **R1 robustness is now comprehensive: every vector passes**
(char-injection 7/8, adv-word 0.123, framing 0.14, full LLM paraphrase 0.07,
multi-turn-naive 0.96, paraphrased reconstruction gap 0). The flashy differentiator
is refuted; the underlying content-robustness is the stronger story. Safeguards:
local model, existing harmful content only, content-blind, paraphrases not saved.

### Phase R2: calibration and operating point
- **Calibrate** on a disjoint calibration split: temperature scaling (primary,
  1-param, robust at small n, preserves ranking); beta calibration as a check; do
  NOT use isotonic (<1000 samples). Report ECE (10-bin) + reliability diagram +
  Brier, before vs after.
- **Operating point**: headline artifact = a recall(harm)-vs-over-refusal(benign)
  tradeoff curve with the chosen threshold marked. Report AUROC + AUPRC with
  bootstrap CIs (AUPRC for rare-positive sensitivity; AUROC for ranking). Pick the
  threshold for a target over-refusal FPR <= 5%; report recall@5%FPR.
- **Distribution-free guarantee**: wrap the threshold in a Learn-then-Test / RCPS
  bound, certifying FNR <= alpha at 95% confidence. Needs the ~500-positive
  calibration split for a tight bound (Hoeffding slack is 0.155 at n=62, 0.055 at
  n=500). This is where v8b can be MORE rigorous than the published guard baselines,
  which publish no operating-point uncertainty.

**R2 RESULT (2026-06-01, `r2_calibrate_v8b.py`; 62 harmful vs 531 real legit
over-refusal).** AUROC **0.970**, AUPRC **0.938** (excellent discrimination of
real harm from real legitimate research). Temperature scaling (T=0.239, fit on a
held-out cal split) cuts **ECE 0.137 -> 0.042** and Brier 0.053 -> 0.035. The
recall-vs-over-refusal curve shows the default **threshold 0.5 is already
near-optimal: recall 0.919 at 2.1% over-refusal** (Wilson 95% CI [0.825, 0.965]);
buying recall 0.95 costs ~15% over-refusal. DONE: temp scaling, ECE/Brier,
operating-point curve, AUROC/AUPRC, Wilson CI. DEFERRED (same scarcity as R3): the
distribution-free conformal FNR bound needs >=500 calibration positives; at n=62
the Hoeffding slack (0.155) makes it uninformative, so report the Wilson CI instead.

### Phase R4: release packaging
- **Stats**: Wilson CIs default; Clopper-Pearson + rule-of-three (3/n) for
  zero-count claims; bootstrap for AUPRC and between-version diffs; exact McNemar
  (paired) for v8b-vs-baseline and version-to-version.
- **Model card**: upgrade `V8B_MODEL_CARD.md` to the HF schema: YAML block
  (`license: cc-by-nc-4.0`, `base_model`, `datasets`, `pipeline_tag`, `model-index`
  for the eval widget, `extra_gated_*`), inline the bio response-harm taxonomy /
  label definition, NeMoGuard-style Bias/Explainability/Privacy/Safety ethics block,
  content warning, responsible-use clauses, disclosure contact.
- **Artifact bundle**: `inference.py` (load + classify), `eval/` (harness + configs
  + RESULTS.md reproducing the card numbers), `train/` (data-prep + label map +
  DeBERTa hyperparameters/seed), threshold + calibration note. Withhold: harmful
  positives, exact threshold, attack harness.
- **Gating**: HF gated repo + click-through Responsible-Use ToU.

## 3. Sequencing and rough effort

1. **R3 data** (harvest + StrongREJECT grade), unblocks everything; the long pole.
2. **R1 robustness** + ship NFKC normalization, the biggest readiness gap and the
   most publishable result (validate the exchange-classifier robustness claim).
3. **R2 calibration**, small once R3's calibration split exists.
4. **R4 packaging**, mechanical once R1-R3 numbers are in.
5. (deferred, see §1a) **Phase C commercial**: retrain without BeaverTails +
   FalseReject; relicense Apache-2.0. Only if a concrete commercial need appears.

R1-R3 are the real work (data harvest + adversarial harness + multi-turn). R2/R4
are light. None requires new harmful generation.

## 4. Decision gates

- After R1: if vanilla-adversarial F1 gap > 0.15, or char-injection ESR stays high
  even with NFKC, or multi-turn windowed does not beat per-turn, v8b is not
  release-grade; iterate (likely add adversarial responses to training).
- After R3: confirm recall/FPR CIs are tight enough (+/-0.05) to make the claims.
- Release: gated non-commercial research release once R1-R4 pass. Commercial only
  after Phase C.

## 5. Consolidated acceptance scorecard (the release bar)

| Dimension | Metric | Pass bar | Current |
|-----------|--------|----------|---------|
| Recall power | bio positives n | >= 140 (min) | 62 |
| Over-refusal power | substantive-benign n | >= 470 | 68 |
| Real harm recall | recall, Wilson CI | >= 0.75, +/-0.05 | 0.92 (+/-0.07) |
| Adversarial | vanilla-adv F1 gap | <= 0.15 | untested |
| Char injection | ESR w/ hardened normalize | < 20% | **DONE: 7/8 pass <=0.105; spacing 0.21 residual** |
| Adv-word sub | ESR (greedy char-swap) | < 40% | **DONE: 0.123 PASS** |
| Obfuscation (framing + LLM paraphrase) | ESR | < 0.20 / 0.40 | **DONE: framing 0.14, LLM-paraphrase 0.07 PASS** |
| Held-out bio | recall; gen gap | >= 0.75; <= 0.15 | untested |
| Multi-turn | windowed; vs per-turn | >= 0.70; +>= 15 pts | **windowed 1.0 PASS; gap +3.6pt naive / 0.0 LLM-paraphrased = NO 15pt differentiator (v8b content-robust)** |
| Over-refusal | FPR substantive | <= 6% | 1.5% (n small) |
| Calibration | ECE reported + reduced | yes | **DONE: 0.137->0.042 (T=0.24); AUROC 0.97/AUPRC 0.94** |
| Operating point | recall@over-refusal<=5% | published | **DONE: tau=0.5 -> recall 0.919 @ 2.1%** |
| License/gating | CC-BY-NC, gated, withholds | in place | **DONE (release/README.md + CHECKLIST)** |
| Model card | HF schema + taxonomy + model-index | complete | **DONE (release/README.md)** |

## 6. What this plan changes about the readiness verdict

Before: "v8b is a strong research artifact but not a drop-in production guard."
This plan makes that precise and costed. A **gated non-commercial research
release** is reachable after R1-R4 (no new harm, reuse-only). **Production /
commercial** additionally needs Phase C (retrain off NC data) and the full
robustness bar. The release posture itself (gated, dual-use-reasoned, threshold
withheld) is a portfolio asset: it demonstrates calibrated release judgment, not
just a benchmark number.

## 7. References (verified across the four research sweeps)

Robustness: Constitutional Classifiers arXiv:2501.18837 + ++ arXiv:2601.04603
(exchange classifier); WildGuard arXiv:2406.18495 (vanilla-vs-adv F1);
Llama Guard arXiv:2312.06674; ShieldGemma arXiv:2407.21772; Aegis 2.0
arXiv:2501.09004; HarmBench arXiv:2402.04249; StrongREJECT arXiv:2402.10260;
MHJ arXiv:2408.15221; Crescendo arXiv:2404.01833; RED QUEEN arXiv:2409.17458;
"Bypassing LLM Guardrails" arXiv:2504.11168 (attacks DeBERTa-v3 family; NFKC fix);
Qwen3Guard generalization-collapse arXiv:2511.22047; Biorisk-Shift (NeurIPS 2025
BioSafe GenAI, OpenReview MwajedbCfX); GeneBreaker arXiv:2505.23839.

Licensing/release: ODC-BY-1.0 (opendatacommons.org/licenses/by/1-0); CC-BY-NC-4.0;
AI2 Responsible Use (allenai.org/responsible-use); HF model-card schema
(huggingface.co/docs/hub/model-cards, model-card-annotated); reference cards
meta-llama/Llama-Guard-3-8B, allenai/wildguard, google/shieldgemma-2b,
nvidia/llama-3.1-nemoguard-8b-content-safety; NC precedent PKU-Alignment/beaver-dam-7b.

Dual-use / responsible release: Structured access Shevlane arXiv:2201.05159;
GPT-2 staged release arXiv:1908.09203; Beyond Release arXiv:2502.16701; marginal
risk / open foundation models arXiv:2403.07918; WMDP arXiv:2403.03218;
SciSafeEval arXiv:2410.03769; Biosecurity-eval best practices arXiv:2510.27629;
NTI|bio managed-access framework (nti.org); Anthropic ASL-3 (deployed, not released).

Calibration/sizing: temperature scaling + ECE, Guo et al. arXiv:1706.04599; beta
calibration, Kull et al. AISTATS 2017; sklearn calibration docs; AUROC-vs-AUPRC
arXiv:2401.06091; conformal RCPS arXiv:2107.07511 + Learn-then-Test arXiv:2110.01052;
Wilson / Clopper-Pearson, Brown Cai DasGupta Stat.Sci. 2001; McNemar (exact when
discordant < 25).

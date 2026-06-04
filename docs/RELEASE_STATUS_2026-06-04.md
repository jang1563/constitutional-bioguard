# Constitutional BioGuard -- release status (2026-06-04)

Single source of truth tying together the dual-mode bio guard program. Per-step detail in the
STEP*_*.md docs. All numbers verified against Cayuga logs/result files (audit commit 4ea4db3).

## System
A dual-mode bio-safety guard = TWO 184M DeBERTa-v3 encoders + a consistency policy (DualModeGuard
artifact, scripts/dual_mode_guard.py):
- PROMPT head (query-only): bio prompt-harm. Recall 0.983, clean-bio over-refusal 0.022.
- RESPONSE head v8b (query+response): bio response-harm. Recall ~0.92, tunable (well-calibrated).

## What was established this session
1. STEP 1 -- FOOTPRINT SOLVED. The 8B generative teacher's bio recall compresses into a 184M
   encoder: student recall 0.983 >= teacher 0.900 (length-norm @0.5). Root-caused + fixed the
   trainer (transformers 5.9.0 loads deberta-v3 fp16 by default -> NaN; force dtype=float32).
2. STEP 1b -- borderline over-refusal is a DATA wall, not capacity. Deep research (20 claims
   verified) + generation (template + LLM-rewrite bio-borderline-benign) cut it 0.671->0.532 but
   PLATEAU -- the query-only head is saturated, cannot separate harmful-bio from dangerous-sounding
   benign-bio at the query level.
3. STEP 2 -- DUAL-MODE resolves it. The response head, seeing the actual safe answer, drives
   borderline over-refusal 0.532 -> 0.076. The two heads have decorrelated failure modes (prompt:
   lexical overfitting; response: density bias) and clear each other's FPs on legit traffic.
4. ARTIFACT -- DualModeGuard (deployable): 4 honest policies (prompt_only/response_only/and/or).
   and = over-refusal-optimal but MISSES jailbreaks (measured: recall 0.919->0.855 on a jailbreak
   set); or (default) = jailbreak-safe. Response head is the workhorse.
5. STEP 3 -- CERTIFIED over-refusal. LTT + Clopper-Pearson gives a distribution-free bound:
   over-refusal <= 10% at 95% confidence, recall 0.80 (and a full alpha/recall table).
6. STEP 4 -- COMPETITIVE, with BOTH halves measured honestly. (a) RECALL: FORTRESS-CBRN bio recall
   0.967 beats WildGuard-7B 0.926 / Llama-Guard-3-8B 0.593 (held out from all); and naive response
   comparison is contaminated -- on the wildguard_test slice held out from WildGuard, WildGuard
   recall collapses to 0.53 while our decontaminated 184M holds 0.94 (we generalize, they memorize;
   small-n). (b) OVER-REFUSAL: on held-out fresh safe responses (FORTRESS rollouts) our guards
   over-refuse 0.26-0.30 vs the competitors' 0.01-0.02 -- they are far better calibrated; we are
   AGGRESSIVE (higher recall, higher over-refusal). At a matched operating point we are
   competitive-to-slightly-behind, NOT dominant.

7. STEP 4b -- the over-refusal weakness is REDUCIBLE, not fundamental. Density-bias debiasing with
   dense-but-safe hard negatives: TARGETED (FORTRESS-only, v8bh) cuts held-out FORTRESS over-refusal
   0.288 -> 0.016 (matches competitors) at -2.4pt recall; BROAD over-augmentation (v8b2) over-corrects
   (-11pt recall, cross-distribution regression). No free universal fix; recipe = debias the served
   distributions + conformal operating point (Step 3). Mechanism proven on held-out data.

## INTEGRITY REVIEW (2026-06-04) -- read before citing any claim
A self-audit + literature audit (INTEGRITY_REVIEW_2026-06-04.md) found the numbers are sound and
leakage is CLEAN, but SEVERAL HEADLINE FRAMINGS were OVERCLAIMED and are corrected there:
- "we generalize, they memorize" (n=17): NOT defensible -- ALL guards degrade on novel prompts;
  drop the causal narrative.
- Native-threshold "best/2nd-best": misleading -- at MATCHED FPR ours is competitive-to-behind on
  the contaminated set; pair every over-refusal with recall (OR-Bench safety/over-ref rho=0.878).
- ShieldGemma undersold (AUROC 0.893), Qwen over-ref inflated by my Controversial=flagged (true 0.005).
- n=30 bio "best" needs CIs/McNemar; density-debias 0.016 is WITHIN-DISTRIBUTION only.
- Must cite WildGuard as prior dual/tri-mode guard; reframe novelty as small-footprint two-encoder policy.
The HONEST headline below is the corrected, defensible version.

## Headline (CORRECTED, post-integrity-review)
A 184M two-head configurable bio guard COMPETITIVE with four 8-9B guards on bio response-harm
(threshold-free AUROC 0.952, on par with or above them; recall in the same band) at ~40x smaller,
with a within-distribution density-debiasing recipe and a response-head over-refusal certificate.
NOT "best/dominant": at matched operating points it does not clearly beat the binary competitors on
the (contamination-affected) response set, and the bio prompt-recall edge rests on n=30. Defensible
as: competitive-bio-guard-at-a-fraction-of-the-size, reported with paired metrics + CIs + per-source
decontamination + a non-causal generalization narrative.

## (superseded) earlier headline
Our 184M bio-specialized guards are competitive-to-best against four 8-9B guards (WildGuard-7B,
Llama-Guard-3-8B, ShieldGemma-9b, Qwen3Guard-8B): BEST bio prompt-recall (0.967, held out from
all), 2nd response recall (0.921, behind only Qwen3Guard 0.956), and -- after density debiasing
(v8bh) -- 2nd-best held-out over-refusal (0.016, near WildGuard/Llama, beating Qwen/ShieldGemma;
was 0.288 worst pre-debias). At 40x smaller. Qwen3Guard-8B is the strongest competitor. Honest
gaps: real_response_bio over-refusal 0.194 is mid-pack (Qwen-like, above WildGuard/Llama 0.05-0.10);
prompt-head saturation; small bio-slice n; SafeRLHF contamination favors competitors on the response
set. Net: best-bio-recall-per-parameter + competitive calibration (debiased) + certifiable, not
universal dominance.

## Open items
- DATA: a larger bio response set held out from ALL guards' training would firm up the
  generalization claim (currently the clean WildGuard slice is n=17 harmful).
- COMPETITORS: ShieldGemma is GATED -- needs JK to accept the HF license (hf.co/google/shieldgemma-*);
  Qwen3Guard not yet attempted.
- DUAL-MODE certificate: Step 3 certifies the response head; extending to the joint policy is future.
- RELEASE (JK actions): HF model cards, GitHub visibility flip, repo polish -- not done by me.

## Key files
docs/STEP1_DISTILL_PILOT, STEP1B_RESEARCH, STEP2_DUALMODE, STEP3_CONFORMAL, STEP4_COMPETITIVE,
DUAL_MODE_GUARD_ARTIFACT. scripts/dual_mode_guard.py (artifact), train_v7c_distill.py,
run_competitor.py, conformal_bound.py. Models (Cayuga): deberta_v7c_distill_bioborder (prompt),
deberta_bioguard_v8b (response).

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
6. STEP 4 -- COMPETITIVE + a contamination finding. FORTRESS-CBRN: our bio recall 0.967 beats
   WildGuard-7B 0.926 and Llama-Guard-3-8B 0.593. Response-harm: naive public-benchmark comparison
   is CONFOUNDED by training contamination (competitors score 0.97 on SafeRLHF they likely trained
   on); on the wildguard_test slice held out from WildGuard, WildGuard recall COLLAPSES to 0.53
   while our decontaminated 184M holds 0.94 -- we generalize, they partly memorize. (held-out n small.)

## Headline
A 184M bio-specialized dual-mode encoder pair matches/beats 40x-larger general guards on bio
safety, with a certified over-refusal bound and a clean unified artifact. Strongest where it
counts (bio recall; generalization on un-memorized data); honest about the gaps below.

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

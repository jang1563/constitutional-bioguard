# Dual-mode bio guard: gap-closing release plan (2026-06-03)

Sequenced plan to take the dual-mode (prompt + response) bio guard from "response head
shipped + prompt head viable candidate" to "release-ready + competitive evidence."
Built from a verified research pass (25 claims, 21 confirmed adversarially). Companion
to `DUAL_MODE_DESIGN.md` and `RESEARCH_REFRESH_2026-06-03.md`.

## State going in
- **Response head (v8b)**: ships. 184M DeBERTa-v3, ~0.919 real harmful-bio recall, 0.02-0.06 real over-refusal.
- **Prompt head (v7.C-aug2)**: viable candidate, NOT release-ready. 8B Llama-3.1 + QLoRA generative; AUROC 0.897; recall 0.85 @ real over-refusal 0.10 / 0.62 @ 0.05 / 0.47 @ 0.021. Targeted benign-aware retrain closed the worst over-refusal; biochem/immuno/chem remain (reuse-only data wall).

## The decisive finding (gates the whole plan)
A generative-8B -> 435M-DeBERTa cross-architecture distillation recipe IS proven (HarmAug,
arXiv:2410.01524: KL+BCE soft-label, lambda=0.5, student F1 0.736 / AUPRC 0.836 beats the
8B teacher at <25% compute). BUT every HarmAug benchmark is general-domain with ZERO bio
test, so there is **no evidence the narrow bio recall signal survives compression**. The
capacity-gap literature (arXiv:2601.10114, 2501.16937, 2305.12129) documents this exact
failure mode: naive forward-KL causes mode averaging, the "curse of capacity gap" means a
bigger teacher does not monotonically help, and a much smaller student incurs a
teacher-favored-subdomain deficit. At our ~18-43x compression, naive KL may NOT preserve
the bio signal -> the 8B footprint stays a release blocker unless verified.

**=> Single highest-leverage move: a cheap bio-distillation PILOT, scored on a held-out
harmful-bio recall set, BEFORE any further prompt-head work.** Its result forks the plan.

## Sequenced plan

### STEP 0 (critical path, ~1-2 wk) - assemble a harmful-bio PROMPT recall set
The positive-class denominator every later step needs, and currently the project's weakest
spot (prompt recall is measured on adversarial CBRN benchmark positives only). Candidate
reuse-only sources surfaced (need a short characterization pass for size/license/contamination):
- **SOSBench** (arXiv:2505.21605) - hazardous-science prompts incl. biology/chemistry.
- **SciSafeEval** (HF Tianhao0x01/SciSafeEval) - biology slice, but largely sequence-templated (per DUAL_MODE_DESIGN P1: text dedups to ~1139; treat as sequence-task, not text-intent).
- **ClearHarm-CBRN** (far.ai) - CBRN jailbreak prompts.
- **HarmBench chemical_biological**, **SALAD-Bench / ALERT bio leaves** - small but genuine intent prompts.
- **WMDP-bio** (arXiv:2403.03218): hazardous-KNOWLEDGE MCQ, NOT operational intent. Do NOT
  label positive (induces over-refusal); use as the dual-use-ambiguous ABSTAIN / over-refusal set.
Leakage-audit byte-disjoint vs train, as before.

### STEP 1 (highest leverage) - bio-distillation pilot -- DONE 2026-06-03 (see STEP1_DISTILL_PILOT_2026-06-03.md)
HarmAug recipe: 8B generative teacher (v7.C-aug2) -> 184M DeBERTa student, hard CE and
soft-CE (lambda=0.5). **RESULT: the fork SPLIT.**
- **RECALL = Fork 1a (transfers).** Student 184M recall 0.983 >= teacher 0.900; clean
  expert legit-bio over-refusal 0.017 ~ teacher 0.023. Footprint is NOT a recall blocker.
- **BORDERLINE-BENIGN OVER-REFUSAL = Fork 1b (materialized).** On the same 739 OR-Bench-health
  borderline prompts: teacher 0.166 vs student ~0.83. Capacity-gap mode-averaging on the
  over-refusal axis; **soft labels did not close it** (pool benign is clean, doesn't cover
  the borderline region). Capacity is sufficient; the gap is DATA COMPOSITION.
- **NEXT (Step 1b):** augment the distill pool with borderline-benign hard negatives
  (OR-Bench-style, leakage-disjoint) + teacher soft labels, re-distill. Fallback if it does
  not close: TAID (arXiv:2501.16937) or a 435M student. Trainer bug fixed (deberta-v3 loaded
  fp16 by default in transformers 5.9.0 -> forced `dtype=float32`).

### STEP 2 (~1-2 wk, LOW risk, well-specified) - dual-mode integration
Wire the two heads. Reference designs: Qwen3Guard-Stream (arXiv:2510.14276 - shared backbone,
two independent jointly-trained heads, prompt scored at end-of-query token, response per-token)
or a two-model decision table (our case is heterogeneous: encoder response head + the Step-1
prompt head). Use the consistency table already in DUAL_MODE_DESIGN §2. Independent,
asymmetric thresholds (stricter response, more permissive prompt).
- **Caution (arXiv:2604.26052)**: a documented prompt-vs-response detection asymmetry that
  few-shot calibration does NOT close. It is one-judge-two-inputs (not two heads) and
  non-bio, so it is a caution, not a verdict. **Measure here** whether adding the second
  axis actually cuts over-refusal (the research did NOT establish that it does - do not
  assume it; this is an open empirical question, Gap 2).

### STEP 3 (~1 wk, LOW-MED risk) - bound over-refusal under the data wall
No new benign labels exist for biochem/immuno/chem. Use **conformal classification with a
reject option** (arXiv:2506.21802): accept only singleton prediction sets, reject empty
(novel) / double (ambiguous); closed-form singleton-error rate sigma = (eps - P(empty))/P(singleton),
a distribution-free finite-sample bound on the accepted-error rate. Caveat: offline-batch
validity is PAC-type/approximate. Pairs with the existing LTT/RCPS plan in DUAL_MODE_DESIGN.

### STEP 4 (~1-2 wk, LOW risk) - competitive evidence
Run the **Qwen3Guard same-corpus protocol** (2510.14276 follows WildGuard settings) against
the exact competitor set - WildGuard-7B (85.8), Llama Guard 3-8B (79.4), ShieldGemma-9B
(70.4) - on FORTRESS (paired ARS/ORS, benign twins) + Health-ORSC-Bench (per-category, incl.
Bio/Chem Hard) + the Step-0 bio recall set. Report a **per-bio axis no competitor reports**
(caveat: Llama Guard S9 bundles bio in CBRNE; true at the reported-metric level). Automation:
**guardbench** (github.com/AmenRa/guardbench) is an existing guard-eval harness to adapt.

## The two forks that change the plan
1. **Distillation does NOT preserve bio recall** (supported by capacity-gap evidence): the
   8B footprint stays a blocker -> pivot to TAID or direct-encoder fine-tuning. This is the
   most likely plan-changer; Step 1 is designed to detect it early and cheaply.
2. **A dual-mode second axis sharply cuts over-refusal**: ~~NOT supported by surviving
   evidence~~ **NOW MEASURED AND CONFIRMED (2026-06-03 bridge experiment).** On the public
   expert set bio-overrefusal-v0.1 (leakage-clean), the response head v8b over-refuses 14.9%
   (27/181, density-bias, worst pathogen_biology 45.5%); the v7.C prompt head over-refuses
   2.3% (4/176) and **clears all 26 of v8b's false-positives that fall in the shared set**
   (v8b flags 27 on its 181-item set; 26 are shared with the prompt head's 176-item set and
   all 26 are cleared; 1, t2_toxicology_0003, is outside the prompt-head set). It never sees
   the dense answer, so it judges the legit query as benign. An AND policy drives over-refusal
   to **0.0%** on the shared (n=176) set. The two axes have COMPLEMENTARY failure modes. This REPRIORITIZES
   dual-mode integration (Step 2) as a validated high-leverage move, not a speculative one.
   CAVEAT: this is the over-refusal side; a naive AND trades recall, so the deployed policy
   is a consistency table (response-harm primary, prompt-benign veto on density-FPs) whose
   RECALL must be verified on a harmful set (Step 0). Scripts: `score_v7c_bridge.py`,
   `analyze_dual_mode_overrefusal.py`.

## Critical path & effort
Steps 0 + 1 are the only HIGH-uncertainty, critical-path items and are co-dependent (Step 1
needs Step 0's recall set). Steps 2-4 are lower-risk and partly parallelizable once the
Step-1 fork resolves. Realistic critical path to release evidence: ~4-7 weeks.

## Refuted / do-not-cite (from verification)
- A small student transferring a "narrow specialist signal" via LoRA (DivScore 2506.06705): REFUTED (it is an AI-text detector, not a guard).
- JurEE F1 0.92 beating LLM judges by ~40% (2410.08442): REFUTED; use only the architecture/single-encoder-suffices point.
- SFS/TFS surpass-theorem (2601.10114): bound overreached; use only the failure-mode characterization.

## Sources
HarmAug 2410.01524 · capacity-gap/SCD 2601.10114 · TAID 2501.16937 · JurEE 2410.08442 ·
conformal reject-option 2506.21802 · prompt/response asymmetry 2604.26052 · Qwen3Guard 2510.14276 ·
WildGuard 2406.18495 · SOSBench 2505.21605 · WMDP 2403.03218 · SciSafeEval (HF) · ClearHarm (far.ai) ·
guardbench (github AmenRa/guardbench).

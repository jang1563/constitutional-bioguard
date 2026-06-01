# Dual-Mode Bio Guard: design (bioguard-deberta, prompt + response)

Goal: extend the shipped response-harm classifier (bioguard-deberta-v2) into a
dual-mode (prompt-harm + response-harm) BIO-specialized guard that genuinely
exceeds the published bar where it matters, not by chasing a leaderboard a 184M
model cannot win, but by owning the bio axis no general guard reports. Synthesized
from four cited research sweeps (competitor bar, prompt-harm data, architecture,
differentiation); references in section 9.

## 0. Thesis and positioning

A 184M DeBERTa will NOT beat Qwen3Guard-8B or WildGuard-7B at general harm
(prompt-avg F1 ~88-90, response-avg ~80-84; our general response recall ~0.69 is
mid-pack). So the strategy is two-pronged:
1. **Match "well enough" on the standard suite** to be a credible dual-mode guard.
2. **Win decisively on the bio axis nobody else reports.** The bio gap is
   STRUCTURAL: every major guard (Llama Guard, WildGuard, ShieldGemma, Aegis/
   NemoGuard, Qwen3Guard, Granite Guardian, OpenAI omni-moderation) either buries
   bio in one "S9 / Indiscriminate Weapons (CBRNE)" bucket or omits it entirely;
   NONE reports a per-bio metric or evaluates on a bio benchmark (WMDP-bio,
   SOSBench, SciSafeEval). The only dual-mode CBRN guard, Anthropic Constitutional
   Classifiers, is closed, unbenchmarked publicly, and chemical-weapons-led.
   => **the first open, bio-specialized, dual-mode guard with a separately-reported
   bio taxonomy is empty space.**

The defensible contribution = (a) bio-selectivity on the dual-use boundary, (b)
low over-refusal on real legitimate science, (c) distribution-free calibration,
(d) cheap first-stage cascade economics, plus (e) defining a free-response bio
guard benchmark.

## 1. The bar to match + the bio gap to win

**Standard suite a credible dual-mode guard reports (2026 de-facto):**
- Prompt-harm: ToxicChat, OpenAI-Mod, Aegis / Aegis2.0-test, SimpleSafetyTests,
  HarmBench-prompt, WildGuardTest-prompt.
- Response-harm: HarmBench-response, SafeRLHF, BeaverTails, XSTest-response,
  WildGuardTest-response.
- Refusal: XSTest + WildGuardTest refusal.

**SOTA to be "credible" against (F1, the bar to approach not beat):** prompt-avg
~88-90 (Qwen3Guard-8B), response-avg ~80-84 (WildGuard-7B / Qwen3Guard-8B). We
report these for honesty; we do not claim to win them.

**Bio benchmarks where we WIN (nobody else reports these for a guard):**
- Recall: WMDP-bio (relabel MCQ stems to intent), SOSBench-bio, SciSafeEval-bio,
  HarmBench chemical_biological, ClearHarm-CBRN.
- Over-refusal: Health-ORSC-Bench Biological/Chemical-Harm Hard tier, FORTRESS
  benign twins, our real legitimate-research set.
- Selectivity / dual-use boundary: RefusalBench matched triples, FORTRESS pairs.

## 2. Architecture decision

> **SUPERSEDED (2026-06-01).** Option 3 (shared-encoder) was NOT built. Shipped
> instead: the **lexicographic hybrid** (see §3 P2d, `dual_mode.py`) -- a standalone
> `pdual_v3` prompt head, consulted only under bio-context, OR'd with the tiered
> lexicon; `v8b` (`query [SEP] response`) as the response head. The prompt head is
> served exactly as trained: pair pipeline with `response=""` (not the single-
> sequence note in line below). The Option-3 rationale here is retained for
> provenance and as future work if a single-model (~184M) footprint is ever needed.

**Chosen: Option 3 = shared DeBERTa-v3-base encoder + two (or three) heads**, init
the encoder from the existing response-harm checkpoint, trained multi-task with a
staged-unfreeze schedule. Fallback to Option 2 (separate prompt model) if the
response head regresses.

Why: it gives one-model serving footprint (~184M) AND independent per-head
thresholds/calibration (Option 1 cannot) AND cross-task transfer (Option 2
cannot). Evidence it works at our scale: WildGuard multi-task BEAT single-task on
response-harm (+6 F1), GLiNER-Guard (147-209M) and Opir (<100M) and Qwen3Guard-
Stream all ship shared-encoder multi-head dual-mode guards. The "unifying will
regress my hard-won response head" fear is the right caution but the wrong default:
joint training tends to HELP; we protect it with the recipe, not by avoiding it.

Heads (tiny linear, ~0.1M each; all real params in the shared encoder):
- **response-harm head**: reuse existing binary head; input `tokenizer(query,
  response)` pair encoding (unchanged, byte-for-byte the shipped behavior).
- **prompt-harm head**: new; input `tokenizer(prompt)` single-sequence.
- **refusal head (optional, recommended, NON-gating)**: did the response refuse?
  Supervised only on pair examples. Makes the consistency table legible; analytics
  only, does not gate.

Loss: `L = lambda_r*CE_response + lambda_p*CE_prompt (+ lambda_f*CE_refusal)` with
per-example head masking (each example only hits its applicable head). Start
lambda_r=1.0, lambda_p=1.0, lambda_f=0.5.

Staged training (the regression-avoidance core):
- Stage A: FREEZE encoder, train only the new prompt (+refusal) head to convergence.
  Zero risk to the response head.
- Stage B: unfreeze with a LOW encoder LR (1-2e-6) + heads LR ~1e-4, multi-task,
  ~2 epochs, with the ORIGINAL response-harm data REPLAYED at >=50% of pair batches.
- Optional Stage B': LoRA (r=16) on the encoder instead of full FT so the original
  response behavior is recoverable by merging/disabling adapters.
- **Regression guard (non-negotiable acceptance test):** gate every checkpoint on
  the existing response-harm validation set; if response F1 drops > ~0.3 from the
  v2 baseline, stop / lower lambda_p / lower encoder LR. If it keeps tripping,
  switch to Option 2 (ship the prompt head as a separate model, zero regression).

Inference routing: pre-generation = encode(prompt) -> prompt head -> tau_p; post-
generation = encode(query,response) -> response head -> tau_r; read refusal head
for analytics. Thresholds independent and asymmetric (stricter tau_r since a missed
uplift response is the costly error; more permissive tau_p to avoid blocking
legitimate dual-use research).

Consistency: handle at the POLICY layer with a decision table, not by forcing head
agreement (which would destroy the refusal signal):

| Prompt-harm | Refusal | Response-harm | Action |
|---|---|---|---|
| UNSAFE | refusal | SAFE | allow (good refusal); log prompt. Heads "disagree" by design. |
| UNSAFE | comply | UNSAFE | BLOCK (the dangerous bio case). |
| UNSAFE | comply | SAFE | INSPECT / soft-block (key bio audit bucket: compliant but called safe). |
| SAFE | - | UNSAFE | block response (benign prompt, harmful generation). |

## 3. Data plan (the new prompt-harm bio head)

Reuse-only, negative-heavy (over-refusal is THE failure mode). Target ~1 : 1.5-2
positive : hard-negative, then easy negatives for base rate.

**Positives (bio-harm intent), ~20-30K:**
- SciSafeEval biology slice (~12.6K, the largest bio prompt-harm set) + its
  jailbreak-augmented variants.
- HarmBench chemical_biological (98) + ClearHarm CBRN (179 + rep40 paraphrases).
- SALAD-Bench "Biological and chemical harms" leaf + ALERT weapon_biological /
  weapon_chemical (paraphrase-expand).
- WMDP-bio stems (1,273) RELABELED MCQ -> operationalized-intent (LLM-rewrite,
  keep only clear-misuse).
- Biorisk-Shift / adversarial transforms as adversarial positives (framing robustness).

**Hard negatives (benign-but-scary bio), ~30-45K (load-bearing):**
- LAB-Bench SeqQA/DbQA/ProtocolQA/CloningScenarios (~1.3K, sharpest collisions,
  upsample) + BioProBench protocol steps.
- FalseReject bio/chem/medical slice; Health-ORSC-Bench Biological/Chemical-Harm
  benign variants (when released).
- PubMedQA / BioASQ / MedQA infectious-disease for bulk benign-bio volume.
- XSTest bio-phrased safe prompts.

**Easy negatives ~20K:** OR-Bench non-bio safe + general benign chat (so the head
isn't only ever shown bio text).

**Held-out eval (never train):** ClearHarm + SciSafeEval-bio (positives);
LAB-Bench + FalseReject-bio + PubMedQA (negatives); HarmBench-non-bio + AdvBench
(the selectivity denominator). Leakage-audit byte-disjoint as before.

Access note: SaladBench/ALERT/HarmBench/AdvBench have open CAIS/GitHub mirrors;
WildJailbreak/MHJ are gated (already requested). SciSafeEval, FalseReject,
LAB-Bench, PubMedQA, OR-Bench are open.

**P1 RESULT (2026-06-01, assessed) - the pool is smaller than the row counts imply:**
- SciSafeEval biology is ~12.6K ROWS but the instruction is TEMPLATED (5 task
  templates: gene/protein classification/generation/structure); the variation lives
  in a `sequence` field (DNA/protein strings), so distinct TEXT prompts dedup to
  ~1139 and the harm is sequence-bound, not natural-language intent. It is really
  sequence-task data -> feeds the BLAST/sequence SELECTIVITY signal, not the text head.
- **WMDP-bio (3272) is NOT a positive.** It is hazardous-KNOWLEDGE MCQ (research
  questions), not operational intent; labeling it positive would INDUCE over-refusal
  (the exact failure mode), and LLM-rewriting it would violate the zero-generated-
  harmful-content principle. Repurpose WMDP-bio as the DUAL-USE-AMBIGUOUS boundary
  set -> ABSTAIN tier + the selectivity / over-refusal eval (this is where claims 1-2
  are proven).
- Clean diverse TEXT intent positives: ~467 (SaladBench-cbrn 247 + ALERT-cbrn 140 +
  AdvBench 21 + HarmBench 10 + ClearHarm 49) + ~1139 SciSafeEval-templated; augmentable
  to ~3-4K with ClearHarm rep40 paraphrases + ALERT/SaladBench adversarial variants
  (all reuse). Hard-negatives ~1000 cached (OR-Bench-health 740 + LAB-Bench 157 + ...)
  expandable via PubMedQA/BioASQ/MedMCQA. Modest but >> response-harm (n=62) and enough
  for a P2 baseline; the model card's small-n honesty framing carries over.

**P2 BASELINE RESULT (2026-06-01, deberta_pdual_v1, standalone prompt-head, 428
pos / 2115 neg).** First prompt-harm number + selectivity baseline, and it
immediately surfaces the crux. In-dist val recall 0.984 / FPR 0 / AUROC 1.0 is a
same-distribution SHORTCUT (63 val positives, trivially separable saladbench/alert
vs or_bench style; not genuine harm-intent learning). The honest OOD signal:
**bio-selectivity ratio S = recall(bio-harm) / flag-rate(non-bio-harm) = 0.984 /
0.589 = 1.67 -> NOT bio-specific**; the head flags 59% of NON-bio harmful prompts,
i.e. it learned GENERIC harm, not bio specifically. WMDP dual-use boundary
over-flagged at 18%. Diagnosis: non-bio harm was absent from the negatives, so the
head never learned "non-bio harm is not the bio head's job." Fix (P2b): add non-bio
harmful prompts as label-0 negatives to directly optimize S (bio-specificity).

**P2b RESULT (2026-06-01, deberta_pdual_v2, + non-bio-harm negatives) - the fix
worked decisively.** Adding non-bio harmful prompts as label-0 crashed the non-bio-
harm flag-rate from 0.589 to **0.002** and lifted **bio-selectivity S from 1.67 to
579**, while bio-harm recall held at 0.982 and benign-bio FPR stayed ~0 (0.004). The
prompt head is now genuinely BIO-SPECIFIC: it flags bio-harm and passes non-bio harm,
which no general guard (bio buried in one CBRNE bucket) can do. This is the first
direct substantiation of differentiation Claim 1. Diagnose->fix->verify worked
cleanly (P2 S=1.67 diagnosed generic-harm -> P2b non-bio-harm negatives -> S=579).
Residual: WMDP dual-use boundary over-flag is 0.21 (the legitimate-dual-use-research
over-refusal axis, Claim 2); next lever = add dual-use-research benign prompts as
negatives (or the ABSTAIN tier) to reduce it.

**P2c RESULT (2026-06-01, deberta_pdual_v3, + all-bio benign-knowledge negatives).**
Adding the full biomedical benign sources (LAB-Bench/PubMedQA/MedQA, ~4300) as
label-0 cut the WMDP dual-use over-flag from 0.21 to **0.132** (passes 87% of
legitimate dual-use research) while recall HELD at 0.983, S improved to 869
(non-bio-harm flag 0.001), benign-bio FPR ~0.006. The standalone prompt-head is now
a strong bio-specialized intent classifier: **recall 0.983, bio-selectivity S=869,
dual-use over-flag 0.132, benign-bio FPR 0.006**. Three diagnose->fix->verify
iterations: P2 generic (S=1.67) -> P2b non-bio-harm-neg (S=579) -> P2c
benign-knowledge-neg (WMDP 0.21->0.132). Residual WMDP over-flag 0.132 (1/8) is
addressable by the ABSTAIN tier or MMLU-bio negatives (diminishing returns).
**NEXT = P3** (shared-encoder dual-mode integration: combine this prompt-head with
the v2 response-head, staged unfreeze + regression guard).

**PRE-P3 VALIDATION (2026-06-01, drift-review payoff) - the prompt-head is a
BIO-KEYWORD SHORTCUT, not a genuine intent classifier.** On HELD-OUT wildguard_test
bio prompts the OOD bio-harm recall is **0.263** (not the in-dist 0.983; the in-dist
number was a saladbench/alert STYLE shortcut). Lexical ablation: masking bio
keywords flips **0.667** of flagged positives to safe -> the head keys on bio
keywords, not harm-intent (embeddings were not frozen). OOD benign-bio FPR 0.667
(n=3). So recall 0.983 / S=869 were MISLEADING (in-distribution + keyword-driven);
the S=869 "bio-specificity" is partly a keyword artifact (non-bio harm lacks bio
keywords). Root cause: the diverse positive pool is only ~467, saladbench/alert-
dominated, too narrow to learn beyond keywords. DECISION: pivot to the section-4
**LEXICOGRAPHIC HYBRID** (tiered lexicon T1 select-agents/toxins + T2 harm-methods +
sequence/BLAST + the learned head as ONE signal; a bare generic-bio keyword never
the sole trigger). A pure learned text classifier cannot be the bio-prompt-harm head
on this data. The drift review caught this BEFORE P3, exactly its purpose (10th
eval-honesty catch of the effort).

**P2d: LEXICOGRAPHIC HYBRID RESULT (2026-06-01).** After 4 iterations (v0 raw lex,
v1 gap-fix adding coronavirus/botulinum/create/make to T1/T2, v2 T2 precision
tightening removing generic verbs, v3 T1U/T1C split), the final 4-tier lexicon:
- T1U (unconditional: bioweapon, gain-of-function, select-agent) flags on mention.
- T1C (contextual: select-agents/toxins) flags only with T2 method co-occurrence.
- T2 (high-precision bio-harm verbs: synthesize, aerosolize, weaponize, etc.).
- T3 (generic bio: virus, pathogen, lab) never flags alone; modulates T2/T1C.
Key design: "what is anthrax" = T1C mention only = PASS; "synthesize anthrax" =
T1C+T2 = FLAG. Generic verbs (create, make, grow) excluded from T2 because they
trigger on non-bio ("create phishing email") and benign-bio ("grow cells").

ALSO DISCOVERED: the OOD eval set (19 "bio-harm" wildguard_test prompts) was
**~80% garbage** (computer-virus/malware/cyber prompts matched by the broad
`is_bio()` keyword filter, substring artifacts like "pricing" matching "ricin",
Plague Inc game references). Cleaned to 9, of which only ~4 are genuine bio-harm.
This is itself a contribution: evaluating bio-specialized classifiers requires
domain-aware filtering, not naive keyword matching.

| Metric | lex | learned | hybrid (lex OR learned-with-bio-ctx) |
|--------|-----|---------|--------------------------------------|
| OOD bio recall (n=9) | 0.000 | 0.444 | **0.444** |
| nonbio-harm flag (n=730) | 0.001 | 0.005 | **0.004** |
| in-dist recall (n=58) | 0.121 | 0.983 | **0.948** |
| in-dist FPR (n=1796) | 0.003 | 0.006 | **0.007** |
| dual-use over-flag (n=3272) | **0.073** | 0.132 | 0.167 |
| nonbio selectivity | 0.001 | 0.001 | **0.001** |

The lex alone is extreme high-precision low-recall (rule catches T1U/T1C+T2 only);
the learned head provides recall; hybrid combines both. Lex's biggest win = dual-use
over-flag 0.073 (vs learned 0.132) because T1C+no-method = PASS. The hybrid
architecture is validated: transparent rules for the certain, learned model for
the rest (with bio-context gating to prevent non-bio FP).

**P2e: T2 STRONG/WEAK SPLIT + DUAL-MODE END-TO-END VALIDATION (2026-06-02).** The
P2d table above is the pre-split lex/learned/hybrid comparison (its own harness,
eval_pdual_hybrid). P2e records the serving wrapper (`dual_mode.py`) and two
refinements found while validating it end-to-end.

*Review-1 fixes (commit eee01f0).* The response head unpacked `predict_batch`'s
3-tuple `(label, conf, prob_unsafe)` into 2 vars -> crash on every response/joint
call, and would have thresholded on confidence=P(argmax) not P(unsafe); fixed.
Also: learned-flag tier sentinel removed, stale "Option 3 chosen" §2 banner added,
and pdual_v3 train==serve confirmed (pair pipeline, response=""). Response head
switched to sliding-window to match serve.py (no effect on short responses;
prevents 512-tok truncation under-recall on long ones).

*T2 strong/weak split (commit fb7bc48).* Benign-bio over-refusal at scale
(eval_overrefusal_bio) localized the residual to ONE pattern: T2 routine verbs
(culture/cultivate/manufacture) x T3 generic-bio firing tier-2 on explicitly
benign prompts ("culture HARMLESS bacteria" -- 8 of 11 lex flags on OR-Bench-
health benign). Fix repartitions T2 (no terms added/removed): T2_STRONG
(synthesize/aerosolize/weaponize/...) escalate with T1C OR generic T3; T2_WEAK
(culture/cultivate/manufacture/propagate/stockpile/scale-up) escalate ONLY named
select agents (T1C), deferring generic-bio to the learned head. "culture anthrax"
still FLAGS (T1C+weak); "culture harmless bacteria" now PASSES. Before -> after:

| prompt-axis metric | before | after |
|--------------------|--------|-------|
| OR-Bench-health benign over-refusal (n=740) | 0.022 (lex 11) | **0.011** (lex 3) |
| in-dist neg FPR (n=1796) | 0.007 (lex 6) | **0.004** (lex 1) |
| in-dist pos recall (n=58) | 0.948 | **0.948** (unchanged) |
| OOD bio recall (n=9) | 0.444 | **0.444** (unchanged) |
| WMDP dual-use stems (n=1273) | 0.125 | **0.119** |

Over-refusal AND negative-FPR improved at zero real-harm recall cost (no positive
relied on the weak+T3 rule; the learned head already caught them) -- the hybrid
thesis in miniature: a more precise lexicon + delegation lowers FP for free.

*Benign-bio over-refusal, final (curated sets, prompt axis, lower=better):*

| set | n | flag-rate |
|-----|---|-----------|
| FalseReject-test bio-context | 142 | **0.000** |
| OR-Bench-Hard-1k benign-bio | 34 | **0.000** |
| OR-Bench-health benign | 740 | 0.011 |
| WMDP-bio stems (dual-use boundary) | 1273 | 0.119 |

The earlier wildguard "bio_benign" 0.667 was an n=3 artifact, now superseded.
Field-best benign-bio FPR is ~0.03 (Meng/Zhang); we sit at 0.000-0.011 on curated
benign-bio WITH a learned recall head, clearing the §4 target.

*Dual-mode end-to-end (`dual_mode.py`; OOD wildguard cleaned + in-dist + joint on
the response-harm test set n=500/pos=331):*

| axis | metric | value |
|------|--------|-------|
| prompt | in-dist recall / nonbio-harm flag / benign-bio | 0.948 / 0.004 / 0.000-0.011 |
| prompt | OOD bio recall (cleaned, n=9) | 0.444 |
| response (v8b, sliding) | TPR / FPR | 0.662 / 0.047 |
| joint OR | TPR / FPR | 0.719 / 0.118 |

*Serving guidance (validated).* joint-OR buys +5.7pt TPR for +7.1pt FPR over
response-alone on response-harm (harmful responses often have benign prompts), so
consume PER AXIS: prompt_flag for input/intent screening, response_flag for output
moderation, joint_flag only for max-recall combined screens. `dual_mode` returns
all three independently.

## 4. Bio-selectivity: the metric (a contribution) + method

The core specialization risk is flagging legitimate dual-use research. We define
and optimize a metric no prior guard publishes:
- **Bio-selectivity ratio S = flag-rate(bio-harm prompts) / flag-rate(non-bio-harm
  prompts).** S >> 1 means bio-SPECIFIC, not a generic harm detector. Denominator
  measured on HarmBench non-chemical_biological + AdvBench + JBB.
- **Benign-bio FPR = flags / benign-bio prompts**, on LAB-Bench + FalseReject-bio +
  PubMedQA. This is the over-refusal axis.
- Target: beat the field-best benign-bio FPR ~0.03 (Meng/Zhang Biosecurity Agent,
  L1/L2 lexicon) at higher recall, using a learned encoder instead of lexicon-only.

Selectivity mechanisms (hybrid, lexicographic priority, from the Biosecurity
Agent's 5-signal stack): sequence/BLAST signals (route nucleotide/protein strings
to BLAST vs a pathogen DB, not text classification) > semantic > fuzzy > tiered
keyword. Build pathogen-lexicon tiers from authoritative lists (US Federal Select
Agents & Toxins, Australia Group, EU dual-use, IGSC Regulated Pathogen DB); a bare
pathogen keyword is the WEAKEST evidence and never the sole trigger (that is what
keeps "what is anthrax" from flagging). For the learned head: 3-way label
ALLOW / BIO-FLAG / ABSTAIN (ABSTAIN routes dual-use ambiguity to review instead of
forcing a call); FREEZE word embeddings during FT (prevents overfitting to
pathogen surface tokens, the "mentions a pathogen => harmful" failure); asymmetric
high tau_flag + margin requirement.

## 5. Differentiation: the defensible claims (and what to drop)

Four claims, each with the proof experiment:
1. **Bio-selectivity (STRONGEST, structural).** General guards have no bio category
   and cannot separate bio-harm from legitimate dual-use. Proof: head-to-head on
   RefusalBench matched-triples + FORTRESS benign-twins; metric = Youden's J /
   tier-discrimination AUC vs the general guards run as bio classifiers. (Evidence:
   RefusalBench shows 0.1-94.6% refusal span on identical bio prompts.)
2. **Low over-refusal on real science (STRONG, fix the framing).** Safety systems
   refuse >80% of hard-benign bio (Health-ORSC-Bench; Claude-Opus 83%). Our ~2% is
   a genuine outlier ONLY if reported on a NAMED corpus with the SAME corpus run
   through competitors. Never compare our 2% to someone else's 80% on a different
   set.
3. **Distribution-free calibration (STRONG, novel for a guard).** Guards are
   miscalibrated (Llama Guard ECE 28.5%, WildGuard 34.9% under jailbreak, F1 still
   92.8% = confidently wrong). Ship ECE/Brier + a conformal (LTT/RCPS) bound on the
   bio-harm miss rate. State the exchangeability assumption honestly.
4. **Cascade efficiency (STRONG but table-stakes).** 184M as a cheap always-on
   first stage routing to a frontier model (GLiGuard precedent: 0.3B within 1-2 F1
   of 7-27B at 16x throughput). Compelling ONLY bundled with bio-selectivity.

Frontier we could OWN (higher risk): multi-turn bio elicitation (Biorisk-Shift
53.5% bypass; all general guards are turn-local) + the dual-use text boundary.
NOT DNA-sequence (GeneBreaker) or agentic (ABC-Bench) alone; we COMPOSE with those.

DROP / heavily qualify:
- "Matches 8B on adversarial robustness" -> most suspicious claim; even Qwen3Guard
  collapses 91 -> 34 on novel attacks. Reframe to "robust within the bio
  distribution under named attack families," report decontaminated novel-attack
  results honestly. Robustness is a guardrail we report, not a differentiator.
- "0.92 recall" -> always with the denominator (N, source, annotator, kappa) and
  the paired precision/over-refusal at that operating point.
- "track record of catching eval artifacts" -> a credibility multiplier and the
  narrative spine (it is HOW we produced claims 1-3: decontaminated splits,
  matched triples, conformal validity, honest FPR denominators), not an
  independently citable claim.

## 6. Pre-registered evaluation gates (fix before training)

Credibility gates (standard suite, prove we are a real dual-mode guard):
- prompt-harm avg F1 within ~10 pts of WildGuard-7B on ToxicChat/OpenAI-Mod/
  HarmBench-prompt/WildGuardTest-prompt.
- response-harm: hold the v2 numbers (no regression).

Winning gates (the bio axis, pre-registered):
- bio prompt-harm recall (SciSafeEval-bio + ClearHarm held-out) >= 0.90.
- bio-selectivity ratio S >= [target, set from baseline run].
- benign-bio FPR <= 0.05 (stretch: approach the 0.03 lexicon floor at higher recall).
- over-refulsal on the real legitimate-research set <= 5%, reported side-by-side
  with general guards on the SAME set.
- ECE <= 0.05 after temperature scaling; report ECE under jailbreak shift vs
  WildGuard's 0.349.

## 7. Phased plan

- P1 Data: harvest + relabel + dedup + leakage-audit the prompt-harm bio set
  (positives + hard-negatives + easy-negatives); build the held-out bio eval suite.
- P2 Baseline: train a STANDALONE prompt-harm bio head (Option 2 fallback first)
  to get a clean prompt-harm number and a selectivity baseline. Low risk, fast.
- P3 Multi-task: Option 3 shared-encoder, staged unfreeze, response-data replay,
  regression guard. Compare to P2.
- P4 Bio-selectivity + lexicographic hybrid (optional sequence/BLAST routing).
- P5 Calibration + conformal bound; the cascade economics ablation.
- P6 Full eval vs the standard suite + the bio suite; honest model card v3
  (dual-mode), gated release as bioguard-deberta-v3.

## 8. References (verified across four research sweeps)

Guards: Llama Guard arXiv:2312.06674; WildGuard arXiv:2406.18495 (multi-task>single
ablation Table 5, prompt-only=empty-response trick Table 12); ShieldGemma
arXiv:2407.21772; Aegis arXiv:2404.05993 / 2.0 arXiv:2501.09004 (LoRA on frozen
base); Qwen3Guard arXiv:2510.14276 (Stream two-head design); Granite Guardian
arXiv:2412.07724; Constitutional Classifiers arXiv:2501.18837. Small encoder
multi-task guards: GLiNER-Guard arXiv:2605.05277; Opir arXiv:2605.29659; GLiGuard
arXiv:2605.07982. Prompt-harm data: SciSafeEval arXiv:2410.03769; SALAD-Bench
arXiv:2402.05044; ALERT arXiv:2404.08676; HarmBench arXiv:2402.04249; WMDP
arXiv:2403.03218; ClearHarm (AlignmentResearch/ClearHarm). Hard negatives:
FalseReject arXiv:2505.08054; OR-Bench arXiv:2405.20947; LAB-Bench arXiv:2407.10362;
XSTest arXiv:2308.01263; PubMedQA arXiv:1909.06146. Bio-selectivity: Meng/Zhang
Biosecurity Agent arXiv:2510.09615; IntentGuard (HF perfecXion blog). Differentiation
evidence: RefusalBench arXiv:2605.21545; Health-ORSC-Bench arXiv:2601.17642;
FORTRESS arXiv:2506.14922; SOSBench arXiv:2505.21605; guard calibration
arXiv:2410.10414; robustness collapse arXiv:2511.22047; cascade SafeRoute
arXiv:2502.12464. Bio frontier (NeurIPS 2025 BioSafe GenAI): Biorisk-Shift,
GeneBreaker, ABC-Bench; BioRiskEval arXiv:2510.27629; LLM Novice Uplift arXiv:2602.23329.

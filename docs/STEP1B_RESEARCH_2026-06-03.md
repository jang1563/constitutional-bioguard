# Step 1b research: closing the borderline-benign over-refusal gap (2026-06-03)

Deep-research pass (25 sources, 121 claims, 20 confirmed / 5 refuted) scoped to the ONE wall
from `STEP1_DISTILL_PILOT_2026-06-03.md`: the distilled 184M student inherits the teacher's
recall (0.983) but over-refuses borderline-benign at ~0.83 vs the teacher's 0.166, and
soft-label distillation did not close it (0.879->0.831).

## Verified mechanism (why soft-CE failed) -- 3-0
Hinton soft-target-CE is a COUPLED loss: the non-target "dark knowledge" (NCKD) term is
weighted by `(1 - p_teacher^T)` (Zhao et al., DKD, CVPR 2022, arXiv:2203.08679, Eq.6
`KD = TCKD + (1 - p_t^T)*NCKD`). At the pilot's teacher harmful soft-mean 0.924, the
benign-boundary signal is attenuated ~13x precisely on confident-harmful examples. This
explains the failed 0.879->0.831. Decoupled KD (DKD) gives TCKD/NCKD independent weights.
NUANCE: NCKD's "non-target classes" is formally degenerate for a strict binary head; it
applies because the teacher is GENERATIVE (continuous soft benign-prob).

## CRITICAL FLAG (strongly implied, not proven)
Naive soft-label distillation on a clean-benign-only pool **fundamentally cannot** close this
gap: the teacher's soft labels never cover the borderline region the student is tested on.
No temperature/lambda change alone resolves it. This is a DATA-COMPOSITION wall. (Synthesis
of the pilot finding + the verified (1-p_t) attenuation; stated as strongly implied.)

## The recipe (prioritized)

### PRIMARY LEVER -- add borderline-benign hard negatives to training (3-0 verified)
- **FalseReject** (Zhang et al., COLM 2025, arXiv:2505.08054; HF `AmazonScience/FalseReject`):
  16k seemingly-toxic-but-benign prompts, 44 safety categories, ships a real **15k train split**
  (Train-Instruct + Train-CoT) + 1.1k human test. SFT on it "substantially reduces unnecessary
  refusals without compromising overall safety or general language capabilities" (benign
  compliance Llama-3-8B 44.58%->64.67%, Qwen-2.5-7B 53.30%->77.48%; toxic-safety held 99-100%).
  Built by a DIFFERENT (graph-informed multi-agent) method than OR-Bench -> distinct from the
  OR-Bench-health eval.
- **XSTest** (Roettger et al., NAACL 2024, arXiv:2308.01263): 250 safe + 200 unsafe MINIMAL-EDIT
  contrastive twins ("kill a Python process" vs "kill a person"). Attributes over-refusal to
  LEXICAL OVERFITTING -- the exact mechanism behind the student over-flagging bio vocab. Small
  (450), best as a contrastive-twin seed alongside FalseReject.
- **Feasibility proof (3-0):** DCR (ICLR 2026, arXiv:2603.03323) lifts OR-Bench compliance
  0.34->0.71, XSTest 0.66->0.93, safety held ~0.94 vs 0.95 -> the over-refusal/recall tradeoff
  is DECOUPLABLE.
- Train these as **HARD-LABEL benign (label 0)** so the student is NOT upper-bounded by the
  teacher's 0.166 (breaks the soft-label ceiling; JurEE-style direct supervision).

### SECONDARY LEVER -- swap the coupled objective (medium, vision-extrapolated)
Once data is fixed, replace Hinton soft-CE with a decoupling/relation-preserving objective:
- **DKD** (arXiv:2203.08679): independent TCKD/NCKD weights -> un-suppress the boundary signal.
- **Logit standardization** (Sun et al., CVPR 2024, arXiv:2403.01427): Z-score logits pre-softmax
  so the student matches logit RELATIONS not magnitude; provably preserves rank/argmax.
- Reverse-KL is mode-seeking but DOUBLE-EDGED (drives overconfidence, can HARDEN the benign
  boundary and worsen FPs) -> not recommended first.
Treat as second-order refinements AFTER the data fix; all are vision/generative extrapolations.

### LEAKAGE-SAFE SOURCING (3-0 verified)
OR-Bench has NO official train/test split and NO health category; "OR-Bench-health" (739) is a
user-derived semantic filter, so training on ANY OR-Bench prompt leaks the eval. Train on
FalseReject-Train / XSTest instead; keep OR-Bench-health purely held-out; decontaminate the
training negatives vs OR-Bench-health with n-gram overlap + embedding-NN thresholds (FalseReject
seeds from existing safety datasets, so decon is required).

## Guardrails (refuted -- do NOT do)
1. Contrastive pairing beating mere negative-addition: REFUTED 0-3 (DCR STL-aug). -> ADD
   negatives is the evidenced default; contrastive twins are an unproven enhancement.
2. Teacher temperature scaling as the fix: REFUTED 0-3. -> do not rely on it.
3. BSS "helps most when data is sparse": REFUTED 0-3 (mechanism stands, rationale does not).

## Dominant uncertainty + ablations to run (open questions)
ALL evidence is vision/many-class or full-generative-LLM SFT; NONE on a 184M DeBERTa encoder
distilled from a generative teacher. Treat the recipe as hypotheses to ablate:
1. **Does FalseReject/DCR over-refusal reduction TRANSFER to a 184M encoder?** (biggest untested
   assumption) -> ablate: encoder trained on harmful+borderline-benign HARD labels vs distilled-only.
2. **Capacity vs data:** does adding borderline-benign close the gap at 184M (data-limited) or
   does it need 435M (capacity-limited)? -> oracle-data ablation + student-size sweep.
3. **Does hard-label borderline-benign break the teacher's 0.166 ceiling?** -> direct test.
UNVERIFIED THIS PASS: JurEE (arXiv:2410.08442) and TAID (arXiv:2501.16937) did not surface in
the confirmed set -> check before committing to those specific levers.

## Concrete next action
1. Fetch FalseReject-Train (HF `AmazonScience/FalseReject`, ~15k). Inspect schema/categories.
2. Decontaminate vs OR-Bench-health (739) + all eval sets (n-gram + embedding-NN).
3. Build augmented pool: bio-harmful + bio-benign + FalseReject borderline-benign (hard label 0).
4. Re-train the 184M student (hard-label CE primary; optional soft-CE on the bio subset only).
5. Eval on held-out OR-Bench-health + bio recall + clean legit-bio over-refusal. This single run
   tests open-Q1 (encoder transfer), Q3 (teacher ceiling), and the data-coverage hypothesis.
6. If the gap closes at 184M -> data-coverage confirmed, ship path clear. If not -> 435M / DKD / TAID.

## EMPIRICAL Step-1b result (2026-06-03) -- general borderline-benign does NOT transfer to bio
Ran the PRIMARY lever: added 2500 decontaminated FalseReject-Train borderline-benign (hard
label 0, stratified across 46 categories) to the 2442 bio pool, re-trained the 184M student
(hard-label CE, unsafe-weight 2.0). Decon: 0 exact + 105 8-gram overlaps dropped (FalseReject
is genuinely disjoint from the eval). Held-out eval (NOT trained on or_bench_health):

| metric | original distill | +FalseReject aug |
|---|---|---|
| recall (120 harmful bio) | 0.983 | 0.967 |
| over-refusal, clean legit bio (181) | 0.017 | 0.022 |
| over-refusal, or_bench 881 ALL | 0.879 | 0.784 |
| over-refusal, cyber-"virus" subset (486) | 0.932 | 0.986 |
| over-refusal, genuinely-biological borderline (79) | 0.671 | 0.709 |
| AUROC | 0.399 | 0.568 |

READ: the data lever moved the GENERAL/non-bio over-refusal (881 0.879->0.784, AUROC up) but
did NOT help the genuinely-biological borderline (0.671->0.709, within noise) and not the cyber
subset. So adding GENERAL borderline-benign does not transfer to the BIO boundary -- it needs
DOMAIN-MATCHED (dangerous-sounding-but-legit BIO) negatives, which FalseReject lacks (only 37
of 14624 prompts are bio). Confirms the research's "domain-matched" caveat empirically.

ALSO: the or_bench_health eval is a poor bio yardstick -- of 739, ~486 are computer/virtual
"virus" prompts (keyword false-match), much of the rest is finance/general; only ~79 are
genuinely biological. The TRUE bio over-refusal signal is: clean legit bio 0.022 (excellent) vs
dangerous-sounding bio borderline 0.671 (lexical overfitting -- flags "dangerous pathogen",
"virus creation", "biological weapon" vocab even with safety/educational/research framing).

CONCLUSION: the residual gap is a BIO-SPECIFIC borderline-benign DATA WALL. Reuse-only bio-
borderline data is thin (the available borderline set IS the eval). Closing it requires either
(A) GENERATING bio-borderline-benign (legit dangerous-sounding research questions, teacher/hard-
labeled benign -- the same LLM-rewriting methodology OR-Bench/FalseReject themselves use; this
is BENIGN generation, distinct from the harmful-positive generation the project avoided in v8d),
or (B) a CASCADE (small student for clear cases, route borderline/uncertain to the 8B teacher,
SafeRoute-style), or (C) ship with the conservative-on-borderline-bio limitation documented
(over-refusing dangerous-sounding bio is the SAFE failure direction for a guard, but less
competitive than the teacher's 0.166). Objective levers (DKD/logit-std) are second-order and
unlikely to fix lexical overfitting without the domain data.

## EXECUTED: option (A) bio-borderline GENERATION (JK-approved 2026-06-03)
Generated bio-borderline-benign prompts two ways (both hard_label=0, decontaminated vs all eval
sets, 0 exact + 0 8-gram overlaps): (1) a TEMPLATE generator (33 dangerous-sounding topics x 10
legit frames x 11 question templates -> 1200 sampled, scripts/experiments/gen_bio_borderline.py); (2) an
LLM-REWRITE generator (Claude Sonnet, 60 seeds x 12 -> 551, scripts/experiments/gen_bio_borderline_llm.py,
more natural/diverse). Trained the 184M student on bio pool + each.

RESULT on the 79 genuinely-biological borderline (realbio_eval): original 0.671 -> +template
0.532 -> +LLM-rewrite 0.544 (recall held: 0.983 template / 0.975 LLM; clean legit-bio 0.022).
Generation HELPED (0.671 -> 0.53) but BOTH approaches PLATEAU at ~0.53 -- the LLM rewrite did not
beat the template. This empirically bounds the query-only head: even with domain-matched generated
borderline data, the saturated prompt head cannot fully separate harmful-bio from dangerous-
sounding-benign-bio at the QUERY level (threshold sweep: 0.481 even at tau=0.99). The residual is
resolved by ARCHITECTURE in Step 2 (the response gate disambiguates: 0.532 -> 0.076), not by more
data. See STEP2_DUALMODE_2026-06-03.md.

## Sources (primary, verified)
DKD 2203.08679 · FalseReject 2505.08054 · DCR 2603.03323 · XSTest 2308.01263 · OR-Bench 2405.20947 ·
logit-std 2403.01427 · teacher-calibration 2508.20224 · BSS 1805.05532 (boundary mechanism only).

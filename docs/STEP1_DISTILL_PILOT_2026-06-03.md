# Step 1 bio-distillation pilot: result + fork resolution (2026-06-03)

Resolves the highest-leverage gate in `RELEASE_PLAN_2026-06-03.md`: does the narrow bio
prompt-harm signal survive compressing the 8B generative teacher (v7.C-aug2) into a
deployable 184M DeBERTa-v3-base encoder? Run on Cayuga (`constitutional-bioguard`), scored
on held-out, leakage-disjoint sets.

## The trainer bug (root cause of every prior NaN / all-zero)
`transformers 5.9.0` loads `microsoft/deberta-v3-base` in **fp16 by default** in
`AutoModelForSequenceClassification.from_pretrained(...)` (the model config has no
`torch_dtype`, but the 5.x default path returns fp16 weights here). DeBERTa-v3's
disentangled attention is numerically unstable in fp16 -> the forward NaNs, which silently
corrupted the student weights (finite logged train_loss, but all-zero / NaN eval). Every
earlier failure mode (Half-vs-Float dtype error, "unscale FP16 gradients", bf16 NaN,
all-zero eval) traces to this single cause. **Fix: `from_pretrained(..., dtype=torch.float32)`
+ pure-fp32 TrainingArguments (`fp16=False, bf16=False`).** Pure fp32 is unconditionally
stable for DeBERTa-v3 and fast enough here (5 epochs x 2442 ex = ~200 s on one A100).
A fresh untrained encoder forward was fine *in fp32*, which is why the earlier isolation
test mislabeled the encoder as innocent: the Trainer ran it in fp16.

## Setup
- Teacher: v7.C-aug2, 8B Llama-3.1 + QLoRA generative prompt-harm head.
- Student: `microsoft/deberta-v3-base` (184M), weighted CE (class0=benign 1.0, class1=harmful 1.5).
- Pool: `distill_pool.jsonl` 2442 bio prompts (harmful 1812 / benign 630), leakage-disjoint
  from every eval set. Soft variant adds teacher `soft_label` (`distill_pool_labeled.jsonl`,
  harmful soft mean 0.924 / benign 0.064).
- Distillation: hard-label CE, and HarmAug soft variant = lam*CE + (1-lam)*soft-target-CE
  (Hinton soft-CE, no log-of-target so NaN-safe), lam=0.5.

## Result (student @0.5)
| metric | hard-label | soft-label | teacher (v7.C-aug2 LN) |
|---|---|---|---|
| pool-self separation (harmful / benign p) | 1.000 / 0.004 | 0.870 / 0.007 | -- |
| **recall, bio_clean_eval 120 harmful** | **0.983** | **0.983** | 0.900 |
| over-refusal, bio-overrefusal-v0.1 (181 expert legit bio) | **0.017** | **0.017** | 0.023 |
| over-refusal, bio_clean_eval 881 benign | 0.879 | 0.831 | 0.232 |
| AUROC, bio_clean_eval | 0.399 | 0.428 | 0.897 |

Breakdown of the bio_clean_eval 881 benign (the over-refusal that looks bad):
- 739 are **OR-Bench-health** (`or_bench_health`): deliberately borderline-phrased benign,
  an over-refusal stress set. ~486 of the 746 keyword-"bio" matches are actually **computer
  /ransomware "virus"** prompts (the `is_bio` keyword filter false-matches "virus").
- student over-refusal: cyber-"virus" subset 0.932; genuine-bio borderline subset 0.785.
- **teacher over-refusal on the same 739 OR-Bench-health: 0.166.** (false_reject_test 142:
  teacher 0.570, hard for both.)

## Fork resolution (splits the plan)
**RECALL at the 0.5 operating point transfers** (student 0.983 >= teacher 0.900), and clean
expert-curated legit-bio over-refusal transfers (0.017 ~ teacher 0.023). HOWEVER, the student's
AUPRC is 0.121 vs teacher 0.605 -- the student achieves high recall by SATURATION (pushing most
items above 0.5, sacrificing precision), not by preserving the teacher's discrimination. The
student is a recall-first pre-generation gate, NOT a calibrated capability-compressed classifier.
"Footprint solved" as originally stated (CLAIM 1) is FALSE at AUPRC; honest framing: "recall at
the chosen threshold transfers, but overall discrimination does not."

**BORDERLINE-BENIGN CALIBRATION: Fork 1b materialized.** On the *same* 739 OR-Bench-health
borderline-benign prompts, teacher over-refuses 16.6% but the student over-refuses ~83%
(genuine-bio 78.5%, cyber 93.2%). This is the documented capacity-gap / mode-averaging
failure on the over-refusal axis: the student inherits the specialist positive signal but
not the teacher's nuanced benign boundary. **Soft-label distillation did NOT close it**
(0.879 -> 0.831) because the distill pool's benign is clean, so the teacher soft labels
never cover the borderline region the student is tested on.

## Implication / next action
The 8B->184M footprint is resolved for **recall and clean over-refusal**, but a naively
distilled small student is **not competitive on adversarial borderline-benign over-refusal**.
The capacity is sufficient (the student fits the pool perfectly); the gap is **data
composition**, not capacity. To ship the small student as the prompt head:
1. **Augment the distill pool with borderline-benign hard negatives** (OR-Bench-style,
   leakage-disjoint from eval) + teacher soft labels on them, then re-distill. This is the
   targeted fix and the most likely to close the 83%->~20% over-refusal gap.
2. If it does not close, fall back to a capacity-gap-aware objective (TAID) or a 435M student.
3. Independent of this, the dual-mode prompt-veto policy already drives the *response* head's
   density-FP over-refusal to 0 (bridge experiment); the prompt head's own borderline
   over-refusal is what #1 addresses.

## Artifacts
- `scripts/experiments/train_v7c_distill.py` (--soft for HarmAug soft-CE), `scripts/experiments/eval_distill_student.py`,
  `scripts/experiments/split_cyber_bio.py`. Results: `results/v7c_distill_pilot.json`,
  `results/distill_student_orefusal_breakdown.json`. Student weights:
  Cayuga `models/deberta_v7c_distill/final` (NOT a release checkpoint; pilot only).

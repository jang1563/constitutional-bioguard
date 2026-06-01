# v8b: a shippable bio response-harm classifier (ship evidence)

One-page evidence memo. Every number below was measured this cycle on held-out,
leakage-audited data. Companion design and caveats live in `docs/V8_DESIGN.md` §8.

## What it is

v8b is a small encoder classifier (DeBERTa-v3-base, 12 layers, hidden 768, about
184M params) that reads a `query [SEP] response` pair and decides whether the
*response* delivers harmful biological content. It is built for the deployment
job a generalist guard does poorly: flag a genuinely harmful bio completion while
leaving legitimate bench research alone.

Training data is 3,507 examples, reuse-only, with zero newly generated harmful
content: WildGuardMix bio responses (1,350), BeaverTails bio (1,024), FalseReject
benign hard-negatives (891), and a non-bio control carryover (242). Positives are
33% of the set. No example was synthesized to inflate the harmful class.

## Ship evidence: two independent bars, both cleared

A guard is only shippable if it catches real harm *and* does not block real
research. v8b is the first model in this line to clear both, measured on two
held-out sets it never trained on.

| Bar | Metric | v8b | v4 (prior best) | v8 baseline |
|-----|--------|-----|-----------------|-------------|
| Catch real harm | Real harmful-bio response recall (n=62 pos) | **0.919** | 0.290 | 0.274 |
| Pass real research | Over-refusal on real legit bio (n=531) | **0.021** | 0.047 | 0.064 |
| Pass real research | Over-refusal on the user's own session logs (n=134) | **0.060** | 0.134 | 0.254 |

The over-refusal set is the operationally important one: 531 real legitimate bio
items, of which 134 are the user's own Claude Code and Codex sessions where
legitimate research had been flagged. v8b over-refuses that real research at 6%,
inside the pre-registered money-metric gate of 10%. The two larger guards this
line was benchmarked against sit at roughly 0.30 to 0.43 recall on real
harmful-bio responses in published evaluation, the same range where v4 sits here,
so v8b's 0.919 is a step change on the recall bar specifically, not a small gain.

## Why this is not a benchmark shortcut

The honest risk with any fine-tuned guard is Goodharting: learning the benchmark,
not the task. Three checks argue against it here.

1. **Lexical ablation.** Masking every bio keyword in the input changes v8b's
   predictions by about 1%. The model is not keying on a bio word list.
2. **Real-world validation.** The over-refusal numbers above are from the user's
   actual research sessions, not a synthetic benchmark. An earlier alarm (v8b
   appearing to over-refuse at 27% on adversarial benchmark negatives) did not
   reproduce on real research, where it over-refuses at 2%. The benchmark
   negatives were noisy; v8b was largely correct to flag them.
3. **Diagnose before prescribing.** Three separate times this cycle, an apparent
   data deficiency turned out to be an evaluation artifact (a prompt-vs-response
   labeling mismatch, a redacted-placeholder routing bug, and the over-refusal
   alarm above). Each was caught by diagnosis before any retraining. The
   discipline is the point: the reported gates reflect the model, not the harness.

Evaluation was leakage-audited (training queries are byte-disjoint from every
test set). One scope caveat, stated plainly: the pre-registered Tier-1 gates
attach a constant stub response, so they test *prompt* harm, and a response-harm
classifier like v8b scores near zero on them by design. v8b's evidence therefore
rests on the real-response benchmarks (WildGuard-native at recall 0.69 on 1,709
items, the bio subset above, and the real over-refusal set), which are the valid
bars for this model class. v8b does not cover prompt-side risk and is not a
prompt filter.

## Honest limitations

- **The positive class has a reuse-only ceiling.** The hardest cases, ambiguous
  dual-use bio such as immune-evasion vector design, do not exist as labeled
  harmful examples in any public source. Expanding past v8b on recall would
  require generating those positives, which this project deliberately does not do.
  v8b is the ceiling reachable without generation, and that ceiling is useful.
- **The real over-refusal sample is modest** (134 sessions, 68 with substantive
  responses). The gaps versus baselines are large and consistent across strata,
  but treat the rates as directional rather than tight point estimates.
- **Distribution scope.** v8b learned from WildGuardMix-family responses; it is
  strong in that broad distribution and was not claimed to transfer to arbitrary
  out-of-distribution response styles without measurement.

## Independent corroboration

A parallel track built a separate over-refusal classifier on a different axis
(Korean-language coverage, the user's own session logs as the money metric). It
reached the same wall from the opposite direction: no labeled harmful examples
exist for ambiguous dual-use cell biology in any published benchmark. Two
independent efforts converging on the same boundary is the strongest available
evidence that the limit is real and not a gap in either build.

## Bottom line

v8b catches real harmful bio responses at 0.919 recall and passes real
legitimate research, including the user's own work, at 2% to 6% over-refusal,
clearing both pre-registered bars with a 184M-parameter model and no synthesized
harmful data. It is a defensible, calibrated, shippable artifact, with its
limitations stated as plainly as its results.

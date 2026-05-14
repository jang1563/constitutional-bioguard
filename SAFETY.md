# Responsible Use and Safety Scope

Constitutional BioGuard is a **prototype** biological dual-use content classifier.
It is intended for safety research, content-moderation pipeline experimentation,
and as a teaching artifact for the Constitutional Classifiers methodology applied
to a single domain. It is **not** a production safeguard.

## In Scope

- Research on dual-use content detection and constitution-driven training pipelines.
- Comparison studies between rule-based, classifier-based, and LLM-judge safeguards.
- Education on NSABB-category classification, calibration vs evasion trade-offs,
  and the limits of small-classifier safety.
- Building integration tests for downstream agent stacks (see AgentShield example).

## Out of Scope

- Sole reliance for any deployment that handles biology queries. The 9.79% mean
  adversarial ASR (and >30% on encoding attacks like ROT13) means this classifier
  must be paired with input filters, response guards, and human review.
- Use as evidence that any production system (Anthropic's, OpenAI's, etc.) is or
  is not "Constitutional-Classifier-equivalent." This repository is a domain
  extension experiment, not a reproduction of any vendor's deployed pipeline.
- Generating, expanding, or sharing the synthetic *unsafe* examples in
  isolation. The `data/` synthetic corpus is gitignored by design; releases
  publish only constitution rules, training scripts, evaluation harness,
  and aggregate metrics.
- Adversarial reuse: probing for evasion vectors against deployed safeguards
  using the published attack taxonomy as a recipe.

## Withheld Content

The following are intentionally **not** in this public repository:

- Generated synthetic unsafe examples (in `data/`, gitignored)
- Trained model weights with the unsafe-class probability head (the published
  HF model is the same architecture; weights are MIT but the unsafe-side
  generations are not redistributed)
- Per-attack ROT13 / encoding payloads at full fidelity
- External validation labels from BioThreat-Eval beyond aggregate kappa

## Reporting Concerns

Open a GitHub issue with the `safety` label for:

- A specific synthetic-example category that should be removed or sanitized
- A NSABB-category framing that is misleading or out of date
- Any artifact that could be repurposed as harmful guidance

For sensitive disclosures, email jak4013@med.cornell.edu directly with
"BIOGUARD SAFETY" in the subject. Do not paste operational biological
detail into public GitHub issues.

## Limitations Recap

- Solo-author classifier; expert circulation pending
- Trained on Claude-generated synthetic data; real-world distribution shift
  is uncharacterized
- English-centric; multilingual coverage limited to code-switching augmentation
- Encoding attacks are a fundamental weakness for any embedding-based classifier;
  they should be handled by an upstream tokenization-aware filter, not by this
  classifier alone

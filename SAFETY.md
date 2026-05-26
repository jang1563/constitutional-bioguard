# Responsible Use and Safety Scope

Constitutional BioGuard is a **research prototype** biological dual-use content
classifier. It is intended for safety research, content-moderation pipeline
experimentation, and as an auditable case study for the Constitutional
Classifiers methodology applied to a single domain. It is **not** a production
safeguard.

## In Scope

- Research on dual-use content detection and constitution-driven training pipelines.
- Comparison studies between rule-based, classifier-based, and LLM-judge safeguards.
- Education on NSABB-category classification, calibration vs evasion trade-offs,
  and the limits of small-classifier safety.
- Building integration tests for downstream agent stacks.

## Out of Scope

- Sole reliance for any deployment that handles biology queries. v4 improves
  over-refusal and response-style shortcut behavior, but still requires input
  filters, response guards, broader safety classifiers, and human review.
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
- Synthetic unsafe training examples and benchmark prompts at full fidelity.
  Model weights may be shared through controlled Hugging Face access for
  research use, but the unsafe-side generated corpus is not redistributed.
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

- Solo-author classifier; independent expert review not yet completed
- Trained on Claude-generated synthetic data; real-world distribution shift
  is uncharacterized
- v4 still over-flags an artificial refusal+compliance hybrid response pattern
- v5 PairCFR was intentionally not released after failing the specialist bio
  recall gate
- English-centric; multilingual coverage limited to code-switching augmentation
- Encoding attacks are a fundamental weakness for any embedding-based classifier;
  they should be handled by an upstream tokenization-aware filter, not by this
  classifier alone

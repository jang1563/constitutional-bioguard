# bioguard-v8b release checklist (R4 packaging)

Status of the gated, non-commercial research release (V8B_RELEASE_PLAN Phase R4).
Posture: gated HF repo, CC-BY-NC-4.0, withhold harmful corpus + exact threshold.

## License (verdict locked)
- [x] Model license CC-BY-NC-4.0 (forced by BeaverTails + FalseReject NonCommercial).
- [x] Each dataset license named in the card (ODC-BY / CC-BY-NC / CC-BY-NC).
- [x] Commercial use disallowed; Phase C (retrain off NC data) deferred by decision.

## Model card (release/README.md)
- [x] HF YAML frontmatter: license, base_model, datasets, pipeline_tag, tags, metrics.
- [x] model-index with structured eval (recall, AUROC, AUPRC, over-refusal FPR).
- [x] extra_gated_prompt + extra_gated_fields (gating + responsible-use acknowledgment).
- [x] Body: model details, intended/out-of-scope use, training data + licenses,
      evaluation (discrimination, over-refusal, calibration, robustness, scope
      boundary, lexical-ablation), limitations, ethics/privacy, get-started snippet.

## Artifact bundle (what ships with the weights)
- [x] `README.md` (the model card).
- [x] Input normalization defense (`constitutional_bioguard/preprocessing.py::normalize_text`),
      hardened for char-injection (ship as the deployed preprocessing).
- [x] `inference.py` (standalone load + normalize + classify; `release/inference.py`).
- [x] `RESULTS.md` (consolidated eval table + per-metric script mapping + reproduce commands).
- [x] `TRAINING.md` (data sources + licenses + hyperparameters + seed; harmful examples withheld).
- [x] Decision-threshold + calibration note (in the card: default 0.5, T=0.24).

## Withhold (do NOT ship)
- [x] Harmful (positive) training examples withheld.
- [x] Exact production/deployment threshold ship default 0.5 + instruct recalibration.
- [x] The author's session-log over-refusal eval data withheld (privacy).
- [x] Any companion attack/red-team harness not open-sourced.

## Gating
- [x] Repo to be created as a Hugging Face **gated** model repo.
- [x] Click-through Responsible-Use terms (in extra_gated_prompt).
- [ ] Responsible-disclosure contact line filled with a real address before push.

## Pending eval (release is shippable now as v1; these are v2 additions)
- [ ] Multi-turn robustness (windowed vs per-turn) blocked on MHJ access (requested).
- [ ] Content obfuscation/reconstruction needs an LLM rewrite endpoint.
- [ ] Held-out bio generalization blocked by the bio real-response scarcity (R3).

## Verdict
The model card + license + gating + withhold list + reproducibility bundle
(`README.md`, `RESULTS.md`, `TRAINING.md`, `inference.py`) are **release-ready as
a v1 gated non-commercial research artifact**. The ONLY remaining pre-push item is
a real responsible-disclosure contact address (JK to provide). The v2 eval
additions (obfuscation / adversarial reconstruction) are documented as untested in
the card; multi-turn naive-split robustness is already reported.

# bioguard-v8b: training reproducibility

Enough to reproduce the model from the same public sources. The harmful (positive)
training examples themselves are **not shipped** (see RELEASE_CHECKLIST withhold
list); this document describes the pipeline, sources, and hyperparameters.

## Data (3,507 examples, reuse-only, zero generated harmful content)

| Source | n | License | Role |
|--------|---|---------|------|
| WildGuardMix (bio responses) | 1,350 | ODC-BY-1.0 | real harmful/benign responses |
| BeaverTails (bio) | 1,024 | CC-BY-NC-4.0 | real responses, response-harm labels |
| FalseReject (benign hard-negatives) | 891 | CC-BY-NC-4.0 | reduce over-refusal |
| non-bio control | 242 | mixed | selectivity control |

- Class balance: 1,163 positive (33%) / 2,344 negative.
- Label scheme: binary response-harm. label=1 iff the RESPONSE is harmful
  (WildGuardMix `response_harm == harmful`; BeaverTails `is_safe == False`).
- Input format: `query [SEP] response`, truncated to 512 tokens.
- Build: `scripts/build_v8_data.py` (writes `data/processed/v8b_{train,val}.jsonl`
  and `v8b_class_weights.json`). Leakage audit: training queries are byte-disjoint
  from every evaluation set.

## Model and hyperparameters

- Base: `microsoft/deberta-v3-base` (12 layers, hidden 768, ~184M params).
- Epochs 5; per-device batch 16 x grad-accum 2 (effective 32); learning rate 2e-5,
  cosine scheduler, warmup ratio 0.1, weight decay 0.01; optimizer adamw_torch.
- Class weighting on (auto SAFE = total/(2*neg); UNSAFE manual); `class_weights: true`.
- Seed 42; early stopping patience 2; best checkpoint by validation F1.
- Train: `scripts/train_v8_bioguard.py --data-prefix v8b --output-name deberta_bioguard_v8b`.

## Shipped preprocessing (part of the model)

`constitutional_bioguard/preprocessing.py::normalize_text` runs before
tokenization at both train and inference time: strips invisible/zero-width/tag/
variation-selector characters, folds homoglyphs (Cyrillic), removes combining
marks, decodes URL/base64/hex/ROT13, applies NFKC. This is the hardened
char-injection defense (see RESULTS.md robustness).

## Notes

- v8c (a v8b variant + PubMedQA benign-bio) was trained and **retired**: on the
  real over-refusal money metric it tied/lost to v8b, so the recall sacrifice
  bought nothing. v8b is the shipped model.
- A commercial (Apache-2.0) variant would require dropping the two CC-BY-NC sources
  (BeaverTails, FalseReject) and retraining; deferred by decision.

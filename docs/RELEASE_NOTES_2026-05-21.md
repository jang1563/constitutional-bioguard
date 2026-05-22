# Constitutional BioGuard — Release Notes (May 21, 2026)

## Release overview

This release updates the local reporting surface and HPC artifact synchronization after a successful full training run on the latest dataset and environment.

- Pipeline run ID: `2959541` (completed successfully)
- Commit range covered:
  - `46f44c0` — README metric refresh and release snapshot
  - `0104a8b` — Sync HPC evaluation artifacts and run metadata
  - `07d9ec5` — Preprocessing + pipeline flow hardening

## Validation summary (latest run)

- Internal held-out metrics:
  - F1: `0.980676`
  - AUROC: `0.997961`
  - FPR: `0.008969` (0.8969%)
- Calibration:
  - Optimal threshold: `0.10`
  - Best score: `0.9852`
  - Validation samples: `697`
- Adversarial robustness:
  - 20 attacks, mean ASR: `0.00%`
- Over-refusal:
  - Benign holdout FPR: `0.00%` (`0 / 100`)
- External cross-check:
  - Primary strategy (`threat_level>=4`) Cohen kappa: `0.41398`
  - F1: `0.5143`
  - Accuracy: `0.7867`

## What changed

### 1) Documentation and release numbers updated
- Updated `README.md` TL;DR and Results section to match current run outputs
- Added a “Latest Run Snapshot (2026-05-21)” section with single-source key metrics

### 2) Artifact synchronization to local workspace
- Synced latest artifacts after full HPC run:
  - `results/metrics/classification_report.txt`
  - `results/metrics/internal_evaluation.json` (contains run metrics)
  - `results/metrics/overrefusal_results.json`
  - `results/metrics/adversarial_results.json`
  - `results/metrics/calibration`-related outputs (`calibration.json`, training metadata)
  - `results/figures/*.png`
  - `pipeline.log`, `data/processed/class_weights.json`, `data/processed/data_summary.json`

### 3) Preprocessing and pipeline reliability
- Added stricter ROT13 decode heuristic in `constitutional_bioguard/preprocessing.py`:
  - Accept decoded text only when English marker score improves, reducing false conversions
- Added dependency:
  - `sentencepiece>=0.2` in `pyproject.toml`
- Fixed full pipeline step ordering in `scripts/run_full_pipeline.sh`:
  - Added explicit training step
  - Aligned labels to the current 7-step flow (`generate`, `augment`, `benign`, `prepare`, `train`, `calibrate`, `evaluate`)

## Notes

- The external validation kappa remains constrained by the query-level vs response-level label mismatch and should be described as a systems-level architectural difference rather than a classifier collapse.
- The full commit history and raw run outputs are retained in git and the local `results/` path for reproducibility.

## Recommended next steps

1. Tag release (for example: `v0.1.1`) after final review.
2. Publish release notes and attach updated `README` metrics.
3. Keep adversarial attack set and `overrefusal` metrics as non-regression gates for future runs.

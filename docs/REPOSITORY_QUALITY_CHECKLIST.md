# Repository Quality Checklist

Use this checklist before treating Constitutional BioGuard as release-ready or
sharing a private preview.

## Required Local Checks

Run these from the repository root:

```bash
python3 -m pytest tests/ -q
python3 scripts/validate_constitution.py
python3 -m ruff check .
```

Expected baseline:

- Unit tests pass without API keys, GPUs, or network access.
- Constitution validation reports 56 rules across the 7 NSABB categories.
- Ruff reports no errors under the repository policy in `pyproject.toml`.

## Release Surface

- `README.md` states that BioGuard is a research prototype, not a production
  safeguard.
- `README.md`, `SAFETY.md`, `CITATION.cff`, the Hugging Face model card, and
  release notes agree on the recommended checkpoint and access mode.
- The v4 checkpoint is described as a private Hugging Face preview unless and
  until model visibility is intentionally changed.
- v5 PairCFR remains documented as a non-release result unless a future run
  passes the specialist bio recall gate.

## Safety and Data Hygiene

- No `.env`, API token, model checkpoint, generated unsafe corpus, or local
  pipeline log is tracked by Git.
- `data/metrics/` contains aggregate metrics and compact prediction metadata,
  not full unsafe synthetic examples.
- Demo code uses benign examples or sanitized placeholders rather than
  operational biological instructions.
- New benchmarks or generated datasets should be added only as scripts,
  aggregate metrics, or access-controlled external assets.

## GitHub/Hugging Face Consistency

- GitHub `main` and the Hugging Face model card point to the same release
  story.
- Hugging Face visibility is checked after upload:

```bash
python3 -c "from huggingface_hub import HfApi; info=HfApi().model_info('jang1563/constitutional-bioguard-v4'); print(info.private, [s.rfilename for s in info.siblings])"
```

- Expected model files: `README.md`, `config.json`, `model.safetensors`,
  tokenizer files, `training_args.bin`, `training_metrics.json`, and
  `v4_provenance.json`.

## Before Public Release

- Replace "private preview" wording with the intended access policy.
- Re-run the full checklist above after any model-card or README edits.
- Confirm that any new metrics have no prompt/example leakage before committing.

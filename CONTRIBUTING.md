# Contributing to Constitutional BioGuard

Thank you for your interest in contributing. This document outlines how to set up a development environment and submit changes.

## Development Setup

```bash
git clone https://github.com/jang1563/constitutional-bioguard
cd constitutional-bioguard
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

## Project Structure

- `constitution/` — The biosafety constitution (YAML rules). Changes here affect all downstream data generation.
- `constitutional_bioguard/generation/` — Synthetic data pipeline (requires Anthropic API key).
- `constitutional_bioguard/training/` — DeBERTa fine-tuning (requires GPU).
- `constitutional_bioguard/evaluation/` — Evaluation suites. Can run on CPU.
- `tests/` — Unit tests. Should run without GPU or API key.

## Submitting Changes

1. Fork the repository and create a feature branch.
2. Make your changes with tests where appropriate.
3. Ensure `pytest tests/ -v` passes and `ruff check .` reports no errors.
4. Open a pull request with a clear description of what changed and why.

## Important Notes

**Constitution changes**: Modifying `constitution/biosafety_constitution.yaml` is the highest-leverage change in this project. Please justify any rule additions or modifications with clear reasoning about the permitted/restricted boundary.

**Data generation**: New synthetic data generation requires the Anthropic API and will incur costs. Do not commit generated data files (covered by `.gitignore`).

**Model weights**: Do not commit model checkpoints. They are covered by `.gitignore` and should be distributed via HuggingFace Hub.

## Responsible Use

This project is intended for biosafety research and AI safety applications. Please do not use it to circumvent biological safety regulations or assist in developing dangerous biological agents.

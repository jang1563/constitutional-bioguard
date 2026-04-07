.PHONY: validate generate augment benign prepare train evaluate external adversarial overrefusal figures all clean test

PYTHON := python

validate:
	$(PYTHON) scripts/validate_constitution.py

generate:
	$(PYTHON) scripts/run_pipeline.py --step generate-synthetic

augment:
	$(PYTHON) scripts/run_pipeline.py --step augment

benign:
	$(PYTHON) scripts/run_pipeline.py --step generate-benign

prepare:
	$(PYTHON) scripts/run_pipeline.py --step prepare-data

train:
	$(PYTHON) scripts/run_pipeline.py --step train --model deberta

evaluate:
	$(PYTHON) scripts/run_pipeline.py --step evaluate

external:
	$(PYTHON) scripts/run_pipeline.py --step external-validate

adversarial:
	$(PYTHON) scripts/run_pipeline.py --step adversarial

overrefusal:
	$(PYTHON) scripts/run_pipeline.py --step overrefusal

figures:
	$(PYTHON) scripts/run_pipeline.py --step figures

all: validate generate augment benign prepare train evaluate external adversarial overrefusal figures

test:
	pytest tests/ -v

clean:
	rm -rf data/raw/*.jsonl data/augmented/*.jsonl data/processed/*.jsonl
	rm -rf results/figures/*.png results/metrics/*.json results/reports/*.md
	rm -rf models/deberta_bioguard_v1/

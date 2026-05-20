"""Tests for benign holdout handling in over-refusal evaluation."""

import json

from constitutional_bioguard.evaluation.overrefusal_test import load_benign_examples
from constitutional_bioguard.models import (
    FineLabel,
    NSABBCategory,
    SafetyLabel,
    SyntheticExample,
)
from constitutional_bioguard.training.prepare_data import split_benign_holdout


def _make_example(example_id: str, fine_label: FineLabel) -> SyntheticExample:
    return SyntheticExample(
        example_id=example_id,
        rule_id="BENIGN" if fine_label == FineLabel.BENIGN else "EH-001",
        category=NSABBCategory.ENHANCE_HARM,
        query=f"Query for {example_id}",
        response=f"Response for {example_id}",
        fine_label=fine_label,
        safety_label=(
            SafetyLabel.SAFE if fine_label != FineLabel.RESTRICTED else SafetyLabel.UNSAFE
        ),
    )


def test_split_benign_holdout_reserves_only_benign_examples():
    examples = [
        _make_example(f"benign_{i}", FineLabel.BENIGN)
        for i in range(5)
    ] + [
        _make_example("restricted_0", FineLabel.RESTRICTED),
        _make_example("permitted_0", FineLabel.PERMITTED),
    ]

    trainable, holdout = split_benign_holdout(examples, holdout_ratio=0.4, seed=1)

    assert len(holdout) == 2
    assert all(ex.fine_label == FineLabel.BENIGN for ex in holdout)
    assert len(trainable) == 5
    assert any(ex.fine_label == FineLabel.RESTRICTED for ex in trainable)
    assert any(ex.fine_label == FineLabel.PERMITTED for ex in trainable)


def test_load_benign_examples_prefers_dedicated_holdout(tmp_path):
    holdout_file = tmp_path / "overrefusal_holdout.jsonl"
    record = {"text": "safe query [SEP] safe response", "query": "safe query", "response": "safe response", "label": 0}
    with open(holdout_file, "w") as f:
        f.write(json.dumps(record) + "\n")

    test_file = tmp_path / "test.jsonl"
    with open(test_file, "w") as f:
        f.write(json.dumps({"text": "legacy test benign", "fine_label": "benign"}) + "\n")

    benign_file = tmp_path / "benign_queries.jsonl"
    with open(benign_file, "w") as f:
        f.write(json.dumps({"query": "legacy raw", "response": "legacy response"}) + "\n")

    examples = load_benign_examples(
        test_file=test_file,
        benign_file=benign_file,
        holdout_file=holdout_file,
    )

    assert examples == [
        {
            "query": "safe query",
            "response": "safe response",
            "text": "safe query [SEP] safe response",
            "source": "overrefusal_holdout",
        }
    ]

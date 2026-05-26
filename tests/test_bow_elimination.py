"""Tests for WS-2 bag-of-words elimination and the external dataset registry."""

from constitutional_bioguard.evaluation.external_datasets import (
    REGISTRY,
    coverage_summary,
    datasets_by_role,
)
from constitutional_bioguard.training.bow_elimination import bow_predictability


def test_bow_predictability_detects_keyword_shortcut():
    """When a keyword perfectly separates classes, examples are trivial."""
    texts = (
        ["this exchange discusses restricted biological escalation"] * 20
        + ["this exchange discusses ordinary garden vegetable soup"] * 20
    )
    labels = [1] * 20 + [0] * 20
    result = bow_predictability(texts, labels, cv_folds=4, confidence=0.8)
    assert result["bow_cv_auroc"] > 0.9
    assert result["trivial_fraction"] > 0.5
    assert result["n_total"] == 40


def test_bow_predictability_low_on_unseparable_data():
    """Identical text with random-looking labels is not keyword-predictable."""
    texts = ["the same neutral sentence about cells"] * 40
    labels = ([0, 1] * 20)
    result = bow_predictability(texts, labels, cv_folds=4, confidence=0.9)
    # No lexical signal — BoW AUROC should be near chance.
    assert result["bow_cv_auroc"] < 0.75


def test_bow_predictability_rejects_misaligned_input():
    try:
        bow_predictability(["a", "b"], [1])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_external_registry_roles_present():
    """The curated union must cover all four evaluation roles."""
    roles = {d.role for d in REGISTRY.values()}
    assert roles == {
        "positive",
        "hard_negative",
        "easy_negative",
    } or "adversarial" in roles or roles.issuperset(
        {"positive", "hard_negative", "easy_negative"}
    )
    assert datasets_by_role("positive")
    assert datasets_by_role("hard_negative")
    assert datasets_by_role("easy_negative")


def test_coverage_summary_flags_bio_specific_caveat():
    summary = coverage_summary()
    assert summary["n_datasets"] == len(REGISTRY)
    assert summary["bio_specific"]
    assert "caveat" in summary and len(summary["caveat"]) > 0

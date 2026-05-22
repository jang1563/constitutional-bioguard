"""Tests for WS-1 escalation calibration."""

from constitutional_bioguard.evaluation.escalation import (
    au_prc,
    curve_recall_span,
    escalation_rate,
    escalation_rate_table,
    find_escalation_threshold,
    temperature_scale_nll,
)


def test_escalation_rate_base_rate_dependence():
    """Flag rate must fall as the unsafe prior falls."""
    high = escalation_rate(recall=0.98, fpr=0.03, base_rate=0.5)
    low = escalation_rate(recall=0.98, fpr=0.03, base_rate=0.01)
    assert high > low
    # At base rate 0, flag rate is exactly the FPR.
    assert abs(escalation_rate(0.98, 0.03, 0.0) - 0.03) < 1e-9
    # At base rate 1, flag rate is exactly the recall.
    assert abs(escalation_rate(0.98, 0.03, 1.0) - 0.98) < 1e-9


def test_escalation_rate_table_keys():
    table = escalation_rate_table(0.98, 0.03)
    assert all(k.startswith("base_rate_") for k in table)
    assert len(table) >= 3


def test_find_escalation_threshold_picks_highest_qualifying():
    """Among thresholds meeting the recall target, the highest is chosen."""
    curve = [
        {"threshold": 0.10, "recall": 0.99, "fpr": 0.30, "precision": 0.8},
        {"threshold": 0.50, "recall": 0.985, "fpr": 0.05, "precision": 0.95},
        {"threshold": 0.90, "recall": 0.97, "fpr": 0.01, "precision": 0.99},
    ]
    result = find_escalation_threshold(curve, target_recall=0.98)
    assert result["achievable"]
    # 0.97 < 0.98, so only the first two qualify; highest threshold is 0.50.
    assert result["operating_point"]["threshold"] == 0.50


def test_find_escalation_threshold_unachievable():
    curve = [{"threshold": 0.5, "recall": 0.5, "fpr": 0.1, "precision": 0.9}]
    result = find_escalation_threshold(curve, target_recall=0.98)
    assert not result["achievable"]
    assert not result["gate_passed"]


def test_gate_uses_realistic_base_rate():
    """The gate must evaluate escalation at the low production base rate."""
    curve = [{"threshold": 0.5, "recall": 0.98, "fpr": 0.03, "precision": 0.95}]
    result = find_escalation_threshold(
        curve, target_recall=0.98, max_escalation_rate=0.15, gate_base_rate=0.01
    )
    # Escalation at base rate 0.01 ~= 0.04, comfortably under 0.15.
    assert result["gate_passed"]
    assert result["escalation_rate_at_gate_base_rate"] < 0.15


def test_au_prc_returns_none_for_narrow_sweep():
    """A sweep that barely spans recall must not yield a fabricated AU-PRC."""
    narrow = [
        {"recall": 0.980, "precision": 0.98},
        {"recall": 0.985, "precision": 0.98},
    ]
    assert au_prc(narrow) is None
    assert curve_recall_span(narrow) < 0.5


def test_au_prc_computes_for_full_sweep():
    full = [
        {"recall": 0.0, "precision": 1.0},
        {"recall": 0.5, "precision": 0.9},
        {"recall": 1.0, "precision": 0.8},
    ]
    area = au_prc(full)
    assert area is not None
    assert 0.0 < area <= 1.0


def test_temperature_scaling_reduces_nll_on_overconfident_scores():
    """Overconfident logits should fit a temperature > 1."""
    # Labels alternate; logits are large-magnitude (overconfident).
    logits = [6.0, -6.0] * 20
    labels = [1, 0] * 20
    result = temperature_scale_nll(logits, labels)
    assert result["temperature"] > 0
    assert result["nll_after"] <= result["nll_before"]

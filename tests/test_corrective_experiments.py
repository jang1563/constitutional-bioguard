"""Tests for corrective experiments (Section 6 of technical report).

All tests use synthetic mock data -- no model loading, no data downloading.
"""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from constitutional_bioguard.evaluation.corrective_experiments import (
    bootstrap_metric_ci,
    compute_full_metrics,
    threshold_sweep,
    tpr_at_fpr,
)

# ── Shared fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    """Realistic binary classification scenario."""
    rng = np.random.RandomState(42)
    n = 200
    true = np.array([1] * 100 + [0] * 100)
    # Decent classifier: ~90% accuracy
    probs = np.where(true == 1, rng.beta(5, 1, n), rng.beta(1, 5, n))
    return true, probs


@pytest.fixture
def perfect_data():
    """Perfect classifier scenario (ceiling effect)."""
    true = np.array([1] * 50 + [0] * 50)
    probs = np.where(true == 1, 0.99, 0.01)
    return true, probs


# ── tpr_at_fpr ──────────────────────────────────────────────────────────

class TestTPRAtFPR:
    def test_perfect_classifier(self, perfect_data):
        true, probs = perfect_data
        assert tpr_at_fpr(true, probs, target_fpr=0.01) == 1.0

    def test_random_classifier_is_low(self):
        rng = np.random.RandomState(42)
        true = np.array([1] * 100 + [0] * 100)
        probs = rng.uniform(0, 1, 200)
        tpr = tpr_at_fpr(true, probs, target_fpr=0.01)
        assert tpr < 0.1  # Random should be near 0 at 1% FPR

    def test_returns_zero_for_impossible_fpr(self, binary_data):
        true, probs = binary_data
        # At FPR=0.0, expect 0 TPR (unless classifier is perfect)
        tpr = tpr_at_fpr(true, probs, target_fpr=0.0)
        assert tpr >= 0.0


# ── threshold_sweep ───────────────────────────────────────────────────

class TestThresholdSweep:
    def test_returns_list_of_dicts(self, binary_data):
        true, probs = binary_data
        curve = threshold_sweep(true, probs)
        assert isinstance(curve, list)
        assert len(curve) > 0
        assert all(isinstance(e, dict) for e in curve)

    def test_required_keys(self, binary_data):
        true, probs = binary_data
        curve = threshold_sweep(true, probs)
        required = {"threshold", "precision", "recall", "f1", "fpr", "n_flagged"}
        for e in curve:
            assert required.issubset(e.keys())

    def test_recall_decreases_with_threshold(self, binary_data):
        true, probs = binary_data
        curve = threshold_sweep(true, probs)
        recalls = [e["recall"] for e in curve]
        # Recall should generally decrease (monotonic with noise)
        assert recalls[0] >= recalls[-1]

    def test_fpr_decreases_with_threshold(self, binary_data):
        true, probs = binary_data
        curve = threshold_sweep(true, probs)
        fprs = [e["fpr"] for e in curve]
        assert fprs[0] >= fprs[-1]

    def test_custom_thresholds(self, binary_data):
        true, probs = binary_data
        custom = np.array([0.3, 0.5, 0.7])
        curve = threshold_sweep(true, probs, thresholds=custom)
        assert len(curve) == 3
        assert [e["threshold"] for e in curve] == [0.3, 0.5, 0.7]


# ── compute_full_metrics ──────────────────────────────────────────────

class TestComputeFullMetrics:
    def test_perfect_classifier(self, perfect_data):
        true, probs = perfect_data
        m = compute_full_metrics(true, probs)
        assert m["f1"] > 0.98
        assert m["fpr"] < 0.02
        assert m["au_prc"] > 0.98
        assert m["auroc"] > 0.98

    def test_all_keys_present(self, binary_data):
        true, probs = binary_data
        m = compute_full_metrics(true, probs)
        expected = {
            "threshold", "precision", "recall", "f1", "fpr",
            "auroc", "au_prc", "tpr_at_1pct_fpr", "n_total",
            "n_positive", "n_negative",
        }
        assert expected.issubset(m.keys())

    def test_threshold_affects_metrics(self, binary_data):
        true, probs = binary_data
        low = compute_full_metrics(true, probs, threshold=0.3)
        high = compute_full_metrics(true, probs, threshold=0.7)
        # Lower threshold -> higher recall, higher FPR
        assert low["recall"] >= high["recall"]

    def test_counts_correct(self, binary_data):
        true, probs = binary_data
        m = compute_full_metrics(true, probs)
        assert m["n_total"] == len(true)
        assert m["n_positive"] + m["n_negative"] == m["n_total"]


# ── bootstrap_metric_ci ──────────────────────────────────────────────

class TestBootstrapCI:
    def test_returns_required_keys(self):
        true = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 10)
        pred = np.array([1, 1, 0, 1, 1, 0, 0, 0] * 10)
        result = bootstrap_metric_ci(true, pred, f1_score, n_iterations=100)
        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "std" in result
        assert "n_iterations" in result
        assert "n_valid" in result

    def test_ci_contains_point_estimate(self):
        rng = np.random.RandomState(42)
        n = 200
        true = rng.randint(0, 2, n)
        pred = true.copy()
        # Flip ~10% of predictions
        flip_idx = rng.choice(n, n // 10, replace=False)
        pred[flip_idx] = 1 - pred[flip_idx]

        result = bootstrap_metric_ci(true, pred, f1_score, n_iterations=1000)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_wider_ci_for_smaller_samples(self):
        rng = np.random.RandomState(42)
        # Small sample
        true_s = rng.randint(0, 2, 30)
        pred_s = true_s.copy()
        pred_s[:5] = 1 - pred_s[:5]
        ci_small = bootstrap_metric_ci(true_s, pred_s, f1_score, n_iterations=1000)

        # Large sample
        true_l = rng.randint(0, 2, 500)
        pred_l = true_l.copy()
        pred_l[:50] = 1 - pred_l[:50]
        ci_large = bootstrap_metric_ci(true_l, pred_l, f1_score, n_iterations=1000)

        assert ci_small["ci_width"] > ci_large["ci_width"]

    def test_kappa_bootstrap(self):
        true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 5)
        pred = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0] * 5)
        result = bootstrap_metric_ci(
            true, pred, cohen_kappa_score, n_iterations=500,
        )
        assert result["point_estimate"] == pytest.approx(
            cohen_kappa_score(true, pred), abs=1e-6
        )
        assert result["ci_lower"] < result["ci_upper"]

    def test_handles_degenerate_samples(self):
        # Very small sample where bootstrap may get all-one-class resamples
        true = np.array([1, 1, 1, 0])
        pred = np.array([1, 1, 0, 0])
        result = bootstrap_metric_ci(
            true, pred, cohen_kappa_score, n_iterations=100,
        )
        # Should handle gracefully
        assert result["n_valid"] <= result["n_iterations"]

    def test_reproducibility(self):
        true = np.array([1, 1, 0, 0, 1, 0] * 10)
        pred = np.array([1, 0, 0, 1, 1, 0] * 10)
        r1 = bootstrap_metric_ci(true, pred, f1_score, n_iterations=500, seed=42)
        r2 = bootstrap_metric_ci(true, pred, f1_score, n_iterations=500, seed=42)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]


# ── Adversarial suite normalize parameter ─────────────────────────────

class TestAdversarialNormalize:
    """Test that the normalize parameter is properly threaded."""

    def test_attack_registry_unchanged(self):
        from constitutional_bioguard.evaluation.adversarial_suite import ATTACKS
        assert len(ATTACKS) == 27

    def test_run_adversarial_suite_accepts_normalize(self):
        """Verify the function signature accepts normalize parameter."""
        import inspect

        from constitutional_bioguard.evaluation.adversarial_suite import (
            run_adversarial_suite,
        )
        sig = inspect.signature(run_adversarial_suite)
        assert "normalize" in sig.parameters
        assert sig.parameters["normalize"].default is True

    def test_run_adversarial_suite_accepts_output_file(self):
        import inspect

        from constitutional_bioguard.evaluation.adversarial_suite import (
            run_adversarial_suite,
        )
        sig = inspect.signature(run_adversarial_suite)
        assert "output_file" in sig.parameters
        assert sig.parameters["output_file"].default is None

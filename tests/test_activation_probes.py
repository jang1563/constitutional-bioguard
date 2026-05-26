"""Unit tests for WS-3 activation probes (CPU-only, no LLM loading)."""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from constitutional_bioguard.evaluation.activation_probes import (
    ProbeResult,
    evaluate_ensemble,
    format_exchange,
    get_target_layer,
    sweep_ensemble_weights,
    tpr_at_fpr,
    train_probe,
)


class TestTargetLayer:
    def test_llama_default(self):
        layer = get_target_layer("llama-3.1-8b")
        assert layer == 12  # 32 * 0.4 = 12

    def test_gemma_default(self):
        layer = get_target_layer("gemma-2-9b")
        assert layer == 16  # 42 * 0.4 = 16

    def test_custom_fraction(self):
        layer = get_target_layer("llama-3.1-8b", fraction=0.75)
        assert layer == 24  # 32 * 0.75

    def test_clamps_to_max(self):
        layer = get_target_layer("llama-3.1-8b", fraction=1.0)
        assert layer == 31  # n_layers - 1


class TestFormatExchange:
    def test_basic(self):
        result = format_exchange("What is CRISPR?", "CRISPR is a gene editing tool.")
        assert "User: What is CRISPR?" in result
        assert "Assistant: CRISPR is a gene editing tool." in result


class TestTrainProbe:
    def test_fits_separable_data(self):
        rng = np.random.RandomState(42)
        n = 200
        d = 64
        X = np.vstack([rng.randn(n // 2, d) - 1, rng.randn(n // 2, d) + 1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        probe = train_probe(X, y)
        assert isinstance(probe, LogisticRegressionCV)
        acc = (probe.predict(X) == y).mean()
        assert acc > 0.9


class TestTPRatFPR:
    def test_perfect_separation(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        tpr = tpr_at_fpr(y, scores, target_fpr=0.01)
        assert tpr > 0.5

    def test_random_scores(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 100)
        scores = rng.rand(100)
        tpr = tpr_at_fpr(y, scores, target_fpr=0.01)
        assert 0.0 <= tpr <= 1.0


class TestEvaluateEnsemble:
    def test_complementary_errors(self):
        rng = np.random.RandomState(42)
        n = 200
        labels = rng.randint(0, 2, n)
        # Probe is right where bioguard is wrong, and vice versa
        probe_probs = labels.astype(float) + rng.normal(0, 0.1, n)
        probe_probs = np.clip(probe_probs, 0, 1)
        # Flip some for bioguard
        bioguard_probs = labels.astype(float) + rng.normal(0, 0.1, n)
        bioguard_probs = np.clip(bioguard_probs, 0, 1)

        result, comp = evaluate_ensemble(
            probe_probs, bioguard_probs, labels,
            model_name="test", target_layer=12, weight_probe=0.5,
        )
        assert isinstance(result, ProbeResult)
        assert result.probe_type == "ensemble"
        assert "spearman_error_correlation" in comp


class TestWeightSweep:
    def test_returns_11_points(self):
        rng = np.random.RandomState(42)
        n = 100
        labels = rng.randint(0, 2, n)
        probe = np.clip(labels + rng.normal(0, 0.2, n), 0, 1)
        bg = np.clip(labels + rng.normal(0, 0.2, n), 0, 1)
        sweep = sweep_ensemble_weights(probe, bg, labels, "test", 12)
        assert len(sweep) == 11
        assert all("weight_probe" in s and "au_prc" in s for s in sweep)


class TestProbeResult:
    def test_to_dict(self):
        r = ProbeResult(
            au_prc=0.85, auroc=0.90, f1=0.82,
            tpr_at_1pct_fpr=0.45, n_total=100,
            probe_type="mean", model_name="test", target_layer=12,
        )
        d = r.to_dict()
        assert d["au_prc"] == 0.85
        assert d["probe_type"] == "mean"

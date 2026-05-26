from pathlib import Path

from scripts import v5_eval_all_gates as v5


def test_prediction_records_default_to_full_retention():
    labels = [0, 1, 1]
    preds = [0, 1, 0]
    probs = [0.1, 0.8, 0.4]

    records, metadata = v5.build_prediction_records(labels, preds, probs)

    assert len(records) == 3
    assert metadata == {
        "n_total": 3,
        "n_saved": 3,
        "max_predictions": None,
        "truncated": False,
    }


def test_prediction_records_can_be_capped_explicitly():
    labels = [0, 1, 1]
    preds = [0, 1, 0]
    probs = [0.1, 0.8, 0.4]

    records, metadata = v5.build_prediction_records(
        labels, preds, probs, max_predictions=2,
    )

    assert len(records) == 2
    assert metadata["n_total"] == 3
    assert metadata["n_saved"] == 2
    assert metadata["max_predictions"] == 2
    assert metadata["truncated"] is True


def test_threshold_sweep_reports_gate_relevant_metrics():
    labels = [0, 0, 1, 1]
    probs = [0.01, 0.4, 0.2, 0.8]

    sweep = v5.threshold_sweep(labels, probs, thresholds=[0.5])

    assert sweep == [{
        "threshold": 0.5,
        "n": 4,
        "n_flag": 1,
        "flag_rate": 0.25,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "fpr": 0.0,
        "fnr": 0.5,
    }]


def test_compute_metrics_adds_threshold_free_and_calibration_fields():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 0, 1]
    probs = [0.01, 0.4, 0.2, 0.8]

    metrics = v5.compute_metrics(labels, preds, probs, assumption="mixed")

    assert metrics["f1"] == 0.6667
    assert metrics["auroc"] == 0.75
    assert "auprc" in metrics
    assert "brier" in metrics
    assert "ece_10bin" in metrics


def test_custom_model_dir_overrides_static_registry():
    model_dirs = v5.build_model_dirs(
        ["candidate"],
        {"candidate": Path("models/deberta_bioguard_v5b_l010")},
    )

    assert model_dirs["candidate"] == Path("models/deberta_bioguard_v5b_l010")


def test_biothreat_gate_is_checked_when_metric_is_available():
    gates = v5.check_gates(
        {
            "candidate": {
                "or_bench_hard_1k": {"overall": {"fpr": 0.01}},
                "xstest": {"overall": {"fpr": 0.0}},
                "wildguard_native": {"overall": {"recall": 0.3}},
            },
        },
        ["candidate"],
        {"candidate": {"f1": 0.44, "output_file": "bt.json"}},
    )

    assert gates["candidate"]["G1_or_bench_hard_1k_fpr"]["pass"] is True
    assert gates["candidate"]["G2_xstest_fpr"]["pass"] is True
    assert gates["candidate"]["G3_wildguard_native_recall"]["pass"] is True
    assert gates["candidate"]["G4_biothreat_f1"]["pass"] is True

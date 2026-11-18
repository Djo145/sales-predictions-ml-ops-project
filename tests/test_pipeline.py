"""Tests de l'ordre du pipeline et de son arrêt en cas de données invalides."""

from types import SimpleNamespace

import numpy as np
import pytest

from retail_forecast import pipeline
from retail_forecast.validation import DataValidationError


def make_config() -> dict:
    return {
        "preprocessing": {
            "top_n": 10,
            "selection_end": "2011-08-31",
            "lookback": 100,
        },
        "models": {
            "lstm": {"hidden_size": 16, "embedding_size": 4},
            "transformer": {
                "d_model": 16,
                "nhead": 2,
                "embedding_size": 4,
            },
        },
    }


def test_pipeline_runs_steps_in_order_and_evaluates_test_only(
    monkeypatch,
    tmp_path,
):
    events = []

    fake_run = SimpleNamespace(info=SimpleNamespace(run_id="test-run"))

    monkeypatch.setattr(
        pipeline,
        "start_tracking_run",
        lambda config: events.append("tracking-start") or fake_run,
    )
    monkeypatch.setattr(
        pipeline,
        "log_successful_run",
        lambda **kwargs: events.append("tracking-log") or {"LSTM": {}, "Transformer": {}},
    )
    monkeypatch.setattr(
        pipeline,
        "finish_tracking_run",
        lambda: events.append("tracking-finish"),
    )
    monkeypatch.setattr(
        pipeline,
        "fail_tracking_run",
        lambda error: pytest.fail("Le pipeline nominal ne doit pas échouer."),
    )
    raw_data = object()
    daily_sales = object()

    X_sales = np.arange(4 * 100, dtype=np.float32).reshape(4, 100, 1)
    product_codes = np.array(["A", "A", "B", "B"])
    y_values = np.array([10, 11, 12, 13], dtype=np.float32)
    target_dates = np.array(
        ["2011-08-30", "2011-08-31", "2011-09-01", "2011-09-02"],
        dtype="datetime64[ns]",
    )

    training = {
        "models": {"LSTM": object(), "Transformer": object()},
        "product_to_index": {"A": 0, "B": 1},
        "train_indices": np.array([0, 1]),
        "test_indices": np.array([2, 3]),
        "product_ids": np.array([0, 0, 1, 1]),
        "loss_histories": {"LSTM": [], "Transformer": []},
    }

    monkeypatch.setattr(
        pipeline,
        "load_current_data",
        lambda path: events.append("ingestion") or raw_data,
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_daily_gross_sales",
        lambda data, **kwargs: events.append("preparation") or daily_sales,
    )
    monkeypatch.setattr(
        pipeline,
        "build_sequences",
        lambda data, **kwargs: events.append("sequences")
        or (X_sales, product_codes, y_values, target_dates),
    )
    monkeypatch.setattr(
        pipeline,
        "train_models",
        lambda *args: events.append("training") or training,
    )

    def fake_evaluate(**kwargs):
        events.append("evaluation")
        np.testing.assert_array_equal(kwargs["X_sales"], X_sales[[2, 3]])
        np.testing.assert_array_equal(kwargs["actual"], y_values[[2, 3]])
        np.testing.assert_array_equal(
            kwargs["product_ids"],
            training["product_ids"][[2, 3]],
        )
        return {"overall": "fake metrics"}

    monkeypatch.setattr(pipeline, "evaluate_models", fake_evaluate)

    def fake_save(model, artifact_dir, **kwargs):
        events.append(f"save-{kwargs['model_type']}")
        return artifact_dir

    monkeypatch.setattr(pipeline, "save_model_artifact", fake_save)

    def fake_save_metadata(output_dir, **kwargs):
        events.append("metadata")
        return {"manifest": output_dir / "run_manifest.json"}

    monkeypatch.setattr(pipeline, "save_run_metadata", fake_save_metadata)

    result = pipeline.run_training_pipeline(
        config=make_config(),
        data_path=tmp_path / "data.xlsx",
        artifact_dir=tmp_path / "output",
    )

    assert events == [
        "tracking-start",
        "ingestion",
        "preparation",
        "sequences",
        "training",
        "evaluation",
        "save-LSTM",
        "save-Transformer",
        "metadata",
        "tracking-log",
        "tracking-finish",
    ]
    assert result["train_count"] == 2
    assert result["test_count"] == 2
    assert result["mlflow_run_id"] == "test-run"


def test_invalid_data_stops_before_training_evaluation_and_saving(
    monkeypatch,
    tmp_path,
):
    fake_run = SimpleNamespace(info=SimpleNamespace(run_id="failed-test-run"))
    tracking_failures = []

    monkeypatch.setattr(
        pipeline,
        "start_tracking_run",
        lambda config: fake_run,
    )
    monkeypatch.setattr(
        pipeline,
        "fail_tracking_run",
        lambda error: tracking_failures.append(error),
    )
    monkeypatch.setattr(
        pipeline,
        "log_successful_run",
        lambda **kwargs: pytest.fail(
            "Un pipeline invalide ne doit pas publier de modèles."
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "finish_tracking_run",
        lambda: pytest.fail("Un pipeline invalide ne doit pas finir en succès."),
    )

    def reject_data(_path):
        raise DataValidationError("données invalides")

    monkeypatch.setattr(pipeline, "load_current_data", reject_data)
    monkeypatch.setattr(
        pipeline,
        "prepare_daily_gross_sales",
        lambda *args, **kwargs: pytest.fail("La préparation ne doit pas être appelée."),
    )
    monkeypatch.setattr(
        pipeline,
        "build_sequences",
        lambda *args, **kwargs: pytest.fail("La création des séquences ne doit pas être appelée."),
    )
    monkeypatch.setattr(
        pipeline,
        "train_models",
        lambda *args, **kwargs: pytest.fail("L'entraînement ne doit pas être appelé."),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_models",
        lambda *args, **kwargs: pytest.fail("L'évaluation ne doit pas être appelée."),
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_artifact",
        lambda *args, **kwargs: pytest.fail("Aucun modèle ne doit être sauvegardé."),
    )

    with pytest.raises(DataValidationError, match="données invalides"):
        pipeline.run_training_pipeline(
            config=make_config(),
            data_path=tmp_path / "invalid.xlsx",
            artifact_dir=tmp_path / "output",
        )

    assert len(tracking_failures) == 1
    assert isinstance(tracking_failures[0], DataValidationError)

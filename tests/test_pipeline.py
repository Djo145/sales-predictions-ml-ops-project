"""Tests de l'ordre du pipeline et de son arrêt en cas de données invalides."""

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
        "ingestion",
        "preparation",
        "sequences",
        "training",
        "evaluation",
        "save-LSTM",
        "save-Transformer",
        "metadata",
    ]
    assert result["train_count"] == 2
    assert result["test_count"] == 2


def test_invalid_data_stops_before_training_evaluation_and_saving(
    monkeypatch,
    tmp_path,
):
    def reject_data(_path):
        raise DataValidationError("données invalides")

    training = pytest.MonkeyPatch()
    training.setattr(pipeline, "load_current_data", reject_data)
    preparation = training.setattr(
        pipeline,
        "prepare_daily_gross_sales",
        lambda *args, **kwargs: pytest.fail("La préparation ne doit pas être appelée."),
    )
    training.setattr(
        pipeline,
        "build_sequences",
        lambda *args, **kwargs: pytest.fail("La création des séquences ne doit pas être appelée."),
    )
    training.setattr(
        pipeline,
        "train_models",
        lambda *args, **kwargs: pytest.fail("L'entraînement ne doit pas être appelé."),
    )
    training.setattr(
        pipeline,
        "evaluate_models",
        lambda *args, **kwargs: pytest.fail("L'évaluation ne doit pas être appelée."),
    )
    training.setattr(
        pipeline,
        "save_model_artifact",
        lambda *args, **kwargs: pytest.fail("Aucun modèle ne doit être sauvegardé."),
    )

    try:
        with pytest.raises(DataValidationError, match="données invalides"):
            pipeline.run_training_pipeline(
                config=make_config(),
                data_path=tmp_path / "invalid.xlsx",
                artifact_dir=tmp_path / "output",
            )
    finally:
        training.undo()

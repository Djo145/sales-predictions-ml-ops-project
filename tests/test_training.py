"""Tests du split temporel et de l'entraînement."""

import numpy as np
import pandas as pd

from retail_forecast.training import split_by_target_date, train_models


def test_split_uses_target_dates_without_mixing_periods():
    dates = pd.to_datetime(
        [
            "2011-08-30",
            "2011-08-31",
            "2011-09-01",
            "2011-09-30",
            "2011-10-01",
        ]
    )

    train_indices, test_indices = split_by_target_date(
        dates,
        train_end="2011-08-31",
        test_start="2011-09-01",
        test_end="2011-09-30",
    )

    np.testing.assert_array_equal(train_indices, [0, 1])
    np.testing.assert_array_equal(test_indices, [2, 3])


def test_train_models_smoke():
    config = {
        "preprocessing": {"lookback": 100},
        "split": {
            "train_end": "2011-08-31",
            "test_start": "2011-09-01",
            "test_end": "2011-09-01",
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "seed": 42,
            "device": "cpu",
        },
        "models": {
            "lstm": {"hidden_size": 4, "embedding_size": 2},
            "transformer": {
                "d_model": 4,
                "nhead": 2,
                "embedding_size": 2,
            },
        },
    }

    X_sales = np.zeros((3, 100, 1), dtype=np.float32)
    products = np.array(["P1", "P2", "P1"])
    targets = np.array([2, 3, 4], dtype=np.float32)
    dates = pd.to_datetime(
        ["2011-08-30", "2011-08-31", "2011-09-01"]
    )

    result = train_models(X_sales, products, targets, dates, config)

    assert set(result["models"]) == {"LSTM", "Transformer"}
    assert result["product_to_index"] == {"P1": 0, "P2": 1}
    np.testing.assert_array_equal(result["train_indices"], [0, 1])
    np.testing.assert_array_equal(result["test_indices"], [2])
    assert len(result["loss_histories"]["LSTM"]) == 1
    assert len(result["loss_histories"]["Transformer"]) == 1
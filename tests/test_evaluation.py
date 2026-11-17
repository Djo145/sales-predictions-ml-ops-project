"""Tests de la baseline et des métriques de régression."""

import numpy as np
import pytest

from retail_forecast.evaluation import (
    four_week_baseline,
    regression_metrics,
)


def test_four_week_baseline_averages_same_weekdays():
    X_sales = np.zeros((1, 100, 1), dtype=np.float32)
    X_sales[0, [-7, -14, -21, -28], 0] = [4, 8, 12, 16]

    baseline = four_week_baseline(X_sales)

    np.testing.assert_allclose(baseline, [10])


def test_regression_metrics_calculate_mae_and_rmse():
    metrics = regression_metrics(
        actual=np.array([2, 4]),
        predicted=np.array([1, 6]),
    )

    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))


def test_regression_metrics_reject_different_lengths():
    with pytest.raises(ValueError, match="même forme"):
        regression_metrics(
            actual=np.array([1, 2]),
            predicted=np.array([1]),
        )
"""Tests unitaires de l'agrégation et de la création des fenêtres."""

import numpy as np
import pandas as pd

from retail_forecast.preprocessing import (
    build_sequences,
    prepare_daily_gross_sales,
)


def make_raw_data(rows):
    """Construit un petit DataFrame de lignes de facture pour un test."""
    return pd.DataFrame(
        rows,
        columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"],
    )


def test_excludes_cancelled_invoices():
    raw = make_raw_data(
        [
            ("100", "P1", 4, "2010-12-01"),
            ("C101", "P1", 20, "2010-12-01"),
        ]
    )

    result = prepare_daily_gross_sales(
        raw,
        top_n=1,
        selection_end="2010-12-01",
    )

    assert result.loc[0, "quantity"] == 4


def test_does_not_count_negative_quantities():
    raw = make_raw_data(
        [
            ("100", "P1", 4, "2010-12-01"),
            ("101", "P1", -2, "2010-12-01"),
        ]
    )

    result = prepare_daily_gross_sales(
        raw,
        top_n=1,
        selection_end="2010-12-01",
    )

    assert result.loc[0, "quantity"] == 4


def test_fills_day_without_sales_with_zero():
    raw = make_raw_data(
        [
            ("100", "P1", 4, "2010-12-01"),
            ("101", "P1", 3, "2010-12-03"),
        ]
    )

    result = prepare_daily_gross_sales(
        raw,
        top_n=1,
        selection_end="2010-12-01",
    )

    day_without_sale = result.loc[
        result["date"] == pd.Timestamp("2010-12-02"),
        "quantity",
    ]

    assert day_without_sale.tolist() == [0]


def test_selects_products_using_only_dates_before_cutoff():
    raw = make_raw_data(
        [
            ("100", "P1", 1, "2010-12-01"),
            ("101", "P2", 2, "2010-12-01"),
            # Cette vente postérieure à la date de classement ne doit pas
            # faire passer P1 devant P2.
            ("102", "P1", 100, "2010-12-03"),
        ]
    )

    result = prepare_daily_gross_sales(
        raw,
        top_n=1,
        selection_end="2010-12-01",
    )

    assert result["StockCode"].unique().tolist() == ["P2"]


def test_builds_window_with_next_day_as_target():
    daily_sales = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-01", periods=4, freq="D"),
            "StockCode": ["P1"] * 4,
            "quantity": [1, 2, 3, 4],
        }
    )

    X_sales, products, y, target_dates = build_sequences(
        daily_sales,
        window_size=3,
    )

    assert X_sales.shape == (1, 3, 1)
    np.testing.assert_array_equal(X_sales[0, :, 0], [1, 2, 3])
    assert products.tolist() == ["P1"]
    np.testing.assert_array_equal(y, [4])
    assert pd.Timestamp(target_dates[0]) == pd.Timestamp("2011-01-04")
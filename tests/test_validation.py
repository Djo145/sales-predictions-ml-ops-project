"""Tests unitaires du contrat des lignes brutes."""

import pandas as pd
import pytest

from retail_forecast.validation import DataValidationError, validate_raw_data


@pytest.fixture
def valid_data() -> pd.DataFrame:
    """Petit jeu conforme, avec une quantité négative et des champs facultatifs nuls."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["536365", "C536366"],
            "StockCode": ["85123A", "71053"],
            "Quantity": [2, -1],
            "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:30:00"],
            "UnitPrice": [2.55, 0.0],
            "Description": ["Produit A", None],
            "CustomerID": [17850.0, None],
            "Country": ["United Kingdom", None],
        }
    )


def test_accepts_valid_data_and_preserves_negative_quantity(valid_data):
    result = validate_raw_data(valid_data)

    assert result is valid_data
    assert result.loc[1, "Quantity"] == -1


def test_rejects_missing_required_column(valid_data):
    data = valid_data.drop(columns=["StockCode"])

    with pytest.raises(DataValidationError, match="StockCode"):
        validate_raw_data(data)


def test_rejects_empty_product_code(valid_data):
    data = valid_data.copy()
    data.loc[0, "StockCode"] = " "

    with pytest.raises(DataValidationError, match="StockCode"):
        validate_raw_data(data)


def test_rejects_missing_quantity(valid_data):
    data = valid_data.copy()
    data.loc[0, "Quantity"] = None

    with pytest.raises(DataValidationError, match="Quantity"):
        validate_raw_data(data)


def test_rejects_fractional_quantity(valid_data):
    data = valid_data.copy()
    data["Quantity"] = data["Quantity"].astype(object)

    data.loc[0, "Quantity"] = 1.5

    with pytest.raises(DataValidationError, match="Quantity"):
        validate_raw_data(data)


def test_rejects_unreadable_date(valid_data):
    data = valid_data.copy()
    data.loc[0, "InvoiceDate"] = "pas une date"

    with pytest.raises(DataValidationError, match="InvoiceDate"):
        validate_raw_data(data)


def test_accepts_missing_optional_price(valid_data):
    data = valid_data.copy()
    data.loc[0, "UnitPrice"] = None

    assert validate_raw_data(data) is data


def test_rejects_non_numeric_price(valid_data):
    data = valid_data.copy()
    data["UnitPrice"] = data["UnitPrice"].astype(object)

    data.loc[0, "UnitPrice"] = "gratuit"

    with pytest.raises(DataValidationError, match="UnitPrice"):
        validate_raw_data(data)


def test_rejects_negative_price(valid_data):
    data = valid_data.copy()
    data.loc[0, "UnitPrice"] = -2.0

    with pytest.raises(DataValidationError, match="UnitPrice"):
        validate_raw_data(data)
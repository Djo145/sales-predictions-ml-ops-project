"""Tests unitaires de la lecture du fichier historique."""

import pandas as pd
import pytest

from retail_forecast.ingestion import load_current_data
from retail_forecast.validation import DataValidationError


@pytest.fixture
def valid_rows() -> pd.DataFrame:
    """Quelques lignes conformes au contrat des données brutes."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["100", "101"],
            "StockCode": ["P1", "P2"],
            "Quantity": [2, -1],
            "InvoiceDate": ["2011-01-01 10:00:00", "2011-01-02 11:00:00"],
            "UnitPrice": [1.5, 2.0],
        }
    )


def test_missing_file_raises_file_not_found_error(tmp_path):
    missing_path = tmp_path / "absent.xlsx"

    with pytest.raises(FileNotFoundError, match="Fichier de données introuvable"):
        load_current_data(missing_path)


def test_reads_and_returns_valid_excel_rows(tmp_path, monkeypatch, valid_rows):
    file_path = tmp_path / "online_retail_actuel.xlsx"
    file_path.touch()
    read_paths = []

    def fake_read_excel(path):
        read_paths.append(path)
        return valid_rows

    monkeypatch.setattr("retail_forecast.ingestion.pd.read_excel", fake_read_excel)

    result = load_current_data(file_path)

    assert read_paths == [file_path]
    pd.testing.assert_frame_equal(result, valid_rows)


def test_rejects_data_returned_by_excel_reader(tmp_path, monkeypatch):
    file_path = tmp_path / "online_retail_actuel.xlsx"
    file_path.touch()
    invalid_rows = pd.DataFrame(
        {
            "InvoiceNo": ["100"],
            "StockCode": ["P1"],
            "Quantity": [2],
            # InvoiceDate est volontairement absent.
        }
    )

    monkeypatch.setattr(
        "retail_forecast.ingestion.pd.read_excel",
        lambda _path: invalid_rows,
    )

    with pytest.raises(DataValidationError, match="InvoiceDate"):
        load_current_data(file_path)

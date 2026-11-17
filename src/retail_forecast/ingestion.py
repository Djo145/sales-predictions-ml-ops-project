"""Lecture et validation du fichier historique."""

from pathlib import Path

import pandas as pd

from retail_forecast.validation import validate_raw_data


def load_current_data(file_path: str | Path) -> pd.DataFrame:
    """Lit un fichier Excel de ventes et valide ses lignes brutes.

    Args:
        file_path: chemin vers le fichier, par exemple
            ``data/online_retail_actuel.xlsx``.

    Returns:
        Les lignes brutes validées, sans les transformer.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        DataValidationError: si les lignes ne respectent pas le contrat.
    """
    path = Path(file_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")

    raw_data = pd.read_excel(path)
    validate_raw_data(raw_data)

    return raw_data
"""Validation des lignes brutes du jeu Online Retail."""

import numpy as np
import pandas as pd

from retail_forecast.schemas import REQUIRED_COLUMNS


class DataValidationError(ValueError):
    """Erreur indiquant que les données ne respectent pas leur contrat."""


def validate_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Vérifie le contrat des lignes brutes sans modifier le DataFrame.

    Les colonnes obligatoires doivent être présentes et renseignées.
    Quantity doit être un nombre entier fini ; les valeurs négatives sont
    autorisées car elles peuvent correspondre à des retours ou corrections.
    UnitPrice est une colonne facultative conservée telle quelle, mais elle
    n'est pas validée car elle n'est pas utilisée pour prévoir les quantités.

    Retourne le DataFrame reçu, inchangé, ou lève DataValidationError.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data doit être un pandas.DataFrame.")

    errors: list[str] = []
    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Colonnes obligatoires absentes : {columns}")

    for column in ("InvoiceNo", "StockCode"):
        values = data[column].astype("string").str.strip()
        invalid = values.isna() | values.eq("")
        if invalid.any():
            errors.append(f"{column} contient {int(invalid.sum())} valeur(s) vide(s).")

    quantity = pd.to_numeric(data["Quantity"], errors="coerce")
    invalid_quantity = (
        quantity.isna()
        | ~np.isfinite(quantity.fillna(0))
        | quantity.mod(1).ne(0)
    )
    if invalid_quantity.any():
        errors.append(
            f"Quantity contient {int(invalid_quantity.sum())} valeur(s) "
            "absente(s), non numérique(s), non finie(s) ou non entière(s)."
        )

    dates = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    invalid_dates = dates.isna()
    if invalid_dates.any():
        errors.append(
            f"InvoiceDate contient {int(invalid_dates.sum())} date(s) illisible(s)."
        )

    if errors:
        raise DataValidationError("Données invalides : " + " ".join(errors))

    return data

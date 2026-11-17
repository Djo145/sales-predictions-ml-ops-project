"""Transformation des lignes de facture en séries quotidiennes et fenêtres."""

import numpy as np
import pandas as pd


def prepare_daily_gross_sales(
    raw_data: pd.DataFrame,
    top_n: int = 10,
    selection_end: str | pd.Timestamp = "2011-08-31",
) -> pd.DataFrame:
    """Agrège les ventes brutes et conserve les top produits.

    Le classement des produits est calculé uniquement jusqu'à
    ``selection_end`` afin que septembre, réservé au test, n'influence
    pas le choix des produits.

    Les journées sans transaction sont complétées avec zéro, à partir de
    la première transaction observée pour chaque produit retenu.

    ``raw_data`` doit avoir passé la validation de l'étape 1.

    Returns:
        Un DataFrame contenant ``date``, ``StockCode`` et ``quantity``.
    """
    required_columns = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate"}
    missing_columns = required_columns - set(raw_data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes nécessaires absentes : {missing}")

    if raw_data.empty:
        raise ValueError("Le DataFrame ne contient aucune ligne.")

    if top_n < 1:
        raise ValueError("top_n doit être supérieur ou égal à 1.")

    data = raw_data[list(required_columns)].copy()
    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    data["StockCode"] = data["StockCode"].astype("string").str.strip()
    data["date"] = pd.to_datetime(data["InvoiceDate"], errors="raise").dt.normalize()
    data["_quantity"] = pd.to_numeric(data["Quantity"], errors="raise").astype("int64")

    last_date = data["date"].max()
    selection_cutoff = pd.Timestamp(selection_end).normalize()

    # Une facture dont le numéro commence par C est une annulation.
    not_cancelled = ~data["InvoiceNo"].str.startswith("C", na=False)

    # La quantité brute ne compte que les quantités strictement positives.
    positive_sales = data.loc[
        not_cancelled & data["_quantity"].gt(0)
    ]

    if positive_sales.empty:
        raise ValueError("Aucune vente positive sur une facture non annulée.")

    # Somme des quantités par produit et par jour.
    gross_by_day = positive_sales.groupby(
        ["StockCode", "date"]
    )["_quantity"].sum()

    # Classement calculé uniquement sur la période de sélection.
    selection_sales = gross_by_day[
        gross_by_day.index.get_level_values("date") <= selection_cutoff
    ]

    product_totals = (
        selection_sales.groupby(level="StockCode")
        .sum()
        .rename("total_quantity")
        .reset_index()
        .sort_values(
            ["total_quantity", "StockCode"],
            ascending=[False, True],
            kind="stable",
        )
    )

    if product_totals.empty:
        raise ValueError(
            f"Aucune vente positive avant le {selection_cutoff.date()}."
        )

    selected_products = product_totals["StockCode"].head(top_n).tolist()

    # Début de la série : première transaction observée, même si elle ne
    # correspond pas à une vente positive.
    first_observation = data.groupby("StockCode")["date"].min()

    daily_frames = []
    for product in selected_products:
        start_date = first_observation.loc[product]
        calendar = pd.date_range(start_date, last_date, freq="D")

        product_sales = gross_by_day.xs(product, level="StockCode")
        product_sales = product_sales.reindex(calendar, fill_value=0)

        daily_frames.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "StockCode": product,
                    "quantity": product_sales.to_numpy(dtype=np.int64),
                }
            )
        )


    # Le résultat est un tableau avec une ligne par produit et par jour 
    return (
        pd.concat(daily_frames, ignore_index=True)
        .sort_values(["StockCode", "date"])
        .reset_index(drop=True)
    )


def build_sequences(
    daily_sales: pd.DataFrame,
    window_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construit les exemples : ventes passées → vente du jour cible.

    Returns:
        Un tuple ``(X_sales, product_codes, y, target_dates)`` :
        - ``X_sales`` : forme ``(N, window_size, 1)`` ;
        - ``product_codes`` : code produit de chaque exemple ;
        - ``y`` : quantité du jour cible ;
        - ``target_dates`` : date correspondant à chaque cible.
    """
    required_columns = {"date", "StockCode", "quantity"}
    missing_columns = required_columns - set(daily_sales.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes nécessaires absentes : {missing}")

    if window_size < 1:
        raise ValueError("window_size doit être supérieur ou égal à 1.")

    data = daily_sales.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["quantity"] = pd.to_numeric(data["quantity"], errors="raise")

    if data.duplicated(["StockCode", "date"]).any():
        raise ValueError("Il existe plusieurs lignes pour un même produit et un même jour.")

    quantities = data["quantity"].to_numpy(dtype=float)
    if not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("Les quantités quotidiennes doivent être finies et positives ou nulles.")

    windows = []
    product_codes = []
    targets = []
    target_dates = []

    for product, product_data in data.groupby("StockCode", sort=True):
        product_data = product_data.sort_values("date").reset_index(drop=True)

        # Une fenêtre de 100 lignes n'est temporellement correcte que si
        # les jours sont consécutifs.
        date_gaps = product_data["date"].diff().dropna()
        if not date_gaps.eq(pd.Timedelta(days=1)).all():
            raise ValueError(
                f"Les dates du produit {product} ne sont pas quotidiennes et continues."
            )

        values = product_data["quantity"].to_numpy(dtype=np.float32)
        dates = product_data["date"].to_numpy(dtype="datetime64[ns]")

        for target_index in range(window_size, len(product_data)):
            windows.append(values[target_index - window_size : target_index])
            product_codes.append(product)
            targets.append(values[target_index])
            target_dates.append(dates[target_index])

    X_sales = np.asarray(windows, dtype=np.float32).reshape(
        -1, window_size, 1
    )
    product_codes_array = np.asarray(product_codes, dtype=str)
    y = np.asarray(targets, dtype=np.float32)
    target_dates_array = np.asarray(target_dates, dtype="datetime64[ns]")

    return X_sales, product_codes_array, y, target_dates_array
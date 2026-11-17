"""Évaluation des prévisions et de la baseline."""

import numpy as np
import pandas as pd
import torch


def four_week_baseline(X_sales: np.ndarray) -> np.ndarray:
    """Prévoit la moyenne des quatre mêmes jours de semaine précédents."""
    X_sales = np.asarray(X_sales, dtype=np.float32)

    if X_sales.ndim != 3 or X_sales.shape[2] != 1:
        raise ValueError("X_sales doit avoir la forme (N, jours, 1).")
    if X_sales.shape[1] < 28:
        raise ValueError("La baseline nécessite au moins 28 jours d'historique.")

    # Les 7, 14, 21 et 28 jours avant la cible.
    lags = X_sales[:, [-7, -14, -21, -28], 0]
    return lags.mean(axis=1)


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calcule MAE et RMSE entre les valeurs réelles et prédites."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError("actual et predicted doivent avoir la même forme.")
    if len(actual) == 0:
        raise ValueError("Impossible de calculer des métriques sans exemple.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Les valeurs réelles et prédites doivent être finies.")

    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def evaluate_models(
    models: dict[str, torch.nn.Module],
    X_sales: np.ndarray,
    product_ids: np.ndarray,
    actual: np.ndarray,
    product_codes: np.ndarray,
    target_dates,
) -> dict[str, pd.DataFrame]:
    """Évalue les modèles et la baseline sur les mêmes exemples."""
    X_sales = np.asarray(X_sales, dtype=np.float32)
    product_ids = torch.as_tensor(product_ids, dtype=torch.long)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    product_codes = np.asarray(product_codes, dtype=str)
    target_dates = pd.to_datetime(target_dates)

    lengths = {
        len(X_sales),
        len(product_ids),
        len(actual),
        len(product_codes),
        len(target_dates),
    }
    if len(lengths) != 1:
        raise ValueError("Toutes les entrées d'évaluation doivent être alignées.")

    predictions = {
        "baseline_4weeks": four_week_baseline(X_sales),
    }

    for name, model in models.items():
        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            predictions[name] = (
                model(
                    torch.as_tensor(X_sales, dtype=torch.float32, device=device),
                    product_ids.to(device),
                )
                .cpu()
                .numpy()
            )

    prediction_table = pd.DataFrame(
        {
            "target_date": target_dates,
            "StockCode": product_codes,
            "actual": actual,
            **predictions,
        }
    )

    metric_rows = []
    for name, predicted in predictions.items():
        metric_rows.append(
            {"model": name, **regression_metrics(actual, predicted)}
        )
    overall = pd.DataFrame(metric_rows).set_index("model")

    by_product_rows = []
    for product, group in prediction_table.groupby("StockCode"):
        for name in predictions:
            by_product_rows.append(
                {
                    "StockCode": product,
                    "model": name,
                    **regression_metrics(
                        group["actual"].to_numpy(),
                        group[name].to_numpy(),
                    ),
                    "count": len(group),
                }
            )
    by_product = pd.DataFrame(by_product_rows)

    return {
        "overall": overall,
        "by_product": by_product,
        "predictions": prediction_table,
    }
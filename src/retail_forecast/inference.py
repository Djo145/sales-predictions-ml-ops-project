"""Prédiction du prochain jour à partir d'un artefact de modèle chargé."""

import numpy as np
import torch


def predict_next_day(
    loaded_artifact: dict,
    sales_history: np.ndarray,
    product_code: str,
) -> float:
    """Prévoit les ventes du lendemain pour un produit connu.

    `sales_history` contient exactement `lookback` quantités, de la plus
    ancienne à la plus récente. La sortie est une quantité continue non négative.
    """
    mapping = loaded_artifact["product_to_index"]
    product_code = str(product_code)
    if product_code not in mapping:
        raise ValueError(f"Produit inconnu dans le modèle : {product_code}")

    history = np.asarray(sales_history, dtype=np.float32).reshape(-1)
    lookback = int(loaded_artifact["lookback"])
    if history.size != lookback:
        raise ValueError(f"L'historique doit contenir exactement {lookback} jours.")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("L'historique doit contenir des quantités finies et positives ou nulles.")

    model = loaded_artifact["model"]
    device = next(model.parameters()).device
    x = torch.as_tensor(history, dtype=torch.float32, device=device).reshape(1, lookback, 1)
    product_ids = torch.tensor([mapping[product_code]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        prediction = float(model(x, product_ids).item())

    if not np.isfinite(prediction):
        raise RuntimeError("Le modèle a produit une prévision non finie.")
    return max(0.0, prediction)

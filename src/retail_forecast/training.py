"""Découpage chronologique et entraînement des modèles."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from retail_forecast.models import SmallLSTM, SmallTransformer


def split_by_target_date(
    target_dates,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne les indices train et test selon la date de la cible."""
    dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()
    train_end_date = pd.Timestamp(train_end).normalize()
    test_start_date = pd.Timestamp(test_start).normalize()
    test_end_date = pd.Timestamp(test_end).normalize()

    if test_start_date <= train_end_date:
        raise ValueError("Le test doit commencer après la fin de l'entraînement.")
    if test_end_date < test_start_date:
        raise ValueError("test_end doit être égal ou postérieur à test_start.")

    train_indices = np.flatnonzero(dates <= train_end_date)
    test_indices = np.flatnonzero(
        (dates >= test_start_date) & (dates <= test_end_date)
    )

    if len(train_indices) == 0:
        raise ValueError("Aucune cible ne se trouve dans la période d'entraînement.")
    if len(test_indices) == 0:
        raise ValueError("Aucune cible ne se trouve dans la période de test.")

    return train_indices, test_indices


def _fit_model(
    model: nn.Module,
    X: torch.Tensor,
    product_ids: torch.Tensor,
    y: torch.Tensor,
    train_indices: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    """Entraîne un modèle et retourne la perte moyenne par epoch."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    train_indices_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loss_history = []

    for _ in range(epochs):
        model.train()
        order = torch.randperm(
            len(train_indices_tensor),
            generator=generator,
        )
        shuffled_indices = train_indices_tensor[order]
        total_loss = 0.0
        examples_seen = 0

        for start in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]

            X_batch = X[batch_indices].to(device)
            ids_batch = product_ids[batch_indices].to(device)
            y_batch = y[batch_indices].to(device)

            optimizer.zero_grad()
            predictions = model(X_batch, ids_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()

            count = len(batch_indices)
            total_loss += loss.item() * count
            examples_seen += count

        loss_history.append(total_loss / examples_seen)

    model.eval()
    return model, loss_history


def train_models(
    X_sales: np.ndarray,
    product_codes: np.ndarray,
    y_values: np.ndarray,
    target_dates,
    config: dict,
) -> dict:
    """Entraîne les deux modèles uniquement sur les cibles de la période train.

    Les exemples de septembre restent isolés et sont retournés par leurs
    indices pour l'évaluation ultérieure.
    """
    preprocessing_config = config["preprocessing"]
    split_config = config["split"]
    training_config = config["training"]
    model_config = config["models"]

    lookback = preprocessing_config["lookback"]
    X_sales = np.asarray(X_sales, dtype=np.float32)
    product_codes = np.asarray(product_codes, dtype=str)
    y_values = np.asarray(y_values, dtype=np.float32).reshape(-1)

    dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()

    if X_sales.ndim != 3 or X_sales.shape[1:] != (lookback, 1):
        raise ValueError(
            f"X_sales doit avoir la forme (N, {lookback}, 1)."
        )
    if not (
        len(X_sales) == len(product_codes) == len(y_values) == len(dates)
    ):
        raise ValueError("Les entrées, produits, cibles et dates doivent être alignés.")

    train_indices, test_indices = split_by_target_date(
        dates,
        train_end=split_config["train_end"],
        test_start=split_config["test_start"],
        test_end=split_config["test_end"],
    )

    # L'ordre trié garantit que les mêmes produits obtiennent les mêmes indices.
    sorted_products = sorted(np.unique(product_codes).tolist())
    product_to_index = {
        product: index for index, product in enumerate(sorted_products)
    }
    product_ids = torch.tensor(
        [product_to_index[product] for product in product_codes],
        dtype=torch.long,
    )

    X_tensor = torch.from_numpy(X_sales)
    y_tensor = torch.from_numpy(y_values)

    device = torch.device(training_config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA est demandé dans la configuration, mais indisponible.")

    seed = training_config["seed"]
    torch.manual_seed(seed)

    model_specs = {
        "LSTM": (SmallLSTM, model_config["lstm"]),
        "Transformer": (SmallTransformer, model_config["transformer"]),
    }

    trained_models = {}
    loss_histories = {}

    for offset, (name, (model_class, specific_config)) in enumerate(
        model_specs.items()
    ):
        model_seed = seed + offset
        torch.manual_seed(model_seed)

        model = model_class(
            num_products=len(product_to_index),
            lookback=lookback,
            **specific_config,
        )

        trained_models[name], loss_histories[name] = _fit_model(
            model,
            X_tensor,
            product_ids,
            y_tensor,
            train_indices,
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            seed=model_seed,
            device=device,
        )

    return {
        "models": trained_models,
        "product_to_index": product_to_index,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "product_ids": product_ids.numpy(),
        "loss_histories": loss_histories,
        "device": str(device),
    }
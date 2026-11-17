"""Architectures LSTM et Transformer pour la prévision des ventes."""

import torch
from torch import nn


def _check_inputs(
    x: torch.Tensor,
    product_ids: torch.Tensor,
    *,
    lookback: int,
) -> None:
    """Vérifie les formes et types attendus par les modèles."""
    if x.ndim != 3 or x.shape[1:] != (lookback, 1):
        raise ValueError(
            f"x doit avoir la forme (batch, {lookback}, 1), reçu : {tuple(x.shape)}"
        )

    if not x.is_floating_point():
        raise TypeError("x doit contenir des valeurs flottantes.")

    if product_ids.ndim != 1 or product_ids.shape[0] != x.shape[0]:
        raise ValueError("product_ids doit contenir un indice par séquence du batch.")

    if product_ids.dtype != torch.long:
        raise TypeError("product_ids doit être un tenseur d'entiers torch.long.")

    if product_ids.device != x.device:
        raise ValueError("x et product_ids doivent être placés sur le même appareil.")


class SmallLSTM(nn.Module):
    """LSTM partagé entre les produits, prédisant la quantité du jour suivant."""

    def __init__(
        self,
        num_products: int,
        hidden_size: int = 16,
        embedding_size: int = 4,
        lookback: int = 100,
    ) -> None:
        super().__init__()

        if num_products < 1:
            raise ValueError("num_products doit être supérieur ou égal à 1.")
        if lookback < 1:
            raise ValueError("lookback doit être supérieur ou égal à 1.")

        self.lookback = lookback
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.product_embedding = nn.Embedding(num_products, embedding_size)
        self.head = nn.Linear(hidden_size + embedding_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Retourne une prévision par séquence, de forme ``(batch,)``."""
        _check_inputs(x, product_ids, lookback=self.lookback)

        sequence, _ = self.lstm(x)
        last_day_features = sequence[:, -1, :]
        product_features = self.product_embedding(product_ids)

        combined = torch.cat(
            (last_day_features, product_features),
            dim=1,
        )
        return self.head(combined).squeeze(-1)


class SmallTransformer(nn.Module):
    """Transformer Encoder à un bloc, partagé entre les produits."""

    def __init__(
        self,
        num_products: int,
        lookback: int = 100,
        d_model: int = 16,
        nhead: int = 2,
        embedding_size: int = 4,
    ) -> None:
        super().__init__()

        if num_products < 1:
            raise ValueError("num_products doit être supérieur ou égal à 1.")
        if lookback < 1:
            raise ValueError("lookback doit être supérieur ou égal à 1.")
        if d_model % nhead != 0:
            raise ValueError("d_model doit être divisible par nhead.")

        self.lookback = lookback
        self.input_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.product_embedding = nn.Embedding(num_products, embedding_size)
        self.head = nn.Linear(d_model + embedding_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Retourne une prévision par séquence, de forme ``(batch,)``."""
        _check_inputs(x, product_ids, lookback=self.lookback)

        x = self.input_projection(x) + self.position
        encoded = self.encoder(x)
        last_day_features = encoded[:, -1, :]
        product_features = self.product_embedding(product_ids)

        combined = torch.cat(
            (last_day_features, product_features),
            dim=1,
        )
        return self.head(combined).squeeze(-1)
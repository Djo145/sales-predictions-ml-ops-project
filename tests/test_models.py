"""Tests unitaires des architectures de prévision."""

import pytest
import torch

from retail_forecast.models import SmallLSTM, SmallTransformer


@pytest.mark.parametrize(
    "model_class",
    [SmallLSTM, SmallTransformer],
)
def test_model_returns_one_prediction_per_example(model_class):
    model = model_class(num_products=10, lookback=100)
    sales = torch.randn(2, 100, 1)
    product_ids = torch.tensor([0, 9], dtype=torch.long)

    predictions = model(sales, product_ids)

    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()


@pytest.mark.parametrize(
    "model_class",
    [SmallLSTM, SmallTransformer],
)
def test_model_supports_a_batch_of_one(model_class):
    model = model_class(num_products=10, lookback=100)
    sales = torch.randn(1, 100, 1)
    product_ids = torch.tensor([3], dtype=torch.long)

    predictions = model(sales, product_ids)

    # squeeze(-1) enlève uniquement la dimension de sortie,
    # pas la dimension batch.
    assert predictions.shape == (1,)


@pytest.mark.parametrize(
    "model_class",
    [SmallLSTM, SmallTransformer],
)
def test_model_rejects_wrong_sequence_length(model_class):
    model = model_class(num_products=10, lookback=100)
    sales = torch.randn(2, 99, 1)
    product_ids = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="forme"):
        model(sales, product_ids)


def test_model_rejects_product_ids_with_wrong_batch_size():
    model = SmallLSTM(num_products=10, lookback=100)
    sales = torch.randn(2, 100, 1)
    product_ids = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="un indice par séquence"):
        model(sales, product_ids)
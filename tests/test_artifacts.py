"""Tests de sauvegarde, rechargement et inférence des artefacts."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from retail_forecast.artifacts import load_model_artifact, save_model_artifact
from retail_forecast.inference import predict_next_day
from retail_forecast.models import SmallLSTM, SmallTransformer


@pytest.mark.parametrize(
    ("model_type", "model_class", "model_config"),
    [
        ("LSTM", SmallLSTM, {"hidden_size": 4, "embedding_size": 2}),
        (
            "Transformer",
            SmallTransformer,
            {"d_model": 4, "nhead": 2, "embedding_size": 2},
        ),
    ],
)
def test_save_reload_preserves_prediction(
    tmp_path, model_type, model_class, model_config
):
    torch.manual_seed(7)
    model = model_class(num_products=2, lookback=5, **model_config).eval()
    mapping = {"PRODUCT-A": 0, "PRODUCT-B": 1}
    history = np.array([0, 2, 1, 4, 3], dtype=np.float32)
    x = torch.tensor(history).reshape(1, 5, 1)
    product_id = torch.tensor([mapping["PRODUCT-A"]])
    with torch.no_grad():
        expected = float(model(x, product_id).item())

    artifact_dir = save_model_artifact(
        model,
        tmp_path / model_type,
        model_type=model_type,
        product_to_index=mapping,
        lookback=5,
        model_config=model_config,
    )
    loaded = load_model_artifact(artifact_dir)

    actual = predict_next_day(loaded, history, "PRODUCT-A")
    assert actual == pytest.approx(max(0.0, expected), abs=1e-6)
    assert loaded["product_to_index"] == mapping
    assert loaded["model"].training is False
    assert loaded["manifest"]["prediction_contract"]["normalization"] == "aucune"


def test_inference_rejects_unknown_product_and_wrong_history_length(tmp_path):
    model_config = {"hidden_size": 4, "embedding_size": 2}
    model = SmallLSTM(num_products=1, lookback=5, **model_config)
    artifact_dir = save_model_artifact(
        model,
        tmp_path / "lstm",
        model_type="LSTM",
        product_to_index={"KNOWN": 0},
        lookback=5,
        model_config=model_config,
    )
    loaded = load_model_artifact(artifact_dir)

    with pytest.raises(ValueError, match="Produit inconnu"):
        predict_next_day(loaded, np.zeros(5), "UNKNOWN")
    with pytest.raises(ValueError, match="exactement 5"):
        predict_next_day(loaded, np.zeros(4), "KNOWN")


def test_inference_rejects_negative_or_non_finite_history(tmp_path):
    model_config = {"hidden_size": 4, "embedding_size": 2}
    model = SmallLSTM(num_products=1, lookback=5, **model_config)
    artifact_dir = save_model_artifact(
        model,
        tmp_path / "lstm",
        model_type="LSTM",
        product_to_index={"KNOWN": 0},
        lookback=5,
        model_config=model_config,
    )
    loaded = load_model_artifact(artifact_dir)

    with pytest.raises(ValueError, match="finies"):
        predict_next_day(loaded, np.array([0, 1, np.nan, 2, 3]), "KNOWN")
    with pytest.raises(ValueError, match="finies"):
        predict_next_day(loaded, np.array([0, 1, -1, 2, 3]), "KNOWN")


def test_artifact_can_be_loaded_in_a_separate_process(tmp_path):
    model_config = {"hidden_size": 4, "embedding_size": 2}
    model = SmallLSTM(num_products=1, lookback=5, **model_config)
    artifact_dir = save_model_artifact(
        model,
        tmp_path / "lstm",
        model_type="LSTM",
        product_to_index={"KNOWN": 0},
        lookback=5,
        model_config=model_config,
    )
    project_src = str(Path(__file__).resolve().parents[1] / "src")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [project_src, environment.get("PYTHONPATH")])
    )
    script = (
        "import json, numpy as np; "
        "from retail_forecast.artifacts import load_model_artifact; "
        "from retail_forecast.inference import predict_next_day; "
        f"bundle = load_model_artifact({str(artifact_dir)!r}); "
        "print(json.dumps(predict_next_day(bundle, np.zeros(5), 'KNOWN')))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert float(result.stdout.strip()) >= 0

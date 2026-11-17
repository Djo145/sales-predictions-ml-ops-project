"""Tests de sauvegarde des preuves d'une exécution."""

import json

import numpy as np
import pandas as pd

from retail_forecast.run_metadata import save_run_metadata, sha256_file


def test_saves_data_fingerprints_split_metrics_and_environment(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (project / "uv.lock").write_text("version = 1\n")

    source_path = project / "sales.xlsx"
    source_path.write_bytes(b"synthetic source file")
    output = project / "runs" / "run_test"
    model_dir = output / "lstm"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text("{}")
    (model_dir / "model_state.pt").write_bytes(b"synthetic weights")

    raw_data = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2011-08-30", "2011-09-01", "2011-09-02"]
            ),
            "Quantity": [2, 3, 4],
        }
    )
    daily_sales = pd.DataFrame(
        {
            "date": pd.to_datetime(["2011-08-30", "2011-08-31", "2011-09-01"]),
            "StockCode": ["A", "A", "A"],
            "quantity": [2, 0, 3],
        }
    )
    evaluation = {
        "overall": pd.DataFrame(
            [{"model": "LSTM", "mae": 1.0, "rmse": 1.2}]
        ).set_index("model"),
        "by_product": pd.DataFrame(
            [{"StockCode": "A", "model": "LSTM", "mae": 1.0, "rmse": 1.2, "count": 1}]
        ),
        "predictions": pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2011-09-01"]),
                "StockCode": ["A"],
                "actual": [3.0],
                "LSTM": [2.0],
            }
        ),
    }

    paths = save_run_metadata(
        output,
        config={"preprocessing": {"lookback": 2}},
        source_path=source_path,
        raw_data=raw_data,
        daily_sales=daily_sales,
        product_codes=np.array(["A", "A", "A"]),
        y_values=np.array([2, 0, 3]),
        target_dates=pd.to_datetime(
            ["2011-08-30", "2011-08-31", "2011-09-01"]
        ),
        training={
            "train_indices": np.array([0, 1]),
            "test_indices": np.array([2]),
            "product_to_index": {"A": 0},
            "loss_histories": {"LSTM": [0.5]},
        },
        evaluation=evaluation,
        artifact_paths={"LSTM": model_dir},
        project_root=project,
    )

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["data"]["source_sha256"] == sha256_file(source_path)
    assert manifest["data"]["daily_row_count"] == 3
    assert manifest["split"]["train_count"] == 2
    assert manifest["split"]["test_count"] == 1
    assert manifest["split"]["excluded_count"] == 0
    assert manifest["models"]["LSTM"]["model_state.pt"]["sha256"] == sha256_file(
        model_dir / "model_state.pt"
    )
    assert paths["daily_snapshot"].is_file()
    assert paths["split"].is_file()
    assert paths["metrics"].is_file()
    assert paths["predictions"].is_file()
    assert (output / "environment" / "uv.lock").is_file()
    assert not (output / source_path.name).exists()

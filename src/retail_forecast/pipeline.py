"""Orchestration du pipeline d'entraînement et d'évaluation."""

from pathlib import Path

import numpy as np

from retail_forecast.artifacts import save_model_artifact
from retail_forecast.evaluation import evaluate_models
from retail_forecast.ingestion import load_current_data
from retail_forecast.preprocessing import (
    build_sequences,
    prepare_daily_gross_sales,
)
from retail_forecast.run_metadata import save_run_metadata
from retail_forecast.training import train_models


def run_training_pipeline(
    config: dict,
    data_path: str | Path,
    artifact_dir: str | Path,
) -> dict:
    """Exécute les étapes du pipeline et sauvegarde les deux modèles.

    La validation est réalisée par load_current_data. Toute erreur remonte
    à l'appelant et interrompt donc les étapes suivantes.
    """
    raw_data = load_current_data(data_path)

    daily_sales = prepare_daily_gross_sales(
        raw_data,
        top_n=config["preprocessing"]["top_n"],
        selection_end=config["preprocessing"]["selection_end"],
    )

    X_sales, product_codes, y_values, target_dates = build_sequences(
        daily_sales,
        window_size=config["preprocessing"]["lookback"],
    )

    training = train_models(
        X_sales,
        product_codes,
        y_values,
        target_dates,
        config,
    )

    # Seules les cibles de la période de test sont évaluées.
    test_indices = training["test_indices"]
    evaluation = evaluate_models(
        models=training["models"],
        X_sales=X_sales[test_indices],
        product_ids=training["product_ids"][test_indices],
        actual=y_values[test_indices],
        product_codes=product_codes[test_indices],
        target_dates=target_dates[test_indices],
    )

    # On sauvegarde après l'évaluation : un échec d'évaluation ne produit
    # pas de nouveaux artefacts de modèles.
    output_dir = Path(artifact_dir)
    artifact_paths = {}

    for model_name, model in training["models"].items():
        artifact_paths[model_name] = save_model_artifact(
            model,
            output_dir / model_name.lower(),
            model_type=model_name,
            product_to_index=training["product_to_index"],
            lookback=config["preprocessing"]["lookback"],
            model_config=config["models"][model_name.lower()],
        )

    metadata_paths = save_run_metadata(
        output_dir,
        config=config,
        source_path=data_path,
        raw_data=raw_data,
        daily_sales=daily_sales,
        product_codes=product_codes,
        y_values=y_values,
        target_dates=target_dates,
        training=training,
        evaluation=evaluation,
        artifact_paths=artifact_paths,
        project_root=Path(__file__).resolve().parents[2],
    )

    return {
        "evaluation": evaluation,
        "artifact_paths": artifact_paths,
        "train_count": len(training["train_indices"]),
        "test_count": len(test_indices),
        "product_to_index": training["product_to_index"],
        "loss_histories": training["loss_histories"],
        "metadata_paths": metadata_paths,
    }

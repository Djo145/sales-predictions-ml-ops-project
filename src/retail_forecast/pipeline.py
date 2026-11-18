"""Étapes métier réutilisables du pipeline de prévision."""

from pathlib import Path

from retail_forecast.artifacts import save_model_artifact
from retail_forecast.evaluation import evaluate_models
from retail_forecast.ingestion import load_current_data
from retail_forecast.preprocessing import build_sequences, prepare_daily_gross_sales
from retail_forecast.run_metadata import save_run_metadata, sha256_file
from retail_forecast.training import train_models
from retail_forecast.tracking import (
    fail_tracking_run,
    finish_tracking_run,
    log_successful_run,
    start_tracking_run,
)


def ingest_data(data_path: str | Path) -> dict:
    """Lit et valide les lignes brutes, puis identifie le fichier source."""
    raw_data = load_current_data(data_path)
    return {
        "raw_data": raw_data,
        "source_sha256": sha256_file(data_path),
        "source_path": str(Path(data_path).expanduser().resolve()),
    }


def prepare_training_data(
    raw_data,
    config: dict,
    source_sha256: str,
) -> dict:
    """Construit les ventes quotidiennes et les fenêtres d'apprentissage."""
    preprocessing = config["preprocessing"]
    daily_sales = prepare_daily_gross_sales(
        raw_data,
        top_n=preprocessing["top_n"],
        selection_end=preprocessing["selection_end"],
    )
    X_sales, product_codes, y_values, target_dates = build_sequences(
        daily_sales,
        window_size=preprocessing["lookback"],
    )
    return {
        "daily_sales": daily_sales,
        "X_sales": X_sales,
        "product_codes": product_codes,
        "y_values": y_values,
        "target_dates": target_dates,
        "source_sha256": source_sha256,
    }


def fit_training_data(prepared: dict, config: dict) -> dict:
    """Entraîne les deux modèles et retourne leurs partitions et historiques."""
    return train_models(
        prepared["X_sales"],
        prepared["product_codes"],
        prepared["y_values"],
        prepared["target_dates"],
        config,
    )


def evaluate_training_data(prepared: dict, training: dict) -> dict:
    """Évalue les modèles et la baseline uniquement sur les indices de test."""
    test_indices = training["test_indices"]
    return evaluate_models(
        models=training["models"],
        X_sales=prepared["X_sales"][test_indices],
        product_ids=training["product_ids"][test_indices],
        actual=prepared["y_values"][test_indices],
        product_codes=prepared["product_codes"][test_indices],
        target_dates=prepared["target_dates"][test_indices],
    )


def save_and_track_training(
    *,
    config: dict,
    data_path: str | Path,
    artifact_dir: str | Path,
    raw_data,
    prepared: dict,
    training: dict,
    evaluation: dict,
    mlflow_run_id: str,
) -> dict:
    """Sauvegarde les artefacts locaux et publie les preuves et modèles MLflow."""
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
        daily_sales=prepared["daily_sales"],
        product_codes=prepared["product_codes"],
        y_values=prepared["y_values"],
        target_dates=prepared["target_dates"],
        training=training,
        evaluation=evaluation,
        artifact_paths=artifact_paths,
        project_root=Path(__file__).resolve().parents[2],
    )

    # La fonction ouvre explicitement le run existant par son identifiant :
    # elle ne dépend pas d'un contexte MLflow actif hérité d'une autre task.
    registered_models = log_successful_run(
        run_id=mlflow_run_id,
        output_dir=output_dir,
        daily_sales=prepared["daily_sales"],
        source_path=data_path,
        training=training,
        evaluation=evaluation,
        project_root=Path(__file__).resolve().parents[2],
    )
    return {
        "artifact_paths": artifact_paths,
        "metadata_paths": metadata_paths,
        "registered_models": registered_models,
    }


def build_pipeline_result(
    prepared: dict,
    training: dict,
    evaluation: dict,
    publication: dict,
    mlflow_run_id: str,
) -> dict:
    """Assemble le résultat public commun au CLI et au flow Prefect."""
    return {
        "evaluation": evaluation,
        "artifact_paths": publication["artifact_paths"],
        "train_count": len(training["train_indices"]),
        "test_count": len(training["test_indices"]),
        "product_to_index": training["product_to_index"],
        "loss_histories": training["loss_histories"],
        "metadata_paths": publication["metadata_paths"],
        "mlflow_run_id": mlflow_run_id,
        "registered_models": publication["registered_models"],
        "source_sha256": prepared["source_sha256"],
    }


def run_training_pipeline(
    config: dict,
    data_path: str | Path,
    artifact_dir: str | Path,
) -> dict:
    """Exécute séquentiellement les étapes métier sans dépendre de Prefect."""
    mlflow_run = start_tracking_run(config)
    run_id = mlflow_run.info.run_id

    try:
        ingested = ingest_data(data_path)
        prepared = prepare_training_data(
            ingested["raw_data"], config, ingested["source_sha256"]
        )
        training = fit_training_data(prepared, config)
        evaluation = evaluate_training_data(prepared, training)
        publication = save_and_track_training(
            config=config,
            data_path=data_path,
            artifact_dir=artifact_dir,
            raw_data=ingested["raw_data"],
            prepared=prepared,
            training=training,
            evaluation=evaluation,
            mlflow_run_id=run_id,
        )
        finish_tracking_run(run_id)
    except Exception as error:
        fail_tracking_run(run_id, error)
        raise

    return build_pipeline_result(prepared, training, evaluation, publication, run_id)

import math
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "retail-forecast"
DEFAULT_TRACKING_URI = "http://localhost:5100"

REGISTERED_NAMES = {
    "LSTM": "retail-forecast-lstm",
    "Transformer": "retail-forecast-transformer",
}

def _flatten_config(values : dict, prefix: str="") -> dict[str, str] :
    """
    Convertit une configuration imbriquée en paramètres MLflow simples
    """
    flattened = {}
    for key, value in values.items() :
        name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict) :
            flattened.update(_flatten_config(value, name))
        elif value is not None : 
            flattened[name] = str(value)
    return flattened


def qualification_decision(model_metrics : dict, baseline_mae : float) -> dict :
    """ Applique une règle pédagogique : MAE meilleur que la baseline"""
    mae = float(model_metrics["mae"])
    rmse = float(model_metrics["rmse"])

    if not math.isfinite(baseline_mae) or not math.isfinite(mae) or not math.isfinite(rmse) :
        return {
            "qualification": "rejected",
            "reason" : "Une métrique est absente ou non finie",
        }

    if mae < baseline_mae :
        return {
            "qualification" : "candidate",
            "reason": "La MAE est inférieure à celle de la baseline.",
        }
    
    return {
        "qualification": "rejected",
        "reason": "La MAE ne bat pas celle de la baseline.",
    }


def start_tracking_run(config :dict) :
    """
    Configure MLflow et ouvre un run pour ce lancement d'entraînement
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run = mlflow.start_run(run_name = "retail-forcast-training")
    mlflow.set_tags(
        {
            "project" : "retail-forcast",
            "workflow" : "training",
            "qualification.policy" : "mae-better-than-four-week-baseling-v1"
        }
    )
    mlflow.log_params(_flatten_config(config))
    return run

def log_successful_run(
        *,
        run,
        output_dir: str | Path,
        daily_sales,
        source_path: str | Path,
        training: dict,
        evaluation: dict,
        project_root: str | Path,
    ) -> dict:
    """Enregistre les données, résultats et deux modèles d'un run réussi."""
    output_dir = Path(output_dir)
    project_root = Path(project_root)

    # Le manifeste contient notamment les SHA-256 du fichier source
    # et du snapshot quotidien déjà calculés par run_metadata.py.
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifeste absent : {manifest_path}")

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha256 = manifest["data"]["source_sha256"]
    snapshot_sha256 = manifest["data"]["daily_snapshot_sha256"]

    # Métadonnées du dataset dans MLflow : cela ne remplace pas le fichier CSV.
    dataset = mlflow.data.from_pandas(
        daily_sales,
        source=str(source_path),
        name="online-retail-daily-top-products",
        targets="quantity",
    )
    mlflow.log_input(dataset, context="prepared-daily-sales")

    mlflow.set_tags(
        {
            "data.source_sha256": source_sha256,
            "data.snapshot_sha256": snapshot_sha256,
        }
    )

    # Envoie config, snapshot quotidien, split, métriques, manifeste,
    # poids locaux et environnement présents dans le dossier du run.
    mlflow.log_artifacts(str(output_dir), artifact_path="training-run")

    # Les métriques du tableau comprennent LSTM, Transformer et baseline.
    for model_name, row in evaluation["overall"].iterrows():
        for metric_name, value in row.items():
            mlflow.log_metric(
                f"{model_name.lower()}.{metric_name}",
                float(value),
            )

    baseline_mae = float(evaluation["overall"].loc["baseline_4weeks", "mae"])
    client = MlflowClient()
    registrations = {}

    for model_name, model in training["models"].items():
        registered_name = REGISTERED_NAMES[model_name]
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name=model_name.lower(),
            registered_model_name=registered_name,
            code_paths=[str(project_root / "src")],
            serialization_format="pickle",
        )

        version = model_info.registered_model_version
        if version is None:
            raise RuntimeError(
                f"MLflow n'a pas retourné de version pour {registered_name}."
            )

        metrics = evaluation["overall"].loc[model_name].to_dict()
        decision = qualification_decision(metrics, baseline_mae)

        # Le tag décrit la version ; le rapport détaillé reste dans le run.
        client.set_model_version_tag(
            registered_name, str(version), "qualification",
            decision["qualification"],
        )
        client.set_model_version_tag(
            registered_name, str(version), "qualification.policy",
            "mae-better-than-four-week-baseline-v1",
        )

        registrations[model_name] = {
            "registered_name": registered_name,
            "version": str(version),
            "model_uri": model_info.model_uri,
            "run_id": run.info.run_id,
            "snapshot_sha256": snapshot_sha256,
            **decision,
        }

    mlflow.log_dict(
        {
            "run_id": run.info.run_id,
            "source_sha256": source_sha256,
            "daily_snapshot_sha256": snapshot_sha256,
            "registered_models": registrations,
        },
        "tracking/registered-models.json",
    )

    return registrations


def finish_tracking_run() -> None:
    """Ferme le run MLflow après l'enregistrement complet."""
    mlflow.end_run(status="FINISHED")


def fail_tracking_run(error: Exception) -> None:
    """Marque le run en échec puis laisse l'erreur remonter au pipeline."""
    if mlflow.active_run() is not None:
        try:
            mlflow.set_tag("failure.type", type(error).__name__)
            mlflow.set_tag("failure.message", str(error)[:500])
        finally:
            mlflow.end_run(status="FAILED")
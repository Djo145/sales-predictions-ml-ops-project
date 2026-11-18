"""Flow Prefect et tâches observables du pipeline de prévision."""

from __future__ import annotations

import errno
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

from retail_forecast.prefect_setup import (
    PREFECT_RESULT_STORAGE_BLOCK,
    ensure_cache_storage_block,
)
from retail_forecast.pipeline import (
    build_pipeline_result,
    evaluate_training_data,
    fit_training_data,
    ingest_data,
    prepare_training_data,
    save_and_track_training,
)
from retail_forecast.tracking import (
    fail_tracking_run,
    finish_tracking_run,
    start_tracking_run,
)

PREPARATION_VERSION = "daily-sales-v1"
TRANSIENT_ERRNOS = {errno.EAGAIN, errno.EINTR, errno.EIO}
if hasattr(errno, "ESTALE"):
    TRANSIENT_ERRNOS.add(errno.ESTALE)


def _is_transient_ingestion_error(error: Exception) -> bool:
    """Reconnaît les pannes système temporaires, jamais les données invalides."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return isinstance(error, OSError) and error.errno in TRANSIENT_ERRNOS


def retry_transient_ingestion(_task, _task_run, state) -> bool:
    """N'autorise le retry Prefect que pour un incident d'E/S temporaire."""
    if not state.is_failed():
        return False
    try:
        error = state.result(raise_on_failure=False)
    except Exception:
        return False
    return isinstance(error, Exception) and _is_transient_ingestion_error(error)


def preparation_cache_key(_context, parameters: dict) -> str:
    """Clé stable liée au contenu source, aux paramètres et au code métier."""
    preprocessing = parameters["config"]["preprocessing"]
    identity = {
        "source_sha256": parameters["ingested"]["source_sha256"],
        "top_n": preprocessing["top_n"],
        "selection_end": preprocessing["selection_end"],
        "lookback": preprocessing["lookback"],
        "preparation_version": PREPARATION_VERSION,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@task(
    name="ingestion-and-validation",
    retries=2,
    retry_delay_seconds=[2, 5],
    retry_condition_fn=retry_transient_ingestion,
    timeout_seconds=300,
    cache_policy=NO_CACHE,
    persist_result=False,
)
def ingestion_task(data_path: str) -> dict:
    """Lit, valide et empreinte le fichier source une fois par flow."""
    task_logger = get_run_logger()
    started = time.perf_counter()
    task_logger.info("Lecture et validation du fichier : %s", data_path)
    result = ingest_data(data_path)
    task_logger.info(
        "Ingestion valide : %s lignes, sha256=%s, durée=%.2fs",
        len(result["raw_data"]),
        result["source_sha256"],
        time.perf_counter() - started,
    )
    return result


@task(
    name="prepare-daily-sales-cached",
    version=PREPARATION_VERSION,
    cache_key_fn=preparation_cache_key,
    cache_expiration=None,
    result_storage=PREFECT_RESULT_STORAGE_BLOCK,
    persist_result=True,
    cache_result_in_memory=True,
    retries=0,
    timeout_seconds=300,
)
def preparation_task(ingested: dict, config: dict) -> dict:
    """Prépare le jeu quotidien et ses séquences, sans effet de bord MLflow."""
    task_logger = get_run_logger()
    started = time.perf_counter()
    task_logger.info(
        "Préparation déterministe : source_sha256=%s, version=%s",
        ingested["source_sha256"],
        PREPARATION_VERSION,
    )
    result = prepare_training_data(
        ingested["raw_data"], config, ingested["source_sha256"]
    )
    task_logger.info(
        "Préparation terminée : %s lignes quotidiennes, %s séquences, durée=%.2fs",
        len(result["daily_sales"]),
        len(result["y_values"]),
        time.perf_counter() - started,
    )
    return result


@task(
    name="train-lstm-and-transformer",
    retries=0,
    cache_policy=NO_CACHE,
    persist_result=False,
    timeout_seconds=7200,
)
def training_task(prepared: dict, config: dict) -> dict:
    """Entraîne les deux modèles ; aucun retry automatique ne duplique le fit."""
    task_logger = get_run_logger()
    started = time.perf_counter()
    task_logger.info("Entraînement du LSTM et du Transformer")
    result = fit_training_data(prepared, config)
    task_logger.info(
        "Entraînement terminé : train=%s, test=%s, durée=%.2fs",
        len(result["train_indices"]),
        len(result["test_indices"]),
        time.perf_counter() - started,
    )
    return result


@task(
    name="evaluate-models",
    retries=0,
    cache_policy=NO_CACHE,
    persist_result=False,
    timeout_seconds=600,
)
def evaluation_task(prepared: dict, training: dict) -> dict:
    """Calcule les métriques sur la partition de test uniquement."""
    task_logger = get_run_logger()
    started = time.perf_counter()
    result = evaluate_training_data(prepared, training)
    task_logger.info(
        "Évaluation terminée sur %s exemples de test, durée=%.2fs",
        len(result["predictions"]),
        time.perf_counter() - started,
    )
    return result


@task(
    name="save-artifacts-and-publish-to-mlflow",
    retries=0,
    cache_policy=NO_CACHE,
    persist_result=False,
)
def publication_task(
    *,
    config: dict,
    data_path: str,
    output_dir: str,
    ingested: dict,
    prepared: dict,
    training: dict,
    evaluation: dict,
    mlflow_run_id: str,
) -> dict:
    """Sauvegarde les sorties et enregistre les modèles une seule fois."""
    task_logger = get_run_logger()
    started = time.perf_counter()
    task_logger.info("Sauvegarde dans %s et publication MLflow run=%s", output_dir, mlflow_run_id)
    result = save_and_track_training(
        config=config,
        data_path=data_path,
        artifact_dir=output_dir,
        raw_data=ingested["raw_data"],
        prepared=prepared,
        training=training,
        evaluation=evaluation,
        mlflow_run_id=mlflow_run_id,
    )
    task_logger.info(
        "Publication terminée : %s modèle(s), durée=%.2fs",
        len(result["registered_models"]),
        time.perf_counter() - started,
    )
    return result


@flow(
    name="retail-sales-training",
    retries=0,
    log_prints=True,
    persist_result=False,
)
def training_flow(
    config_path: str = "config.json",
    output_dir: str | None = None,
) -> dict:
    """Orchestre les tâches dans l'ordre et ferme le run MLflow correctement."""
    flow_logger = get_run_logger()
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    project_dir = resolved_config_path.parent

    data_path = Path(config["data"]["current_file"]).expanduser()
    if not data_path.is_absolute():
        data_path = project_dir / data_path
    data_path = data_path.resolve()

    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        resolved_output_dir = project_dir / "runs" / f"run_{timestamp}"
    else:
        resolved_output_dir = Path(output_dir).expanduser()
        if not resolved_output_dir.is_absolute():
            resolved_output_dir = project_dir / resolved_output_dir
        resolved_output_dir = resolved_output_dir.resolve()

    flow_logger.info("Configuration : %s", resolved_config_path)
    flow_logger.info("Source : %s", data_path)
    flow_logger.info("Dossier de sortie : %s", resolved_output_dir)

    # Le block doit exister avant le lancement de la tâche persistée. Le créer
    # au premier flow évite de lire tout le classeur puis d'échouer au cache.
    cache_created, block_id = ensure_cache_storage_block()
    if cache_created:
        flow_logger.info(
            "Block de stockage du cache créé : %s (résultats locaux : %s)",
            block_id,
            "<project>/.prefect-cache/task-results",
        )
    else:
        flow_logger.info("Block de stockage du cache Prefect vérifié.")

    run_id = None
    try:
        current_flow_run_id = flow_run.id
        mlflow_run = start_tracking_run(
            config,
            flow_run_id=str(current_flow_run_id) if current_flow_run_id else None,
        )
        run_id = mlflow_run.info.run_id

        ingested = ingestion_task(str(data_path))
        prepared = preparation_task(ingested, config)
        training = training_task(prepared, config)
        evaluation = evaluation_task(prepared, training)
        publication = publication_task(
            config=config,
            data_path=str(data_path),
            output_dir=str(resolved_output_dir),
            ingested=ingested,
            prepared=prepared,
            training=training,
            evaluation=evaluation,
            mlflow_run_id=run_id,
        )
        finish_tracking_run(run_id)
    except Exception as error:
        if run_id is not None:
            try:
                fail_tracking_run(run_id, error)
            except Exception:
                flow_logger.exception(
                    "Impossible de finaliser le run MLflow %s après l'échec",
                    run_id,
                )
        flow_logger.exception("Le flow échoue : %s", error)
        raise

    flow_logger.info("Flow terminé ; mlflow_run_id=%s", run_id)
    return build_pipeline_result(prepared, training, evaluation, publication, run_id)

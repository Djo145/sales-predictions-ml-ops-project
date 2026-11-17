"""Écriture des preuves et métadonnées d'une exécution d'entraînement."""

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_file(path: str | Path) -> str:
    """Calcule le SHA-256 d'un fichier sans le charger entièrement en mémoire."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Type non sérialisable en JSON : {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _git_state(project_root: Path) -> dict:
    """Retourne la révision et l'état Git, ou indique que Git est indisponible."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "."],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "available": True,
            "commit": commit,
            "working_tree_clean": not changes,
            "changed_paths_count": len(changes),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "available": False,
            "commit": None,
            "working_tree_clean": None,
            "changed_paths_count": None,
        }


def save_run_metadata(
    output_dir: str | Path,
    *,
    config: dict,
    source_path: str | Path,
    raw_data: pd.DataFrame,
    daily_sales: pd.DataFrame,
    product_codes: np.ndarray,
    y_values: np.ndarray,
    target_dates,
    training: dict,
    evaluation: dict,
    artifact_paths: dict,
    project_root: str | Path,
) -> dict[str, Path]:
    """Sauvegarde configuration, données utiles, split, métriques et provenance.

    Le fichier Excel source n'est pas recopié. Son chemin, sa taille et son
    empreinte sont consignés ; le petit snapshot quotidien réellement préparé
    est, lui, conservé dans le dossier du run.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fichier source introuvable : {source}")
    project = Path(project_root).expanduser().resolve()

    config_path = output / "config.json"
    _write_json(config_path, config)

    environment_dir = output / "environment"
    environment_dir.mkdir(exist_ok=True)
    environment_files = {}
    for name in ("pyproject.toml", "uv.lock"):
        original = project / name
        if original.is_file():
            copied = environment_dir / name
            shutil.copy2(original, copied)
            environment_files[name] = {
                "path": str(copied.relative_to(output)),
                "sha256": sha256_file(copied),
            }

    snapshot_path = output / "daily_sales.csv"
    daily_sales.sort_values(["StockCode", "date"]).to_csv(
        snapshot_path,
        index=False,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )

    codes = np.asarray(product_codes, dtype=str)
    targets = np.asarray(y_values, dtype=np.float64).reshape(-1)
    dates = pd.to_datetime(target_dates).normalize()
    if not (len(codes) == len(targets) == len(dates)):
        raise ValueError("Produits, cibles et dates doivent être alignés pour l'audit.")

    train_indices = np.asarray(training["train_indices"], dtype=int)
    test_indices = np.asarray(training["test_indices"], dtype=int)
    partition = np.full(len(targets), "excluded", dtype=object)
    partition[train_indices] = "train"
    partition[test_indices] = "test"
    lookback = int(config["preprocessing"]["lookback"])
    split = pd.DataFrame(
        {
            "sample_index": np.arange(len(targets)),
            "partition": partition,
            "StockCode": codes,
            "history_start_date": dates - pd.to_timedelta(lookback, unit="D"),
            "history_end_date": dates - pd.Timedelta(days=1),
            "target_date": dates,
            "actual_quantity": targets,
        }
    )
    split_path = output / "split.csv"
    split.to_csv(split_path, index=False, date_format="%Y-%m-%d")

    predictions_path = output / "test_predictions.csv"
    evaluation["predictions"].to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    metrics_path = output / "metrics.json"
    _write_json(
        metrics_path,
        {
            "overall": evaluation["overall"].rename_axis("model").reset_index().to_dict(
                orient="records"
            ),
            "by_product": evaluation["by_product"].to_dict(orient="records"),
        },
    )
    losses_path = output / "training_losses.json"
    _write_json(losses_path, training["loss_histories"])

    timestamps = pd.to_datetime(raw_data["InvoiceDate"], errors="raise")
    model_files = {}
    for model_name, model_dir in artifact_paths.items():
        model_dir = Path(model_dir)
        model_files[model_name] = {}
        for filename in ("manifest.json", "model_state.pt"):
            file_path = model_dir / filename
            model_files[model_name][filename] = {
                "path": str(file_path.relative_to(output)),
                "sha256": sha256_file(file_path),
            }

    source_size = source.stat().st_size
    manifest = {
        "run_id": output.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "source_size_bytes": source_size,
            "raw_row_count": len(raw_data),
            "raw_columns": list(raw_data.columns),
            "raw_dtypes": {column: str(dtype) for column, dtype in raw_data.dtypes.items()},
            "raw_min_invoice_date": timestamps.min().isoformat(),
            "raw_max_invoice_date": timestamps.max().isoformat(),
            "daily_snapshot_path": snapshot_path.name,
            "daily_snapshot_sha256": sha256_file(snapshot_path),
            "daily_row_count": len(daily_sales),
            "sequence_count": len(targets),
            "product_to_index": training["product_to_index"],
        },
        "split": {
            "train_count": int((partition == "train").sum()),
            "test_count": int((partition == "test").sum()),
            "excluded_count": int((partition == "excluded").sum()),
            "train_target_min": dates[train_indices].min().isoformat(),
            "train_target_max": dates[train_indices].max().isoformat(),
            "test_target_min": dates[test_indices].min().isoformat(),
            "test_target_max": dates[test_indices].max().isoformat(),
            "split_file": split_path.name,
            "split_sha256": sha256_file(split_path),
        },
        "configuration_file": config_path.name,
        "configuration_sha256": sha256_file(config_path),
        "environment": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "files": environment_files,
        },
        "code": _git_state(project),
        "evaluation": {
            "metrics_file": metrics_path.name,
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_file": predictions_path.name,
            "predictions_sha256": sha256_file(predictions_path),
            "training_losses_file": losses_path.name,
        },
        "models": model_files,
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)

    return {
        "manifest": manifest_path,
        "config": config_path,
        "daily_snapshot": snapshot_path,
        "split": split_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "training_losses": losses_path,
        "environment": environment_dir,
    }

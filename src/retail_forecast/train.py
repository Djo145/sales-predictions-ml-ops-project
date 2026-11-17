"""Point d'entrée en ligne de commande pour l'entraînement."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from retail_forecast.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne et évalue les modèles de prévision des ventes."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Chemin vers le fichier de configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        help="Dossier de sortie des modèles ; créé automatiquement si absent.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_dir = config_path.parent

    data_path = Path(config["data"]["current_file"]).expanduser()
    if not data_path.is_absolute():
        data_path = project_dir / data_path

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = project_dir / output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = project_dir / "runs" / f"run_{timestamp}"

    result = run_training_pipeline(
        config=config,
        data_path=data_path,
        artifact_dir=output_dir,
    )

    print("Évaluation sur la période de test :")
    print(result["evaluation"]["overall"].round(3).to_string())
    print(f"\nExemples d'entraînement : {result['train_count']}")
    print(f"Exemples de test : {result['test_count']}")
    print("\nModèles sauvegardés :")
    for model_name, path in result["artifact_paths"].items():
        print(f"- {model_name} : {path}")


if __name__ == "__main__":
    main()
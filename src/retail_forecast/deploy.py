"""Point d'entrée du déploiement local Prefect du pipeline d'entraînement."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from retail_forecast.flows import training_flow

DEPLOYMENT_NAME = "retail-sales-training-local"
DEPLOYMENT_VERSION = "0.1.0"
DEPLOYMENT_DESCRIPTION = (
    "Entraîne, évalue et publie les modèles de prévision des ventes. "
    "Le processus serve s'exécute localement et lance les runs demandés depuis Prefect."
)


def positive_minutes(value: str) -> int:
    """Convertit un intervalle CLI en entier strictement positif."""
    minutes = int(value)
    if minutes <= 0:
        raise argparse.ArgumentTypeError("L'intervalle doit être strictement positif.")
    return minutes


def build_serve_kwargs(
    config_path: str | Path,
    deployment_name: str = DEPLOYMENT_NAME,
    interval_minutes: int | None = None,
) -> dict:
    """Construit la configuration du déploiement sans lancer de processus.

    Le chemin absolu rend le déploiement indépendant du dossier depuis lequel un
    run est demandé. ``output_dir`` n'est volontairement pas fixé : le flow crée
    un nouveau dossier horodaté pour chaque exécution lancée depuis Prefect.
    """
    kwargs = {
        "name": deployment_name,
        "parameters": {"config_path": str(Path(config_path).expanduser().resolve())},
        "description": DEPLOYMENT_DESCRIPTION,
        "version": DEPLOYMENT_VERSION,
        "limit": 1,
    }
    if interval_minutes is not None:
        kwargs["interval"] = timedelta(minutes=interval_minutes)
    return kwargs


def main(argv: list[str] | None = None) -> None:
    """Enregistre le déploiement puis attend les demandes de run Prefect."""
    parser = argparse.ArgumentParser(
        description="Expose le flow d'entraînement à l'interface Prefect locale."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Chemin du fichier de configuration utilisé par défaut.",
    )
    parser.add_argument(
        "--name",
        default=DEPLOYMENT_NAME,
        help="Nom du déploiement visible dans Prefect.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=positive_minutes,
        help="Planification optionnelle, en minutes. Sans cette option : aucun planning.",
    )
    args = parser.parse_args(argv)

    # serve() ne lance aucun entraînement immédiatement. Il enregistre le
    # déploiement, puis garde ce processus disponible pour les runs demandés.
    training_flow.serve(
        **build_serve_kwargs(
            config_path=args.config,
            deployment_name=args.name,
            interval_minutes=args.interval_minutes,
        )
    )


if __name__ == "__main__":
    main()

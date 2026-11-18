"""Point d'entrée en ligne de commande pour l'entraînement."""

import argparse

from retail_forecast.flows import training_flow


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

    result = training_flow(
        config_path=args.config,
        output_dir=args.output_dir,
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

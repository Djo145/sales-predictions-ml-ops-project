"""Sauvegarde et rechargement des modèles avec leur contrat d'inférence."""

import json
from pathlib import Path

import torch
from torch import nn

from retail_forecast.models import SmallLSTM, SmallTransformer

ARTIFACT_FORMAT_VERSION = 1


def save_model_artifact(
    model: nn.Module,
    artifact_dir: str | Path,
    *,
    model_type: str,
    product_to_index: dict[str, int],
    lookback: int,
    model_config: dict,
) -> Path:
    """Sauvegarde les poids et les informations nécessaires à l'inférence.

    Le dossier créé contient `model_state.pt` (poids uniquement) et
    `manifest.json` (architecture, mapping et contrat d'entrée).
    """
    model_classes = {"LSTM": SmallLSTM, "Transformer": SmallTransformer}
    if model_type not in model_classes or not isinstance(model, model_classes[model_type]):
        raise ValueError("model_type ne correspond pas à une architecture prise en charge.")
    if lookback < 1:
        raise ValueError("lookback doit être supérieur ou égal à 1.")
    if not product_to_index:
        raise ValueError("product_to_index ne peut pas être vide.")

    normalized_mapping = {str(code): int(index) for code, index in product_to_index.items()}
    expected_indices = list(range(len(normalized_mapping)))
    if sorted(normalized_mapping.values()) != expected_indices:
        raise ValueError("Les indices produit doivent être uniques et commencer à zéro.")

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)

    manifest = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": model_type,
        "model_config": model_config,
        "lookback": int(lookback),
        "product_to_index": normalized_mapping,
        "prediction_contract": {
            "input": "ventes brutes quotidiennes, ordre chronologique ancien vers récent",
            "input_shape": [int(lookback), 1],
            "input_unit": "unités vendues par jour",
            "normalization": "aucune",
            "output": "prévision continue des unités vendues le jour suivant, bornée à zéro",
        },
    }

    # Le format weights-only limite le chargement aux tenseurs, sans pickle arbitraire.
    torch.save(model.state_dict(), destination / "model_state.pt")
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def load_model_artifact(
    artifact_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict:
    """Reconstruit le réseau, restaure ses poids et retourne son mapping produit."""
    source = Path(artifact_dir)
    manifest_path = source / "manifest.json"
    weights_path = source / "model_state.pt"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Artefact incomplet : {source}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Version de format d'artefact inconnue.")

    model_type = manifest.get("model_type")
    model_classes = {"LSTM": SmallLSTM, "Transformer": SmallTransformer}
    if model_type not in model_classes:
        raise ValueError(f"Architecture non prise en charge : {model_type!r}")

    mapping = manifest.get("product_to_index", {})
    num_products = len(mapping)
    if not mapping or sorted(mapping.values()) != list(range(num_products)):
        raise ValueError("Le mapping produit de l'artefact est invalide.")

    model = model_classes[model_type](
        num_products=num_products,
        lookback=int(manifest["lookback"]),
        **manifest["model_config"],
    )
    selected_device = torch.device(device)
    state_dict = torch.load(weights_path, map_location=selected_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(selected_device)
    model.eval()

    return {
        "model": model,
        "model_type": model_type,
        "product_to_index": {str(code): int(index) for code, index in mapping.items()},
        "lookback": int(manifest["lookback"]),
        "manifest": manifest,
        "artifact_dir": source,
    }

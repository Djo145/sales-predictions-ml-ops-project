"""Provisionne le stockage local persistant utilisé par le cache Prefect."""

from pathlib import Path

from prefect.filesystems import LocalFileSystem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFECT_CACHE_DIR = PROJECT_ROOT / ".prefect-cache" / "task-results"
PREFECT_RESULT_STORAGE_BLOCK = "local-file-system/retail-forecast-cache"


def configure_cache_storage() -> str:
    """Enregistre dans Prefect un block pointant vers le cache local du projet."""
    PREFECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    storage = LocalFileSystem(basepath=str(PREFECT_CACHE_DIR))
    block_id = storage.save(name="retail-forecast-cache", overwrite=True)
    return str(block_id)


def ensure_cache_storage_block() -> tuple[bool, str | None]:
    """Vérifie le block et le crée au premier lancement s'il est absent.

    Returns:
        ``(created, block_id)`` ; ``block_id`` vaut ``None`` si le block existait.

    Raises:
        Exception: si le serveur Prefect est inaccessible ou si le block ne peut
            pas être chargé/créé. Ces erreurs sont volontairement bloquantes :
            le flow ne démarre pas avec un cache annoncé mais inutilisable.
    """
    try:
        LocalFileSystem.load("retail-forecast-cache")
    except ValueError as error:
        if "Unable to find block document named retail-forecast-cache" not in str(error):
            raise
        return True, configure_cache_storage()
    return False, None


if __name__ == "__main__":
    print(f"Block de cache Prefect enregistré : {configure_cache_storage()}")
    print(f"Résultats locaux persistés dans : {PREFECT_CACHE_DIR}")

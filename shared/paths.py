# Rôle : Centraliser les chemins absolus des principaux dossiers du projet.

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
PROCESSING_DIR = PROJECT_ROOT / "processing"
PROCESSING_PIPELINE_DIR = PROCESSING_DIR / "pipeline"


def portable_project_path(path: str | Path) -> str:
    """Store paths inside HERAMAN relative to the project root."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        # External files cannot be made portable with respect to HERAMAN.
        return str(resolved)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a stored project-relative path on the current computer."""
    stored = Path(path)
    return stored if stored.is_absolute() else PROJECT_ROOT / stored


def resolve_stored_path(path: str | Path | None, fallback: str | Path) -> Path:
    """Resolve a saved path and recover from a stale absolute path."""
    fallback_path = Path(fallback)
    if path is None or not str(path).strip():
        return fallback_path
    candidate = resolve_project_path(path)
    if candidate.is_file() or not fallback_path.is_file():
        return candidate
    return fallback_path


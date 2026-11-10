"""Local catalogue of ENVI sources. Stores paths only, never cube copies."""
import hashlib
import json
import os
from pathlib import Path
from shared.envi import load_spectral_cube, resolve_envi_header
from shared.paths import PROJECT_ROOT, portable_project_path, resolve_project_path

CATALOG = PROJECT_ROOT/'config'/'envi_sources.json'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def registered_sources(catalog_path=None):
    catalog = Path(catalog_path or CATALOG)
    if not catalog.is_file():
        return []
    values = json.loads(catalog.read_text(encoding='utf-8'))
    if not isinstance(values, list):
        raise ValueError('Invalid ENVI source catalogue.')
    unique = {}
    for value in values:
        if isinstance(value, str) and resolve_project_path(value).is_file():
            resolved = str(resolve_project_path(value).resolve())
            unique.setdefault(os.path.normcase(resolved), resolved)
    return list(unique.values())


def register_envi_source(path, *, catalog_path=None):
    header = resolve_envi_header(path)
    cube, _ = load_spectral_cube(header)
    mapping = getattr(cube, '_mmap', None)
    if mapping is not None:
        mapping.close()
    catalog = Path(catalog_path or CATALOG)
    sources = registered_sources(catalog)
    source = str(header)
    if os.path.normcase(source) not in {os.path.normcase(value) for value in sources}:
        sources.append(source)
    stored = [portable_project_path(value) for value in sources]
    previous = json.loads(catalog.read_text(encoding='utf-8')) if catalog.is_file() else None
    if stored != previous:
        catalog.parent.mkdir(parents=True, exist_ok=True)
        temporary = catalog.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(stored, indent=2), encoding='utf-8')
        temporary.replace(catalog)
    return source

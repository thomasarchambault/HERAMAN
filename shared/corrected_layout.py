"""Stable corrected specimen layout and guarded classification replacement."""
from pathlib import Path
import shutil
import stat


def _linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def specimen_root(cube_path):
    path = Path(cube_path).resolve()
    return path.parent.parent if path.parent.name == 'cube' else path.parent


def nearby_image_folders(path):
    path = Path(path)
    parent = path.parent if path.is_file() else path
    return list(dict.fromkeys([parent/'corrected_images', parent, parent.parent/'corrected_images', parent.parent]))


def prepare_classification_folder(path, source_cube):
    """Replace only a dedicated result directory, never its source or ancestors."""
    target = Path(path).resolve()
    source = Path(source_cube).resolve()
    if target == target.parent or target == source.parent or target in source.parents:
        raise ValueError(f'Unsafe classification output directory: {target}')
    if Path(path).exists() and _linked(Path(path)):
        raise ValueError('Classification output cannot be a link or junction.')
    if target.exists():
        if target.name != 'classification' and any(target.iterdir()) and not (target/'classification_source.json').is_file():
            raise ValueError(f'Refusing to clear an unrecognised output directory: {target}')
        # The resolved target above is the explicitly selected output, and it
        # cannot contain the cube. Never traverse linked subdirectories.
        for item in target.rglob('*'):
            if _linked(item):
                raise ValueError(f'Linked item in classification output: {item}')
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target

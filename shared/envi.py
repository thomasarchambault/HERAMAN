"""Strict ENVI reader. Values are preserved; wavelengths are normalized to nm."""
import re
from pathlib import Path
import numpy as np

TYPES = {1:'u1', 2:'i2', 3:'i4', 4:'f4', 5:'f8', 12:'u2', 13:'u4', 14:'i8', 15:'u8'}

def resolve_envi_binary(header, fields):
    """Resolve the same binary for reading and cache invalidation."""
    header = Path(header)
    if 'data file' in fields:
        binary = header.parent / fields['data file'].strip('"')
    else:
        candidates = [header.with_suffix(''), *(header.with_suffix(ext) for ext in ('.raw','.img','.dat','.bin'))]
        candidates = list(dict.fromkeys(p for p in candidates if p.is_file()))
        if len(candidates)!=1:
            raise ValueError('Binary file missing or ambiguous; specify data file in header.')
        binary = candidates[0]
    return binary.resolve()

def read_envi(path, *, require_wavelengths=True):
    header = resolve_envi_header(path)
    text = header.read_text(encoding='utf-8-sig')
    if not text.strip().startswith('ENVI'):
        raise ValueError('Missing ENVI signature.')
    fields = {m.group(1).strip().lower(): m.group(2).strip().strip('{}').strip()
              for m in re.finditer(r'(?m)^\s*([^;\n={}]+?)\s*=\s*(\{[^}]*\}|[^\n]*)', text)}
    h, w, n = (int(fields[k]) for k in ('lines','samples','bands'))
    if min(h,w,n) < 1:
        raise ValueError('Invalid ENVI dimensions.')
    code = int(fields['data type'])
    if code not in TYPES:
        raise ValueError(f'Unsupported ENVI data type: {code}')
    order = int(fields.get('byte order','0'))
    offset = int(fields.get('header offset','0'))
    if order not in (0,1) or offset < 0:
        raise ValueError('Invalid byte order or header offset.')
    dtype = np.dtype(TYPES[code]).newbyteorder('>' if order else '<')
    wl = np.asarray([float(v) for v in fields.get('wavelength','').split(',') if v.strip()], dtype=np.float64)
    if wl.size or require_wavelengths:
        unit = fields.get('wavelength units','').lower()
        if unit in ('micrometers','micrometer','microns','um','µm'):
            wl *= 1000
        elif unit not in ('nanometers','nanometer','nm'):
            raise ValueError('Explicit wavelength units (nm or micrometers) required.')
        if len(wl)!=n or not np.isfinite(wl).all() or len(np.unique(wl))!=n:
            raise ValueError('Missing, duplicate or invalid ENVI wavelengths.')
    else:
        wl = None
    binary = resolve_envi_binary(header, fields)
    if binary.stat().st_size != offset+h*w*n*dtype.itemsize:
        raise ValueError('ENVI binary size does not match header.')
    layout = fields['interleave'].lower()
    shapes = {'bip':(h,w,n),'bil':(h,n,w),'bsq':(n,h,w)}
    if layout not in shapes:
        raise ValueError('Unsupported ENVI interleave.')
    cube = np.memmap(binary, mode='r', dtype=dtype, offset=offset, shape=shapes[layout])
    if layout=='bil': cube=cube.transpose(0,2,1)
    if layout=='bsq': cube=cube.transpose(1,2,0)
    indices=np.argsort(wl) if wl is not None else np.arange(n)
    # Preserve native mapped storage for the usual ascending spectral axis.
    if not np.array_equal(indices,np.arange(n)): cube=cube[:,:,indices]
    return cube, wl[indices].astype(np.float32) if wl is not None else None, fields


def resolve_envi_header(path):
    """Accept either member of an ENVI header/binary pair."""
    path = Path(path).resolve()
    if path.suffix.lower() == '.hdr':
        return path
    candidates = {p.resolve() for p in (path.with_suffix('.hdr'), Path(str(path)+'.hdr')) if p.is_file()}
    # Windows paths are case-insensitive; this also supports uppercase headers on other OSes.
    candidates.update(p.resolve() for p in path.parent.iterdir()
                      if p.is_file() and p.name.lower() in (path.with_suffix('.hdr').name.lower(), (path.name+'.hdr').lower()))
    if len(candidates) != 1:
        raise ValueError('Select the ENVI .hdr file; the binary header is missing or ambiguous.')
    header = candidates.pop()
    # Verify the chosen header actually points at this binary, not a neighbour.
    text = header.read_text(encoding='utf-8-sig')
    fields = {m.group(1).strip().lower(): m.group(2).strip().strip('{}').strip()
              for m in re.finditer(r'(?m)^\s*([^;\n={}]+?)\s*=\s*(\{[^}]*\}|[^\n]*)', text)}
    if resolve_envi_binary(header, fields) != path:
        raise ValueError('The ENVI header refers to a different binary file.')
    return header


def load_spectral_cube(path):
    cube, wavelengths, _ = read_envi(path)
    return cube, wavelengths


def write_envi(path, values, *, wavelengths=None, band_names=None, description='HERAMAN raster'):
    """Write a real-valued H x W x bands raster as ENVI BIP, without NumPy files."""
    import os
    path = Path(path).with_suffix('.hdr')
    array = np.asarray(values)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError('ENVI output must have shape H x W x bands.')
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    code = next((code for code, dtype in TYPES.items() if np.dtype(dtype).kind == array.dtype.kind and np.dtype(dtype).itemsize == array.dtype.itemsize), None)
    if code is None:
        raise ValueError(f'Unsupported ENVI output dtype: {array.dtype}')
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=float)
        if wavelengths.shape != (array.shape[2],) or not np.isfinite(wavelengths).all() or np.unique(wavelengths).size != wavelengths.size:
            raise ValueError('Invalid output wavelength axis.')
    if band_names is not None and len(band_names) != array.shape[2]:
        raise ValueError('One band name is required per output band.')
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = path.with_suffix('.raw')
    if path.resolve() == binary.resolve():
        raise ValueError('Header and binary must be distinct.')
    temporary = binary.with_suffix('.raw.tmp')
    try:
        with temporary.open('wb') as stream:
            for row in range(0, array.shape[0], 32):
                np.asarray(array[row:row+32], dtype=np.dtype(TYPES[code]).newbyteorder('<')).tofile(stream)
        os.replace(temporary, binary)
        clean = lambda value: re.sub(r'[{},\r\n]', ' ', str(value))
        fields = ['ENVI', 'description = {'+clean(description)+'}',
                  f'samples = {array.shape[1]}', f'lines = {array.shape[0]}', f'bands = {array.shape[2]}',
                  'header offset = 0', 'file type = ENVI Standard', f'data type = {code}',
                  'interleave = bip', 'byte order = 0', 'data file = '+binary.name]
        if wavelengths is not None:
            fields += ['wavelength units = Nanometers', 'wavelength = {'+', '.join(f'{v:.12g}' for v in wavelengths)+'}']
        if band_names is not None:
            fields += ['band names = {'+', '.join(clean(v) for v in band_names)+'}']
        header_temporary = path.with_suffix('.hdr.tmp')
        header_temporary.write_text('\n'.join(fields)+'\n', encoding='utf-8')
        os.replace(header_temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

"""Three reproducible 1024-square ENVI scenes; reruns reuse intact files."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from shared.envi_catalog import file_digest, register_envi_source

VERSION = 3
SCENES = ('scene_01_mosaic', 'scene_02_particles', 'scene_03_layers')


def _scene_fields(number, size, rng):
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)].astype(np.float32)
    if number == 0:
        sample = (x/.89)**2+(y/.82)**2 < 1
        labels = 1+((np.floor((x+1)*4)+np.floor((y+1)*4)*3).astype(int)%4)
    elif number == 1:
        labels = np.zeros((size,size), dtype=np.uint8)
        for index in range(35):
            cx, cy = rng.uniform(-.8,.8,2)
            radius = rng.uniform(.045,.19)
            labels[((x-cx)**2+(y-cy)**2)<radius**2] = 1+index%4
        sample = labels > 0
    else:
        radius = np.hypot(x/.88,y/.8)
        sample = radius < 1+.06*np.sin(np.arctan2(y,x)*7)
        labels = 1+(np.floor((y+.12*np.sin(x*9)+1)*5).astype(int)%4)
        sample &= ~(((x-.42)**2+(y+.28)**2)<.012)
    labels = np.where(sample,labels,0).astype(np.uint8)
    # Soft boundaries provide mixed pixels; labels remain the nominal material.
    weights = np.stack([gaussian_filter((labels==k).astype(np.float32), 1.6+number) for k in range(5)])
    texture = gaussian_filter(rng.normal(size=(size,size)).astype(np.float32), 2)
    texture /= max(float(texture.std()), 1e-6)
    light = .82+.14*x-.1*y+.05*np.sin(x*13+y*7)+.025*texture
    light *= 1-.25*np.exp(-((x+.15)**2/.035+(y-.3)**2/.35))
    slope = .035*np.sin(x*7-y*4)
    return labels, weights, light, slope


def _generate_scene(folder, number, size):
    folder.mkdir(parents=True, exist_ok=True)
    stem = SCENES[number]
    header, binary = folder/(stem+'.hdr'), folder/(stem+'.raw')
    truth, manifest = folder/'ground_truth.png', folder/'synthetic_manifest.json'
    config = {'version': VERSION, 'size': size, 'bands':121, 'scene': number}
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        old_truth = folder/'ground_truth.npy'
        old_config = dict(config, version=2)
        old_files = (header,binary,old_truth)
        if saved.get('config')==old_config and all(p.is_file() and file_digest(p)==saved.get('sha256',{}).get(p.name) for p in old_files):
            if not cv2.imwrite(str(truth),np.load(old_truth,allow_pickle=False)):
                raise OSError('Could not migrate the synthetic label image to PNG.')
            saved['config'] = config
            saved['sha256'] = {p.name:file_digest(p) for p in (header,binary,truth)}
            manifest.write_text(json.dumps(saved,indent=2),encoding='utf-8')
            old_truth.unlink()
        files = (header,binary,truth)
        if saved.get('config')==config and all(p.is_file() and file_digest(p)==saved.get('sha256',{}).get(p.name) for p in files):
            print(f'Reused {stem}: {size} x {size}', flush=True)
            return header
    rng = np.random.default_rng(421+number)
    wl = np.linspace(400,1000,121,dtype=np.float32)
    labels, weights, light, slope = _scene_fields(number,size,rng)
    gauss = lambda center,width: np.exp(-((wl-center)/width)**2)
    spectra = np.array([
        np.full(wl.size,.012),
        .20+.28*gauss(550,85)+.20*(wl-400)/600,
        .58-.24*gauss(675,38)-.20*gauss(965,44),
        .16+.48/(1+np.exp(-(wl-715)/30))-.09*gauss(925,35),
        .45+.16*gauss(465,32)-.25*gauss(580,65)+.08*gauss(830,55)],dtype=np.float32)
    spectra[1:] *= 1+(number-1)*.025
    mapped = np.memmap(binary, mode='w+', dtype='<f4', shape=(size,size,len(wl)))
    try:
        for row in range(0,size,16):
            stop = min(size,row+16)
            block = np.einsum('khw,kb->hwb',weights[:,row:stop],spectra,optimize=True)
            block *= light[row:stop,:,None]
            block += slope[row:stop,:,None]*((wl-700)/600)[None,None,:]*(labels[row:stop,:,None]>0)
            block += rng.normal(0,.004,block.shape).astype(np.float32)
            mapped[row:stop] = np.maximum(block,0)
        mapped.flush()
    finally:
        mapped._mmap.close()
        del mapped
    header.write_text('ENVI\ndescription = {SYNTHETIC '+stem+'; artificial materials, not measured HERA data}\n'
        +f'samples = {size}\nlines = {size}\nbands = 121\nheader offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bip\nbyte order = 0\ndata file = {binary.name}\nwavelength units = Nanometers\nwavelength = '
        +'{'+', '.join(f'{v:g}' for v in wl)+'}\n', encoding='ascii')
    if not cv2.imwrite(str(truth),labels):
        raise OSError('Could not save synthetic labels.')
    manifest.write_text(json.dumps({'config':config, 'classes':{'0':'background','1':'material_A','2':'material_B','3':'material_C','4':'material_D'},
        'variations':['spatial texture','illumination gradient','shadow','mixed boundaries','spectral noise','between-scene intensity variation'],
        'sha256':{p.name:file_digest(p) for p in (header,binary,truth)}},indent=2),encoding='utf-8')
    print(f'Created {stem}: {size} x {size}',flush=True)
    return header


def generate(root, *, size=1024, populate=True):
    root=Path(root)
    headers=[_generate_scene(root/name,number,size) for number,name in enumerate(SCENES)]
    if populate:
        for header in headers:
            print('GUI cube:', register_envi_source(header), flush=True)
    return headers


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'ENVI files'/'synthetic_data')
    args=parser.parse_args()
    generate(args.output)

"""Classify native ENVI sources. No cube conversion, acquisition or radiometric correction."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
from shared.image_io import read_image, write_image
import numpy as np
from shared.paths import PROJECT_ROOT, portable_project_path
from shared.cube_preview import save_rgb_from_cube
from shared.envi import load_spectral_cube, resolve_envi_header, resolve_envi_binary, read_envi, write_envi
from shared.corrected_layout import prepare_classification_folder
from shared.tissue_segmentation import segment_cube
from processing.pipeline.svm_classifier import unmix_from_cube


def classify_envi(args):
    source = resolve_envi_header(args.cube_path)
    print('[1/3] Reading ENVI source directly; original values preserved', flush=True)
    cube, wavelengths, metadata = read_envi(source)
    binary = resolve_envi_binary(source, metadata)
    key = hashlib.sha256(portable_project_path(source).casefold().encode()).hexdigest()[:12]
    output = Path(args.output_dir or PROJECT_ROOT/'results'/f'{source.stem}_{key}'/'classification').resolve()
    protected = (source, binary, Path(args.dataset_path).resolve(), Path(args.library_path).resolve(), Path(args.model_path).resolve())
    try:
        if any(path == output or output in path.parents for path in protected):
            raise ValueError('The classification folder cannot contain an input file.')
        result_dir = prepare_classification_folder(output, source)
        provenance = dict(cube_path=portable_project_path(source), binary_path=portable_project_path(binary), path_base="project", wavelengths=wavelengths.tolist(),
                          model_path=portable_project_path(args.model_path), normalization=args.norm,
                          selected_wavelengths=args.selected_wavelengths_nm, classification_mode=args.classification_mode or args.svm_type,
                          input_format='ENVI', input_values='unchanged', source_header=metadata)
        (result_dir/'classification_source.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        save_rgb_from_cube(cube,result_dir,wavelengths=wavelengths)
        print(f'[2/3] ENVI cube {cube.shape}; native resolution preserved',flush=True)
        mask = None
        if args.tissue_only:
            _, mask, threshold, fallback = segment_cube(cube,wavelengths,args)
            if not mask.any():
                raise ValueError('Empty sample mask; adjust the segmentation settings.')
            write_envi(result_dir/'tissue_mask.hdr',mask,band_names=['sample_mask'])
            write_image(str(result_dir/'background_mask.png'),mask.astype(np.uint8)*255)
            (result_dir/'tissue_mask_settings.json').write_text(json.dumps(dict(
                band_nm=args.segmentation_band_nm,method=args.segmentation_method,threshold=float(threshold),
                log_intensity=args.segmentation_log_intensity,erosion_px=args.tissue_mask_erosion_px,
                frame_fallback=bool(fallback)),indent=2),encoding='utf-8')
        print('[3/3] SVM classification from the ENVI-mapped array',flush=True)
        result = unmix_from_cube(cube=cube,lib_path=args.library_path,out_dir=result_dir,
            cube_wavelengths_nm=wavelengths,prediction_mask=mask,dataset_path=args.dataset_path,
            model_path=args.model_path,normalization_mode=args.norm,svm_type=args.classification_mode or args.svm_type,
            selected_wavelengths_nm=args.selected_wavelengths_nm,force_retrain=args.force_retrain,
            chunk_size=50_000 if (args.classification_mode or args.svm_type)=='rbf' else 250_000)
        print(f'[3/3] Classification complete -> {result_dir}',flush=True)
        for key,field in [('T_PNG_PATH','overlay_path'),('M_PNG_PATH','mixture_side_by_side_path'),('S_DATA_PATH','spectra_data_path')]:
            if result.get(field): print(f'{key} {Path(result[field]).resolve()}',flush=True)
        return result_dir
    finally:
        mapping = getattr(cube,'_mmap',None)
        if mapping is not None: mapping.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cube_path',required=True,help='ENVI header (.hdr) or associated binary')
    parser.add_argument('--output_dir')
    parser.add_argument('--norm',default='none')
    parser.add_argument('--svm_type',choices=['linear','rbf'],default='linear')
    parser.add_argument('--classification_mode',choices=['linear','rbf'])
    parser.add_argument('--dataset_path',default=str(PROJECT_ROOT/'roi_outputs/roi_dataset.npz'))
    parser.add_argument('--library_path',default=str(PROJECT_ROOT/'roi_outputs/roi_dataset_library.json'))
    parser.add_argument('--model_path',default=str(PROJECT_ROOT/'svm_models/roi_dataset_svm.joblib'))
    parser.add_argument('--selected_wavelengths_nm',nargs='*',type=float)
    parser.add_argument('--target_resolution',choices=['native'],default='native')
    parser.add_argument('--force_retrain',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--tissue_only',action='store_true',help='Segment the sample before classification')
    parser.add_argument('--segmentation_band_nm',type=float,default=720.)
    parser.add_argument('--segmentation_method',choices=['otsu','threshold'],default='otsu')
    parser.add_argument('--segmentation_threshold',type=float,default=10.)
    parser.add_argument('--segmentation_log_intensity',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--tissue_mask_erosion_px',type=int,default=5)
    args=parser.parse_args()
    if args.tissue_mask_erosion_px<0: parser.error('Erosion must be non-negative.')
    classify_envi(args)


if __name__=='__main__':
    main()

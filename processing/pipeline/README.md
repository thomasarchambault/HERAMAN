# HERAMAN processing pipeline

- `postprocessing_pipeline.py`: native ENVI reading, optional segmentation, classification, and ENVI outputs.
- `roi_selection_pipeline.py`: RGB generation and ROI selection; training datasets and libraries.
- `svm_classifier.py`: training and applying linear/RBF models, with optional Platt calibration.
- `fat_calibration.py`: dependency retained for reading legacy calibrators; the HERAMAN workflow uses generic calibration across all classes.

The launchers in `processing/` use the GUI's 64-bit Python environment. No hardware drivers are required.

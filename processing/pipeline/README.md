# Traitements HERAMAN

- `postprocessing_pipeline.py` : lecture ENVI native, segmentation optionnelle, classification et résultats ENVI.
- `roi_selection_pipeline.py` : génération RGB et sélection ROI ; jeux d'apprentissage et bibliothèques.
- `svm_classifier.py` : entraînement et application des modèles Linear/RBF, avec calibration Platt optionnelle.
- `fat_calibration.py` : dépendance conservée pour la lecture des anciens calibrateurs ; le parcours HERAMAN utilise la calibration générique de toutes les classes.

Les lanceurs de `processing/` utilisent l'environnement Python 64 bits du GUI. Aucun pilote matériel n'est requis.

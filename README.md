# HERAMAN

Application Python 64 bits pour explorer et classifier des cubes spectraux HERA déjà reconstruits et corrigés en amont. Aucun pilote de caméra ou filtre n'est nécessaire.

## Démarrage

Lancer `HERAMAN LAUNCHER.bat`, ou `python -m gui.main` depuis ce dossier. Pour créer l’environnement dédié : **INSTALL HERAMAN.bat**. Voir [INSTALLATION.md](INSTALLATION.md) pour installer, diagnostiquer et transférer le projet.

Par défaut, les traitements ROI/SVM et les contrôles de modèles utilisent le Python 64 bits qui exécute le GUI. `config/runtime_64.json` utilise `python64: null` pour cette sélection automatique ; un ancien chemin de venv absent revient à cet interpréteur actif. Un chemin existant dans la configuration ou la variable explicite `MSI_PYTHON64` peut sélectionner un autre Python, dont l'architecture 64 bits est vérifiée. Un override explicite inexistant produit une erreur claire. Le lanceur `tools/run_gui.py` suit la même règle.

## Parcours

- Visualisation : ouvrir un cube ENVI sans modèle, afficher une bande ou une composition RGB dans le viewer existant de Postprocessing. Le survol trace les spectres à droite ; le clic fige le spectre et sélectionne le pixel pour l'export CSV/NPZ. La classification ajoute les classes en overlay sur cette même image, avec opacité, zoom et déplacement. Aucun sous-onglet ou second display.
- ROI selection : créer un profil nommé par projet/type d'échantillon, définir ses classes et constituer son jeu d'entraînement. Chaque lancement génère automatiquement un RGB natif à partir des bandes les plus proches de 650/550/450 nm et l'affiche en premier dans la fenêtre de sélection. L'aperçu PNG et sa provenance sont conservés sous `roi_outputs/previews/<source>_<identifiant>/`, au même emplacement à chaque lancement. Aucun RGB préalable n'est nécessaire et les valeurs ENVI restent inchangées. Chaque profil conserve ses propres données et modèles.
- SVM Models : examiner et réentraîner les modèles Linear/RBF. L'option Enable Platt scaling calibre toutes les classes à partir de scores obtenus sur des cubes exclus de l'entraînement. Elle est également disponible dans ROI selection pour les nouveaux modèles. Un modèle enregistré conserve sa calibration ; modifier l'option nécessite de le réentraîner. Une calibration impossible est signalée explicitement.
- Segmentation Tool : segmentation d'intensité optionnelle de l'échantillon, avec aperçu sans classification. Le masque restreint les pixels à classifier, sans modifier les classes ou le jeu d'entraînement.

Les contrôles P(fat)/non-fat et les règles de classification spécifiques à la graisse sont retirés du parcours HERAMAN. Les nouveaux modèles utilisent les décisions SVM brutes ou l'argmax des probabilités calibrées pour toutes les classes. Les anciennes variantes MARGIN sont conservées sur disque mais doivent être réentraînées pour le protocole générique HERAMAN.

Les exports de pixels incluent les coordonnées (x, y), avec origine à zéro, l'axe spectral en nm et les valeurs importées sans transformation d'affichage. CSV : une ligne par pixel ; NPZ : `coordinates_xy`, `spectra`, `wavelengths`, `cube_path`, `coordinate_origin`.

## Import

ENVI BSQ/BIL/BIP, types numériques réels, little/big-endian et header offset. Sélectionner l'en-tête `.hdr`, avec son binaire à proximité. `data file` est respecté ; sinon un binaire non ambigu est recherché. L'axe wavelength et ses unités explicites sont obligatoires. Les micromètres sont convertis en nm et les bandes sont ordonnées par longueur d'onde.

Les fichiers ENVI sont le format natif des cubes : visualisation, ROI, segmentation et SVM lisent directement le `.raw` associé au `.hdr`, avec accès mémoire mappé en lecture seule. Sélectionner le `.hdr` ou son binaire `.raw`/`.img`/`.dat`/`.bin` ; le GUI mémorise uniquement le chemin de l'en-tête dans `config/envi_sources.json`. Aucune conversion NPY ou copie complète du cube n'est créée. Les entrées NPY/NPZ héritées de MARGIN ne sont plus proposées comme cubes.

Les valeurs restent inchangées : le contraste d'affichage ne modifie jamais les spectres. Les résultats sont séparés des sources dans `results/<source>_<identifiant>/classification/` : cartes de classes `abundances.hdr/.raw`, probabilités calibrées `class_probabilities.hdr/.raw` si Platt est actif, et masque d'échantillon `tissue_mask.hdr/.raw` si la segmentation est active. Les bandes de ces cartes représentent des classes, pas des longueurs d'onde. Les images PNG restent des aperçus.

Les NPZ de ROI, d'export de spectres et de diagnostics regroupent des tableaux et métadonnées ; les modèles utilisent Joblib. Ces fichiers ne sont pas des caches de cubes. `classification_source.json` relie chaque résultat au fichier ENVI d'origine. Garder sa paire `.hdr`/binaire disponible pour réafficher les spectres. Les anciennes cartes NPY doivent être recalculées depuis les sources ENVI.

## Données synthétiques

Exécuter `python tools/generate_synthetic_envi.py`. Trois scènes distinctes de **1024 × 1024 pixels, 121 bandes de 400 à 1000 nm**, sont générées et disponibles dans le sélecteur du GUI :

- `scene_01_mosaic` : mosaïque de quatre matériaux dans une forme elliptique.
- `scene_02_particles` : particules de tailles variées, contacts et superpositions.
- `scene_03_layers` : couches ondulées, contour irrégulier et trou dans l'échantillon.

Chaque scène se trouve dans `synthetic_data/<nom>/<nom>.hdr`, avec son binaire ENVI BIP float32, `ground_truth.png` (0 = fond, 1–4 = matériaux A–D) et un manifeste de génération. Les scènes comprennent textures spatiales, gradients d'éclairage, ombres, bruit spectral, mélanges aux frontières et variations entre scènes. Les classes ont une signification artificielle et commune aux trois scènes ; aucune signature ne représente un matériau réel certifié ou une mesure HERA.

Le générateur est déterministe et réutilise les fichiers intacts. Le relancer ou rouvrir un ENVI ne crée aucun doublon. Les tests utilisent de petites fixtures dans des dossiers de test séparés, sans remplir le sélecteur. Les anciens petits cubes et tous les caches NPY synthétiques ont été supprimés. Le nombre de 121 bandes est un choix de test, pas une contrainte du GUI.

## État et limites

Vérifiés sur données synthétiques : import ENVI, GUI fusionné sans matériel, bande/RGB, sélection et export des pixels, segmentation, entraînement et réutilisation des modèles Linear/RBF avec et sans Platt, refus explicite d’une calibration sans provenance suffisante. Ces vérifications ne constituent pas une validation scientifique sur des données réelles.

Tests automatisés : lecture BSQ/BIL/BIP, big-endian et offset ; extraction ROI depuis ENVI ; SVM Linear/RBF avec et sans Platt ; worker CLI ; résultat ENVI relu dans le GUI ; survol et export de pixels. À valider sur données réelles : interactions ROI manuelles, performances de classification sur grands cubes et véritable export NIREOS. Les champs bbl, data ignore value et reflectance scale factor sont conservés dans la provenance mais ne sont pas encore appliqués automatiquement : utiliser actuellement des cubes valides déjà corrigés et des bandes exploitables. Les anciens modèles MARGIN ne sont pas garantis compatibles avec HERA. Les noms de certains fichiers et arguments internes restent historiques pour la compatibilité des résultats existants.

La lecture ENVI est mappée directement sur le binaire ; l'import GUI est synchrone et certaines analyses existantes peuvent encore allouer le cube complet. La composition RGB utilise les bandes les plus proches de 650/550/450 nm ; les longueurs d'onde effectivement utilisées sont affichées. Les statistiques ROI intégrées à cette vue restent à compléter.

Voir `docs/NIREOS_ENVI.md` pour les sources. Les données et modèles locaux existants ont été conservés.

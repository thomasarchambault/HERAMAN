# Export NIREOS et fixtures ENVI

Sources consultées le 10 novembre 2026 :

- https://nireos.com/hera-software/ : HERA Acquisition App sauvegarde en ENVI ; HyperLAB importe des cubes ENVI et fournit analyse/export.
- https://nireos.com/faq/ : export ENVI décrit comme binaire plat avec en-tête ASCII, ou MATLAB ; jeux de données sur demande auprès de NIREOS.
- https://nireos.com/product/hera-vnir/ : acquisition, analyse et export standard ; calcul de réflectance dans HyperLAB.
- https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html : spécification des dimensions, types, interleave, offset, byte order, wavelength et unités.
- https://www.nv5geospatialsoftware.com/docs/enviimagefiles.html : stockage BSQ/BIL/BIP.

Aucun manuel officiel complet librement téléchargeable trouvé dans cette recherche. Une copie tierce d'un manuel a été repérée, mais n'a pas servi de spécification technique. Il reste donc à confirmer sur un vrai export NIREOS le nom du binaire, les champs présents, les unités et la nature scientifique des valeurs. Les fixtures générées sont des fichiers ENVI standards, pas une reproduction certifiée des sorties NIREOS. Aucune grille spectrale uniforme HERA n'est supposée par le lecteur.


## Implémentation native HERAMAN

Le lecteur utilise directement l'en-tête et le binaire ENVI (lecture seule, aucune conversion NPY). La même représentation H × W × bandes est partagée par la visualisation, les ROI et les SVM. Les images de classification et les masques sont écrits en ENVI BIP avec les noms de classes comme noms de bandes, sans inventer un axe spectral pour ces cartes. Les jeux ROI NPZ et les modèles Joblib restent des enregistrements d'apprentissage distincts des cubes. Les résultats référencent le `.hdr` source dans `classification_source.json` et sont écrits sous `results/`.

# Installer et transférer HERAMAN

HERAMAN cible **Windows x64**, avec **Python 3.10, 3.11 ou 3.12 x64**. L'environnement a été installé et testé ici avec Python 3.10. Les pilotes HERA/Hamamatsu/VariSpec et le logiciel NIREOS ne sont pas nécessaires pour analyser des exports ENVI.

## Première installation sur un PC

1. Installer Python x64 dans la plage ci-dessus s'il n'est pas déjà présent, avec les composants standard `venv` et `pip`.
2. Placer le dossier HERAMAN dans un emplacement où vous pouvez écrire, puis double-cliquer **INSTALL HERAMAN.bat**.
3. L'installateur détecte Python, crée **.venv-heraman**, installe les versions définies dans `tools/requirements-64bit.txt` et vérifie les imports ainsi que Qt. Internet est nécessaire pour télécharger les dépendances. Les certificats de confiance Windows sont utilisés pour pip ; la vérification TLS reste active.
4. Lancer **HERAMAN LAUNCHER.bat**. Il choisit l'environnement dédié ; les processus ROI/SVM utilisent le même Python.

Python n'est pas téléchargé/installé automatiquement par ce script. Aucune dépendance n'est installée dans le Python global.

## Transférer un projet

Copier le dossier HERAMAN **sans les environnements virtuels** `.venv-heraman`, `.venv64`, `.venv` ou `venv`. Recréer l'environnement sur chaque ordinateur avec l'installateur ; un environnement virtuel n'est pas un paquet portable.

Pour conserver vos données et travaux, copier aussi les dossiers utiles :

- `ENVI files/` : un sous-dossier par groupe d'acquisitions, dont `synthetic_data/` pour les données synthétiques et `acquisitions_reelles/` pour les mesures. Conserver les paires ENVI `.hdr` + binaire et le champ `data file` de l'en-tête. Les anciens dossiers `data/` et `synthetic_data/` restent accessibles s'ils existent.
- `roi_outputs/` : annotations, jeux d'apprentissage et bibliothèques.
- `svm_models/` : modèles et profils ; conserver les mêmes versions de bibliothèques pour leur relecture.
- `results/` : classifications et provenance.
- `config/envi_sources.json` : catalogue local facultatif.

Les nouveaux chemins internes du catalogue, de la provenance de classification et des jeux ROI sont relatifs au dossier HERAMAN. Le dossier peut changer de nom et d'emplacement. Les fichiers externes à HERAMAN restent des références absolues : les copier et les sélectionner de nouveau sur le nouveau PC. Un ancien résultat contenant un chemin absolu périmé peut devoir être recalculé depuis son ENVI. Les en-têtes ENVI externes utilisant eux-mêmes un `data file` absolu doivent être corrigés si cette adresse change.

Les gros fichiers de données et les modèles sont ignorés par Git : cloner le dépôt ne les transfère pas. Pour obtenir les trois scènes de démonstration après clonage, lancer :

```powershell
.\.venv-heraman\Scripts\python.exe tools\generate_synthetic_envi.py
```

## Vérifier ou réparer

Dans PowerShell, depuis HERAMAN :

```powershell
& '.\INSTALL HERAMAN.bat' -Check
& '.\INSTALL HERAMAN.bat' -Python 'D:\Python312\python.exe'
```

Pour recréer uniquement l'environnement, fermer HERAMAN puis lancer :

```powershell
& '.\INSTALL HERAMAN.bat' -Recreate
```

`-Recreate` supprime seulement le dossier `.venv-heraman` vérifié dans HERAMAN, puis réinstalle les dépendances. Les données, annotations et modèles ne sont pas supprimés.

Un réglage explicite `MSI_PYTHON64` ou `python64` dans `config/runtime_64.json` peut choisir un autre interpréteur. L'installation remet la configuration en mode automatique ; si elle remplace un chemin explicite, elle conserve une copie `runtime_64.json.before-install.json`. Une variable d'environnement explicite conserve la priorité sur le lanceur.

## Vérifications réalisées

Installation réelle dans un environnement isolé, diagnostic des dépendances et initialisation Qt ; copie d'un petit projet vers un dossier renommé avec espaces et accents, suppression de l'ancienne copie, puis relecture du catalogue, du modèle et de l'overlay ENVI. Cela vérifie la relocalisation sur ce poste ; aucun second ordinateur physique n'a été utilisé.

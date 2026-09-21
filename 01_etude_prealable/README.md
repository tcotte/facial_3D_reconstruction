# Scripts d'analyse — reconstruction 3D VISIA-CR

Scripts Python utilisés pour l'étude de faisabilité. Chaque script est
autonome, documenté en tête de fichier, et rappelle les résultats obtenus
ainsi que les réserves associées.

## Installation

```
pip install opencv-python numpy
```

Aucun GPU requis. Testé avec OpenCV 4.13.

## Configuration

Éditer `IMG_DIR` en tête de `visia_lib.py` :

```python
IMG_DIR = r"C:\donnees\visia\D0"
```

Les scripts attendent la convention de nommage VISIA :
`<sujet>_<angle>_<modalite>.jpg`, par exemple
`alban_D0_Left_Oblique_Cross-Polarized.jpg`.

## Ordre d'exécution

| Script | Objet | Durée |
|---|---|---|
| `01_check_integrity.py` | Détecte les exports blancs (bug VISIA) | secondes |
| `02_modality_benchmark.py` | Classe les 4 modalités d'éclairage | 2 min / +10 min avec `--asift` |
| `03_stability.py` | Mouvement du sujet entre prises | ~1 min |
| `04_coverage_map.py` | Cartographie de la couverture faciale | ~10 min |
| `05_chart_calibration.py` | Échelle métrique par la charte | ~1 min |
| `06_reconstruction.py` | Triangulation 3D + résolution de focale | ~10 min |

`visia_lib.py` regroupe les fonctions communes (E/S, ASIFT, appariement,
filtrage géométrique, photométrie). Il n'est pas exécutable directement.

**Lancer `01_check_integrity.py` en premier, systématiquement.** Six fichiers
sur douze sont arrivés blancs lors de la collecte initiale, sans message
d'erreur.

## Points de vigilance

**Métrique « sur la peau ».** La charte et l'appui frontal sont rigides et mats :
ils s'apparient très bien et gonflent artificiellement les comptages globaux.
Seuls les appariements situés sur la peau renseignent sur la faisabilité. C'est
le rôle de `skin_mask()`, et ce filtrage change les conclusions du tout au tout
(par exemple : le parallèle-polarisé produit ~260 inliers totaux mais 6 sur la peau).

**Erreur d'échelle et volume.** Un volume varie comme le cube d'une longueur :
une erreur d'échelle linéaire de x % induit environ 3x % d'erreur de volume.
Les constantes `PITCH_X` / `PITCH_Y` de `05_chart_calibration.py` sont donc
critiques et doivent être confirmées (idéalement par la fiche technique
Calibrite plutôt que par mesure manuelle).

**ASIFT est un substitut.** Il a été retenu pour sa disponibilité sous OpenCV
seul. L'état de l'art (LoFTR, RoMa, SuperPoint+LightGlue via hloc) fait
généralement mieux sur les grandes bases angulaires. Les chiffres produits par
ces scripts sont donc des **planchers de performance**.

**Verrou ouvert.** `06_reconstruction.py` produit deux focales incompatibles à
28 % entre les paires gauche et droite. Tant que cet écart n'est pas expliqué,
aucune mesure volumétrique n'est défendable. L'étape suivante est un ajustement
de faisceaux conjoint sur les trois vues, avec focale unique commune, angles
imposés et point principal libre.

## Sorties produites

- `couverture_frontal.png` — appariements projetés sur la vue frontale
  (vert : frontal ↔ oblique gauche ; orange : frontal ↔ oblique droit)
- `reconstruction_profondeur.png` — points 3D colorés par profondeur
  (bleu : proche ; rouge : éloigné)

## Références

- Lowe, D. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*, 60(2).
- Morel, J.-M. & Yu, G. (2009). ASIFT: A new framework for fully affine invariant image comparison. *SIAM J. Imaging Sciences*, 2(2).
- Fischler, M. & Bolles, R. (1981). Random Sample Consensus. *Comm. ACM*, 24(6).
- Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE TPAMI*, 22(11).
- Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2e éd.

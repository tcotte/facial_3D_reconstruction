# Reconstruction 3D du visage à partir d'images VISIA-CR — scripts Python

Ensemble des algorithmes développés au cours de l'étude de faisabilité, organisés
par finalité. Chaque script est autonome, documenté en tête de fichier, et
rappelle **les résultats obtenus ainsi que les réserves associées** — de sorte
qu'un lecteur reprenant le code sache non seulement ce qu'il fait, mais ce qu'il
a montré.

---

## Avertissement à lire avant toute utilisation

**La focale est vérifiée** : objectif 50 mm à focale fixe, f/13, sur Canon R5
avec recadrage (pas de rééchantillonnage). Soit `FOCAL_REL = 1,42222 × côté long`
en pixels, et une distorsion nulle. La reconstruction est donc **euclidienne en
forme**.

**Mais le facteur d'échelle global reste inconnu.** La tentative de le récupérer
par la charte ColorChecker a échoué (écart de 57 % entre les deux paires stéréo,
voir `02_calibration/metric_scale_chart.py`).

> **Aucune valeur ne peut être exprimée en mm ou mm³** tant que la distance de
> travail n'a pas été mesurée.

Ce qui reste valide : les **variations relatives** entre deux sessions,
exprimées comme rapport de deux grandeurs homogènes issues de la même
reconstruction. Cette robustesse a été validée par simulation puis **confirmée
expérimentalement** : le retraitement complet avec la focale vérifiée (−9 % sur
la focale, distorsion ramenée de −0,20 à 0) n'a déplacé les indicateurs que de
moins de 5 %.

---

## Organisation

| Dossier | Finalité | Quand l'utiliser |
|---|---|---|
| `00_pipeline/` | **Production** : reconstruction et comparaison D0/Dx | usage courant |
| `01_etude_prealable/` | Diagnostics de faisabilité | nouveau dispositif, nouveau protocole |
| `02_calibration/` | Estimation et diagnostic des paramètres optiques | une fois, puis si le banc change |
| `03_appariement/` | Moteurs d'appariement | mise au point, comparaison de méthodes |
| `04_validation/` | Validation méthodologique | avant toute revendication de résultat |

---

## Installation

```bash
pip install opencv-python numpy scipy
pip install torch kornia                              # appariement appris (recommandé)
pip install git+https://github.com/Parskatt/RoMa.git  # optionnel, densification
```

Sans `kornia`, le pipeline bascule automatiquement sur ASIFT **et le signale
explicitement** — jamais de dégradation silencieuse.

GPU non obligatoire, mais fortement recommandé : les scripts détectent CUDA
automatiquement. Sur CPU, compter quelques minutes par paire d'images.

---

## `00_pipeline/` — chaîne de production

Reconstruction d'une session et comparaison entre deux sessions.

| Script | Rôle |
|---|---|
| `config.py` | **constantes gelées** et versionnées (`CALIB_VERSION`) |
| `face_mask.py` | **masque facial robuste** au cadrage et au vêtement |
| `fuse_modalities.py` | **fusion des quatre modalités** d'éclairage |
| `outlier_filter.py` | **rejet des points aberrants** (4 critères) |
| `visia_core.py` | E/S, appariement, densification, géométrie |
| `visia_compare.py` | recalage, métriques adimensionnelles, test à blanc |
| `run_calibrate.py` | estime et **gèle** la géométrie du banc (une fois pour l'étude) |
| `run_session.py` | reconstruit une session |
| `run_compare.py` | compare deux sessions |
| `make_masks.py` | tracé interactif de la ROI et de la zone stable |
| `selftest.py` | validation sur déformation simulée |

```bash
python run_calibrate.py alban D0 --out rig.json   # une seule fois
python make_masks.py alban D0                     # ROI + zone stable
python run_session.py alban D0 --rig rig.json
python run_session.py alban Dx --rig rig.json
python selftest.py alban_D0.npz                   # avant toute vraie mesure
python run_compare.py alban_D0.npz alban_Dx.npz \
    --subject alban --ref-session D0 --test-session Dx \
    --roi roi.png --stable stable.png
```

**Quatre conditions de validité**, sans lesquelles les résultats ne valent rien :

1. **Constantes gelées** — `config.py` ne doit pas être modifié en cours
   d'étude. Toute modification invalide rétroactivement les comparaisons déjà
   produites.
2. **Rig gelé** — les poses de caméra sont une propriété du banc, pas du sujet.
3. **Recalage sur zone stable**, jamais sur la ROI évaluée.
4. **Test à blanc** — toute variation inférieure au plancher de bruit doit être
   déclarée non significative.

---

## `01_etude_prealable/` — diagnostics de faisabilité

À exécuter face à un nouveau dispositif, avant d'investir dans un pipeline.

| Script | Objet | Résultat obtenu |
|---|---|---|
| `01_check_integrity.py` | détecte les exports blancs | 6 fichiers sur 12 lors de la collecte initiale |
| `02_modality_benchmark.py` | classe les modalités d'éclairage | standard retenu ; PP et raked exclus |
| `03_stability.py` | mouvement du sujet entre prises | 0,06 % du champ (intra-session) |
| `04_coverage_map.py` | couverture faciale exploitable | cartographie par cellules |
| `05_chart_calibration.py` | échelle métrique par la charte | non concluant en oblique |
| `06_reconstruction.py` | triangulation, résolution de focale | écart de 28 % entre paires |

**Lancer `01_check_integrity.py` en premier, systématiquement.**

**Métrique clé — « sur la peau ».** La charte et les appuis mécaniques sont
rigides et mats : ils s'apparient très bien et gonflent artificiellement les
comptages. Le parallèle-polarisé produit ~260 inliers au total mais **6
seulement sur le visage**. Sans ce filtrage, les conclusions s'inversent.

---

## `02_calibration/` — paramètres optiques

| Script | Objet | Résultat obtenu |
|---|---|---|
| `sensor_check.py` | cohérence focale ↔ capteur | hypothèse 24 mm **exclue** ; optique estimée ≈ 56 mm (vraie : 50 mm) |
| `focal_sweep.py` | identifiabilité de la focale | écart de 16 % entre deux critères ; le critère « données » était juste à 10 % |
| `bundle_adjustment.py` | ajustement conjoint 3 vues | reprojection 0,40–0,75 px ; point principal centré |
| `distortion_search.py` | la distorsion explique-t-elle l'écart ? | **non** ; la vraie distorsion est nulle |
| `metric_scale_chart.py` | échelle métrique par la charte | **échec** : 57 % d'écart entre paires |

**Ordre recommandé** : `focal_sweep` → `sensor_check` → `bundle_adjustment` →
`distortion_search`.

**Piège documenté dans `bundle_adjustment.py`** : sans recalage préalable de
l'échelle entre les deux branches stéréo, l'optimisation stagne à 23,9 px de
résidu. Après recalage (facteur 0,9497), l'écart 3D tombe à 0,14 % et
l'optimisation converge.

**Épilogue de cette famille.** La focale a finalement été obtenue auprès du
fournisseur : **50 mm f/13**. Confrontation aux estimations :

| Critère | Estimation | Écart à la vérité |
|---|---|---|
| Minimum de reprojection (données) | 6250 px | **+9,9 %** |
| Critère angulaire 45°+45° (mécanique) | 7250 px | +27,4 % |
| Valeur vraie (50 mm) | 5689 px @4000 | — |

Le critère fondé sur les données était le bon. L'hypothèse mécanique des 45° se
trompait de 27 % — et les angles réels mesurés valent 32° à 38° selon la
configuration, écart non expliqué à ce jour.

**Reste ouvert** : le facteur d'échelle global. La voie la plus simple est de
**mesurer la distance de travail** (objectif → plan du visage) ; avec la focale
connue, cette seule mesure suffit.

---

## `03_appariement/` — moteurs d'appariement

| Script | Objet |
|---|---|
| `tiled_matching.py` | DISK + LightGlue par tuiles (recommandé) |
| `roma_matching.py` | RoMa, modes `sample` et `grid` |

**Progression mesurée** (appariements sur la peau, F↔L / F↔R) :

| Moteur | F↔L | F↔R |
|---|---|---|
| SIFT | 17 | 26 |
| ASIFT | 435 | 206 |
| DISK + LightGlue (1280 px) | 768 | 779 |
| **DISK + LightGlue par tuiles (4000 px)** | **3616** | **3770** |
| tiny RoMa par tuiles | 9873 | 13 484 |

Le facteur limitant de toute la chaîne n'était ni les images ni l'éclairage,
mais la capacité du moteur à absorber 45° de rotation : **facteur 140 à 210**
entre SIFT et l'appariement appris par tuiles.

**Répartition des rôles retenue :**

- **LightGlue pour la calibration** — points répétables, chaînables entre les
  deux paires, précis au sous-pixel.
- **RoMa pour la densification** une fois le rig fixé — plus de points, 5 à 6
  fois plus rapide, meilleure couverture.

RoMa en mode `sample` détruit le chaînage 3 vues (50 pistes contre 272–580) car
deux appels indépendants ne tirent pas les mêmes points frontaux. Le mode `grid`
restaure le chaînage mais le filtrage épipolaire rejette 93 % des pistes : la
précision de localisation d'un champ dense reste inférieure à celle d'un
détecteur de points.

---

## `04_validation/` — validation méthodologique

| Script | Objet | Résultat obtenu |
|---|---|---|
| `validate_relative_metric.py` | robustesse de la mesure relative | biais **< 3 %** pour ±20 % d'erreur de focale |
| `pairwise_comparison.py` | comparaison D0/Dx par paire stéréo | corrige un biais antisymétrique |
| `intersession_link.py` | **liaison entre sessions** par tuiles | 1 760 → **30 970** appariements |
| `volume_map.py` | **carte de variation de volume** | seuil de détection 1,13 % |
| `export_pointcloud.py` | export PLY coloré d'une session | ~780 000 points par session |
| `benchmark.py` | **temps de calcul et mémoire** par étape | ≈ 10 min par session sur 1 cœur |

### La formulation du résultat détermine sa validité

| Erreur de focale | A : déformation / distance de travail | B : déformation / relief de la ROI |
|---|---|---|
| −20 % | **+47 %** | **+2,9 %** |
| +20 % | **−27 %** | **−2,1 %** |

Numérateur et dénominateur étant deux grandeurs de même nature issues de la même
reconstruction, la distorsion se simplifie. **Le mécanisme ne fonctionne pas avec
une référence externe** (distance de travail, mesure anthropométrique).

### Pourquoi une chaîne de mesure par paire stéréo

Chaque joue n'est reconstruite que par **une seule** paire stéréo — conséquence
de l'absence de recouvrement entre les deux obliques. Fusionner les deux nuages
avant recalage injecte le désaccord résiduel entre paires sous forme de **biais
antisymétrique**, du même ordre que le signal recherché :

| | Avant correction | Après correction |
|---|---|---|
| Joue côté image droite | +0,118 | **+0,181** |
| Joue côté image gauche | −0,061 | **+0,146** |
| | *signes opposés* | *même signe, écart 21 %* |

---

## Limites connues

- **Couverture faciale partielle** (50–60 % en haute confiance) ; nez, paupières
  et cavités orbitaires souvent absents.
- **Pas de recouvrement significatif entre les deux obliques** : chaque
  demi-visage repose sur une seule paire stéréo, sans contrôle croisé.
- **Profondeur mal conditionnée** : l'étendue du visage représente 3–4 % de la
  distance de prise de vue — c'est l'axe où se mesure un gonflement.
- **Repositionnement inter-session** : 0,28 % du champ, soit cinq fois la
  stabilité intra-session. C'est le facteur limitant principal, et il n'est
  atténué par aucune normalisation.
- **Marge de détection** : sur un gonflement volontaire des joues, le signal
  atteint au mieux 1,16 fois le plancher de bruit. Le dispositif ne mesurera pas
  plus fin.
- **Dépendance au type de peau** : couverture et bruit varient d'un **facteur 2**
  entre les deux sujets testés. La sensibilité n'est pas une constante
  instrumentale et doit être établie par sujet ou par strate de phototype.
- **Chaînes spatialement disjointes** : les deux paires stéréo ne se recouvrent
  que sur 1,9 % du visage. Comparer leurs signes n'a pas valeur de contrôle
  croisé.
- **Échelle métrique non établie** : la forme est correcte, la taille ne l'est
  pas. Mesurer la distance de travail lèverait ce point.

---

## Améliorations du protocole, par ordre d'impact

1. **Vues intermédiaires (±22°)** — supprimerait l'absence de recouvrement entre
   obliques et le biais antisymétrique qui en découle.
2. **Projection d'un motif de texture aléatoire (speckle)** — solution retenue
   par les systèmes de référence pour l'appariement sur peau peu texturée.
3. **Mire de calibration** placée à la position du visage.
4. **Capture simultanée multi-appareils** — lèverait le risque non-rigide.


---

## Temps de calcul

Mesures sur **Intel Xeon 2,10 GHz — 1 cœur, 4 Go de RAM, sans GPU**
(PyTorch et OpenCV limités à 1 thread). Voir `04_validation/benchmark.py`.

| Étape | Temps | Pic mémoire |
|---|---|---|
| Appariement pleine image 1280 px (1 paire) | **68,9 s** | **3 334 Mo** |
| Appariement par tuiles (1 paire, ≈14 tuiles) | ≈ 93 s | 3 334 Mo |
| Densification (1 paire) | 1,9 s | 395 Mo |
| Triangulation de 400 000 points | 11,8 s | 300 Mo |
| Fusion et filtrage (1 session) | 7,2 s | 300 Mo |
| Liaison inter-session (1 modalité) | ≈ 200 s | 3 300 Mo |
| Cartes qualité / redondance / variation | 1,6 s | 300 Mo |
| **Reconstruction d'une session (4 modalités)** | **≈ 10 min** | |
| **Comparaison complète (2 sessions)** | **≈ 28 min** | |

**L'appariement représente 90 % du temps** et c'est le seul poste qui sature la
mémoire — ce qui a imposé le tuilage. Sur GPU, DISK et LightGlue sont
typiquement 20 à 50 fois plus rapides : une session passerait sous la minute.
Les quatre modalités et les deux paires étant indépendantes, une parallélisation
sur 8 cœurs diviserait aussi le temps par 8 sans GPU.

---

## Licences des composants

Relevé de septembre 2026. **Une validation juridique reste nécessaire avant tout
usage commercial** : les licences des poids pré-entraînés sont moins bien
documentées que celles du code et changent d'une version à l'autre.

| Composant | Version | Licence |
|---|---|---|
| OpenCV | 4.13.0 | Apache 2.0 |
| NumPy | 2.4.4 | BSD 3-clauses |
| SciPy | 1.17.1 | BSD 3-clauses |
| PyTorch | 2.14.0 | BSD 3-clauses |
| Kornia | 0.8.3 | Apache 2.0 |
| LightGlue (code) | — | Apache 2.0 |
| DISK (poids) | — | MIT (fiche `kornia/disk`) |

**Point d'attention.** LightGlue est distribué avec plusieurs jeux de poids.
Ceux entraînés sur **SuperPoint** héritent de la licence Magic Leap, **non
commerciale**. La chaîne retenue ici utilise **DISK + LightGlue**, la
combinaison signalée comme commercialement permissive. Ce choix doit être
maintenu explicitement : basculer sur SuperPoint changerait le régime juridique.

**RoMa** (évalué, non retenu) : code MIT, DINOv2 en Apache 2.0 *depuis 2024*
seulement — les versions antérieures étaient explicitement non commerciales.
XFeat, base de `tiny_roma`, est en Apache 2.0.

Aucun composant à copyleft fort (GPL, AGPL) sur le périmètre retenu.

---

## Références

- Lowe, D. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*, 60(2).
- Morel, J.-M. & Yu, G. (2009). ASIFT. *SIAM J. Imaging Sciences*, 2(2).
- Fischler, M. & Bolles, R. (1981). Random Sample Consensus. *Comm. ACM*, 24(6).
- Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE TPAMI*, 22(11).
- Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2e éd.
- Lindenberger, P. et al. (2023). LightGlue. *ICCV*.
- Edstedt, J. et al. (2024). RoMa: Robust dense feature matching. *CVPR*.
- Umeyama, S. (1991). Least-squares estimation of transformation parameters. *IEEE TPAMI*, 13(4).

*Références à vérifier avant citation formelle dans un rapport d'étude.*

# RoMa v2 — extraction sur machine GPU et intégration au pipeline

Chaîne en trois temps : **exporter** le champ dense sur la machine H100,
**télécharger** un fichier unique, **relire** ce fichier dans le pipeline à la
place de l'étape de densification. Le reste de la chaîne — rig gelé, focale,
seuils, triangulation, filtrage — n'est pas touché.

```
  machine H100                     fichier .npz                 poste de travail
  ────────────                     ────────────                 ────────────────
  roma_v2_export.py   ──────────►  roma2_<sujet>_<session>  ──►  run_session.py --roma
  (romav2 + images)                _<modalité>.npz               (roma_field.py)
```

---

## 1. Ce qui est extrait, et pourquoi ce format

Le champ RoMa est lu **aux pixels de la vue frontale**, pas échantillonné
aléatoirement. C'est le point décisif : en interrogeant les **mêmes** points
frontaux pour F↔L et pour F↔R, le chaînage 3 vues est conservé par
construction. C'est exactement ce que le mode `sample` de RoMa v1 détruisait
(50 pistes contre 272–580, cf. `roma_matching.py`).

| Champ | Contenu | À quoi ça sert |
|---|---|---|
| `idx` | index du pixel frontal | référentiel commun aux deux paires |
| `<L\|R>_xy` | position correspondante dans l'oblique, px OpenCV | la donnée utile |
| `<L\|R>_cert` | *overlap* RoMa, dans [0, 1] | seuillage |
| `<L\|R>_sigma` | écart-type de localisation, **en pixels**, sur les deux axes propres | **nouveau en v2** |
| `<L\|R>_theta` | orientation du grand axe | idem |
| `<L\|R>_spread` | désaccord entre tuiles voyant le même pixel | estimateur d'erreur **indépendant** |
| `<L\|R>_src` | 0 = passe pleine image, 1 = tuile | traçabilité |
| `meta` | JSON : modèle, réglage, empreintes des jpg, GPU, date | contrôle de compatibilité |

**`sigma` est la nouveauté qui compte.** RoMa v2 prédit une matrice de
précision par pixel : une incertitude de localisation **anisotrope et
calibrée**, là où le pipeline ne dispose aujourd'hui que d'un seuil binaire sur
le résidu épipolaire. Elle permet de pondérer la triangulation et de distinguer
un point mal contraint d'un point faux.

Le **résidu épipolaire n'est pas calculé sur la machine GPU** : il doit l'être
avec le rig gelé, côté pipeline, pour rester une validation géométrique
indépendante de l'appariement.

---

## 2. Sur la machine H100

```bash
pip install romav2 opencv-python-headless numpy torch   # poids ~1 Go au 1er appel
```

Deux scripts, même format de sortie :

| Script | Quand |
|---|---|
| `roma_v2_simple.py` | **commencer par là.** Un seul `model.match()` par paire, pas de tuilage, quelques secondes |
| `roma_v2_export.py` | si la couverture est insuffisante : passe grossière puis tuiles guidées, résolution native |

```bash
# version simple
python roma_v2_simple.py --img-dir /data/visia --subject alban --session D0

# version tuilée
python roma_v2_export.py --subject alban --session D0 \
    --img-dir /data/visia --out roma2_alban_D0_Standard_1.npz
```

`roma_v2_simple.py` accepte aussi des chemins explicites, hors convention de
nommage VISIA :

```bash
python roma_v2_simple.py --front F.jpg --right R.jpg --left L.jpg \
    --subject alban --session D0 --out roma2_alban_D0.npz
```

Une seule vue oblique suffit : `run_session.py` reconstruit alors la paire
présente et signale l'autre comme absente.

Options utiles :

| Option | Défaut | Effet |
|---|---|---|
| `--mode tiled\|full` | `tiled` | `full` = une passe pleine image (secondes) ; `tiled` = passe grossière puis tuiles guidées, à la résolution native |
| `--step` | `2` | pas de la grille frontale exportée. `1` quadruple le volume |
| `--setting` | `precise` | `precise` = 1280², bidirectionnel. `base`/`fast`/`turbo` plus rapides |
| `--cert-min` | `0.02` | seuil d'export (le seuil de travail est appliqué côté pipeline) |
| `--long` | `4000` | **doit valoir `config.WORK_LONG`**, sinon les coordonnées ne correspondent pas |

Pour les quatre modalités (nécessaire à `fuse_modalities.py`) :

```bash
for m in Standard_1 Cross-Polarized Raked Parallel-Polarized; do
  python roma_v2_export.py --subject alban --session D0 --modality $m \
      --out roma2_alban_D0_$m.npz
done
```

Ordre de grandeur : ~30 Mo par session et par modalité à `--step 2`,
~110 Mo à `--step 1`.

### Pourquoi le tuilage reste utile malgré le H100

Ce n'est pas une question de mémoire mais de **résolution du champ**. RoMa
redimensionne les images à un carré fixe — 1280×1280 en réglage `precise`. Sur
une image de travail à 4000 px de côté long, cela sous-échantillonne d'un
facteur ≈ 3 : le champ ne peut pas localiser plus finement que sa propre
grille. Le tuilage traite chaque tuile de 1024 px à la pleine résolution du
modèle et restitue la précision native.

Si `--mode full` suffit pour votre usage, le fichier est produit en quelques
secondes — comparez les deux avec `roma_v2_compare.py` avant de trancher.

---

## 3. Sur le poste de travail

```bash
# inspecter le fichier téléchargé
python 00_pipeline/roma_field.py --info roma2_alban_D0_Standard_1.npz

# reconstruire la session avec ce champ
python 00_pipeline/run_session.py alban D0 --rig rig.json --step 1 \
    --roma roma2_alban_D0_Standard_1.npz
```

`--step 1` : le sous-échantillonnage a déjà été fait à l'export ; le
réappliquer déciderait deux fois.

Options de `run_session.py` propres à RoMa :

| Option | Défaut | Effet |
|---|---|---|
| `--roma-cert` | `0.05` | seuil sur l'*overlap* |
| `--roma-sigma-max` | inactif | rejet sur l'écart-type prédit (px) |
| `--roma-spread-max` | inactif | rejet sur le désaccord inter-tuiles (px) |
| `--roma-fill` | désactivé | interpole le champ entre les points exportés |
| `--epi-from rig\|field` | `rig` | origine de la matrice fondamentale |

### `--epi-from rig` : le résidu redevient une vraie validation

Par défaut la matrice fondamentale est **dérivée du rig gelé**,
`F = K⁻ᵀ [t]ₓ R K⁻¹`. Elle ne doit donc rien à RoMa.

C'est plus strict que la chaîne actuelle, qui ré-estime `F` à chaque session
sur sa propre passe grossière : le résidu y est en partie auto-validé. Un
résidu élevé signale alors réellement quelque chose — dérive du banc, ou
mouvement du sujet entre les trois prises.

### `--roma-fill` : à manier avec précaution

Sans cette option, les points reconstruits sont **exactement** ceux qui ont été
appariés. Avec, le champ est interpolé sur tous les pixels du masque : c'est
défendable — le champ RoMa est continu et produit en interne à 1280 px — mais
cela **gonfle le nombre de points sans ajouter d'information indépendante**. Ne
jamais s'en servir pour annoncer un nombre d'appariements.

---

## 4. Avant de basculer la production : mesurer

```bash
python 03_appariement/roma_v2_compare.py --subject alban --session D0 \
    --rig rig.json --roma roma2_alban_D0_Standard_1.npz
```

Compare RoMa v2 et la densification actuelle **avec la même matrice
fondamentale** (celle du rig) — comparer chaque méthode à sa propre `F`
ré-estimée avantagerait mécaniquement celle qui a servi à l'estimer.

Quatre critères : couverture, résidu épipolaire, **chaînage 3 vues**, et accord
entre les deux méthodes. Ce dernier point mérite attention : un désaccord
important *sans* différence de résidu épipolaire signale un glissement **le
long de la droite épipolaire**, invisible au contrôle épipolaire et pourtant
directement converti en erreur de profondeur.

Puis, dans l'ordre : `selftest.py` sur déformation simulée, et un test à blanc
D0/D0. Tant que ces deux-là ne sont pas passés, le gain n'est pas établi.

---

## 5. Garde-fous

**Stamp de version.** Une session reconstruite avec RoMa porte
`CALIB_VERSION + '+roma2'`. `run_compare.py` refuse de comparer deux sessions
dont les stamps diffèrent. Changer la densification change les valeurs : c'est
le même principe que le gel de `config.py`.

**Contrôles au chargement**, qui lèvent une erreur : côté long différent de
`config.WORK_LONG`, sujet ou session ne correspondant pas, taille de l'image
frontale différente, taille des jpg sources différente de celle vue sur la
machine GPU.

**Tests sans GPU**, à rejouer après toute modification :

```bash
python 04_validation/test_roma_coords.py   # conventions de coordonnées
python 04_validation/test_roma_field.py    # chaîne complète, géométrie synthétique
```

---

## 6. Deux pièges corrigés au passage

**Lecture du champ au plus proche voisin.** L'ancien `roma_matching.py` en mode
`grid` arrondissait la position à l'entier de la grille du champ
(`np.round(u).astype(int)`). Sur un champ 1280² lu depuis une image de travail
à 4000 px, cela introduit une erreur **médiane de 1,42 px** et de 2,88 px au
95ᵉ centile — pour un seuil `EPI_HIGH` de **1,0 px**. C'est une explication
plausible des 93 % de rejet épipolaire constatés à l'époque, et le motif du
verdict « RoMa inférieur pour la calibration ». La lecture est désormais
bilinéaire. *Le verdict mériterait d'être réexaminé sur cette base* — mais
c'est une autre étude, et le rig reste gelé en attendant.

**Extrapolation au bord du champ.** Avec `align_corners=False`, le champ n'est
défini que pour |n| ≤ 1 − 1/N. Au-delà, `grid_sample` renvoie une valeur
extrapolée ou nulle, qui se traduit par une correspondance **plausible et
totalement fausse** (jusqu'à 680 px d'erreur, vers le centre de l'image). Ces
points sont désormais invalidés explicitement, comme le fait RoMa dans son
propre `sample()`. En mode tuilé, où les bords sont nombreux, l'oubli serait
coûteux.

---

## 7. Licence — point ouvert

RoMa v2 est en **MIT**, mais son descripteur est **DINOv3**, sous licence
propre à Meta — pas Apache 2.0, contrairement à DINOv2 depuis 2024. Le README
principal classe déjà RoMa comme « évalué, non retenu » pour des raisons de
licence : l'arrivée de DINOv3 ne lève pas ce point, **elle le déplace**.

Une validation juridique est nécessaire avant tout usage commercial. Tant
qu'elle n'est pas faite, la position prudente est celle du README : RoMa pour
la mise au point et la mesure de faisabilité, DISK + LightGlue pour ce qui doit
être livrable.

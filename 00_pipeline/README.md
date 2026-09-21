# Pipeline de comparaison 3D VISIA-CR — D0 / Dx

Reconstruction 3D partielle du visage a partir des 3 vues VISIA-CR, et
comparaison entre deux sessions d'acquisition.

## Nature des resultats — a lire avant toute utilisation

Les parametres optiques du banc (focale, angles) ne sont pas connus : ils sont
**estimes a partir des images**. La reconstruction n'est donc **pas metrique**.

**Aucune valeur ne peut etre exprimee en mm ou mm³.**

Ce qui reste valide : les **variations relatives** entre deux sessions,
exprimees comme rapport de deux grandeurs homogenes issues de la meme
reconstruction. Verification par simulation :

| Formulation | Biais pour ±20 % d'erreur de focale |
|---|---|
| deformation / distance de travail | ±27 % |
| **deformation / relief propre de la ROI** | **±3 %** |

C'est la seconde qui est implementee (`ratio_relief_moyen`,
`variation_volume_relative`).

## Conditions de validite

1. **Constantes gelees** — `config.py` ne doit pas etre modifie en cours
   d'etude. Toute modification invalide retroactivement les comparaisons deja
   produites. Le champ `CALIB_VERSION` est verifie a chaque chargement.
2. **Rig gele** — les poses de camera sont estimees une fois
   (`run_calibrate.py`) et reutilisees pour toutes les sessions.
3. **Recalage sur zone stable** — jamais sur la ROI evaluee.
4. **Verification par test a blanc** — toute variation inferieure au plancher
   de bruit doit etre declaree non significative.

## Installation

```
pip install opencv-python numpy scipy
pip install torch kornia          # fortement recommande (x5 sur l'appariement)
```

Sans `kornia`, le pipeline bascule sur ASIFT et le signale explicitement.

## Utilisation

```bash
# 1. Geler la geometrie du banc (une seule fois pour toute l'etude)
python run_calibrate.py alban D0 --out rig.json

# 2. Definir ROI et zone stable sur la vue frontale de reference
python make_masks.py alban D0

# 3. Reconstruire chaque session
python run_session.py alban D0 --rig rig.json
python run_session.py alban Dx --rig rig.json

# 4. Valider le pipeline sur deformation simulee (avant toute vraie mesure)
python selftest.py alban_D0.npz

# 5. Comparer
python run_compare.py alban_D0.npz alban_Dx.npz \
    --subject alban --ref-session D0 --test-session Dx \
    --roi roi.png --stable stable.png
```

## Sorties

| Fichier | Contenu |
|---|---|
| `rig.json` | geometrie gelee du banc |
| `<sujet>_<session>.npz` | nuage 3D, pixels, confiance |
| `<sujet>_<session>_confiance.png` | zones mesurees / faibles / non mesurees |
| `comparaison_deviation.png` | carte de deviation signee |
| `comparaison_metriques.json` | indicateurs sans dimension + test a blanc |

## Interpretation

- `ratio_relief_moyen` — variation moyenne de relief sur la ROI, rapportee au
  relief propre de cette ROI. Positif = gain de volume apparent.
- `variation_volume_relative` — integrale de la deviation sur la ROI rapportee
  a l'integrale du relief. Proxy de variation volumique relative.
- `controle_zone_stable_moyen` — **doit etre proche de zero**. S'il ne l'est
  pas, le recalage ou la zone stable est en cause : la mesure n'est pas fiable.
- `test_a_blanc.seuil_detection_2sigma` — plancher de bruit empirique. Toute
  variation inferieure n'est pas interpretable.

## Limites connues

- couverture faciale partielle (typiquement 50-60 % en haute confiance) ;
  nez, paupieres et cavites orbitaires souvent absents
- pas de recouvrement significatif entre les deux vues obliques : chaque
  demi-visage repose sur une seule paire stereo
- profondeur mal conditionnee (l'etendue du visage represente 3-4 % de la
  distance de prise de vue)
- repetabilite inter-visite non etablie : elle est la principale source
  d'incertitude restante et n'est attenuee par aucune normalisation

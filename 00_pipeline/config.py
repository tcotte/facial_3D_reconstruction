"""
config.py — constantes GELEES du pipeline de reconstruction 3D VISIA-CR.

=============================================================================
AVERTISSEMENT — NE PAS MODIFIER EN COURS D'ETUDE
=============================================================================
Les paramètres optiques ci-dessous ne sont PAS mesures : ils sont estimes a
partir des images elles-memes, faute d'acces aux EXIF de l'appareil et a la
geometrie exacte du banc.

Toute la validite du protocole D0/Dx repose sur le fait que ces constantes
sont IDENTIQUES pour toutes les acquisitions comparees. Les modifier en cours
d'etude invaliderait retroactivement toutes les comparaisons deja produites.

Si les vraies valeurs (focale EXIF, angles mecaniques) deviennent disponibles :
creer une NOUVELLE version (CALIB_VERSION), retraiter l'integralite du jeu de
donnees, et ne jamais melanger deux versions dans une meme analyse.
=============================================================================
"""

CALIB_VERSION = "v2.0-2026-09-focale-verifiee"

# --- Chemins (a adapter) ---------------------------------------------------
IMG_DIR = r"C:\donnees\visia"
OUT_DIR = r"C:\donnees\visia\resultats"

# --- Convention de nommage -------------------------------------------------
# <sujet>_<session>_<angle>_<modalite>.jpg
ANGLES = {"F": "Frontal", "L": "Left_Oblique", "R": "Right_Oblique"}
MODALITY_GEOM = "Standard_1"       # meilleure modalite geometrique (cf. etude)
MODALITY_COLOR = "Cross-Polarized" # reference colorimetrique

# --- Resolution de travail -------------------------------------------------
WORK_LONG = 4000        # cote long en px (natif 8000 -> facteur 2)

# --- Parametres optiques GELES ---------------------------------------------
# FOCALE VERIFIEE aupres du fournisseur : objectif 50 mm a focale fixe, f/13.
# Le VISIA RECADRE les pixels (pas de reechantillonnage) : le pas pixel reste
# celui du capteur R5, soit 36 mm / 8192 px = 4.3945 um.
#   f_px = 50 mm / 4.3945 um = 11378 px a la resolution native (8000 px)
#   -> FOCAL_REL = 11378 / 8000 = 1.42222, independant de la resolution
FOCAL_REL = 1.42222
# Distorsion radiale : NULLE. Le residu de reprojection est minimal a k1 = 0 et
# se degrade de facon monotone quand on l'augmente. La valeur -0.20 utilisee
# precedemment etait un artefact de l'ajustement conjoint focale-distorsion.
K1 = 0.0
# Point principal : centre de l'image (confirme par ajustement de faisceaux,
# decalage mesure < 1 px ; le recadrage VISIA est uniforme et centre).
PP_REL = (0.5, 0.5)

# Angles optiques recouvres, avec la focale verifiee. NON verifies mecaniquement
# et INSTABLES selon la configuration de traitement (32 a 41 deg au fil de
# l'etude). A titre informatif uniquement : ne pas interpreter comme une mesure
# de la geometrie du banc. L'ecart aux 45 deg annonces reste inexplique.
ANGLE_L_DEG = 32.4
ANGLE_R_DEG = 31.7

# Echelle metrique : NON DETERMINEE.
# La tentative par la charte ColorChecker donne deux valeurs incompatibles a
# 57 % entre les deux paires stereo (mire trop petite et trop peripherique).
# Tant que ce facteur est inconnu, AUCUNE valeur en mm ou mm^3 n'est produite.
SCALE_MM_PER_UNIT = None

# --- Appariement -----------------------------------------------------------
TILE_SIZE, TILE_STRIDE = 640, 512
MAX_FEATURES = 2048
RANSAC_F_THRESH = 3.0
RANSAC_E_THRESH = 2.0

# --- Densification / confiance ---------------------------------------------
EPI_HIGH = 1.0          # residu epipolaire (px) : seuil "mesure"
EPI_MED = 3.0           # seuil "faiblement contraint"
TEXTURE_MIN = 5.0       # contraste local minimal
SKIN_TOP, SKIN_BOTTOM = 0.28, 0.88

# --- Comparaison D0/Dx -----------------------------------------------------
# Le recalage est estime sur la ZONE STABLE, jamais sur la ROI mesuree.
REGISTRATION = "similarity"   # 'similarity' (echelle libre) ou 'rigid'
MIN_STABLE_POINTS = 300
MIN_ROI_POINTS = 100

def focal_px(long_side=None):
    return FOCAL_REL * (long_side or WORK_LONG)

def principal_point(w, h):
    return PP_REL[0] * w, PP_REL[1] * h

def summary():
    return (f"CALIB {CALIB_VERSION} | focale {FOCAL_REL:.4f}*L "
            f"({focal_px():.0f} px @ {WORK_LONG}) | k1 ={K1} | "
            f"angles {ANGLE_L_DEG}/{ANGLE_R_DEG} deg")

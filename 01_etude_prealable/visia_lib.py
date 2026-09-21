"""
visia_lib.py — fonctions communes pour l'analyse des images VISIA-CR.

Dépendances : opencv-python, numpy
    pip install opencv-python numpy

Auteur : étude de faisabilité reconstruction 3D VISIA-CR
"""

import os
import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Configuration : adapter ce chemin à votre poste
# ----------------------------------------------------------------------------
IMG_DIR = r"C:\donnees\visia\D0"       # dossier contenant les JPEG VISIA-CR

# Taille du fichier blanc de substitution produit par le bug d'export VISIA.
# Voir check_integrity(). À réajuster si la valeur change.
BLANK_FILE_SIZE = 251258

# Résolutions de travail (côté long, en pixels)
LONG_SIFT = 3200      # tests SIFT : compromis vitesse / détail
LONG_ASIFT = 2000     # ASIFT : coûteux (simulation affine), on réduit

# Bande verticale considérée comme "peau" (fraction de la hauteur d'image).
# Exclut la charte + appui frontal en haut, et la mentonnière en bas.
SKIN_TOP, SKIN_BOTTOM = 0.30, 0.86


# ----------------------------------------------------------------------------
# Entrées / sorties
# ----------------------------------------------------------------------------
def path(filename):
    return os.path.join(IMG_DIR, filename)


def check_integrity(filename):
    """Détecte les fichiers blancs produits par le bug d'export VISIA.

    Retourne (ok: bool, message: str).
    Deux critères indépendants : taille exacte du fichier de substitution,
    et écart-type nul (image uniforme).
    """
    full = path(filename)
    if not os.path.exists(full):
        return False, "fichier absent"
    size = os.path.getsize(full)
    im = cv2.imread(full)
    if im is None:
        return False, "illisible"
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    std = float(g.std())
    if size == BLANK_FILE_SIZE:
        return False, f"taille = {BLANK_FILE_SIZE} o -> fichier blanc de substitution"
    if std < 1.0:
        return False, f"ecart-type = {std:.2f} -> image uniforme (blanche)"
    return True, f"OK  {im.shape[1]}x{im.shape[0]}  moyenne={g.mean():.1f}  ecart-type={std:.1f}"


def load_gray(filename, long_side=LONG_SIFT):
    """Charge en niveaux de gris, rééchantillonné à côté long fixe.

    INTER_AREA est le filtre correct pour une réduction (moyennage de bloc) :
    il évite le repliement de spectre qui fausserait la détection de points.
    """
    im = cv2.imread(path(filename))
    if im is None:
        raise IOError(f"lecture impossible : {filename}")
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def load_color(filename, long_side=LONG_SIFT):
    im = cv2.imread(path(filename))
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Détection / description
# ----------------------------------------------------------------------------
def make_sift(nfeatures=20000, contrast=0.015, edge=15):
    """SIFT réglé permissif : la peau est peu contrastée, un seuil de contraste
    standard (0.04) élimine trop de points valides."""
    return cv2.SIFT_create(nfeatures=nfeatures,
                           contrastThreshold=contrast,
                           edgeThreshold=edge)


def _affine_skew(tilt, phi, img):
    """Simule une vue inclinée : rotation de phi degrés puis compression d'un
    facteur `tilt` selon x. Retourne (image simulée, matrice affine A)."""
    h, w = img.shape[:2]
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi_r = np.deg2rad(phi)
        s, c = np.sin(phi_r), np.cos(phi_r)
        A = np.float32([[c, -s], [s, c]])
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        tc = corners @ A.T
        x, y, w2, h2 = cv2.boundingRect(tc.reshape(1, -1, 2).astype(np.float32))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w2, h2), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        # anti-crénelage avant sous-échantillonnage anisotrope
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0,
                         interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    return img, A


def asift_detect(img, detector=None):
    """ASIFT (Morel & Yu, SIAM J. Imaging Sci. 2009).

    Simule un jeu de points de vue inclinés, applique SIFT sur chacun, puis
    ramène les points détectés dans le repère de l'image d'origine.
    C'est ce qui permet d'absorber les 45 degres d'écart du banc VISIA, là où
    SIFT seul décroche (facteur 7 à 12 sur le nombre d'appariements).

    ATTENTION : coûteux (une dizaine de simulations). Travailler à 2000 px.
    """
    if detector is None:
        detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    params = [(1.0, 0.0)]
    for t in 2 ** (0.5 * np.arange(1, 4)):          # tilts ~1.4, 2.0, 2.8
        for phi in np.arange(0, 180, 72.0 / t):     # rotations
            params.append((t, phi))
    kps, descs = [], []
    for t, phi in params:
        timg, A = _affine_skew(t, phi, img)
        k, d = detector.detectAndCompute(timg, None)
        if d is None:
            continue
        Ai = cv2.invertAffineTransform(A)
        for kp in k:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        kps.extend(k)
        descs.append(d)
    return kps, np.vstack(descs)


# ----------------------------------------------------------------------------
# Appariement et vérification géométrique
# ----------------------------------------------------------------------------
def match(des_a, des_b, kp_a, kp_b, ratio=0.78, flann=False):
    """Appariement au plus proche voisin + test de ratio de Lowe (IJCV 2004).

    Le test de ratio rejette les appariements ambigus : si le meilleur voisin
    n'est pas nettement meilleur que le second, l'appariement est écarté.
    """
    if flann:
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                        dict(checks=64))
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    pa = np.float32([kp_a[m.queryIdx].pt for m in good])
    pb = np.float32([kp_b[m.trainIdx].pt for m in good])
    return pa, pb


def geometric_filter(pa, pb, thresh=3.0, conf=0.99):
    """Filtre RANSAC par matrice fondamentale (Fischler & Bolles 1981).

    Ne conserve que les appariements compatibles avec une même géométrie
    épipolaire : élimine les faux appariements que le seul test de ratio laisse
    passer (peau répétitive, poils, reflets).
    """
    if len(pa) < 8:
        return pa[:0], pb[:0], None
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, thresh, conf)
    if mask is None:
        return pa[:0], pb[:0], None
    inl = mask.ravel().astype(bool)
    return pa[inl], pb[inl], F


def skin_mask(points, image_height):
    """Vrai pour les points situés dans la bande cutanée.

    MÉTRIQUE CLÉ de l'étude : la charte et l'appui frontal sont rigides et mats,
    ils s'apparient très bien et gonflent artificiellement les comptages
    globaux. Seuls les appariements SUR LA PEAU renseignent sur la faisabilité
    d'une reconstruction faciale.
    """
    y = points[:, 1] / image_height
    return (y > SKIN_TOP) & (y < SKIN_BOTTOM)


# ----------------------------------------------------------------------------
# Métriques photométriques
# ----------------------------------------------------------------------------
def photometry(filename, long_side=LONG_SIFT):
    """Fraction spéculaire et contraste de texture sur la région cutanée.

    - fraction spéculaire : pixels clairs ET peu saturés (V>200, S<60).
      Un reflet spéculaire est une réflexion directe de la source : il conserve
      la couleur de la lumière (donc peu saturé) et sature en luminance.
    - contraste de texture : écart-type du laplacien, proxy du contenu
      haute fréquence (micro-relief, pores, poils).
    """
    im = load_color(filename, long_side)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    skin = V > 40                                   # exclut le fond noir
    spec = 100.0 * ((V > 200) & (S < 60) & skin).sum() / max(skin.sum(), 1)
    tex = float(cv2.Laplacian(g, cv2.CV_32F, ksize=3)[skin].std())
    return {"specular_pct": spec, "texture_lap_std": tex}

"""
face_mask.py — masque facial robuste au cadrage et au vetement.

OBJET
    Delimiter la surface a reconstruire. Trois pieges ont ete rencontres et
    corriges au fil de l'etude ; chacun donnait un masque plausible mais faux.

PIEGE 1 — bande verticale fixe
    Une bande [0,28 ; 0,88] de la hauteur d'image coupait le haut du front, donc
    les rides frontales. Le bord superieur est desormais LIBRE : l'appui frontal
    et le fond, sombres, sortent naturellement par selection de la plus grande
    composante connexe.

PIEGE 2 — fenetre de recherche fixe pour la coupe du cou
    Le minimum de largeur de machoire etait cherche dans une fenetre fixe. Des
    que le cadrage change (second sujet), la fenetre tombe APRES le minimum reel
    et la coupe se fait en plein milieu du menton. La recherche part maintenant
    de la ligne la plus large et descend jusqu'a la transition machoire/cou.

PIEGE 3 — ordre des operations
    Le remplissage des trous internes (sourcils, narines) comblait aussi
    l'espace sombre entre le menton et un vetement clair, rendant le profil de
    largeur PARFAITEMENT CONSTANT : plus aucune transition detectable. Il faut
    couper le cou AVANT de remplir les trous.

DEUX AUTRES REGLES
    - largeur mesuree sur une BANDE CENTRALE : le cou et le vetement
      s'elargissent sur les cotes, le menton est central
    - critere de SATURATION : la peau est claire ET saturee (S ~ 80-100), un
      vetement blanc est clair mais desature (S ~ 24), le fond est sombre

USAGE
    from face_mask import face_mask
    masque, ycut, ymax, mini, wmax = face_mask(image_bgr)
"""
import cv2
import numpy as np

SEUIL_LUMINANCE = 45      # separe le visage du fond noir
SEUIL_SATURATION = 45     # separe la peau d'un vetement clair
SEUIL_CHARTE_S = 110      # patchs colores de la charte ColorChecker
SEUIL_CHARTE_V = 80


def cut_neck(m, marge_frac=0.02, seuil_reprise=1.10):
    """Coupe a la transition machoire/cou, sans fenetre de recherche fixe."""
    h, w = m.shape
    cols = np.nonzero(m.any(axis=0))[0]
    xc = 0.5 * (cols.min() + cols.max())
    demi = 0.28 * (cols.max() - cols.min())
    bande = np.zeros(w, bool)
    bande[int(xc - demi):int(xc + demi)] = True
    prof = m[:, bande].sum(axis=1).astype(float)
    sm = np.convolve(prof, np.ones(31) / 31, mode='same')

    # la ligne la plus large se cherche dans la zone du VISAGE : un vetement
    # clair peut occuper toute la largeur d'image plus bas
    ymax = int(np.argmax(sm[:int(0.72 * h)]))
    wmax = sm[ymax]

    # depart sous la bouche : plus haut, les trous des yeux et des sourcils
    # (pas encore combles) creent de faux minima locaux
    rows = np.nonzero(m.any(axis=1))[0]
    depart = max(ymax + 40, int(rows.min() + 0.62 * (rows.max() - rows.min())))

    ycut, mini = h, wmax
    for y in range(depart, h):
        if sm[y] < 0.04 * wmax:                 # fin naturelle : plus de peau
            ycut = y
            break
        if sm[y] < mini:
            mini = sm[y]
        elif sm[y] > seuil_reprise * mini and mini < 0.85 * wmax:
            ycut = y                            # la largeur repart : cou
            break
    ycut = min(h, ycut + int(marge_frac * h))   # marge pour la pointe du menton

    out = m.copy()
    out[ycut:, :] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    if n > 1:
        out = (lab == 1 + np.argmax(st[1:, 4]))
    return out, ycut, ymax, mini, wmax


def face_mask(A):
    """Masque facial complet. Retourne (masque, ycut, ymax, mini, wmax)."""
    h, w = A.shape[:2]
    g = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)

    # charte ColorChecker : exclue par sa BOITE ENGLOBANTE, pas pixel a pixel
    # (un seuillage colorimetrique mordrait sur la peau du front)
    sat = ((hsv[:, :, 1] > SEUIL_CHARTE_S) & (hsv[:, :, 2] > SEUIL_CHARTE_V)).astype(np.uint8)
    sat[int(0.15 * h):, :] = 0
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    block = np.zeros((h, w), bool)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 400:
            continue
        mg = int(0.03 * h)
        block[max(0, y-mg):min(h, y+bh+mg), max(0, x-mg):min(w, x+bw+mg)] = True

    S = hsv[:, :, 1].astype(float)
    m = ((g > SEUIL_LUMINANCE) & (S > SEUIL_SATURATION) & ~block).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)

    # ORDRE IMPORTANT : couper AVANT de remplir (voir piege 3 en tete de fichier)
    mb, ycut, ymax, mini, wmax = cut_neck(m.astype(bool))
    mb = mb.astype(np.uint8)
    ff = mb.copy()
    mk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 1)
    mb = (mb | (1 - ff)).astype(np.uint8)
    mb[ycut:, :] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mb, 8)
    if n > 1:
        mb = (lab == 1 + np.argmax(st[1:, 4]))
    return mb.astype(bool), ycut, ymax, mini, wmax


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--long", type=int, default=2000)
    ap.add_argument("--out", default="masque.npy")
    a = ap.parse_args()
    A = cv2.imread(a.image)
    h0, w0 = A.shape[:2]
    s = a.long / max(h0, w0)
    A = cv2.resize(A, (int(w0*s), int(h0*s)), interpolation=cv2.INTER_AREA)
    m, ycut, ymax, mini, wmax = face_mask(A)
    ys = np.nonzero(m.any(axis=1))[0]
    print(f"masque : {m.sum()} px | y {ys.min()}-{ys.max()} "
          f"({ys.min()/A.shape[0]:.3f}-{ys.max()/A.shape[0]:.3f})")
    print(f"coupe du cou a y={ycut} | largeur max {wmax:.0f} px a y={ymax} | "
          f"minimum machoire {mini:.0f} px")
    np.save(a.out, m)

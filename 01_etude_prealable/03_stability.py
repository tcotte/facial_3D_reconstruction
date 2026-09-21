"""
03_stability.py — stabilité du sujet entre deux prises successives.

OBJET
    La capture VISIA est séquentielle. Or toutes les méthodes multi-vues
    supposent une scène rigide et statique (Hartley & Zisserman, 2004).
    Un visage se déforme (expression, mâchoire, respiration) et la tête peut
    bouger : il faut quantifier ce mouvement AVANT de conclure quoi que ce soit
    sur la faisabilité.

PRINCIPE
    On apparie deux modalités du MÊME angle (standard vs cross-polarisé), donc
    deux déclenchements successifs sans repositionnement. On ajuste une
    homographie et on mesure le déplacement résiduel.

RÉSULTAT OBTENU
    2346 appariements, 2139 inliers, déplacement médian 1,5 px à 2400 px de
    côté long, soit environ 5 px à la résolution native (8000 px) : ~0,06 % de
    la hauteur du champ. La contention (appui frontal + mentonnière) est très
    efficace.

RÉSERVES
    (a) mesure le mouvement entre deux prises du MÊME angle, pas entre angles ;
    (b) une homographie 2D ne capte pas le mouvement hors-plan (profondeur).
    À refaire entre visites (D0 vs D28) pour valider la répétabilité longitudinale.

USAGE
    python 03_stability.py
"""

import numpy as np
import visia_lib as vl

SUBJECT = "alban_D0"
LONG = 2400


def stability(angle):
    a = vl.load_gray(f"{SUBJECT}_{angle}_Standard_1.jpg", LONG)
    b = vl.load_gray(f"{SUBJECT}_{angle}_Cross-Polarized.jpg", LONG)
    sift = vl.make_sift()
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    pa, pb = vl.match(da, db, ka, kb, ratio=0.80)
    if len(pa) < 20:
        return None
    H, mask = cv2_findHomography(pa, pb)
    inl = mask.ravel().astype(bool)
    d = pb[inl] - pa[inl]
    med = float(np.median(np.linalg.norm(d, axis=1)))
    return {
        "matches": len(pa),
        "inliers": int(inl.sum()),
        "median_px": med,
        "std_x": float(d[:, 0].std()),
        "std_y": float(d[:, 1].std()),
        "native_px": med * 8000.0 / LONG,
        "pct_field": 100.0 * med / LONG,
    }


def cv2_findHomography(pa, pb):
    import cv2
    return cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)


def main():
    for angle in ["Frontal", "Left_Oblique", "Right_Oblique"]:
        r = stability(angle)
        if r is None:
            print(f"{angle}: appariements insuffisants")
            continue
        print(f"\n=== {angle} (standard vs cross-polarisé, même pose) ===")
        print(f"  appariements     : {r['matches']}  (inliers {r['inliers']})")
        print(f"  déplacement médian : {r['median_px']:.2f} px @ {LONG} px")
        print(f"  dispersion         : x={r['std_x']:.2f}  y={r['std_y']:.2f} px")
        print(f"  équivalent natif   : {r['native_px']:.1f} px @ 8000 px "
              f"({r['pct_field']:.3f} % du champ)")


if __name__ == "__main__":
    main()

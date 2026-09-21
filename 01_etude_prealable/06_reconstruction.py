"""
06_reconstruction.py — reconstruction 3D préliminaire par paires stéréo.

OBJET
    Triangulation à deux vues (frontal <-> oblique), en exploitant la propriété
    la plus utile du banc VISIA : les angles sont MÉCANIQUEMENT INDEXÉS ET
    RÉPÉTABLES (0, +45, -45 degres). On peut donc inverser le problème et
    chercher la focale qui reproduit exactement la rotation connue de 45 degres,
    au lieu de calibrer avec une mire.

RÉSULTATS OBTENUS
    Focale résolue :  F<->Obl.G = 5327 px    F<->Obl.D = 4170 px
    Triangulation  :  435 points, reprojection 0.51 px  (paire gauche)
                      261 points, reprojection 0.37 px  (paire droite)
    Étendue de profondeur du visage : 3.4 % et 4.4 % de la distance de prise
    de vue -> problème intrinsèquement mal conditionné selon l'axe Z, qui est
    précisément l'axe où se mesure un gonflement.

!! VERROU PRINCIPAL !!
    Les deux paires donnent des focales incompatibles à 28 %, alors qu'il
    s'agit du même appareil à la même focale. Les solutions sont pourtant
    numériquement stables (5 relances : 45.0 degres à chaque fois).
    Hypothèses non départagées :
      1. les deux angles ne sont pas rigoureusement symétriques à 45 degres ;
      2. le point principal n'est pas au centre (recadrage asymétrique VISIA ?) ;
      3. la distorsion optique n'est pas négligeable ;
      4. la géométrie à deux vues reste mal conditionnée sur ce nombre de points.

    TANT QUE CE VERROU N'EST PAS LEVÉ, AUCUN VOLUME NE PEUT ÊTRE REVENDIQUÉ.
    Étape suivante : ajustement de faisceaux CONJOINT sur les trois vues
    (focale unique commune, angles imposés, point principal libre).

USAGE
    python 06_reconstruction.py
"""

import cv2
import numpy as np
import visia_lib as vl

SUBJECT = "alban_D0"
MODALITY = "Standard_1"
RIG_ANGLE_DEG = 45.0
OUT = "reconstruction_profondeur.png"


def correspondences(frontal_gray, ka, da, angle):
    b = vl.load_gray(f"{SUBJECT}_{angle}_{MODALITY}.jpg", vl.LONG_ASIFT)
    kb, db = vl.asift_detect(b)
    pa, pb = vl.match(da, db, ka, kb, ratio=0.80, flann=True)
    ia, ib, _ = vl.geometric_filter(pa, pb, thresh=2.0, conf=0.999)
    m = vl.skin_mask(ia, frontal_gray.shape[0])
    return ia[m], ib[m]


def recovered_angle(pa, pb, focal, W, H):
    """Angle de rotation recouvré pour une focale supposée."""
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], float)
    E, _ = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.5)
    if E is None or E.shape != (3, 3):
        return None
    _, R, _, _ = cv2.recoverPose(E, pa, pb, K)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def solve_focal(pa, pb, W, H, target=RIG_ANGLE_DEG):
    """Dichotomie sur la focale pour retrouver l'angle mécanique connu."""
    lo, hi = 1500.0, 12000.0
    for _ in range(28):
        mid = (lo + hi) / 2
        a = recovered_angle(pa, pb, mid, W, H)
        if a is None:
            break
        if a < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def triangulate(pa, pb, focal, W, H):
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], float)
    E, _ = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.5)
    _, R, t, _ = cv2.recoverPose(E, pa, pb, K)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = cv2.triangulatePoints(P1, P2, pa.T, pb.T)
    X = (X[:3] / X[3]).T
    ok = np.isfinite(X).all(1) & (X[:, 2] > 0)
    X, p = X[ok], pa[ok]
    rp, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, None)
    err = float(np.linalg.norm(rp.reshape(-1, 2) - p, axis=1).mean())
    return X, p, err


def main():
    frontal = f"{SUBJECT}_Frontal_{MODALITY}.jpg"
    a = vl.load_gray(frontal, vl.LONG_ASIFT)
    H, W = a.shape
    ka, da = vl.asift_detect(a)
    print(f"frontal : {len(ka)} points ASIFT ({W}x{H})\n")

    vis = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    vis = (vis * 0.45).astype(np.uint8)
    focals = {}

    for angle in ["Left_Oblique", "Right_Oblique"]:
        pa, pb = correspondences(a, ka, da, angle)
        f = solve_focal(pa, pb, W, H)
        focals[angle] = f
        checks = [recovered_angle(pa, pb, f, W, H) for _ in range(3)]
        X, p, err = triangulate(pa, pb, f, W, H)
        z = X[:, 2]
        p5, p95 = np.percentile(z, 5), np.percentile(z, 95)
        print(f"=== Frontal <-> {angle} ===")
        print(f"  points sur la peau      : {len(pa)}")
        print(f"  focale résolue à 45 deg : {f:.0f} px")
        print(f"  contrôle de stabilité   : " + ", ".join(f"{c:.1f}" for c in checks))
        print(f"  points 3D valides       : {len(X)}")
        print(f"  reprojection RMS        : {err:.2f} px")
        print(f"  étendue de profondeur   : {100 * (p95 - p5) / np.median(z):.1f} % "
              f"de la distance\n")

        zn = np.clip((z - p5) / (p95 - p5 + 1e-9), 0, 1)
        for pt, t_ in zip(p, zn):
            c = cv2.applyColorMap(np.uint8([[t_ * 255]]), cv2.COLORMAP_JET)[0, 0].tolist()
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, c, -1)

    cv2.imwrite(OUT, vis)
    print(f"carte de profondeur écrite : {OUT}")

    fl, fr = focals["Left_Oblique"], focals["Right_Oblique"]
    disc = 100.0 * abs(fl - fr) / ((fl + fr) / 2)
    print(f"\n>>> ÉCART ENTRE LES DEUX FOCALES : {disc:.0f} %")
    if disc > 5:
        print(">>> INCOHÉRENT : la calibration n'est pas résolue.")
        print(">>> Aucune mesure volumétrique ne peut être revendiquée à ce stade.")


if __name__ == "__main__":
    main()

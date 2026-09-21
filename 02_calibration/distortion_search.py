"""
distortion_search.py — la distorsion explique-t-elle l'ecart de focale ?

OBJET
    Balayage conjoint (focale x coefficient radial k1), en cherchant la
    combinaison qui reconcilie les deux criteres de focal_sweep.py.

POURQUOI NE PAS PRENDRE UN COEFFICIENT DE LA LITTERATURE
    La focale estimee (~56 mm) est incompatible avec l'hypothese d'un 24 mm
    (voir sensor_check.py). Appliquer les coefficients publies pour un 24 mm
    (-2,5 % a -7,5 % de distorsion en barillet) introduirait une erreur
    systematique majeure. Le coefficient doit etre estime SUR LES DONNEES.

RESULTAT OBTENU
    L'ecart entre criteres reste entre 12 % et 18 % sur toute la plage de k1.
    Le meilleur accord (k1 = -0,20) correspond a une distorsion tres forte,
    incoherente avec une optique de 56 mm (typiquement < 1 %).
    -> la distorsion est ECARTEE comme explication de l'ecart.

USAGE
    python distortion_search.py --dir donnees --width 2667 --height 4000
"""
import argparse, numpy as np, cv2


def undistort(p, K, k1):
    d = np.array([k1, 0, 0, 0], float)
    return cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64), K, d, P=K).reshape(-1, 2)


def evaluate(f, k1, pa, pb, t3, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    o = {}
    for k in "LR":
        ua, ub = undistort(pa[k], K, k1), undistort(pb[k], K, k1)
        E, _ = cv2.findEssentialMat(ua, ub, K, cv2.RANSAC, 0.999, 2.0)
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, _ = cv2.recoverPose(E, ua, ub, K)
        o[k] = (R, t.ravel())
    ang = [float(np.degrees(np.arccos(np.clip((np.trace(o[k][0]) - 1) / 2, -1, 1)))) for k in "LR"]
    u0 = undistort(t3[:, 0:2], K, k1)
    u1 = undistort(t3[:, 2:4], K, k1)
    u2 = undistort(t3[:, 4:6], K, k1)

    def tri(R, t, p1, p2):
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(3, 1)])
        X = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        return (X[:3] / X[3]).T
    XL = tri(*o["L"], u0, u1); XR = tri(*o["R"], u0, u2)
    ok = (XL[:, 2] > 0) & (XR[:, 2] > 0)
    if ok.sum() < 50:
        return None
    s = np.median(XL[ok, 2] / XR[ok, 2])
    Xc = XL @ o["R"][0].T + o["R"][1] * s
    z = np.clip(Xc[:, 2], 1e-6, None)
    pr = np.column_stack([f * Xc[:, 0] / z + cx, f * Xc[:, 1] / z + cy])
    return ang[0] + ang[1], float(np.median(np.linalg.norm(pr - u2, axis=1)[ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    a = ap.parse_args()
    cx, cy = a.width / 2, a.height / 2
    pa = {k: np.load(f"{a.dir}/pa_{k}.npy") for k in "LR"}
    pb = {k: np.load(f"{a.dir}/pb_{k}.npy") for k in "LR"}
    t3 = np.load(f"{a.dir}/t3.npy")
    print(f"{'k1':>7s} {'f (reproj)':>12s} {'erreur':>8s} {'f (angle 90)':>13s} {'ecart':>8s}")
    best = None
    for k1 in [-0.20, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12]:
        rows = []
        for f in np.arange(4500, 10000, 250):
            r = evaluate(float(f), k1, pa, pb, t3, cx, cy)
            if r:
                rows.append((f, r[0], r[1]))
        if len(rows) < 5:
            continue
        r = np.array(rows)
        i = int(np.argmin(r[:, 2])); j = int(np.argmin(np.abs(r[:, 1] - 90.0)))
        gap = 100 * abs(r[i, 0] - r[j, 0]) / r[i, 0]
        print(f"{k1:+7.2f} {r[i,0]:12.0f} {r[i,2]:8.2f} {r[j,0]:13.0f} {gap:7.1f}%")
        if best is None or gap < best[0]:
            best = (gap, k1, r[i, 0])
    print(f"\nmeilleur accord : k1 = {best[1]:+.2f}, f = {best[2]:.0f} px, ecart {best[0]:.1f} %")
    if best[0] > 5:
        print(">>> la distorsion NE reconcilie PAS les deux criteres.")
        print(">>> chercher ailleurs : angle mecanique reel, EXIF d'origine.")

if __name__ == "__main__":
    main()

"""
focal_sweep.py — la focale est-elle identifiable par les donnees ?

OBJET
    Sans EXIF ni mesure du banc, la focale doit etre estimee. Deux criteres
    INDEPENDANTS sont confrontes :
      (A) minimum de l'erreur de reprojection sur les pistes vues dans les 3 vues
      (B) somme des angles recouvres L+R = 90 deg (contrainte mecanique supposee)
    Leur ecart mesure directement le degre d'identifiabilite.

RESULTAT OBTENU
    critere (A) : 6250 px    critere (B) : 7250 px    ECART 16 %
    -> la focale n'est PAS identifiable de facon fiable. L'ecart a resiste a
       trois moteurs d'appariement differents (16 %, 18 %, 21 %) : ce n'est pas
       un probleme d'algorithme.

ENTREES
    pa_L.npy, pb_L.npy, pa_R.npy, pb_R.npy : appariements verifies par paire
    t3.npy : pistes 3 vues, colonnes [xF, yF, xL, yL, xR, yR]

USAGE
    python focal_sweep.py --dir donnees --width 2667 --height 4000
"""
import argparse, numpy as np, cv2


def poses(pa, pb, K, thresh=2.0):
    E, _ = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, thresh)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, pa, pb, K)
    return R, t.ravel()


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def triangulate(K, R, t, p1, p2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (X[:3] / X[3]).T


def evaluate(f, pa, pb, t3, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    o = {}
    for k in "LR":
        r = poses(pa[k], pb[k], K)
        if r is None:
            return None
        o[k] = r
    aL, aR = angle_deg(o["L"][0]), angle_deg(o["R"][0])
    XL = triangulate(K, *o["L"], t3[:, 0:2], t3[:, 2:4])
    XR = triangulate(K, *o["R"], t3[:, 0:2], t3[:, 4:6])
    ok = (XL[:, 2] > 0) & (XR[:, 2] > 0)
    if ok.sum() < 50:
        return None
    # les deux branches sont normalisees independamment : recaler leur echelle
    s = np.median(XL[ok, 2] / XR[ok, 2])
    tR = o["R"][1] * s
    Xc = XL @ o["R"][0].T + tR
    z = np.clip(Xc[:, 2], 1e-6, None)
    pr = np.column_stack([f * Xc[:, 0] / z + cx, f * Xc[:, 1] / z + cy])
    err = float(np.median(np.linalg.norm(pr - t3[:, 4:6], axis=1)[ok]))
    return aL, aR, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--fmin", type=float, default=4500)
    ap.add_argument("--fmax", type=float, default=10500)
    ap.add_argument("--step", type=float, default=250)
    a = ap.parse_args()
    cx, cy = a.width / 2, a.height / 2
    pa = {k: np.load(f"{a.dir}/pa_{k}.npy") for k in "LR"}
    pb = {k: np.load(f"{a.dir}/pb_{k}.npy") for k in "LR"}
    t3 = np.load(f"{a.dir}/t3.npy")
    print(f"appariements : L {len(pa['L'])}, R {len(pa['R'])} | pistes 3 vues {len(t3)}\n")
    print(f"{'focale':>8s} {'angle L':>9s} {'angle R':>9s} {'L+R':>8s} {'reproj 3 vues':>14s}")
    rows = []
    for f in np.arange(a.fmin, a.fmax, a.step):
        r = evaluate(float(f), pa, pb, t3, cx, cy)
        if r is None:
            continue
        aL, aR, e = r
        rows.append((f, aL, aR, aL + aR, e))
        print(f"{f:8.0f} {aL:9.2f} {aR:9.2f} {aL+aR:8.2f} {e:14.2f}")
    r = np.array(rows)
    i = int(np.argmin(r[:, 4]))
    j = int(np.argmin(np.abs(r[:, 3] - 90.0)))
    gap = 100 * abs(r[i, 0] - r[j, 0]) / r[i, 0]
    print(f"\n(A) minimum de reprojection : f = {r[i,0]:.0f} px (erreur {r[i,4]:.2f} px, "
          f"angles {r[i,1]:.1f}/{r[i,2]:.1f})")
    print(f"(B) critere angulaire 90 deg : f = {r[j,0]:.0f} px (erreur {r[j,4]:.2f} px)")
    print(f"ECART ENTRE CRITERES         : {gap:.1f} %")
    if gap > 5:
        print(">>> focale NON identifiable de facon fiable.")
        print(">>> Aucune mesure metrique n'est defendable ; se limiter au relatif.")

if __name__ == "__main__":
    main()

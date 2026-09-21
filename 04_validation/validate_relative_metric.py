"""
validate_relative_metric.py — la mesure relative resiste-t-elle a une focale fausse ?

OBJET
    La focale n'etant pas identifiable, la question decisive est : une variation
    RELATIVE entre deux sessions reste-t-elle valide malgre une focale erronee ?
    Ce script le verifie par simulation : deformation d'amplitude CONNUE
    appliquee a une reconstruction reelle, reprojection, puis reconstruction
    avec une focale FAUSSE.

RESULTAT OBTENU — la formulation compte enormement
    erreur focale | A : defo / distance de travail | B : defo / relief de la ROI
       -20 %      |        +47 %                   |        +2,9 %
       -10 %      |        +20 %                   |        +1,3 %
       +10 %      |        -15 %                   |        -1,1 %
       +20 %      |        -27 %                   |        -2,1 %

    Formulation A : biais ~1,5 x l'erreur de focale.
    Formulation B : biais < 3 % pour +/- 20 % d'erreur -> attenuation x15.

    Numerateur et denominateur etant deux grandeurs de MEME NATURE issues de la
    MEME reconstruction, la distorsion se simplifie. Le mecanisme ne fonctionne
    PAS avec une reference externe (distance de travail, mesure anthropometrique).

PIEGE RENCONTRE
    Un premier essai sans RECALAGE donnait des resultats aberrants (jusqu'a
    +364 %). Deux reconstructions vivent dans des reperes arbitraires distincts
    et ne peuvent etre comparees directement.

USAGE
    python validate_relative_metric.py --cloud sess_D0.npz --focal 6250 \
        --width 1333 --height 2000
"""
import argparse, numpy as np, cv2


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(S)
    Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[2, 2] = -1
    R = U @ Sg @ Vt
    s = np.trace(np.diag(D) @ Sg) / ((Xc ** 2).sum() / len(X))
    return s, R, my - s * R @ mx


def plane_fit(P):
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    c, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    return c


def relief(P, c):
    return P[:, 2] - (c[0] * P[:, 0] + c[1] * P[:, 1] + c[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True, help="npz contenant X (N,3)")
    ap.add_argument("--focal", type=float, required=True)
    ap.add_argument("--k1", type=float, default=-0.20)
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--amp", type=float, default=0.004,
                    help="amplitude de la bosse, en fraction de la distance")
    a = ap.parse_args()
    cx, cy = a.width / 2, a.height / 2
    X = np.load(a.cloud)["X"].astype(float)
    X = X[np.isfinite(X).all(1) & (X[:, 2] > 0)]
    z = X[:, 2]; lo, hi = np.percentile(z, [2, 98]); X = X[(z > lo) & (z < hi)]

    c = X.mean(0)
    rad = 0.025 * np.median(X[:, 2])
    r2 = (X[:, 0] - c[0]) ** 2 + (X[:, 1] - c[1]) ** 2
    roi = r2 < (1.5 * rad) ** 2
    stable = r2 > (4.0 * rad) ** 2
    print(f"{len(X)} points | ROI {roi.sum()} | zone stable {stable.sum()}")
    if roi.sum() < 100 or stable.sum() < 300:
        raise SystemExit("couverture insuffisante")

    amp = a.amp * np.median(X[:, 2])
    Xd = X.copy()
    Xd[:, 2] -= amp * np.exp(-r2 / (2 * rad ** 2))

    # pose de reference (approximation : rotation de 40 deg autour de Y)
    ang = np.deg2rad(40.0)
    R0 = cv2.Rodrigues(np.array([0, ang, 0]))[0]
    t0 = np.array([np.sin(ang), 0, 1 - np.cos(ang)]) * np.median(X[:, 2])

    def project(P, R, t, f, k1):
        Xc = P @ R.T + t
        zz = np.clip(Xc[:, 2], 1e-9, None)
        xn, yn = Xc[:, 0] / zz, Xc[:, 1] / zz
        g = 1 + k1 * (xn ** 2 + yn ** 2)
        return np.column_stack([f * xn * g + cx, f * yn * g + cy])

    def reconstruct(P, f):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        d = np.array([a.k1, 0, 0, 0], float)
        p1 = project(P, np.eye(3), np.zeros(3), a.focal, a.k1)
        p2 = project(P, R0, t0, a.focal, a.k1)
        q1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, d, P=K).reshape(-1, 2)
        q2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), K, d, P=K).reshape(-1, 2)
        E, _ = cv2.findEssentialMat(q1, q2, K, cv2.RANSAC, 0.9999, 0.5, maxIters=20000)
        _, R, t, _ = cv2.recoverPose(E, q1, q2, K)
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(3, 1)])
        Y = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
        return (Y[:3] / Y[3]).T

    print(f"\n{'err. focale':>12s} {'A: /distance':>15s} {'B: /relief ROI':>16s}")
    ref = None
    for err in [0.0, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20]:
        f = a.focal * (1 + err)
        X0 = reconstruct(X, f); X1 = reconstruct(Xd, f)
        good = np.isfinite(X0).all(1) & np.isfinite(X1).all(1) & (X0[:, 2] > 0) & (X1[:, 2] > 0)
        st, rr = stable & good, roi & good
        if st.sum() < 200 or rr.sum() < 50:
            print(f"{err*100:+11.0f} %   echec"); continue
        s, R, t = umeyama(X1[st], X0[st])
        X1a = s * (R @ X1.T).T + t
        dev = float(np.mean(X0[rr, 2] - X1a[rr, 2]))
        A_ = dev / float(np.median(X0[good, 2]))
        co = plane_fit(X0[st])
        B_ = dev / float(np.std(relief(X0[rr], co)))
        if ref is None:
            ref = (A_, B_)
        print(f"{err*100:+11.0f} %   {A_/ref[0]:14.3f}  {B_/ref[1]:15.3f}")
    print("\nA = biais proportionnel a l'erreur de focale -> NE PAS UTILISER")
    print("B = grandeur adimensionnelle, robuste -> formulation retenue")

if __name__ == "__main__":
    main()

"""
bundle_adjustment.py — ajustement de faisceaux conjoint sur les 3 vues.

OBJET
    Un traitement paire par paire ne peut pas lever une incoherence portant sur
    un parametre PARTAGE (la focale). L'ajustement optimise simultanement :
    focale unique, point principal, poses des deux obliques, et positions 3D.

PIEGE MAJEUR — initialisation
    Les deux branches (F<->L et F<->R) sont triangulees avec des translations
    normalisees INDEPENDAMMENT : elles vivent a des echelles differentes. Sans
    recalage prealable de ces echelles, l'optimisation stagne (residu observe :
    23,9 px). Apres recalage sur les points vus dans les 3 vues (facteur 0,9497),
    l'ecart 3D relatif tombe a 0,14 % et l'optimisation converge.

RESULTAT OBTENU
    reprojection mediane : frontal 0,40 px | oblique G 0,68 px | oblique D 0,75 px
    point principal      : au centre (ecart < 1 px) -> pas de recadrage decentre
    somme des angles     : 89,81 deg (proche des 90 deg mecaniques)
    focale               : NON deplacee par l'optimisation -> voir focal_sweep.py

USAGE
    python bundle_adjustment.py --dir donnees --width 1333 --height 2000 --focal0 4000
"""
import argparse, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

NCAM = 15   # f, cx, cy, rvec_L(3), t_L(3), rvec_R(3), t_R(3)


def project(X, r, t, f, cx, cy):
    R = cv2.Rodrigues(np.asarray(r, dtype=float))[0]
    Xc = X @ R.T + t
    z = np.clip(Xc[:, 2], 1e-6, None)
    return np.column_stack([f * Xc[:, 0] / z + cx, f * Xc[:, 1] / z + cy])


def triangulate(K, R, t, p1, p2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (X[:3] / X[3]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--focal0", type=float, required=True)
    ap.add_argument("--angle", type=float, default=45.0,
                    help="angle mecanique suppose, en degres (contrainte souple)")
    ap.add_argument("--weight-angle", type=float, default=300.0)
    a = ap.parse_args()
    W, H = a.width, a.height
    t3 = np.load(f"{a.dir}/t3.npy")
    t2L = np.load(f"{a.dir}/t2L.npy")
    t2R = np.load(f"{a.dir}/t2R.npy")
    n3, n2L, n2R = len(t3), len(t2L), len(t2R)
    N = n3 + n2L + n2R
    f0 = a.focal0
    K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]], float)

    def pose(pa, pb):
        E, _ = cv2.findEssentialMat(pa, pb, K0, cv2.RANSAC, 0.999, 1.5)
        _, R, t, _ = cv2.recoverPose(E, pa, pb, K0)
        return R, t.ravel()
    RL, tL = pose(np.load(f"{a.dir}/pa_L.npy"), np.load(f"{a.dir}/pb_L.npy"))
    RR, tR = pose(np.load(f"{a.dir}/pa_R.npy"), np.load(f"{a.dir}/pb_R.npy"))

    # --- ETAPE CRITIQUE : recaler l'echelle des deux branches
    X3L = triangulate(K0, RL, tL, t3[:, 0:2], t3[:, 2:4])
    X3R = triangulate(K0, RR, tR, t3[:, 0:2], t3[:, 4:6])
    ok = (X3L[:, 2] > 0) & (X3R[:, 2] > 0)
    s = float(np.median(X3L[ok, 2] / X3R[ok, 2]))
    tR = tR * s
    X3R = triangulate(K0, RR, tR, t3[:, 0:2], t3[:, 4:6])
    d = np.linalg.norm(X3L - X3R, axis=1) / np.abs(X3L[:, 2])
    print(f"echelle branche L/R = {s:.4f} | ecart 3D relatif median apres recalage "
          f"= {100*np.median(d[ok]):.2f} %")

    X0 = np.vstack([(X3L + X3R) / 2,
                    triangulate(K0, RL, tL, t2L[:, 0:2], t2L[:, 2:4]),
                    triangulate(K0, RR, tR, t2R[:, 0:2], t2R[:, 2:4])])
    rL = cv2.Rodrigues(RL)[0].ravel(); rR = cv2.Rodrigues(RR)[0].ravel()
    p0 = np.hstack([f0, W / 2, H / 2, rL, tL, rR, tR, X0.ravel()])

    obsF = np.vstack([t3[:, 0:2], t2L[:, 0:2], t2R[:, 0:2]])
    obsL = np.vstack([t3[:, 2:4], t2L[:, 2:4]])
    obsR = np.vstack([t3[:, 4:6], t2R[:, 2:4]])
    iL = np.r_[0:n3, n3:n3 + n2L]
    iR = np.r_[0:n3, n3 + n2L:N]
    TARGET = np.deg2rad(a.angle)

    def unpack(p):
        return p[0], p[1], p[2], p[3:6], p[6:9], p[9:12], p[12:15], p[NCAM:].reshape(-1, 3)

    def residuals(p):
        f, cx, cy, rl, tl, rr, tr, X = unpack(p)
        e1 = (project(X, np.zeros(3), np.zeros(3), f, cx, cy) - obsF).ravel()
        e2 = (project(X[iL], rl, tl, f, cx, cy) - obsL).ravel()
        e3 = (project(X[iR], rr, tr, f, cx, cy) - obsR).ravel()
        ang = np.array([a.weight_angle * (np.linalg.norm(rl) - TARGET),
                        a.weight_angle * (np.linalg.norm(rr) - TARGET)])
        return np.concatenate([e1, e2, e3, ang])

    mF, mL, mR = 2 * N, 2 * (n3 + n2L), 2 * (n3 + n2R)
    J = lil_matrix((mF + mL + mR + 2, len(p0)), dtype=int)
    J[:, :NCAM] = 1
    for i in range(N):
        J[2 * i:2 * i + 2, NCAM + 3 * i:NCAM + 3 * i + 3] = 1
    for k, i in enumerate(iL):
        J[mF + 2 * k:mF + 2 * k + 2, NCAM + 3 * i:NCAM + 3 * i + 3] = 1
    for k, i in enumerate(iR):
        J[mF + mL + 2 * k:mF + mL + 2 * k + 2, NCAM + 3 * i:NCAM + 3 * i + 3] = 1

    print(f"residu initial RMS = {np.sqrt(np.mean(residuals(p0)[:-2]**2)):.3f} px")
    sol = least_squares(residuals, p0, jac_sparsity=J, x_scale="jac", method="trf",
                        loss="huber", f_scale=1.5, max_nfev=120, xtol=1e-10, ftol=1e-10)
    f, cx, cy, rl, tl, rr, tr, X = unpack(sol.x)
    rr_ = residuals(sol.x)
    eF = rr_[:mF].reshape(-1, 2)
    eL = rr_[mF:mF + mL].reshape(-1, 2)
    eR = rr_[mF + mL:mF + mL + mR].reshape(-1, 2)
    print("\n=== RESULTAT ===")
    print(f"reprojection mediane : frontal {np.median(np.linalg.norm(eF,axis=1)):.2f} px | "
          f"obl. G {np.median(np.linalg.norm(eL,axis=1)):.2f} px | "
          f"obl. D {np.median(np.linalg.norm(eR,axis=1)):.2f} px")
    print(f"focale         : {f:.0f} px (init {f0:.0f})")
    print(f"point principal: ({cx:.1f}, {cy:.1f}) | centre ({W/2:.1f}, {H/2:.1f}) "
          f"| ecart ({cx-W/2:+.1f}, {cy-H/2:+.1f}) px")
    print(f"angles         : {np.degrees(np.linalg.norm(rl)):.2f} / "
          f"{np.degrees(np.linalg.norm(rr)):.2f} deg "
          f"(somme {np.degrees(np.linalg.norm(rl))+np.degrees(np.linalg.norm(rr)):.2f})")
    if abs(f - f0) / f0 < 0.01:
        print("\nATTENTION : la focale n'a pas bouge. Verifier son identifiabilite "
              "avec focal_sweep.py avant de lui accorder du credit.")
    np.savez(f"{a.dir}/ba_result.npz", focal=f, cx=cx, cy=cy, rL=rl, tL=tl, rR=rr, tR=tr, X=X)

if __name__ == "__main__":
    main()

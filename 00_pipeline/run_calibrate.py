"""
run_calibrate.py — estime et GELE la geometrie du banc (une seule fois).

Les positions de camera sont une propriete du VISIA (angles indexes et
repetables), pas du sujet. Les estimer une fois puis les reutiliser supprime
une source majeure de variabilite entre sessions.

    python run_calibrate.py <sujet> <session> [--out rig.json]

Le fichier rig.json produit doit ensuite etre utilise pour TOUTES les sessions
comparees. Il est estampille avec CALIB_VERSION : toute incompatibilite est
detectee au chargement.
"""
import argparse, numpy as np, cv2
import config as C, visia_core as vc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject"); ap.add_argument("session")
    ap.add_argument("--out", default="rig.json")
    a = ap.parse_args()
    print(C.summary())

    paths = {k: vc.img_path(a.subject, a.session, k) for k in "FLR"}
    for k, p in paths.items():
        ok, msg = vc.check_file(p)
        print(f"  {k}: {msg}")
        if not ok:
            raise SystemExit("fichier invalide — corriger l'export avant de continuer")

    A = vc.load(paths["F"])
    K = vc.K_matrix(A.shape[1], A.shape[0])
    matcher = vc.Matcher()
    poses = {}
    for k in "LR":
        B = vc.load(paths[k])
        print(f"\n-- appariement F<->{k} (passe grossiere)")
        r = vc.match_pair(matcher, A, B, guide=None)
        if r is None:
            raise SystemExit(f"appariement F<->{k} echoue")
        pa, pb, F = r
        print(f"   {len(pa)} appariements verifies")
        R, t = vc.estimate_pose(pa, pb, K)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print(f"   angle recouvre : {ang:.2f} deg")
        poses[k] = (R, t)

    # mise a l'echelle relative des deux branches : indispensable, sinon les
    # deux demi-visages vivent a des echelles independantes.
    print("\n-- mise a l'echelle relative des deux branches")
    B_L = vc.load(paths["L"]); B_R = vc.load(paths["R"])
    rL = vc.match_pair(matcher, A, B_L, None); rR = vc.match_pair(matcher, A, B_R, None)
    from scipy.spatial import cKDTree
    tree = cKDTree(rR[0])
    d, idx = tree.query(rL[0], k=1)
    common = d < 3.0
    if common.sum() < 50:
        print("   ATTENTION : trop peu de points communs aux 3 vues, echelle relative peu fiable")
        s = 1.0
    else:
        XL = vc.triangulate(K, *poses["L"], rL[0][common], rL[1][common])
        XR = vc.triangulate(K, *poses["R"], rL[0][common], rR[1][idx[common]])
        ok = (XL[:, 2] > 0) & (XR[:, 2] > 0)
        s = float(np.median(XL[ok, 2] / XR[ok, 2]))
        print(f"   {common.sum()} points communs -> facteur d'echelle L/R = {s:.4f}")
    poses["R"] = (poses["R"][0], poses["R"][1] * s)

    vc.save_rig(a.out, poses, note=f"estime sur {a.subject}/{a.session}")
    print(f"\nrig gele -> {a.out}")
    print("Ce fichier doit etre utilise pour TOUTES les sessions comparees.")

if __name__ == "__main__":
    main()

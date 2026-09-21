"""
fuse_modalities.py — fusion des quatre modalites d'eclairage.

OBJET
    Reconstruire une chaine 3D COMPLETE par modalite, puis fusionner les cartes
    de profondeur. Les quatre modalites partagent le meme point de vue (1,5 px
    de decalage sur 2400, soit 0,06 % du champ), donc le meme rig s'applique et
    les nuages sont directement superposables.

CE QUI MARCHE, CE QUI NE MARCHE PAS
    Apparier ENTRE modalites differentes (frontal standard <-> oblique raked)
    echoue : chaque modalite porte une information dependante du point de vue.
    Ce qui marche est de reconstruire une chaine independante PAR modalite, puis
    de fusionner les resultats 3D.

RESULTAT — la couverture depend fortement du sujet
    modalite            sujet 1   sujet 2
    standard             51,9 %    32,9 %
    cross-polarise       45,5 %    41,8 %
    raked                41,9 %    27,0 %
    parallele-polarise   34,6 %    22,4 %
    UNION                74,6 %    64,4 %

    Le classement S'INVERSE selon le type de peau : sur un visage mature et
    texture le standard domine, sur un visage jeune et lisse c'est le
    cross-polarise. Le CP est la seule modalite stable entre sujets.

CE QUE LA FUSION N'APPORTE PAS
    Le moyennage des chaines ne reduit PAS le bruit : les erreurs sont
    correlees (rapport observe 1,004 contre 0,707 attendu si independantes).
    La fusion apporte de la COUVERTURE et de la REDONDANCE DE CONTROLE, pas de
    la precision.

USAGE
    python fuse_modalities.py --subject alban --session D0 --rig rig.json
"""
import argparse
import json
import cv2
import numpy as np
from face_mask import face_mask

MODALITES = [("Standard_1", "std"), ("Cross-Polarized", "cp"),
             ("Raked", "raked"), ("Parallel-Polarized", "pp")]


def triangulate(K, R, t, p1, p2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (X[:3] / X[3]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--dense-dir", default=".",
                    help="dossier des sorties de densification (obx, oby, epi)")
    ap.add_argument("--img-dir", default=".")
    ap.add_argument("--long", type=int, default=2000)
    ap.add_argument("--focal-rel", type=float, default=1.42222)
    ap.add_argument("--epi-max", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    A = cv2.imread(f"{a.img_dir}/{a.subject}_{a.session}_Frontal_Standard_1.jpg")
    h0, w0 = A.shape[:2]
    s = a.long / max(h0, w0)
    h, w = int(h0 * s), int(w0 * s)
    A = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    face = face_mask(A)[0]
    g = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    tex = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (15, 15))

    f = a.focal_rel * max(w, h)
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], float)
    D = np.zeros(4)
    und = lambda p: cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64),
                                        K, D, P=K).reshape(-1, 2)
    rig = json.load(open(a.rig))
    POSE = {k: (np.array(v["R"]), np.array(v["t"])) for k, v in rig["poses"].items()}

    def chain(pfx):
        Z = np.full((h, w), np.nan, np.float32)
        for tag in "LR":
            obx = np.load(f"{a.dense_dir}/{a.session}_{pfx}_obx_{tag}.npy")
            oby = np.load(f"{a.dense_dir}/{a.session}_{pfx}_oby_{tag}.npy")
            epi = np.load(f"{a.dense_dir}/{a.session}_{pfx}_epi_{tag}.npy")
            hi = face & (~np.isnan(obx)) & (epi < a.epi_max) & (tex > 5)
            ys, xs = np.nonzero(hi)
            p1 = np.column_stack([xs, ys]).astype(np.float32)
            p2 = np.column_stack([obx[ys, xs], oby[ys, xs]]).astype(np.float32)
            X = triangulate(K, *POSE[tag], und(p1), und(p2))
            m = np.isfinite(X).all(1) & (X[:, 2] > 0)
            Z[ys[m], xs[m]] = X[m, 2]
        return Z

    Zs, noms = [], []
    Zref = None
    for mod, pfx in MODALITES:
        try:
            Zi = chain(pfx)
        except FileNotFoundError:
            print(f"  {mod:20s} : donnees absentes, ignoree")
            continue
        if Zref is None:
            Zref = Zi
        else:
            # recalage d'echelle sur la zone commune avec la chaine de reference
            both = np.isfinite(Zref) & np.isfinite(Zi)
            if both.sum() > 1000:
                u, v = Zi[both], Zref[both]
                Zi = (Zi - u.mean()) * (v.std() / max(u.std(), 1e-9)) + v.mean()
        cov = 100 * np.isfinite(Zi).sum() / face.sum()
        print(f"  {mod:20s} : couverture {cov:5.1f} % du masque")
        Zs.append(Zi)
        noms.append(mod)

    Z = np.array(Zs)
    nch = np.sum(np.isfinite(Z), axis=0)
    Zf = np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=0)
    print(f"\nUNION : {100*np.isfinite(Zf).sum()/face.sum():5.1f} % du masque")
    print("redondance :")
    for k in range(len(Zs) + 1):
        print(f"  {k} modalite(s) : {100*((nch==k)&face).sum()/face.sum():5.1f} %")
    out = a.out or f"fusion_{a.subject}_{a.session}.npz"
    np.savez(out, Z=Zf, Zall=Z, nch=nch, face=face, modalites=noms, focal=f)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

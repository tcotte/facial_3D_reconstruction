"""
run_session.py — reconstruit une session (D0 ou Dx) avec le rig GELE.

    python run_session.py <sujet> <session> --rig rig.json

Produit <sujet>_<session>.npz :
    pix      (N,2) position des points dans le repere de la vue frontale
    X        (N,3) coordonnees 3D (unites arbitraires, NON metriques)
    conf     (N,)  residu epipolaire, en px (plus petit = plus fiable)
    src      (N,)  'L' ou 'R' selon la paire d'origine
plus une carte de confiance PNG.
"""
import argparse, numpy as np, cv2
import config as C, visia_core as vc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject"); ap.add_argument("session")
    ap.add_argument("--rig", default="rig.json")
    ap.add_argument("--step", type=int, default=2, help="sous-echantillonnage des pixels")
    a = ap.parse_args()
    print(C.summary())
    rig = vc.load_rig(a.rig)

    paths = {k: vc.img_path(a.subject, a.session, k) for k in "FLR"}
    for k, p in paths.items():
        ok, msg = vc.check_file(p)
        print(f"  {k}: {msg}")
        if not ok:
            raise SystemExit("fichier invalide")

    A = vc.load(paths["F"])
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    h, w = gA.shape
    K = vc.K_matrix(w, h)
    fmask = vc.face_mask(gA)
    tex = cv2.blur(np.abs(cv2.Laplacian(gA, cv2.CV_32F, ksize=3)), (15, 15))
    matcher = vc.Matcher()

    PIX, X3, CONF, SRC = [], [], [], []
    conf_img = (A * 0.5).astype(np.uint8)
    for k in "LR":
        B = vc.load(paths[k])
        print(f"\n-- paire F<->{k}")
        r1 = vc.match_pair(matcher, A, B, guide=None)
        if r1 is None:
            print("   passe grossiere echouee"); continue
        pa, pb, F = r1
        print(f"   passe grossiere : {len(pa)} appariements")
        obx, oby, epi, wm = vc.densify(A, B, pa, pb, F)
        print(f"   densification : residu epipolaire median {np.median(epi[np.isfinite(epi)]):.2f} px")

        high = fmask & wm & (epi < C.EPI_HIGH) & (tex > C.TEXTURE_MIN)
        med = fmask & wm & (epi < C.EPI_MED) & ~high
        print(f"   mesure {100*high.sum()/fmask.sum():.1f} % | faible {100*med.sum()/fmask.sum():.1f} %")

        ys, xs = np.nonzero(high)
        ys, xs = ys[::a.step], xs[::a.step]
        p1 = np.column_stack([xs, ys]).astype(np.float64)
        p2 = np.column_stack([obx[ys, xs], oby[ys, xs]]).astype(np.float64)
        u1, u2 = vc.undistort(p1.astype(np.float32), K), vc.undistort(p2.astype(np.float32), K)
        Xk = vc.triangulate(K, *rig[k], u1, u2)
        ok = np.isfinite(Xk).all(1) & (Xk[:, 2] > 0)
        PIX.append(p1[ok]); X3.append(Xk[ok])
        CONF.append(epi[ys, xs][ok]); SRC.append(np.full(ok.sum(), k))
        conf_img[high] = (0.3 * conf_img[high] + 0.7 * np.array([90, 220, 90])).astype(np.uint8)
        conf_img[med] = (0.55 * conf_img[med] + 0.45 * np.array([60, 190, 255])).astype(np.uint8)

    if not PIX:
        raise SystemExit("aucune reconstruction produite")
    pix = np.vstack(PIX); X = np.vstack(X3)
    conf = np.concatenate(CONF); src = np.concatenate(SRC)
    out = f"{a.subject}_{a.session}"
    np.savez_compressed(out + ".npz", pix=pix, X=X, conf=conf, src=src,
                        shape=np.array([h, w]), calib=C.CALIB_VERSION)
    cv2.imwrite(out + "_confiance.png", conf_img)
    print(f"\n{len(X)} points 3D -> {out}.npz")
    print(f"carte de confiance -> {out}_confiance.png")

if __name__ == "__main__":
    main()

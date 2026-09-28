"""
run_session.py — reconstruit une session (D0 ou Dx) avec le rig GELE.

    python run_session.py <sujet> <session> --rig rig.json

    # densification par le champ RoMa v2 exporte depuis la machine GPU
    python run_session.py <sujet> <session> --rig rig.json --step 1 \
        --roma roma2_<sujet>_<session>_Standard_1.npz

Avec --roma, la densification (pre-recalage Delaunay + flot optique DIS) est
remplacee par la lecture du champ RoMa v2. TOUT LE RESTE EST IDENTIQUE : meme
rig gele, meme focale, memes seuils, meme triangulation. Le fichier produit
porte alors le stamp CALIB_VERSION + '+roma2' : run_compare.py refusera de le
comparer a une session reconstruite autrement, ce qui est le comportement
voulu (changer la densification change les valeurs).

Avec --roma, passer --step 1 : le sous-echantillonnage a deja ete fait a
l'export (parametre `step` du fichier), le reappliquer ici deciderait deux
fois.

Produit <sujet>_<session>.npz :
    pix      (N,2) position des points dans le repere de la vue frontale
    X        (N,3) coordonnees 3D (unites arbitraires, NON metriques)
    conf     (N,)  residu epipolaire, en px (plus petit = plus fiable)
    src      (N,)  'L' ou 'R' selon la paire d'origine
    cert     (N,)  overlap RoMa dans [0,1]           (--roma uniquement)
    sigma    (N,)  ecart-type de localisation, en px (--roma uniquement)
plus une carte de confiance PNG.
"""
import argparse, os
import numpy as np, cv2
import config as C, visia_core as vc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject"); ap.add_argument("session")
    ap.add_argument("--rig", default="rig.json")
    ap.add_argument("--step", type=int, default=2, help="sous-echantillonnage des pixels")
    ap.add_argument("--roma", default=None,
                    help="champ RoMa v2 exporte (.npz) ; remplace la densification")
    ap.add_argument("--roma-fill", action="store_true",
                    help="interpole le champ entre les points exportes "
                         "(gonfle le comptage, a eviter pour un chiffre d'appariements)")
    ap.add_argument("--roma-cert", type=float, default=0.05,
                    help="seuil d'overlap RoMa")
    ap.add_argument("--roma-sigma-max", type=float, default=None,
                    help="rejet sur l'ecart-type de localisation predit (px)")
    ap.add_argument("--roma-spread-max", type=float, default=None,
                    help="rejet sur le desaccord inter-tuiles (px)")
    ap.add_argument("--epi-high", type=float, default=None,
                    help="remplace config.EPI_HIGH pour CETTE reconstruction. "
                         "La valeur est inscrite dans le stamp : deux sessions "
                         "reconstruites avec des seuils differents ne pourront "
                         "pas etre comparees.")
    ap.add_argument("--epi-med", type=float, default=None,
                    help="remplace config.EPI_MED (idem)")
    ap.add_argument("--texture-min", type=float, default=None,
                    help="remplace config.TEXTURE_MIN (idem). Ce filtre rejette "
                         "la peau lisse ; c'est souvent le poste de perte le "
                         "plus lourd sur un visage jeune.")
    ap.add_argument("--epi-from", choices=["rig", "field"], default="rig",
                    help="origine de F pour le residu epipolaire (--roma uniquement). "
                         "'rig' = independant de l'appariement, recommande.")
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
    epi_high = a.epi_high if a.epi_high is not None else C.EPI_HIGH
    epi_med = a.epi_med if a.epi_med is not None else C.EPI_MED
    tex_min = a.texture_min if a.texture_min is not None else C.TEXTURE_MIN
    field = None
    stamp = C.CALIB_VERSION
    if a.roma:
        import roma_field
        field = roma_field.load(a.roma)
        field.check(a.subject, a.session, (h, w), strict=True)
        stamp = C.CALIB_VERSION + roma_field.ROMA_STAMP
        print(f"\ndensification : champ RoMa v2 ({os.path.basename(a.roma)})")
        field.report()
        if a.step != 1:
            print(f"   NOTE : --step {a.step} s'ajoute au pas {field.step} de "
                  f"l'export, soit un pas effectif de {a.step * field.step}.")
    else:
        print("\ndensification : pre-recalage Delaunay + flot optique DIS")

    if a.epi_high is not None or a.epi_med is not None or a.texture_min is not None:
        suff = f"+epi{epi_high:g}-{epi_med:g}"
        if a.texture_min is not None:
            suff += f"-tex{tex_min:g}"
        stamp += suff
        print(f"\nSEUILS MODIFIES : EPI_HIGH={epi_high} (config {C.EPI_HIGH}), "
              f"EPI_MED={epi_med} (config {C.EPI_MED}), "
              f"TEXTURE_MIN={tex_min} (config {C.TEXTURE_MIN})")
        print(f"   inscrits dans le stamp ('{suff}') : ces sessions ne se compareront")
        print(f"   qu'entre elles. config.py reste inchange.")

    matcher = vc.Matcher() if field is None else None

    PIX, X3, CONF, SRC, CERT, SIG = [], [], [], [], [], []
    conf_img = (A * 0.5).astype(np.uint8)
    for k in "LR":
        print(f"\n-- paire F<->{k}")
        if field is not None:
            if k not in field.pairs:
                print(f"   paire absente du champ exporte"); continue
            obx, oby, epi, wm = field.densify_like(
                k, K, *rig[k], fill=a.roma_fill, epi_from=a.epi_from,
                cert_min=a.roma_cert, sigma_max=a.roma_sigma_max,
                spread_max=a.roma_spread_max)
            cert_k = field.cert_map(k, fill=a.roma_fill)
            sig_k = field.sigma_map(k, fill=a.roma_fill)[..., 0]
            print(f"   champ RoMa : {int(wm.sum())} pixels apparies | "
                  f"residu epipolaire median (F issue du {a.epi_from}) "
                  f"{np.median(epi[np.isfinite(epi)]):.2f} px")
        else:
            B = vc.load(paths[k])
            r1 = vc.match_pair(matcher, A, B, guide=None)
            if r1 is None:
                print("   passe grossiere echouee"); continue
            pa, pb, F = r1
            print(f"   passe grossiere : {len(pa)} appariements")
            obx, oby, epi, wm = vc.densify(A, B, pa, pb, F)
            cert_k = sig_k = None
            print(f"   densification : residu epipolaire median "
                  f"{np.median(epi[np.isfinite(epi)]):.2f} px")

        # distribution du residu : de quoi choisir un seuil en connaissance de cause
        ef = epi[wm & fmask]
        ef = ef[np.isfinite(ef)]
        if len(ef):
            qs = [np.percentile(ef, q) for q in (25, 50, 75, 90)]
            print(f"   residu epipolaire : 25e {qs[0]:.2f} | median {qs[1]:.2f} | "
                  f"75e {qs[2]:.2f} | 90e {qs[3]:.2f} px")
            for seuil in (0.5, 1.0, 1.5, 2.0, 3.0):
                print(f"      sous {seuil:.1f} px : {100*(ef < seuil).mean():5.1f} %"
                      + ("   <== seuil retenu" if abs(seuil - epi_high) < 1e-9 else ""))

        # cascade des rejets : ou partent les points, et a cause de quel filtre
        nm = int(fmask.sum())
        n_app = int((fmask & wm).sum())
        n_epi = int((fmask & wm & (epi < epi_high)).sum())
        n_tex = int((fmask & wm & (tex > tex_min)).sum())
        high = fmask & wm & (epi < epi_high) & (tex > tex_min)
        med = fmask & wm & (epi < epi_med) & ~high
        print(f"   cascade sur {nm} pixels du masque facial :")
        print(f"      apparies par le champ      {n_app:9d}  {100*n_app/nm:5.1f} %"
              f"   (perte : echantillonnage et couverture)")
        print(f"      + residu < {epi_high:<4g} px          {n_epi:9d}  {100*n_epi/nm:5.1f} %"
              f"   (ce seul filtre coute {100*(n_app-n_epi)/max(1,n_app):.0f} %)")
        print(f"      + texture > {tex_min:<4g}            {n_tex:9d}  {100*n_tex/nm:5.1f} %"
              f"   (ce seul filtre coute {100*(n_app-n_tex)/max(1,n_app):.0f} %)")
        print(f"      les deux ensemble          {int(high.sum()):9d}  "
              f"{100*high.sum()/nm:5.1f} %   <= 'mesure'")
        plafond = 100.0 * n_app / nm
        print(f"   mesure {100*high.sum()/fmask.sum():.1f} % | faible {100*med.sum()/fmask.sum():.1f} %"
              f"  (plafond impose par l'echantillonnage : {plafond:.1f} %)")

        ys, xs = np.nonzero(high)
        ys, xs = ys[::a.step], xs[::a.step]
        p1 = np.column_stack([xs, ys]).astype(np.float64)
        p2 = np.column_stack([obx[ys, xs], oby[ys, xs]]).astype(np.float64)
        u1, u2 = vc.undistort(p1.astype(np.float32), K), vc.undistort(p2.astype(np.float32), K)
        Xk = vc.triangulate(K, *rig[k], u1, u2)
        ok = np.isfinite(Xk).all(1) & (Xk[:, 2] > 0)
        PIX.append(p1[ok]); X3.append(Xk[ok])
        CONF.append(epi[ys, xs][ok]); SRC.append(np.full(ok.sum(), k))
        if cert_k is not None:
            CERT.append(cert_k[ys, xs][ok]); SIG.append(sig_k[ys, xs][ok])
        conf_img[high] = (0.3 * conf_img[high] + 0.7 * np.array([90, 220, 90])).astype(np.uint8)
        conf_img[med] = (0.55 * conf_img[med] + 0.45 * np.array([60, 190, 255])).astype(np.uint8)

    if not PIX:
        raise SystemExit("aucune reconstruction produite")
    pix = np.vstack(PIX); X = np.vstack(X3)
    conf = np.concatenate(CONF); src = np.concatenate(SRC)
    out = f"{a.subject}_{a.session}"
    extra = {}
    if CERT:
        extra["cert"] = np.concatenate(CERT)
        extra["sigma"] = np.concatenate(SIG)
        extra["roma_source"] = os.path.basename(a.roma)
    np.savez_compressed(out + ".npz", pix=pix, X=X, conf=conf, src=src,
                        shape=np.array([h, w]), calib=stamp, **extra)
    cv2.imwrite(out + "_confiance.png", conf_img)
    print(f"\n{len(X)} points 3D -> {out}.npz  [{stamp}]")
    print(f"carte de confiance -> {out}_confiance.png")

if __name__ == "__main__":
    main()

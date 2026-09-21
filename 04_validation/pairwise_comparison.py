"""
pairwise_comparison.py — comparaison D0/Dx avec une chaine par PAIRE STEREO.

DECOUVERTE STRUCTURELLE QUI MOTIVE CE SCRIPT
    Chaque joue n'est reconstruite que par UNE seule paire stereo :
        joue cote image gauche  -> paire F<->R a 100 %
        joue cote image droite  -> paire F<->L a  99 %
    (consequence de l'absence de recouvrement entre les deux obliques)

    Fusionner les deux nuages AVANT recalage injecte le desaccord residuel entre
    paires sous forme de biais ANTISYMETRIQUE. Mesure sur zone stable, qui n'a
    par definition pas bouge :
        points issus de la paire L : +0,049 %
        points issus de la paire R : -0,048 %
    Cet artefact est DU MEME ORDRE que le signal recherche, et produit des
    signes opposes sur les deux joues.

CORRECTION
    Chaque paire devient une chaine de mesure independante, avec son propre
    recalage sur sa propre portion de zone stable. Les deux mesures deviennent
    alors mutuellement controlables : leur concordance est un indicateur qualite.

RESULTAT OBTENU (gonflement volontaire des joues)
    avant : +0,118 / -0,061  (SIGNES OPPOSES)
    apres : +0,181 / +0,146  (meme signe, ecart 21 %)

USAGE
    python pairwise_comparison.py --ref sess_D0.npz --test sess_Dx.npz \
        --link link_D0_Dx.npz --stable m_stable.npy \
        --roi-left m_joueL.npy --roi-right m_joueR.npy
"""
import argparse, json, numpy as np, cv2
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# la paire L observe la joue cote image DROITE, et reciproquement
CHEEK = {"L": "right", "R": "left"}


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


def noise_floor(X0, Xx, stable_idx, pix, nblocks=4):
    """Plancher de bruit : recalage sur 3/4 de la zone stable, mesure sur 1/4."""
    ps = pix[stable_idx]
    gx = np.digitize(ps[:, 0], np.linspace(ps[:, 0].min(), ps[:, 0].max(), nblocks))
    gy = np.digitize(ps[:, 1], np.linspace(ps[:, 1].min(), ps[:, 1].max(), nblocks))
    cell = gx * 10 + gy
    vals = []
    for c_ in np.unique(cell):
        te, tr = stable_idx[cell == c_], stable_idx[cell != c_]
        if len(te) < 150 or len(tr) < 800:
            continue
        s, R, t = umeyama(Xx[tr], X0[tr])
        Xa = s * (R @ Xx.T).T + t
        co = plane_fit(X0[tr])
        dn = np.std(relief(X0[te], co))
        if dn > 0:
            vals.append(np.mean(X0[te, 2] - Xa[te, 2]) / dn)
    v = np.array(vals)
    return (2 * v.std() if len(v) > 2 else np.nan), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--link", required=True,
                    help="npz avec p0/p1 : appariement des vues frontales des deux sessions")
    ap.add_argument("--stable", required=True)
    ap.add_argument("--roi-left", required=True); ap.add_argument("--roi-right", required=True)
    ap.add_argument("--out", default="comparaison")
    a = ap.parse_args()

    d0, dx = np.load(a.ref), np.load(a.test)
    lk = np.load(a.link)
    tgt = griddata(lk["p0"], lk["p1"], d0["pix"], method="linear")
    ok = np.isfinite(tgt).all(1)
    dist, idx = cKDTree(dx["pix"]).query(tgt[ok], k=1)
    paired = dist < 2.0
    i0 = np.nonzero(ok)[0][paired]; ix = idx[paired]
    X0, Xx, pix = d0["X"][i0].astype(float), dx["X"][ix].astype(float), d0["pix"][i0]
    src0, srcx = d0["src"][i0], dx["src"][ix]
    print(f"points apparies entre sessions : {len(X0)} "
          f"({100*len(X0)/len(d0['X']):.1f} % de la reference)")

    h, w = d0["shape"]
    xi = np.clip(pix[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pix[:, 1].astype(int), 0, h - 1)
    M = {"stable": np.load(a.stable)[yi, xi],
         "left": np.load(a.roi_left)[yi, xi],
         "right": np.load(a.roi_right)[yi, xi]}
    zmed = np.median(X0[:, 2])
    res = {}
    for p in "LR":
        sel = (src0 == p) & (srcx == p)
        st = M["stable"] & sel
        roi = M[CHEEK[p]] & sel
        if st.sum() < 500 or roi.sum() < 300:
            print(f"paire {p} : couverture insuffisante"); continue
        s, R, t = umeyama(Xx[st], X0[st])
        Xa = s * (R @ Xx.T).T + t
        dev = X0[:, 2] - Xa[:, 2]
        co = plane_fit(X0[st])
        ind = float(np.mean(dev[roi]) / np.std(relief(X0[roi], co)))
        seuil, nb = noise_floor(X0, Xx, np.nonzero(st)[0], pix)
        res[p] = dict(roi=CHEEK[p], n_stable=int(st.sum()), n_roi=int(roi.sum()),
                      scale=float(s), pct=float(100 * np.mean(dev[roi]) / zmed),
                      indicateur=ind, seuil_2sigma=float(seuil), n_blocs=nb,
                      controle_stable=float(100 * np.mean(dev[st]) / zmed))
        print(f"\n--- paire F<->{p} (joue {CHEEK[p]}) ---")
        print(f"  zone stable {st.sum()} pts | ROI {roi.sum()} pts | echelle {s:.4f}")
        print(f"  deviation   : {res[p]['pct']:+.4f} % | indicateur {ind:+.4f}")
        print(f"  plancher 2s : {seuil:.4f} ({nb} blocs)")
        print(f"  controle zone stable : {res[p]['controle_stable']:+.4f} % (attendu 0)")
        print(f"  --> {'SIGNIFICATIF' if abs(ind) > seuil else 'non significatif'} "
              f"({abs(ind)/seuil:.2f}x)")
    if len(res) == 2:
        iL, iR = res["L"]["indicateur"], res["R"]["indicateur"]
        ec = 100 * abs(iL - iR) / np.mean([abs(iL), abs(iR)])
        print(f"\n=== CONCORDANCE DES DEUX CHAINES ===")
        print(f"meme signe : {'OUI' if iL*iR > 0 else 'NON — resultat non fiable'}")
        print(f"ecart entre chaines : {ec:.0f} %")
        res["concordance"] = dict(meme_signe=bool(iL * iR > 0), ecart_pct=float(ec))
    with open(a.out + "_metriques.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n-> {a.out}_metriques.json")

if __name__ == "__main__":
    main()

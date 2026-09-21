"""
volume_map.py — carte de variation de volume local entre deux sessions.

OBJET
    Produire une carte signee des variations, exprimee en % du relief facial,
    avec un seuil de detection mesure sur la zone stable.

TROIS REGLES METHODOLOGIQUES

1. UNE CHAINE DE MESURE PAR PAIRE STEREO
   Chaque joue n'est reconstruite que par UNE paire (absence de recouvrement
   entre obliques). Fusionner avant recalage injecte un biais ANTISYMETRIQUE du
   meme ordre que le signal. Chaque paire est donc recalee independamment sur sa
   propre portion de zone stable.

2. NORMALISATION GLOBALE, PAS LOCALE
   Diviser par le relief LOCAL explose sur les zones plates (plancher de bruit
   mesure a 24,9 % — inexploitable). La deviation est rapportee a l'amplitude de
   relief du VISAGE ENTIER (P3-P97).

3. LE PLANCHER DE BRUIT SE MESURE SUR DES BLOCS, PAS PIXEL A PIXEL
   L'ecart-type pixel a pixel surestime fortement l'incertitude sur une MOYENNE
   REGIONALE : le bruit est spatialement correle, mais pas totalement. Mesure
   par blocs de taille comparable aux ROI :
       blocs de  80 px : seuil 2 sigma = 1,43 %
       blocs de 160 px : seuil 2 sigma = 1,06 %   <- retenu
   Contre 4,71 % avec l'ecart-type pixel a pixel, qui declarait tout non
   significatif.

LIMITE NON RESOLUE
   Les deux chaines sont SPATIALEMENT DISJOINTES : recouvrement de 1,9 % du
   visage seulement, chaque paire ayant sa zone stable ET sa ROI du meme cote.
   Comparer leurs signes n'a donc pas la valeur d'un controle croise. C'est le
   verrou structurel V5 (pas de recouvrement entre obliques).

USAGE
    python volume_map.py --ref fusion_alban_D0_filtre.npz \
        --test fusion_alban_D14_filtre.npz --link link_alban_D0_D14.npz \
        --stable m_stable.npy
"""
import argparse
import json
import cv2
import numpy as np


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(S)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    sc = np.trace(np.diag(D) @ E) / ((Xc ** 2).sum() / len(X))
    return sc, R, my - sc * R @ mx


def plancher_bloc(vol, stable, tailles=(80, 120, 160, 200)):
    """Plancher de bruit pour une MOYENNE REGIONALE, mesure par blocs."""
    h, w = vol.shape
    out = {}
    for B in tailles:
        vals = []
        for i in range(0, h - B, B):
            for j in range(0, w - B, B):
                m = stable[i:i+B, j:j+B]
                if m.sum() < B * B * 0.45:
                    continue
                vals.append(float(np.nanmean(vol[i:i+B, j:j+B][m])))
        if len(vals) >= 4:
            out[B] = 2 * np.std(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-L", required=True); ap.add_argument("--ref-R", required=True)
    ap.add_argument("--test-L", required=True); ap.add_argument("--test-R", required=True)
    ap.add_argument("--link", required=True)
    ap.add_argument("--stable", required=True)
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--focal-rel", type=float, default=1.42222)
    ap.add_argument("--out", default="variation_volume")
    a = ap.parse_args()

    A = cv2.imread(a.frontal)
    h, w = A.shape[:2]
    f = a.focal_rel * max(w, h)
    lk = np.load(a.link)
    mapx, mapy = lk["mapx"], lk["mapy"]
    stable = np.load(a.stable)

    DEV, SIG = {}, {}
    for tag, fr, ft in [("L", a.ref_L, a.test_L), ("R", a.ref_R, a.test_R)]:
        Z0 = np.load(fr)["Z"]
        Zx = cv2.remap(np.load(ft)["Z"], mapx, mapy, cv2.INTER_LINEAR,
                       borderValue=np.nan)
        ok = np.isfinite(Z0) & np.isfinite(Zx)
        st = ok & stable
        ys, xs = np.nonzero(st)
        P0 = np.column_stack([(xs-w/2)/f*Z0[ys, xs], (ys-h/2)/f*Z0[ys, xs], Z0[ys, xs]])
        PX = np.column_stack([(xs-w/2)/f*Zx[ys, xs], (ys-h/2)/f*Zx[ys, xs], Zx[ys, xs]])
        sc, R, t = umeyama(PX, P0)
        ys2, xs2 = np.nonzero(ok)
        PA = np.column_stack([(xs2-w/2)/f*Zx[ys2, xs2], (ys2-h/2)/f*Zx[ys2, xs2], Zx[ys2, xs2]])
        PA = sc * (R @ PA.T).T + t
        dev = np.full((h, w), np.nan, np.float32)
        dev[ys2, xs2] = Z0[ys2, xs2] - PA[:, 2]
        DEV[tag] = dev
        zm = float(np.nanmedian(Z0[ok]))
        print(f"paire {tag} : {int(ok.sum())} px comparables | echelle {sc:.4f} | "
              f"controle zone stable {100*np.nanmean(dev[st])/zm:+.4f} % "
              f"(ecart-type {100*np.nanstd(dev[st])/zm:.4f} %)")
        SIG[tag] = zm

    dev = np.where(np.isfinite(DEV["L"]), DEV["L"], DEV["R"])
    ok = np.isfinite(dev)
    Z0 = np.load(a.ref_R)["Z"]
    ys, xs = np.nonzero(stable & np.isfinite(Z0))
    M = np.column_stack([xs, ys, np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(M, Z0[ys, xs], rcond=None)
    ref = coef[0]*np.arange(w)[None, :] + coef[1]*np.arange(h)[:, None] + coef[2]
    relief = np.where(ok, ref - Z0, np.nan)
    AMP = float(np.nanpercentile(relief[ok], 97) - np.nanpercentile(relief[ok], 3))
    vol = 100 * dev / AMP
    seuils = plancher_bloc(vol, stable & ok)
    S = seuils.get(160, np.nan)
    print(f"\namplitude de relief du visage : {AMP:.4f} u.a.")
    print("plancher de bruit (2 sigma) par taille de bloc :")
    for B, v in seuils.items():
        print(f"   {B:4d} px : {v:.2f} %" + ("   <- retenu" if B == 160 else ""))
    np.savez(a.out + ".npz", vol=vol, dev=dev, ok=ok, seuil=S, amplitude=AMP)
    json.dump({"seuil_2sigma_pct": float(S), "amplitude_relief_ua": AMP},
              open(a.out + ".json", "w"), indent=2)
    print(f"-> {a.out}.npz / .json")


if __name__ == "__main__":
    main()

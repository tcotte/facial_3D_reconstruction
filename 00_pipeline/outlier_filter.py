"""
outlier_filter.py — rejet des points aberrants.

OBJET
    Les zones a faible redondance produisent des points fortement ecartes des
    moyennes locales, principalement autour des yeux, du nez et des bords du
    visage. Quatre criteres INDEPENDANTS sont combines par union.

CRITERES ET TAUX DE REJET MESURES
    ecart a la mediane locale (15 px, 3,5 x MAD)   5,7 - 6,0 %
    desaccord entre modalites > 1 %                3,5 - 3,6 %
    points isoles (< 35 voisins sur 121)           0,4 - 0,6 %
    gradient de profondeur excessif                6,6 - 7,9 %
    UNION                                         12,2 - 13,4 %

    L'union rejette 12-13 % alors que la somme depasse 16 % : les criteres se
    recoupent largement, ce qui est rassurant sur la coherence du diagnostic.

POINT DE VIGILANCE
    Le critere de gradient est le plus prolifique ET le plus risque : un
    gradient fort est legitime sur l'arete du nez ou le bord des paupieres. Le
    seuil retenu (2 % de la distance de travail par pixel) preserve le relief
    anatomique reel, mais c'est le parametre a surveiller en priorite si vous
    constatez une perte de detail.

USAGE
    python outlier_filter.py --fusion fusion_alban_D0.npz
"""
import argparse
import cv2
import numpy as np
from scipy.ndimage import median_filter


def filtre(Z, Zall, nch, face, win=15, k_mad=3.5, plancher=0.0035,
           desaccord=0.010, voisins_min=35, grad_max=0.02):
    ok = np.isfinite(Z) & face
    zm = float(np.nanmedian(Z[ok]))

    filled = np.where(ok, Z, zm)
    med = median_filter(filled, size=win, mode="nearest")
    dev = np.abs(Z - med)
    mad = median_filter(np.where(ok, dev, 0), size=win, mode="nearest")
    r1 = ok & (dev > np.maximum(k_mad * 1.4826 * mad, plancher * zm))

    with np.errstate(invalid="ignore"):
        disp = np.nanstd(np.where(np.isfinite(Zall), Zall, np.nan), axis=0)
    r2 = ok & (nch >= 2) & (disp > desaccord * zm)

    cnt = cv2.blur(ok.astype(np.float32), (11, 11)) * 121
    r3 = ok & (cnt < voisins_min)

    Zn = np.where(ok, Z, np.nan).astype(np.float32)
    gx = cv2.Sobel(Zn, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Zn, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(np.nan_to_num(gx)**2 + np.nan_to_num(gy)**2)
    r4 = ok & (gm > grad_max * zm)

    return ok, {"mediane locale": r1, "desaccord modalites": r2,
                "isolement": r3, "gradient": r4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = np.load(a.fusion, allow_pickle=True)
    Z, Zall, nch, face = d["Z"], d["Zall"], d["nch"], d["face"]
    ok, crit = filtre(Z, Zall, nch, face)
    print(f"{'critere':34s} {'rejetes':>9s} {'%':>7s}")
    rej = np.zeros_like(ok)
    for nom, r in crit.items():
        print(f"{nom:34s} {int(r.sum()):9d} {100*r.sum()/ok.sum():6.1f}%")
        rej |= r
    keep = ok & ~rej
    print(f"\n{'REJET TOTAL (union)':34s} {int(rej.sum()):9d} {100*rej.sum()/ok.sum():6.1f}%")
    print(f"{'CONSERVES':34s} {int(keep.sum()):9d} {100*keep.sum()/ok.sum():6.1f}%")
    print(f"couverture finale : {100*keep.sum()/face.sum():.1f} % du masque")
    out = a.out or a.fusion.replace(".npz", "_filtre.npz")
    np.savez(out, Z=np.where(keep, Z, np.nan), keep=keep, rej=rej,
             nch=nch, face=face, focal=d["focal"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()

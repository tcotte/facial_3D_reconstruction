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
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "04_validation"))


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
    ap.add_argument("--grad-max", type=float, default=0.02,
                    help="gradient de profondeur max, en fraction de la distance "
                         "de travail par pixel. C'est LE parametre a surveiller : "
                         "un gradient fort est legitime sur l'arete du nez et le "
                         "bord des paupieres.")
    ap.add_argument("--k-mad", type=float, default=3.5,
                    help="ecart a la mediane locale, en MAD")
    ap.add_argument("--desaccord", type=float, default=0.010,
                    help="desaccord entre modalites tolere")
    ap.add_argument("--voisins-min", type=int, default=35,
                    help="voisins minimum sur 121")
    ap.add_argument("--par-zone", action="store_true",
                    help="detailler les rejets par region du visage")
    a = ap.parse_args()
    d = np.load(a.fusion, allow_pickle=True)
    Z, Zall, nch, face = d["Z"], d["Zall"], d["nch"], d["face"]
    ok, crit = filtre(Z, Zall, nch, face, k_mad=a.k_mad, desaccord=a.desaccord,
                      voisins_min=a.voisins_min, grad_max=a.grad_max)
    print(f"{'critere':34s} {'rejetes':>9s} {'%':>7s}")
    rej = np.zeros_like(ok)
    for nom, r in crit.items():
        print(f"{nom:34s} {int(r.sum()):9d} {100*r.sum()/ok.sum():6.1f}%")
        rej |= r
    keep = ok & ~rej
    print(f"\n{'REJET TOTAL (union)':34s} {int(rej.sum()):9d} {100*rej.sum()/ok.sum():6.1f}%")
    print(f"{'CONSERVES':34s} {int(keep.sum()):9d} {100*keep.sum()/ok.sum():6.1f}%")
    print(f"couverture finale : {100*keep.sum()/face.sum():.1f} % du masque")
    if a.par_zone:
        try:
            from coverage_by_zone import zone_maps
        except ImportError:
            print("\n(coverage_by_zone.py introuvable, detail par zone ignore)")
        else:
            zones = zone_maps(face)
            larg = max(len(n) for n in zones) + 1
            noms = list(crit)
            ent = f"{'region':{larg}s}" + "".join(f"{n[:11]:>13s}" for n in noms) \
                  + f"{'TOTAL':>9s}"
            print(f"\nrejets par region, en % des points mesures de la region\n{ent}")
            print("-" * len(ent))
            for nom, z in zones.items():
                base = int((ok & z).sum())
                if base == 0:
                    continue
                ligne = f"{nom:{larg}s}"
                for n in noms:
                    ligne += f"{100*(crit[n] & z).sum()/base:12.1f}%"
                print(ligne + f"{100*(rej & z).sum()/base:8.1f}%")
            print("\nUn rejet massif concentre sur une seule region trahit un")
            print("critere mal calibre pour l'anatomie, pas des points aberrants.")

    out = a.out or a.fusion.replace(".npz", "_filtre.npz")
    np.savez(out, Z=np.where(keep, Z, np.nan), keep=keep, rej=rej,
             nch=nch, face=face, focal=d["focal"],
             **{f"rej_{n.replace(' ', '_')}": v for n, v in crit.items()})
    print(f"-> {out}")


if __name__ == "__main__":
    main()

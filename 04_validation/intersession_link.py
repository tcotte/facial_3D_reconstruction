"""
intersession_link.py — mise en correspondance des vues frontales de deux sessions.

OBJET
    Etablir le champ de correspondance D0 -> Dx qui permet de comparer les deux
    reconstructions point a point. Etape critique : c'est elle qui determine la
    surface comparable.

DEUX ERREURS RENCONTREES, TOUTES DEUX CORRIGEES ICI

1. APPARIEMENT PLEINE IMAGE INSUFFISANT
   Un seul appel LightGlue en modalite standard donnait 1760 appariements, dont
   l'enveloppe convexe ne couvrait pas le bas du visage. L'interpolation
   lineaire renvoyait NaN au-dela -> trous a BORD RECTILIGNE dans la carte de
   comparaison, signature d'une frontiere d'enveloppe convexe et non d'un
   phenomene physique.
       transfert defini            83,0 % -> 93,6 %
       bas de joue comparable      54,8 % -> 94,9 %
       appariements                 1 760 -> 30 970

2. FILTRAGE PAR HOMOGRAPHIE
   Il retirait les points a forte parallaxe, c'est-a-dire LE RELIEF REEL DU
   VISAGE. Une homographie ne decrit qu'une surface plane ; l'appliquer comme
   filtre biaise la liaison vers le plan moyen. Seul le filtrage par matrice
   fondamentale est conserve.

3. NE PAS EXTRAPOLER AUX BORDS
   Combler les zones hors enveloppe convexe par plus proche voisin degradait le
   recalage (ecart-type de controle 0,233 % -> 0,617 %). Mieux vaut une zone non
   comparable qu'une correspondance inventee.

USAGE
    python intersession_link.py --subject alban --ref D0 --test D14 \
        --modalites Standard_1 Cross-Polarized
"""
import argparse
import time
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

TILE, STRIDE, MARGE = 640, 448, 180


class Matcher:
    def __init__(self):
        import torch
        import kornia as K
        import kornia.feature as KF
        self.torch, self.K, self.KF = torch, K, KF
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.disk = KF.DISK.from_pretrained("depth").eval().to(self.dev)
        self.lg = KF.LightGlueMatcher("disk").eval().to(self.dev)

    def _prep(self, c, cap=608):
        hh, ww = c.shape[:2]
        sc = min(1.0, cap / max(hh, ww))
        nh = max(16, int(round(hh * sc / 16)) * 16)
        nw = max(16, int(round(ww * sc / 16)) * 16)
        r = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
        t = self.K.color.bgr_to_rgb(self.K.image_to_tensor(r, False).float() / 255.)
        return t.to(self.dev), (ww / nw, hh / nh)

    def match(self, ca, cb, nf=2048):
        ta, sa = self._prep(ca)
        tb, sb = self._prep(cb)
        with self.torch.inference_mode():
            fa = self.disk(ta, nf, pad_if_not_divisible=True)[0]
            fb = self.disk(tb, nf, pad_if_not_divisible=True)[0]
            if len(fa.keypoints) < 8 or len(fb.keypoints) < 8:
                return None
            la = self.KF.laf_from_center_scale_ori(fa.keypoints[None])
            lb = self.KF.laf_from_center_scale_ori(fb.keypoints[None])
            _, idx = self.lg(fa.descriptors, fb.descriptors, la, lb)
        if len(idx) < 8:
            return None
        i = idx.cpu().numpy()
        return (fa.keypoints.cpu().numpy()[i[:, 0]] * np.array(sa),
                fb.keypoints.cpu().numpy()[i[:, 1]] * np.array(sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--modalites", nargs="+", default=["Standard_1", "Cross-Polarized"])
    ap.add_argument("--img-dir", default=".")
    ap.add_argument("--match-long", type=int, default=4000)
    ap.add_argument("--work-long", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = Matcher()

    def load(sess, mod):
        im = cv2.imread(f"{a.img_dir}/{a.subject}_{sess}_Frontal_{mod}.jpg")
        hh, ww = im.shape[:2]
        s = a.match_long / max(hh, ww)
        return cv2.resize(im, (int(ww*s), int(hh*s)), interpolation=cv2.INTER_AREA)

    PA, PB = [], []
    t0 = time.time()
    for mod in a.modalites:
        A, B = load(a.ref, mod), load(a.test, mod)
        HA, WA = A.shape[:2]
        HB, WB = B.shape[:2]
        face = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY) > 45
        n = c = 0
        for y0 in range(int(0.10 * HA), int(0.94 * HA) - TILE, STRIDE):
            for x0 in range(0, WA - TILE, STRIDE):
                if face[y0:y0+TILE, x0:x0+TILE].mean() < 0.25:
                    continue
                bx0, by0 = max(0, x0-MARGE), max(0, y0-MARGE)
                bx1, by1 = min(WB, x0+TILE+MARGE), min(HB, y0+TILE+MARGE)
                r = m.match(A[y0:y0+TILE, x0:x0+TILE], B[by0:by1, bx0:bx1])
                n += 1
                if r is None:
                    continue
                PA.append(r[0] + np.array([x0, y0]))
                PB.append(r[1] + np.array([bx0, by0]))
                c += len(r[0])
        print(f"{mod:20s}: {n} tuiles -> {c} appariements ({time.time()-t0:.0f}s)", flush=True)

    PA, PB = np.vstack(PA), np.vstack(PB)
    # matrice fondamentale SEULEMENT : l'homographie rejetterait la parallaxe
    F, mk = cv2.findFundamentalMat(PA, PB, cv2.FM_RANSAC, 3.0, 0.999, maxIters=60000)
    inl = mk.ravel().astype(bool)
    sc = a.work_long / a.match_long
    pa, pb = PA[inl] * sc, PB[inl] * sc
    print(f"\n{len(PA)} bruts -> {len(pa)} inliers (matrice fondamentale)")

    A0 = cv2.imread(f"{a.img_dir}/{a.subject}_{a.ref}_Frontal_Standard_1.jpg")
    h0, w0 = A0.shape[:2]
    s = a.work_long / max(h0, w0)
    h, w = int(h0 * s), int(w0 * s)
    yy, xx = np.mgrid[0:h, 0:w]
    Q = LinearNDInterpolator(pa, pb)(np.column_stack([xx.ravel(), yy.ravel()]))
    mapx = Q[:, 0].reshape(h, w).astype(np.float32)
    mapy = Q[:, 1].reshape(h, w).astype(np.float32)
    ok = np.isfinite(mapx)
    print(f"transfert defini sur {100*ok.mean():.1f} % de l'image "
          f"(interpolation lineaire seule, PAS d'extrapolation)")
    d = np.sqrt((mapx - xx)**2 + (mapy - yy)**2)
    print(f"deplacement median : {np.nanmedian(d[ok]):.2f} px a {w}x{h}")
    out = a.out or f"link_{a.subject}_{a.ref}_{a.test}.npz"
    np.savez(out, mapx=mapx, mapy=mapy, pa=pa, pb=pb)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

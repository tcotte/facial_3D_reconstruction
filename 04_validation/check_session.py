"""
check_session.py — la reconstruction ressemble-t-elle au visage ?

OBJET
    Reprojeter le nuage 3D dans le repere de la VUE FRONTALE et en faire des
    images. C'est le seul controle qui ne depend d'aucune convention de
    visualiseur : si la carte de profondeur ressemble a un visage — nez en
    avant, joues en retrait, orbites creusees — la geometrie est bonne, et un
    nuage qui "ne colle pas" dans MeshLab n'est qu'un probleme de repere
    d'affichage.

SORTIES (prefixe --out)
    _profondeur.png   carte de profondeur, palette du proche au lointain
    _relief.png       ombrage du relief. C'EST L'IMAGE A REGARDER EN PREMIER :
                      un visage y est immediatement reconnaissable, ou pas.
    _branches.png     profondeur par paire stereo, cote a cote
    _desaccord.png    ecart entre les deux branches la ou elles se recouvrent
    _superposition.png  contours de profondeur sur la photo

CE QUE LES CHIFFRES DOIVENT DONNER
    relief          3 a 4 % de la distance de travail (cf. README)
    desaccord L/R   faible devant le relief ; sinon les deux demi-visages ne
                    sont pas a la meme echelle et le nuage fusionne est
                    trompeur (biais antisymetrique, cf. pairwise_comparison.py)

USAGE
    python check_session.py --session alban_D0.npz \
        --frontal /donnees/visia/alban_D0_Frontal_Standard_1.jpg --out check_D0
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "00_pipeline"))
import config as C                      # noqa: E402


def carte(pix, val, h, w, step=1):
    """Nuage disperse -> carte dense (NaN la ou rien n'est mesure)."""
    m = np.full((h, w), np.nan, np.float32)
    xi = np.clip(pix[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pix[:, 1].astype(int), 0, h - 1)
    m[yi, xi] = val
    return m


def colorise(Z, lo=2, hi=98):
    ok = np.isfinite(Z)
    if ok.sum() == 0:
        return np.zeros(Z.shape + (3,), np.uint8)
    a, b = np.percentile(Z[ok], [lo, hi])
    n = np.clip((Z - a) / max(b - a, 1e-9), 0, 1)
    img = cv2.applyColorMap((255 * (1 - np.nan_to_num(n))).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    img[~ok] = (30, 30, 30)
    return img


def relief(Z, exagere=1.0):
    """Ombrage : eclairage rasant sur la surface de profondeur.

    Les trous sont bouches par un flou UNIQUEMENT pour le calcul des normales.
    Aucune valeur interpolee n'est exportee ni mesuree ; c'est une aide a l'oeil.
    """
    ok = np.isfinite(Z)
    if ok.sum() == 0:
        return np.zeros(Z.shape, np.uint8)
    z = np.where(ok, Z, 0).astype(np.float32)
    w8 = ok.astype(np.float32)
    num = cv2.GaussianBlur(z, (0, 0), 9)
    den = cv2.GaussianBlur(w8, (0, 0), 9)
    lisse = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    Zf = np.where(ok, Z, lisse)
    Zf = cv2.GaussianBlur(np.nan_to_num(Zf, nan=float(np.nanmedian(Z))), (0, 0), 2.0)
    ech = exagere / max(np.nanpercentile(Z[ok], 98) - np.nanpercentile(Z[ok], 2), 1e-9)
    gx = cv2.Sobel(Zf * ech, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(Zf * ech, cv2.CV_32F, 0, 1, ksize=5)
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    L = np.array([-0.45, -0.60, 0.66]); L /= np.linalg.norm(L)
    sh = np.clip(n @ L, 0, 1)
    out = (255 * sh).astype(np.uint8)
    out[~ok] = 25
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--out", default="check")
    ap.add_argument("--reduire", type=int, default=3,
                    help="facteur de reduction des images produites")
    ap.add_argument("--exagere", type=float, default=1.0,
                    help="exageration du relief dans l'ombrage")
    a = ap.parse_args()

    d = np.load(a.session, allow_pickle=True)
    h, w = (int(v) for v in d["shape"])
    X, pix, src = d["X"].astype(float), d["pix"], d["src"]
    stamp = str(d["calib"]) if "calib" in d.files else "?"
    print(f"session {os.path.basename(a.session)} | {len(X)} points | {stamp}")
    print(f"image de travail {w}x{h}")

    im = cv2.imread(a.frontal)
    if im is None:
        raise SystemExit(f"image frontale illisible : {a.frontal}")
    if im.shape[:2] != (h, w):
        s = C.WORK_LONG / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)),
                        interpolation=cv2.INTER_AREA)
    if im.shape[:2] != (h, w):
        raise SystemExit(f"image frontale {im.shape[1]}x{im.shape[0]} != "
                         f"session {w}x{h}")

    ok = np.isfinite(X).all(1) & (X[:, 2] > 0)
    X, pix, src = X[ok], pix[ok], src[ok]
    lo, hi = np.percentile(X[:, 2], [1, 99])
    k = (X[:, 2] > lo) & (X[:, 2] < hi)
    X, pix, src = X[k], pix[k], src[k]
    Z = X[:, 2]
    Zmed = float(np.median(Z))

    rel = (np.percentile(Z, 97) - np.percentile(Z, 3)) / Zmed
    print(f"\nprofondeur mediane        {Zmed:.4f} (unites arbitraires)")
    print(f"relief (3e-97e centile)   {100*rel:.2f} % de la distance de travail")
    print(f"   attendu 3 a 4 % -> {'COHERENT' if 0.015 < rel < 0.08 else 'HORS PLAGE'}")

    Zmap = carte(pix, Z, h, w)
    maps = {k2: carte(pix[src == k2], Z[src == k2], h, w) for k2 in ("L", "R")}

    both = np.isfinite(maps["L"]) & np.isfinite(maps["R"])
    print(f"\nrecouvrement des deux branches : {int(both.sum())} pixels "
          f"({100*both.sum()/max(1,np.isfinite(Zmap).sum()):.1f} % des mesures)")
    if both.sum() > 100:
        dz = np.abs(maps["L"][both] - maps["R"][both]) / Zmed
        print(f"desaccord L/R  mediane {100*np.median(dz):.2f} % | "
              f"90e centile {100*np.percentile(dz,90):.2f} %")
        print(f"   a comparer au relief de {100*rel:.2f} % : "
              f"{'exploitable' if np.median(dz) < 0.25*rel else 'DESACCORD DU MEME ORDRE QUE LE SIGNAL'}")

    r = a.reduire
    def sauve(nom, img):
        p = f"{a.out}_{nom}.png"
        cv2.imwrite(p, cv2.resize(img, (w // r, h // r), interpolation=cv2.INTER_AREA))
        print(f"   {p}")

    print("\nimages produites :")
    sauve("profondeur", colorise(Zmap))
    sh = relief(Zmap, a.exagere)
    sauve("relief", cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))
    sauve("branches", np.hstack([colorise(maps["L"]), colorise(maps["R"])]))
    if both.sum() > 100:
        dmap = np.full((h, w), np.nan, np.float32)
        dmap[both] = (maps["L"][both] - maps["R"][both]) / Zmed
        sauve("desaccord", colorise(dmap, 5, 95))
    base = (im * 0.55).astype(np.uint8)
    base[..., 1] = np.where(np.isfinite(Zmap),
                            np.clip(base[..., 1].astype(int) + sh // 2, 0, 255),
                            base[..., 1])
    sauve("superposition", base)

    print("\nRegarder _relief.png en premier : un visage doit y etre")
    print("immediatement reconnaissable. Si oui, la geometrie est bonne et un")
    print("nuage qui semble faux dans MeshLab n'est qu'un probleme de repere")
    print("d'affichage (voir export_pointcloud.py --frame).")


if __name__ == "__main__":
    main()

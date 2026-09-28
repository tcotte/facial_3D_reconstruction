"""
diagnose_field_frame.py — le champ et le rig vivent-ils dans le meme repere ?

SYMPTOME TRAITE
    run_session.py --roma annonce un residu epipolaire de plusieurs centaines
    de pixels et ne retient presque aucun point, ALORS QUE :
      - rig_check.py declare le rig physiquement plausible,
      - le champ RoMa affiche une certitude elevee et une bonne couverture.

    Quand ces trois faits coexistent, ni le rig ni l'appariement ne sont en
    cause : les correspondances sont geometriquement DEPLACEES, parce qu'elles
    ont ete produites dans un repere image different de celui du rig.

CAUSE LA PLUS FREQUENTE
    Un tag EXIF d'orientation. `cv2.imread` applique l'orientation EXIF,
    `PIL.Image.open` ne l'applique pas. Un script qui construit sa grille avec
    cv2 mais laisse RoMa ouvrir le fichier avec PIL travaille donc dans deux
    reperes differents. Si le tag vaut 3 (180 deg) ou 2/4 (miroir), les
    DIMENSIONS sont identiques des deux cotes et rien ne le signale.

    C'etait le defaut de roma_v2_simple.py avant le 2026-09-22 : il passait des
    CHEMINS a model.match(). Corrige depuis — il passe des tableaux deja
    charges par cv2. roma_v2_export.py (mode tuile) n'a jamais eu ce defaut.

METHODE
    On reprend les correspondances telles qu'elles ont ete enregistrees, on
    leur applique tour a tour chaque hypothese de repere, et on mesure le
    residu epipolaire contre la matrice fondamentale DU RIG. L'hypothese qui
    fait tomber le residu au niveau du pixel est la bonne. Aucune n'y arrive :
    le desaccord est ailleurs, et le script le dit.

USAGE
    python diagnose_field_frame.py --roma roma2_alban_D0.npz --rig rig.json
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
import roma_field as rf                 # noqa: E402
import visia_core as vc                 # noqa: E402


# --- transformations dihedrales du plan image ------------------------------
DIHEDRAL = {
    "identite":        (lambda x, y, w, h: (x, y), False),
    "miroir horizontal": (lambda x, y, w, h: (w - 1 - x, y), False),
    "miroir vertical": (lambda x, y, w, h: (x, h - 1 - y), False),
    "rotation 180":    (lambda x, y, w, h: (w - 1 - x, h - 1 - y), False),
    "transposee":      (lambda x, y, w, h: (y, x), True),
    "rotation 90 h":   (lambda x, y, w, h: (h - 1 - y, x), True),
    "rotation 90 ah":  (lambda x, y, w, h: (y, w - 1 - x), True),
    "anti-transposee": (lambda x, y, w, h: (h - 1 - y, w - 1 - x), True),
}


def exif_repair(tag):
    """Inverse exact du mecanisme cv2/PIL : les coordonnees normalisees ont ete
    calculees avec les dimensions AFFICHEES alors que le champ etait defini sur
    l'image STOCKEE, d'orientation differente."""
    def f(x, y, w, h):
        ws, hs = h, w                        # l'image stockee est transposee
        sx, sy = x * (ws / w), y * (hs / h)  # ce que le champ a reellement lu
        if tag == 6:
            return hs - 1 - sy, sx
        return sy, ws - 1 - sx
    # la reparation RAMENE vers le repere affiche : les dimensions de sortie
    # sont celles de depart, il n'y a pas d'echange largeur/hauteur.
    return f, False


def residu(K, R, t, ax, ay, bx, by):
    F = rf.fundamental_from_rig(K, R, t)
    e = rf.epipolar_residual(F, ax, ay, bx, by)
    return e[np.isfinite(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roma", required=True)
    ap.add_argument("--rig", default="rig.json")
    ap.add_argument("--cert", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=60000, help="points echantillonnes")
    a = ap.parse_args()

    field = rf.load(a.roma)
    rig = vc.load_rig(a.rig)
    W, H = field.W, field.H
    ys0 = (field.idx // W).astype(np.float64)
    xs0 = (field.idx % W).astype(np.float64)
    print(f"champ  : {os.path.basename(a.roma)}  grille {W}x{H} "
          f"(largeur x hauteur), pas {field.step}")
    print(f"rig    : {os.path.basename(a.rig)}")
    print(f"loader : {field.meta.get('loader', 'NON RENSEIGNE (export anterieur au correctif)')}")
    if field.meta.get("loader") in (None, "path-pil"):
        print("         -> ce champ a peut-etre ete produit en passant des CHEMINS")
        print("            a model.match() : cv2 et PIL peuvent differer sur l'EXIF.")

    rng = np.random.default_rng(0)
    for k in field.pairs:
        if k not in rig:
            continue
        xy = field._d[f"{k}_xy"].astype(np.float64)
        cert = field._d[f"{k}_cert"].astype(np.float64)
        m = np.isfinite(xy[:, 0]) & (cert > a.cert)
        if m.sum() < 1000:
            print(f"\npaire {k} : trop peu de points"); continue
        sel = np.nonzero(m)[0]
        if len(sel) > a.n:
            sel = rng.choice(sel, a.n, replace=False)
        ax0, ay0 = xs0[sel], ys0[sel]
        bx0, by0 = xy[sel, 0], xy[sel, 1]

        print(f"\n{'='*72}\npaire F<->{k}   ({len(sel)} points, cert > {a.cert})\n{'='*72}")

        # 1. le jeu est-il coherent avec lui-meme ?
        Fm, _ = cv2.findFundamentalMat(np.column_stack([ax0, ay0]),
                                       np.column_stack([bx0, by0]),
                                       cv2.USAC_MAGSAC, 1.0, 0.9999, 100000)
        if Fm is not None:
            Fm = Fm[:3] / (np.linalg.norm(Fm[:3]) + 1e-30)
            e = rf.epipolar_residual(Fm, ax0, ay0, bx0, by0)
            print(f"coherence interne du champ (F re-estimee) : mediane "
                  f"{np.median(e):.3f} px")
            if np.median(e) < 2.0:
                print("   -> les correspondances sont coherentes entre elles.")
                print("      L'appariement RoMa n'est pas en cause.")
            else:
                print("   -> meme sa propre F ne les explique pas : l'appariement")
                print("      lui-meme est douteux, inutile d'aller plus loin.")

        # 2. balayage des hypotheses de repere
        cands = []
        for nom, (f, swap) in DIHEDRAL.items():
            cands.append((f"les deux images : {nom}", f, f, swap, swap))
        for nom, (f, swap) in DIHEDRAL.items():
            if nom == "identite":
                continue
            cands.append((f"oblique seule  : {nom}", None, f, False, swap))
        for tag in (6, 8):
            f, swap = exif_repair(tag)
            cands.append((f"les deux images : reparation EXIF {tag}", f, f, swap, swap))

        res = []
        for nom, fa, fb, swa, swb in cands:
            if fa is None:
                ax, ay, Wa, Ha = ax0, ay0, W, H
            else:
                ax, ay = fa(ax0, ay0, W, H)
                Wa, Ha = (H, W) if swa else (W, H)
            bx, by = fb(bx0, by0, W, H)
            Wb, Hb = (H, W) if swb else (W, H)
            if (Wa, Ha) != (Wb, Hb):
                continue
            K = vc.K_matrix(Wa, Ha)
            e = residu(K, *rig[k], ax, ay, bx, by)
            if len(e):
                res.append((float(np.median(e)), nom, float(np.percentile(e, 90))))

        res.sort()
        print(f"\n{'hypothese de repere':44s} {'residu median':>14s} {'90e c.':>9s}")
        print("-" * 70)
        for med, nom, p90 in res[:8]:
            marque = "  <== LE BON" if med < 3.0 else ""
            print(f"{nom:44s} {med:13.2f} px {p90:8.1f}{marque}")

        best = res[0]
        print()
        if best[0] < 3.0:
            if best[1].startswith("les deux images : identite"):
                print("Le repere est correct : le desaccord vient d'ailleurs.")
            else:
                print(f"REPERE FAUTIF IDENTIFIE : « {best[1]} » ramene le residu")
                print(f"a {best[0]:.2f} px. Re-exporter le champ avec la version")
                print("corrigee de roma_v2_simple.py (qui passe des tableaux cv2")
                print("a model.match() au lieu de chemins) doit suffire.")
        else:
            print("Aucune hypothese de repere n'explique l'ecart.")
            print("Le champ est coherent avec lui-meme mais incompatible avec le rig :")
            print("comparer la pose que le champ implique a celle du rig —")
            print(f"   python ../02_calibration/rig_check.py --rig {a.rig} --roma {a.roma}")
            # pose impliquee par le champ
            K = vc.K_matrix(W, H)
            u1 = vc.undistort(np.column_stack([ax0, ay0]).astype(np.float32), K)
            u2 = vc.undistort(np.column_stack([bx0, by0]).astype(np.float32), K)
            E, _ = cv2.findEssentialMat(u1, u2, K, method=cv2.USAC_MAGSAC,
                                        prob=0.9999, threshold=1.0)
            if E is not None:
                _, Rf, tf, _ = cv2.recoverPose(E[:3], u1, u2, K)
                ang = np.degrees(np.arccos(np.clip((np.trace(Rf) - 1) / 2, -1, 1)))
                w_, v_ = np.linalg.eig(Rf)
                axv = np.real(v_[:, np.argmin(np.abs(w_ - 1))])
                axv /= np.linalg.norm(axv)
                Rr = rig[k][0]
                angr = np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))
                dd = np.degrees(np.arccos(np.clip((np.trace(Rf @ Rr.T) - 1) / 2, -1, 1)))
                print(f"\n   pose impliquee par le champ : {ang:.1f}° | "
                      f"lacet {100*abs(axv[1]):.0f} % | roulis {100*abs(axv[2]):.0f} %")
                print(f"   pose du rig                 : {angr:.1f}°")
                print(f"   ecart entre les deux poses  : {dd:.1f}°")


if __name__ == "__main__":
    main()

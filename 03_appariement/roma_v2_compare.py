"""
roma_v2_compare.py — RoMa v2 contre la densification actuelle, a armes egales.

OBJET
    Avant de basculer la chaine de production, mesurer. Ce script reconstruit
    la MEME session deux fois — flot optique DIS d'un cote, champ RoMa v2
    exporte de l'autre — avec le meme rig gele, la meme focale, les memes
    seuils, et compare sur quatre criteres.

CRITERES
    1. Couverture     % du masque facial classe "mesure" (epi < EPI_HIGH).
    2. Residu epipolaire  mediane et 90e centile, avec la MEME matrice
       fondamentale des deux cotes : celle DERIVEE DU RIG GELE. C'est la seule
       facon de comparer honnetement deux appariements — comparer chacun a sa
       propre F re-estimee avantage mecaniquement celui qui a servi a
       l'estimer.
    3. Chainage 3 vues    nombre de pixels frontaux apparies A LA FOIS dans L
       et dans R. C'est le point ou RoMa v1 echouait (50 pistes en mode
       'sample' contre 272-580 pour LightGlue) et ce que le mode grille est
       cense restaurer. Un chainage faible interdit toute mise a l'echelle
       relative des deux branches stereo.
    4. Accord des deux methodes   ecart, en pixels, entre les positions
       obliques predites par l'une et par l'autre, la ou les deux repondent.
       Un desaccord important sans difference de residu epipolaire signale un
       glissement LE LONG de la droite epipolaire — invisible au controle
       epipolaire, mais directement converti en erreur de profondeur.

CE QUE LE SCRIPT NE TRANCHE PAS
    Il ne dit rien de l'exactitude absolue : les deux methodes peuvent etre
    coherentes et fausses ensemble. Le seul juge reste selftest.py sur
    deformation simulee, puis le test a blanc D0/D0.

USAGE
    python roma_v2_compare.py --subject alban --session D0 --rig rig.json \
        --roma roma2_alban_D0_Standard_1.npz --out compare_roma
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "00_pipeline"))
import config as C                      # noqa: E402
import visia_core as vc                 # noqa: E402
import roma_field as rfmod              # noqa: E402


def stats(name, obx, oby, wm, fmask, tex, F, epi):
    ok = wm & fmask
    high = ok & (epi < C.EPI_HIGH) & (tex > C.TEXTURE_MIN)
    med = ok & (epi < C.EPI_MED) & ~high
    e = epi[ok & np.isfinite(epi)]
    return {
        "nom": name,
        "apparies": int(ok.sum()),
        "couverture_%": 100.0 * ok.sum() / fmask.sum(),
        "mesure_%": 100.0 * high.sum() / fmask.sum(),
        "faible_%": 100.0 * med.sum() / fmask.sum(),
        "epi_median": float(np.median(e)) if e.size else float("nan"),
        "epi_p90": float(np.percentile(e, 90)) if e.size else float("nan"),
        "high": high,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--rig", default="rig.json")
    ap.add_argument("--roma", required=True)
    ap.add_argument("--out", default="compare_roma")
    ap.add_argument("--roma-cert", type=float, default=0.05)
    ap.add_argument("--skip-dis", action="store_true",
                    help="ne pas recalculer la densification actuelle (lente)")
    a = ap.parse_args()
    print(C.summary())
    rig = vc.load_rig(a.rig)

    A = vc.load(vc.img_path(a.subject, a.session, "F"))
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    h, w = gA.shape
    K = vc.K_matrix(w, h)
    fmask = vc.face_mask(gA)
    tex = cv2.blur(np.abs(cv2.Laplacian(gA, cv2.CV_32F, ksize=3)), (15, 15))

    field = rfmod.load(a.roma)
    field.check(a.subject, a.session, (h, w))
    print()
    field.report()

    matcher = None if a.skip_dis else vc.Matcher()
    rows = []
    chain = {}
    for k in "LR":
        print(f"\n===== paire F<->{k} =====")
        F_rig = rfmod.fundamental_from_rig(K, *rig[k])

        # --- RoMa v2 --------------------------------------------------------
        obx_r, oby_r, epi_r, wm_r = field.densify_like(
            k, K, *rig[k], epi_from="rig", cert_min=a.roma_cert)
        s_r = stats("RoMa v2", obx_r, oby_r, wm_r, fmask, tex, F_rig, epi_r)
        rows.append((k, s_r))
        chain.setdefault("roma", {})[k] = s_r["high"]

        # --- densification actuelle ----------------------------------------
        if matcher is not None:
            B = vc.load(vc.img_path(a.subject, a.session, k))
            r1 = vc.match_pair(matcher, A, B, guide=None)
            if r1 is None:
                print("   passe grossiere echouee, comparaison impossible")
                continue
            pa, pb, Fd = r1
            obx_d, oby_d, _, wm_d = vc.densify(A, B, pa, pb, Fd)
            # meme F que pour RoMa : celle du rig
            epi_d = np.full((h, w), np.inf, np.float32)
            ys, xs = np.nonzero(wm_d & np.isfinite(obx_d))
            epi_d[ys, xs] = rfmod.epipolar_residual(
                F_rig, xs.astype(float), ys.astype(float),
                obx_d[ys, xs].astype(float), oby_d[ys, xs].astype(float)).astype(np.float32)
            s_d = stats("DIS actuel", obx_d, oby_d, wm_d & np.isfinite(obx_d),
                        fmask, tex, F_rig, epi_d)
            rows.append((k, s_d))
            chain.setdefault("dis", {})[k] = s_d["high"]

            both = wm_r & wm_d & np.isfinite(obx_d) & np.isfinite(obx_r) & fmask
            if both.sum():
                d = np.hypot(obx_r[both] - obx_d[both], oby_r[both] - oby_d[both])
                print(f"\n   accord des deux methodes sur {int(both.sum())} pixels : "
                      f"mediane {np.median(d):.2f} px, 90e centile "
                      f"{np.percentile(d, 90):.2f} px, max {d.max():.1f} px")

    # --- tableau ------------------------------------------------------------
    print(f"\n{'paire':6s} {'methode':12s} {'apparies':>10s} {'couv.%':>8s} "
          f"{'mesure%':>9s} {'epi med':>9s} {'epi p90':>9s}")
    for k, s in rows:
        print(f"F<->{k:<2s} {s['nom']:12s} {s['apparies']:10d} "
              f"{s['couverture_%']:7.1f}% {s['mesure_%']:8.1f}% "
              f"{s['epi_median']:8.2f} {s['epi_p90']:8.2f}")

    # --- chainage 3 vues ----------------------------------------------------
    print(f"\n{'chainage 3 vues (pixels frontaux vus dans L ET dans R)':55s}")
    for tag, label in (("roma", "RoMa v2"), ("dis", "DIS actuel")):
        if tag in chain and set(chain[tag]) == {"L", "R"}:
            c = chain[tag]["L"] & chain[tag]["R"]
            nl, nr = chain[tag]["L"].sum(), chain[tag]["R"].sum()
            print(f"   {label:12s} : {int(c.sum()):8d} "
                  f"({100*c.sum()/max(1, min(nl, nr)):.1f} % de la plus petite branche)")
    print("   Rappel : sans chainage, la mise a l'echelle relative des deux")
    print("   branches stereo (run_calibrate.py) n'est pas estimable.")

    # --- carte --------------------------------------------------------------
    vis = (A * 0.45).astype(np.uint8)
    if "dis" in chain:
        for k in "LR":
            if k in chain["dis"]:
                vis[chain["dis"][k]] = (0.4 * vis[chain["dis"][k]]
                                        + 0.6 * np.array([60, 190, 255])).astype(np.uint8)
    for k in "LR":
        if k in chain.get("roma", {}):
            m = chain["roma"][k]
            vis[m] = (0.4 * vis[m] + 0.6 * np.array([90, 220, 90])).astype(np.uint8)
    cv2.imwrite(a.out + "_couverture.png", vis)
    print(f"\ncarte de couverture -> {a.out}_couverture.png "
          f"(vert = RoMa v2, bleu = DIS seul)")


if __name__ == "__main__":
    main()

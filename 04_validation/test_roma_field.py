"""
test_roma_field.py — validation bout en bout de l'integration RoMa v2, sans GPU.

On fabrique une scene synthetique dont la verite terrain est connue :
    - une grille de pixels frontaux, a qui on attribue une profondeur,
    - retroprojection en 3D, puis projection dans l'oblique avec un rig connu.
Les correspondances obtenues sont EXACTES par construction. On les empaquette
au format d'export, puis on les relit par roma_field.py et on verifie :

    T1  le residu epipolaire calcule a partir du rig est nul
        -> valide la convention de fundamental_from_rig (une erreur de signe
           ou d'ordre des vues donnerait des residus de plusieurs centaines de px)
    T2  la triangulation retrouve la profondeur injectee
    T3  l'interpolation (fill=True) reste exacte sur un champ affine
    T4  les garde-fous de compatibilite se declenchent bien
    T5  une correspondance volontairement fausse est bien rejetee par le
        residu epipolaire

    python 04_validation/test_roma_field.py
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "00_pipeline"))

import config as C                      # noqa: E402

# La scene de test travaille a 1000 px de cote long. On l'annonce comme telle
# a la config : les controles de compatibilite portent sur la COHERENCE entre
# l'export et le pipeline, pas sur une valeur particuliere.
C.WORK_LONG = 1000

import visia_core as vc                 # noqa: E402
import roma_field as rf                 # noqa: E402

W, H = 1000, 750
STEP = 2
fail = 0


def make_rig(angle_deg=32.4, baseline=0.55):
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), 0, np.sin(th)],
                  [0, 1, 0],
                  [-np.sin(th), 0, np.cos(th)]])
    t = np.array([-baseline, 0.02, 0.08])
    return R, t


def build_scene(K, R, t, rng):
    """Correspondances exactes sur une grille de pas STEP."""
    ys, xs = np.mgrid[0:H:STEP, 0:W:STEP]
    xs, ys = xs.ravel(), ys.ravel()
    # profondeur : plan incline + bosse, pour ne pas tester un cas degenere
    Z = (8.0 + 0.0012 * (xs - W / 2) + 0.0008 * (ys - H / 2)
         + 0.6 * np.exp(-(((xs - W / 2) ** 2 + (ys - H / 2) ** 2) / (2 * 120.0 ** 2))))
    Ki = np.linalg.inv(K)
    rays = (Ki @ np.column_stack([xs, ys, np.ones_like(xs)]).T).T
    X = rays * Z[:, None]
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    p2h = (P2 @ np.column_stack([X, np.ones(len(X))]).T).T
    p2 = p2h[:, :2] / p2h[:, 2:3]
    return xs, ys, X, p2


def write_export(path, xs, ys, xyL, xyR=None, work_long=None, grid_shape=None):
    idx = (ys.astype(np.int64) * W + xs.astype(np.int64)).astype(np.int32)
    n = len(idx)
    out = {"idx": idx}
    for k, xy in (("L", xyL), ("R", xyR)):
        if xy is None:
            continue
        out[f"{k}_xy"] = xy.astype(np.float32)
        out[f"{k}_cert"] = np.full(n, 0.9, np.float16)
        out[f"{k}_sigma"] = np.full((n, 2), 0.4, np.float16)
        out[f"{k}_theta"] = np.zeros(n, np.float16)
        out[f"{k}_spread"] = np.full(n, np.nan, np.float16)
        out[f"{k}_src"] = np.ones(n, np.int8)
        out[f"{k}_shape"] = np.array([H, W], np.int32)
    meta = {
        "format": "visia-roma2-field/1", "subject": "test", "session": "D0",
        "modality": C.MODALITY_GEOM, "pairs": "L" + ("R" if xyR is not None else ""),
        "work_long": work_long or C.WORK_LONG, "step": STEP, "mode": "tiled",
        "setting": "precise", "tile": 1024, "stride": 768, "margin": 96,
        "cert_min": 0.02, "mask_margin": 40,
        "grid_shape": list(grid_shape or [H, W]),
        "coord_convention": "OpenCV pixel centers (integer), bilinear field readout",
        "model": {"name": "RoMa v2", "setting": "precise", "H_lr": 800, "W_lr": 800,
                  "H_hr": 1280, "W_hr": 1280, "bidirectional": True},
        "sources": {}, "env": {"host": "test", "python": "-", "torch": "-",
                               "cuda": None, "gpu": None},
        "stats": {k: {"points": n, "couverture_grille_%": 100.0, "cert_mediane": 0.9,
                      "sigma_median_px": 0.4, "spread_median_px": None, "tuiles": 1}
                  for k in ("L" + ("R" if xyR is not None else ""))},
        "date": "2026-01-01T00:00:00",
    }
    out["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **out)
    return path

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    K = vc.K_matrix(W, H)
    R, t = make_rig()
    xs, ys, X_true, p2 = build_scene(K, R, t, rng)
    tmp = tempfile.mkdtemp()
    path = write_export(os.path.join(tmp, "roma2_test_D0.npz"), xs, ys, p2, p2)

    field = rf.load(path)
    field.check("test", "D0", (H, W))

    # --- T1 : residu epipolaire issu du rig -----------------------------------
    obx, oby, epi, wm = field.densify_like("L", K, R, t, epi_from="rig", cert_min=0.05)
    e = epi[wm]
    ok = np.median(e) < 1e-3 and np.percentile(e, 99.9) < 1e-2
    fail += not ok
    print(f"T1 residu epipolaire (F issue du rig) sur correspondances exactes :")
    print(f"   mediane {np.median(e):.2e} px, 99.9e centile {np.percentile(e,99.9):.2e} px, "
          f"max {e.max():.2e} px  {'OK' if ok else 'ECHEC'}")

    # contre-epreuve : rig errone (vues inversees) -> le residu doit exploser
    _, _, epi_bad, wm_bad = field.densify_like("L", K, R.T, -R.T @ t, epi_from="rig")
    print(f"   contre-epreuve, rig inverse : residu median "
          f"{np.median(epi_bad[wm_bad]):.1f} px (doit etre >> 1)")
    fail += not (np.median(epi_bad[wm_bad]) > 5.0)

    # --- T2 : la triangulation retrouve la profondeur --------------------------
    yy, xx = np.nonzero(wm)
    p1 = np.column_stack([xx, yy]).astype(np.float64)
    pb = np.column_stack([obx[yy, xx], oby[yy, xx]]).astype(np.float64)
    Xr = vc.triangulate(K, R, t, p1, pb)
    order = np.lexsort((xs, ys))
    order_r = np.lexsort((xx, yy))
    err = np.abs(Xr[order_r, 2] - X_true[order, 2])
    rel = err / X_true[order, 2]
    ok = np.percentile(rel, 99.9) < 1e-6
    fail += not ok
    print(f"T2 profondeur triangulee : erreur relative mediane {np.median(rel):.2e}, "
          f"99.9e centile {np.percentile(rel,99.9):.2e}  {'OK' if ok else 'ECHEC'}")

    # --- T3 : interpolation du champ ------------------------------------------
    obxf, obyf, epif, wmf = field.densify_like("L", K, R, t, fill=True, epi_from="rig")
    ef = epif[wmf]
    gain = wmf.sum() / wm.sum()
    ok = np.percentile(ef, 99) < 0.05
    fail += not ok
    print(f"T3 champ interpole (fill=True) : {int(wmf.sum())} pixels "
          f"({gain:.1f}x) | residu 99e centile {np.percentile(ef,99):.2e} px  "
          f"{'OK' if ok else 'ECHEC'}")

    # --- T4 : garde-fous -------------------------------------------------------
    checks = []
    bad1 = write_export(os.path.join(tmp, "bad_long.npz"), xs, ys, p2,
                        work_long=C.WORK_LONG + 1)
    try:
        rf.load(bad1).check("test", "D0", (H, W)); checks.append(("cote long", False))
    except RuntimeError:
        checks.append(("cote long different", True))
    try:
        field.check("test", "Dx", (H, W)); checks.append(("session", False))
    except RuntimeError:
        checks.append(("session differente", True))
    try:
        field.check("test", "D0", (H + 4, W)); checks.append(("taille image", False))
    except RuntimeError:
        checks.append(("taille image differente", True))
    try:
        field.densify_like("R", K, R, t); has_R = True
    except KeyError:
        has_R = False
    for nom, good in checks:
        print(f"T4 garde-fou '{nom}' : {'declenche OK' if good else 'NON DECLENCHE — ECHEC'}")
        fail += not good

    # --- T5 : une correspondance fausse est-elle rejetee ? ---------------------
    p2_bad = p2.copy()
    n_bad = len(p2_bad) // 10
    sel = rng.choice(len(p2_bad), n_bad, replace=False)
    p2_bad[sel] += rng.normal(0, 25, (n_bad, 2))          # 25 px de bruit
    path2 = write_export(os.path.join(tmp, "roma2_test_D0_bruite.npz"), xs, ys, p2_bad)
    f2 = rf.load(path2)
    _, _, epi2, wm2 = f2.densify_like("L", K, R, t, epi_from="rig")
    flat = np.zeros(H * W, bool); flat[(ys.astype(np.int64) * W + xs.astype(np.int64))[sel]] = True
    is_bad = flat.reshape(H, W)
    kept = wm2 & (epi2 < C.EPI_HIGH)
    taux = 100.0 * (kept & is_bad).sum() / max(1, is_bad.sum())
    ok = taux < 15.0
    fail += not ok
    print(f"T5 correspondances bruitees a 25 px : {taux:.1f} % passent encore le "
          f"seuil EPI_HIGH={C.EPI_HIGH} px  {'OK' if ok else 'ECHEC'}")
    print("   (le reliquat glisse LE LONG de la droite epipolaire : le controle")
    print("    epipolaire ne peut pas le voir — c'est une limite connue, pas un bug)")

    print(f"\n{'TOUS LES TESTS PASSENT' if fail == 0 else str(fail) + ' TEST(S) EN ECHEC'}")
    sys.exit(1 if fail else 0)

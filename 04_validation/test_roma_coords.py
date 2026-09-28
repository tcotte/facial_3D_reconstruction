"""test_roma_coords.py — validation des conventions de coordonnees RoMa v2.

A executer apres toute modification de roma_v2_export.py ou roma_field.py.
Ne necessite ni GPU, ni torch, ni romav2 : quelques secondes sur n'importe
quel poste.

    python 04_validation/test_roma_coords.py

On ne dispose pas de torch ici : on reimplemente grid_sample(bilinear,
align_corners=False) en numpy a partir de sa DEFINITION, et on verifie que les
conversions pixel <-> normalise sont exactement celles de romav2 :

    romav2.geometry.get_normalized_grid : linspace(-1+1/N, 1-1/N, N)
    romav2.geometry.to_pixel            : (x+1)/2 * N

Trois tests :
  T1  px_to_norm reproduit exactement get_normalized_grid
  T2  echantillonner le warp IDENTITE redonne les positions de depart
  T3  echantillonner un warp AFFINE (exact en bilineaire) redonne la valeur
      analytique, y compris a des positions non alignees sur la grille du champ
"""
import os
import sys

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "03_appariement"))
from roma_v2_export import px_to_norm, norm_to_px


def get_normalized_grid(H, W):
    """Copie exacte de romav2.geometry.get_normalized_grid (B=1)."""
    ys = np.linspace(-1 + 1 / H, 1 - 1 / H, H)
    xs = np.linspace(-1 + 1 / W, 1 - 1 / W, W)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx, gy], axis=-1)          # (H, W, 2) = (x, y)


def grid_sample_bilinear(field, grid_xy):
    """F.grid_sample(mode='bilinear', align_corners=False, padding_mode='zeros').

    field   (H, W, C)
    grid_xy (N, 2) coordonnees normalisees
    """
    H, W, C = field.shape
    # definition align_corners=False : n -> (n+1)/2*N - 0.5
    fx = (grid_xy[:, 0] + 1) / 2 * W - 0.5
    fy = (grid_xy[:, 1] + 1) / 2 * H - 0.5
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    wx = fx - x0; wy = fy - y0
    out = np.zeros((len(fx), C))
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
            v = np.zeros((len(fx), C))
            v[inside] = field[np.clip(yi, 0, H - 1)[inside],
                              np.clip(xi, 0, W - 1)[inside]]
            out += (w * inside)[:, None] * v
    return out

if __name__ == '__main__':
    fail = 0

    # --- T1 --------------------------------------------------------------------
    for (H, W) in [(1280, 1280), (2667, 4000), (1024, 777)]:
        g = get_normalized_grid(H, W)
        ii, jj = np.array([0, 1, H // 3, H - 1]), np.array([0, 1, W // 2, W - 1])
        un, vn = px_to_norm(jj.astype(float), ii.astype(float), W, H)
        e = max(np.abs(un - g[0, jj, 0]).max(), np.abs(vn - g[ii, 0, 1]).max())
        # aller-retour
        xb, yb = norm_to_px(un, vn, W, H)
        e2 = max(np.abs(xb - jj).max(), np.abs(yb - ii).max())
        ok = e < 1e-12 and e2 < 1e-9
        fail += not ok
        print(f"T1 {H}x{W}: ecart a get_normalized_grid = {e:.2e} | "
              f"aller-retour = {e2:.2e}  {'OK' if ok else 'ECHEC'}")

    # --- T2 : warp identite ----------------------------------------------------
    Hf, Wf = 1280, 1280          # resolution interne du champ RoMa ('precise')
    HA, WA = 2667, 4000          # image de travail frontale
    HB, WB = 2667, 4000
    field = get_normalized_grid(Hf, Wf)          # warp identite en normalise
    rng = np.random.default_rng(0)
    gx = rng.integers(0, WA, 4000).astype(float)
    gy = rng.integers(0, HA, 4000).astype(float)
    un, vn = px_to_norm(gx, gy, WA, HA)
    w = grid_sample_bilinear(field, np.stack([un, vn], -1))
    xb, yb = norm_to_px(w[:, 0], w[:, 1], WB, HB)
    # Invalidation appliquee par sample_field : hors du domaine interpolable du
    # champ (|n| > 1 - 1/N), grid_sample extrapole et produit une correspondance
    # fausse mais plausible. Sans cette exclusion l'erreur monte a ~700 px.
    valid = (np.abs(un) <= 1 - 1 / Wf) & (np.abs(vn) <= 1 - 1 / Hf)
    e_all = max(np.abs(xb - gx).max(), np.abs(yb - gy).max())
    e = max(np.abs(xb[valid] - gx[valid]).max(), np.abs(yb[valid] - gy[valid]).max())
    ok = e < 1e-6
    fail += not ok
    print(f"T2 warp identite, champ {Hf}x{Wf} lu depuis une image {WA}x{HA} :")
    print(f"   points valides : erreur max = {e:.2e} px  {'OK' if ok else 'ECHEC'}")
    print(f"   sans l'exclusion des bords : erreur max = {e_all:.1f} px "
          f"({int((~valid).sum())} points concernes) -> justifie l'invalidation")

    # --- T3 : warp affine (bilineaire exact) -----------------------------------
    M = np.array([[0.83, 0.11], [-0.07, 0.94]])
    b = np.array([0.031, -0.017])
    base = get_normalized_grid(Hf, Wf).reshape(-1, 2)
    field3 = (base @ M.T + b).reshape(Hf, Wf, 2)
    w = grid_sample_bilinear(field3, np.stack([un, vn], -1))
    truth = np.stack([un, vn], -1) @ M.T + b
    # bord : grid_sample met 0 hors champ, on exclut la couronne d'un demi-pixel
    inner = (np.abs(un) < 1 - 1 / Wf) & (np.abs(vn) < 1 - 1 / Hf)
    e_norm = np.abs(w[inner] - truth[inner]).max()
    xb, yb = norm_to_px(w[:, 0], w[:, 1], WB, HB)
    xt, yt = norm_to_px(truth[:, 0], truth[:, 1], WB, HB)
    e_px = max(np.abs(xb[inner] - xt[inner]).max(), np.abs(yb[inner] - yt[inner]).max())
    ok = e_px < 1e-6
    fail += not ok
    print(f"T3 warp affine : erreur max = {e_norm:.2e} (normalise) / {e_px:.2e} px  "
          f"{'OK' if ok else 'ECHEC'}")

    # --- Contre-epreuve : ce que donnerait l'ancienne lecture au plus proche ----
    ui = np.clip(np.round((gx / WA) * (Wf - 1)).astype(int), 0, Wf - 1)
    vi = np.clip(np.round((gy / HA) * (Hf - 1)).astype(int), 0, Hf - 1)
    w_nn = field[vi, ui]
    x_nn, y_nn = norm_to_px(w_nn[:, 0], w_nn[:, 1], WB, HB)
    err_nn = np.hypot(x_nn - gx, y_nn - gy)
    print(f"\nContre-epreuve — lecture au plus proche voisin (ancien roma_matching.py,"
          f" mode grid) sur le MEME champ identite :")
    print(f"   erreur mediane {np.median(err_nn):.2f} px, 95e centile "
          f"{np.percentile(err_nn, 95):.2f} px, max {err_nn.max():.2f} px")
    print(f"   -> a comparer au seuil epipolaire EPI_HIGH = 1.0 px du pipeline")

    print(f"\n{'TOUS LES TESTS PASSENT' if fail == 0 else str(fail) + ' TEST(S) EN ECHEC'}")
    sys.exit(1 if fail else 0)

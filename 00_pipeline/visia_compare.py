"""
visia_compare.py — comparaison entre deux sessions (D0 vs Dx).

PRINCIPE ET LIMITES
-------------------
La focale et les angles du banc ne sont pas connus : la reconstruction n'est
donc PAS metrique. Aucune valeur en mm ou mm^3 ne peut etre produite.

En revanche, une mesure RELATIVE reste valide, a deux conditions :
  1. les memes constantes optiques sont utilisees aux deux sessions (config.py) ;
  2. le resultat est exprime comme un rapport de deux grandeurs HOMOGENES
     issues de la MEME reconstruction.

Verification numerique (simulation avec deformation connue) :
    deformation / distance de travail   -> biais ~1,5 x l'erreur de focale
                                           (soit ~25 % pour 16 % d'incertitude)
    deformation / relief propre de ROI  -> biais < 3 % pour +/- 20 % d'erreur

C'est la seconde formulation qui est implementee ici (indice `ratio_relief`).

RECALAGE
--------
Le recalage est estime UNIQUEMENT sur la zone stable, jamais sur la ROI.
Recaler sur une zone qui evolue avec le traitement fausserait la mesure.
"""

import numpy as np
import cv2
import config as C


def umeyama(X, Y, with_scale=True):
    """Similitude alignant X sur Y (Umeyama 1991). Retourne (s, R, t)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    Sig = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(Sig)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / ((Xc ** 2).sum() / len(X))) if with_scale else 1.0
    t = my - s * R @ mx
    return s, R, t


def apply_sim(X, s, R, t):
    return s * (R @ X.T).T + t


def register(X_src, X_dst, stable_mask):
    """Recale la session source sur la session de reference, via la zone stable."""
    if stable_mask.sum() < C.MIN_STABLE_POINTS:
        raise RuntimeError(
            f"zone stable insuffisante ({stable_mask.sum()} points, "
            f"minimum {C.MIN_STABLE_POINTS}) : recalage non fiable")
    s, R, t = umeyama(X_src[stable_mask], X_dst[stable_mask],
                      with_scale=(C.REGISTRATION == "similarity"))
    Xa = apply_sim(X_src, s, R, t)
    resid = np.linalg.norm(Xa[stable_mask] - X_dst[stable_mask], axis=1)
    return Xa, {"scale": float(s),
                "residu_median": float(np.median(resid)),
                "residu_p95": float(np.percentile(resid, 95)),
                "n_stable": int(stable_mask.sum())}


def plane_fit(P):
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    return coef


def relief(P, coef):
    return P[:, 2] - (coef[0] * P[:, 0] + coef[1] * P[:, 1] + coef[2])


def compare(X0, Xx, roi_mask, stable_mask):
    """Compare deux nuages APPARIES point a point (meme indexation).

    Retourne un dictionnaire de metriques sans dimension et la carte de
    deviation signee (unites de reconstruction, non metriques).
    """
    Xa, reg = register(Xx, X0, stable_mask)

    dev = X0[:, 2] - Xa[:, 2]        # deviation signee selon la profondeur
    roi = roi_mask
    if roi.sum() < C.MIN_ROI_POINTS:
        raise RuntimeError(f"ROI trop peu couverte ({roi.sum()} points)")

    coef = plane_fit(X0[stable_mask])
    rel0 = relief(X0[roi], coef)
    denom = float(np.std(rel0))      # grandeur homogene, meme reconstruction

    dev_roi = dev[roi]
    # controle interne : la meme mesure sur la zone stable doit etre ~0
    dev_stable = dev[stable_mask]

    m = {
        "calib_version": C.CALIB_VERSION,
        "n_roi": int(roi.sum()),
        "recalage": reg,
        # --- indicateur principal, sans dimension ---
        "ratio_relief_moyen": float(np.mean(dev_roi) / denom),
        "ratio_relief_median": float(np.median(dev_roi) / denom),
        # --- integrale surfacique (proxy de variation de volume), sans dimension ---
        "variation_volume_relative": float(np.sum(dev_roi) / np.sum(np.abs(rel0))),
        # --- dispersion ---
        "dev_roi_ecart_type": float(np.std(dev_roi) / denom),
        # --- controle interne (doit etre proche de 0) ---
        "controle_zone_stable_moyen": float(np.mean(dev_stable) / denom),
        "controle_zone_stable_ecart_type": float(np.std(dev_stable) / denom),
    }
    return m, dev, Xa


def null_test(X0, Xx, stable_mask, n_splits=8, seed=0):
    """Plancher de bruit : on scinde la zone stable en blocs, on recale sur une
    moitie et on mesure sur l'autre. Donne l'amplitude minimale credible."""
    rng = np.random.default_rng(seed)
    idx = np.nonzero(stable_mask)[0]
    rng.shuffle(idx)
    blocks = np.array_split(idx, n_splits)
    vals = []
    for i in range(n_splits):
        test = blocks[i]
        train = np.concatenate([blocks[j] for j in range(n_splits) if j != i])
        m_tr = np.zeros(len(X0), bool); m_tr[train] = True
        try:
            Xa, _ = register(Xx, X0, m_tr)
        except RuntimeError:
            continue
        coef = plane_fit(X0[m_tr])
        denom = float(np.std(relief(X0[test], coef)))
        if denom <= 0:
            continue
        vals.append(float(np.mean(X0[test, 2] - Xa[test, 2]) / denom))
    if not vals:
        return None
    vals = np.array(vals)
    return {"n": len(vals), "biais_moyen": float(vals.mean()),
            "ecart_type": float(vals.std()),
            "seuil_detection_2sigma": float(2 * vals.std())}


def deviation_map(shape, pix, dev, roi_mask=None, vmax=None):
    """Rend une carte couleur de la deviation signee (bleu = perte, rouge = gain)."""
    h, w = shape
    img = np.zeros((h, w, 3), np.uint8)
    acc = np.full((h, w), np.nan, np.float32)
    for (x, y), d in zip(pix, dev):
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        acc[yi, xi] = d
    m = ~np.isnan(acc)
    if vmax is None:
        vmax = np.nanpercentile(np.abs(acc[m]), 95) if m.any() else 1.0
    norm = np.clip((acc + vmax) / (2 * vmax + 1e-12), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_COOLWARM 
                            if hasattr(cv2, "COLORMAP_COOLWARM") else cv2.COLORMAP_JET)
    img[m] = col[m]
    return img, float(vmax), m

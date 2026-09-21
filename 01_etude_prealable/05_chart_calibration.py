"""
05_chart_calibration.py — échelle métrique par la charte ColorChecker Nano.

OBJET
    La charte est plane, rigide, à motif régulier, solidaire de l'appui frontal
    (donc fixe par rapport à la tête) et visible dans les 3 vues sous 3
    orientations différentes : c'est la configuration requise par la méthode de
    Zhang (IEEE TPAMI, 2000). Elle est donc candidate au double rôle de mire de
    calibration et de référence d'échelle métrique.

GÉOMÉTRIE PHYSIQUE (mesurée au pied à coulisse — À CONFIRMER)
    patch  : 4.40 x 3.40 mm
    pas    : 5.90 x 5.03 mm (moyenné sur plusieurs itérations)

    !! POINT CRITIQUE !!
    Un volume varie comme le CUBE d'une longueur : une erreur d'échelle
    linéaire de x % induit une erreur de volume d'environ 3x %.
    - Ne JAMAIS dériver l'échelle de l'image (une première estimation image
      donnait 5.74 x 4.62 mm, soit -2 % et -6.9 %, à cause d'un artefact de
      recadrage : troisième rangée de patches tronquée).
    - Mesurer la distance sur 5 intervalles (~29.5 mm) plutôt qu'un pas isolé :
      l'incertitude relative tombe à ~0.2 %, soit ~0.5 % sur le volume.
    - Cohérence à vérifier : avec ces valeurs, les intervalles valent 1.50 mm (H)
      et 1.63 mm (V). Sur une charte industrielle on attendrait des intervalles
      identiques -> incertitude résiduelle sur l'une des dimensions.
    - Idéalement : obtenir la fiche technique Calibrite.

ÉTAT
    NON CONCLUANT en l'état. Frontal : RMS de reprojection 1.46 px (seuil visé
    ~1 px). Obliques : détection insuffisante (charte fortement inclinée,
    partiellement rognée, patches neutres sombres confondus avec le corps noir).
    Une part de l'échec est imputable à la détection par centroïdes ; une
    détection par modèle de grille global devrait faire mieux.

USAGE
    python 05_chart_calibration.py
"""

import cv2
import numpy as np
import visia_lib as vl

SUBJECT = "alban_D0"
MODALITY = "Standard_1"

PATCH_W, PATCH_H = 4.40, 3.40      # mm
PITCH_X, PITCH_Y = 5.90, 5.03      # mm (centre à centre)


def locate_chart(filename):
    """Localise grossièrement la charte : région saturée compacte, en haut."""
    im = cv2.imread(vl.path(filename))
    Hf, Wf = im.shape[:2]
    small = cv2.resize(im, (Wf // 8, Hf // 8), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    m = ((S > 90) & (V > 70)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 200:
            continue
        if 1.2 < w / max(h, 1) < 4.0 and y < small.shape[0] * 0.45:
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    pad = int(0.45 * max(w, h)) * 8
    x0, y0 = max(0, x * 8 - pad), max(0, y * 8 - pad)
    x1, y1 = min(Wf, x * 8 + w * 8 + pad), min(Hf, y * 8 + h * 8 + pad)
    return im[y0:y1, x0:x1]


def detect_patches(crop):
    """Détecte les patches comme quadrilatères. Rejette ceux qui touchent le
    bord du recadrage : c'est exactement le biais qui avait faussé la première
    estimation du pas vertical."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    mask = (((S > 60) & (V > 40)) | (V > 95)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in cnts:
        if cv2.contourArea(c) < 1200:
            continue
        p = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(p) != 4:
            continue
        x, y, w, h = cv2.boundingRect(p)
        if not (0.7 < w / h < 2.6):
            continue
        if x <= 1 or y <= 1 or x + w >= crop.shape[1] - 1 or y + h >= crop.shape[0] - 1:
            continue                                    # patch tronqué -> rejet
        q = p.reshape(4, 2).astype(np.float64)
        quads.append((q, q.mean(0)))
    return quads


def fit_grid(quads):
    """Ordonne les patches en grille (via PCA, pour gérer la charte inclinée
    en oblique) puis ajuste une homographie vers le modèle métrique."""
    if len(quads) < 10:
        return None
    C = np.array([c for _, c in quads])
    X = C - C.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u, v = X @ Vt[0], X @ Vt[1]
    order = np.argsort(v)
    vs = v[order]
    gaps = np.diff(vs)
    thr = np.median(gaps) * 2.5 if len(gaps) else 1.0
    rows, cur = [], [order[0]]
    for i in range(1, len(order)):
        if vs[i] - vs[i - 1] > thr:
            rows.append(cur)
            cur = [order[i]]
        else:
            cur.append(order[i])
    rows.append(cur)
    rows = [r for r in rows if len(r) >= 4]
    rows.sort(key=lambda r: v[r].mean())
    if len(rows) < 2:
        return None

    img_pts, obj_pts = [], []
    for ri, r in enumerate(rows):
        r = sorted(r, key=lambda i: u[i])
        for ci, i in enumerate(r):
            q = quads[i][0]
            qc = q - q.mean(0)
            idx = np.argsort(np.arctan2(qc @ Vt[1], qc @ Vt[0]))
            cx, cy = ci * PITCH_X, ri * PITCH_Y
            model = np.array([[cx - PATCH_W / 2, cy - PATCH_H / 2],
                              [cx + PATCH_W / 2, cy - PATCH_H / 2],
                              [cx + PATCH_W / 2, cy + PATCH_H / 2],
                              [cx - PATCH_W / 2, cy + PATCH_H / 2]])
            mc = model - model.mean(0)
            midx = np.argsort(np.arctan2(mc[:, 1], mc[:, 0]))
            for a_, b_ in zip(q[idx], model[midx]):
                img_pts.append(a_)
                obj_pts.append(b_)

    img_pts = np.float32(img_pts)
    obj_pts = np.float32(obj_pts)
    H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, 3.0)
    proj = cv2.perspectiveTransform(obj_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    inl = mask.ravel().astype(bool)

    o = obj_pts.mean(0)
    p0 = cv2.perspectiveTransform(np.float32([[o]]), H)[0, 0]
    px = cv2.perspectiveTransform(np.float32([[[o[0] + 1, o[1]]]]), H)[0, 0]
    scale_px_per_mm = float(np.linalg.norm(px - p0))

    return {"rows": [len(r) for r in rows], "corners": len(img_pts),
            "inliers": int(inl.sum()), "rms": float(err[inl].mean()),
            "max": float(err[inl].max()), "px_per_mm": scale_px_per_mm}


def main():
    for angle in ["Frontal", "Left_Oblique", "Right_Oblique"]:
        fname = f"{SUBJECT}_{angle}_{MODALITY}.jpg"
        crop = locate_chart(fname)
        if crop is None:
            print(f"{angle}: charte non localisée")
            continue
        quads = detect_patches(crop)
        res = fit_grid(quads)
        print(f"\n=== {angle} ===")
        print(f"  patches détectés : {len(quads)}/18")
        if res is None:
            print("  ajustement de grille : ÉCHEC (détection insuffisante)")
            continue
        print(f"  rangées          : {res['rows']}")
        print(f"  coins utilisés   : {res['corners']} (inliers {res['inliers']})")
        print(f"  reprojection RMS : {res['rms']:.2f} px (max {res['max']:.2f})")
        print(f"  échelle          : {res['px_per_mm']:.1f} px/mm  "
              f"({1000.0 / res['px_per_mm']:.1f} um/px)")
        if res['rms'] > 1.0:
            print("  --> AU-DESSUS du seuil de 1 px : non exploitable en métrologie")


if __name__ == "__main__":
    main()

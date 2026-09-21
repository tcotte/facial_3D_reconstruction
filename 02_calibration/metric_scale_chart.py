"""
metric_scale_chart.py — tentative d'echelle metrique par la charte ColorChecker.

OBJET
    La focale etant connue (50 mm verifie), la reconstruction est euclidienne
    A UN FACTEUR D'ECHELLE GLOBAL PRES. Ce script tente de recuperer ce facteur
    en reconstruisant la charte, dont la geometrie est connue.

RESULTAT OBTENU — ECHEC
    paire F<->L : 215,3 mm/u.a.
    paire F<->R : 338,0 mm/u.a.
    -> ECART DE 57 % entre deux mesures censees donner la meme grandeur,
       avec 28 % de dispersion interne sur les pas d'une meme chaine.

    Causes : 17 patchs detectes seulement, regroupement en grille defaillant,
    8 et 7 pas horizontaux exploitables, un seul pas vertical. Surtout, la
    charte est PETITE et PROCHE DU BORD du champ : sa profondeur reconstruite
    est tres bruitee.

    L'echelle metrique n'est PAS etablie par cette voie.

ALTERNATIVES RECOMMANDEES
    1. Mesurer la DISTANCE DE TRAVAIL (objectif -> plan du visage). Avec la
       focale connue, cette seule mesure donne l'echelle. Un metre ruban ferait
       vraisemblablement mieux que ce script.
    2. Placer un objet de calibration a la position du visage, occupant une part
       significative du champ.

Ce script est conserve pour documenter la tentative et permettre de la rejouer
si la charte devenait plus grande ou mieux placee.

USAGE
    python metric_scale_chart.py --subject alban --session D0 --rig rig.json
"""
import argparse, json, numpy as np, cv2
from scipy.spatial import cKDTree

PITCH_X, PITCH_Y = 5.90, 5.03      # mm, mesure au pied a coulisse — A CONFIRMER
CHART_BAND = 0.30                  # fraction haute de l'image contenant la charte


def detect_patches(top):
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    m = (((S > 70) & (V > 50)) | (V > 110)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
    C = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 60 or not (0.6 < bw / max(bh, 1) < 2.8):
            continue
        if x <= 1 or y <= 1 or x + bw >= top.shape[1] - 1 or y + bh >= top.shape[0] - 1:
            continue          # patch tronque : biaise les centroides
        C.append(cent[i])
    return np.array(C)


def group_grid(C):
    """Ordonne les centres en grille via ACP (gere la charte inclinee)."""
    mu = C.mean(0); X = C - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u, v = X @ Vt[0], X @ Vt[1]
    o = np.argsort(v); vs = v[o]
    gaps = np.diff(vs)
    thr = np.median(gaps) * 2.5 if len(gaps) else 1
    rows, cur = [], [o[0]]
    for i in range(1, len(o)):
        if vs[i] - vs[i - 1] > thr:
            rows.append(cur); cur = [o[i]]
        else:
            cur.append(o[i])
    rows.append(cur)
    rows = [r for r in rows if len(r) >= 4]
    rows.sort(key=lambda r: v[r].mean())
    return rows, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart-points", required=True,
                    help="npz avec p (2D frontal) et X (3D) de la zone charte")
    ap.add_argument("--frontal", required=True, help="image frontale a la resolution de travail")
    a = ap.parse_args()
    A = cv2.imread(a.frontal)
    h = A.shape[0]
    C = detect_patches(A[0:int(CHART_BAND * h), :])
    print(f"patchs detectes : {len(C)}/18")
    if len(C) < 10:
        raise SystemExit("detection insuffisante — l'echelle ne peut pas etre etablie")
    rows, u = group_grid(C)
    print(f"rangees formees : {[len(r) for r in rows]}  (attendu [6, 6, 6])")
    if [len(r) for r in rows] != [6, 6, 6]:
        print("ATTENTION : la grille n'est pas correctement reconstituee, "
              "le resultat sera peu fiable")

    d = np.load(a.chart_points)
    P, X3 = d["p"], d["X"]
    tree = cKDTree(P)
    pts, lab = [], []
    for ri, r in enumerate(rows):
        for ci, i in enumerate(sorted(r, key=lambda k: u[k])):
            dd, j = tree.query(C[i], k=8)
            if dd.min() > 25:
                continue
            wg = 1 / np.maximum(dd, 1e-3); wg /= wg.sum()
            pts.append((X3[j] * wg[:, None]).sum(0)); lab.append((ri, ci))
    pts, lab = np.array(pts), np.array(lab)
    print(f"centres triangules : {len(pts)}")
    dx, dy = [], []
    for k in range(len(pts)):
        for l in range(len(pts)):
            if lab[k, 0] == lab[l, 0] and lab[l, 1] - lab[k, 1] == 1:
                dx.append(np.linalg.norm(pts[l] - pts[k]))
            if lab[k, 1] == lab[l, 1] and lab[l, 0] - lab[k, 0] == 1:
                dy.append(np.linalg.norm(pts[l] - pts[k]))
    dx, dy = np.array(dx), np.array(dy)
    if len(dx) < 3:
        raise SystemExit("pas assez de paires adjacentes")
    sx = PITCH_X / np.median(dx)
    disp = 100 * np.std(dx) / np.median(dx)
    print(f"\npas horizontal : {len(dx)} mesures, dispersion {disp:.1f} %")
    print(f"FACTEUR D'ECHELLE : {sx:.2f} mm/u.a.")
    if len(dy) >= 2:
        sy = PITCH_Y / np.median(dy)
        print(f"  controle par le pas vertical : {sy:.2f} mm/u.a. "
              f"(coherence {100*abs(sx-sy)/np.mean([sx,sy]):.1f} %)")
    if disp > 10:
        print("\n>>> dispersion trop elevee : facteur d'echelle NON FIABLE.")
        print(">>> Preferer une mesure directe de la distance de travail.")

if __name__ == "__main__":
    main()

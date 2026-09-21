"""
04_coverage_map.py — cartographie de la couverture faciale exploitable.

OBJET
    Le nombre d'appariements ne suffit pas : leur RÉPARTITION SPATIALE décide
    de ce qui est réellement mesurable. Ce script projette les appariements
    validés sur la vue frontale et quantifie la couverture par quadrillage.

RÉSULTAT OBTENU
    Zone faciale centrale (grille 14x14, colonnes centrales, bandes 30-86 %) :
    44 cellules sur 64 contiennent au moins un point, soit 69 %.

    NOTE MÉTHODOLOGIQUE : une première évaluation donnait 31 %, mais elle
    incluait à tort les bords et le fond noir, qui ne peuvent par construction
    contenir aucun point. La valeur retenue est 69 %.

    Constat qualitatif : forte densité autour du nez, des sillons et de la
    barbe ; faible densité sur les joues latérales, le front et le menton.

USAGE
    python 04_coverage_map.py
"""

import cv2
import numpy as np
import visia_lib as vl

SUBJECT = "alban_D0"
MODALITY = "Standard_1"          # meilleure modalité géométrique
GRID = 14
OUT = "couverture_frontal.png"


def main():
    frontal = f"{SUBJECT}_Frontal_{MODALITY}.jpg"
    a = vl.load_gray(frontal, vl.LONG_ASIFT)
    ka, da = vl.asift_detect(a)
    print(f"frontal : {len(ka)} points ASIFT")

    H, W = a.shape
    vis = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    vis = (vis * 0.55).astype(np.uint8)
    colors = {"Left_Oblique": (80, 255, 80), "Right_Oblique": (80, 160, 255)}
    all_pts = []

    for angle, col in colors.items():
        b = vl.load_gray(f"{SUBJECT}_{angle}_{MODALITY}.jpg", vl.LONG_ASIFT)
        kb, db = vl.asift_detect(b)
        pa, pb = vl.match(da, db, ka, kb, ratio=0.80, flann=True)
        ia, ib, _ = vl.geometric_filter(pa, pb)
        print(f"  {angle}: {len(pa)} bruts -> {len(ia)} inliers")
        all_pts.append(ia)
        for p in ia:
            cv2.circle(vis, (int(p[0]), int(p[1])), 4, col, -1)

    cv2.imwrite(OUT, vis)
    print(f"\ncarte écrite : {OUT}")

    # occupation par cellule, restreinte à la zone faciale centrale
    occ = np.zeros((GRID, GRID), int)
    for p in np.vstack(all_pts):
        occ[min(int(p[1] / H * GRID), GRID - 1),
            min(int(p[0] / W * GRID), GRID - 1)] += 1

    rows = range(int(vl.SKIN_TOP * GRID), int(vl.SKIN_BOTTOM * GRID))
    cells = [(r, c) for r in rows for c in range(3, GRID - 3)]
    filled = sum(1 for r, c in cells if occ[r, c] > 0)
    print(f"couverture faciale centrale : {filled}/{len(cells)} = "
          f"{100.0 * filled / len(cells):.0f} %")

    print("\noccupation (haut -> bas) :")
    for r in range(GRID):
        print(" ".join(f"{v:3d}" for v in occ[r]))


if __name__ == "__main__":
    main()

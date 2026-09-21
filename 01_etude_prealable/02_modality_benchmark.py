"""
02_modality_benchmark.py — classement des modalités d'éclairage pour la 3D.

OBJET
    Mesurer, pour chaque modalité VISIA-CR, le nombre d'appariements
    géométriquement validés SUR LA PEAU entre paires de vues. C'est la mesure
    qui détermine si les images contiennent l'information nécessaire à une
    reconstruction — indépendamment de l'algorithme de reconstruction choisi.

RÉSULTATS OBTENUS (ASIFT, appariements sur la peau)
    Modalité              F<->Obl.G   F<->Obl.D
    Standard                  435         206     <- retenue
    Cross-polarisé            341         201     <- complément / colorimétrie
    Parallèle-polarisé         10           6     <- exclu
    Raked                       7          19     <- exclu

INTERPRÉTATION
    L'éclairage VISIA est solidaire de la caméra. Les modalités qui exposent
    l'ALBÉDO (pigment : standard, cross-polarisé) donnent une information
    indépendante du point de vue, donc appariable. Celles qui exposent la
    RÉPONSE DIRECTIONNELLE (spéculaire en parallèle-polarisé, ombres portées en
    raked) donnent une information qui se déplace sur le visage quand la caméra
    tourne, donc non appariable.

USAGE
    python 02_modality_benchmark.py            # SIFT seul, rapide
    python 02_modality_benchmark.py --asift    # ajoute ASIFT (lent : ~10 min)
"""

import sys
import numpy as np
import visia_lib as vl

SUBJECT = "alban_D0"
SETS = {
    "Standard":           "Standard_1",
    "Cross-polarise":     "Cross-Polarized",
    "Parallele-polarise": "Parallel-Polarized",
    "Raked":              "Raked",
}
PAIRS = [("Frontal", "Left_Oblique"), ("Frontal", "Right_Oblique"),
         ("Left_Oblique", "Right_Oblique")]


def bench_sift(suffix):
    sift = vl.make_sift()
    g, kp, des = {}, {}, {}
    for angle in ["Frontal", "Left_Oblique", "Right_Oblique"]:
        g[angle] = vl.load_gray(f"{SUBJECT}_{angle}_{suffix}.jpg", vl.LONG_SIFT)
        kp[angle], des[angle] = sift.detectAndCompute(g[angle], None)
    out = {}
    for a, b in PAIRS:
        pa, pb = vl.match(des[a], des[b], kp[a], kp[b], ratio=0.78)
        ia, ib, _ = vl.geometric_filter(pa, pb)
        n_skin = int(vl.skin_mask(ia, g[a].shape[0]).sum()) if len(ia) else 0
        out[(a, b)] = (len(pa), len(ia), n_skin)
    return out


def bench_asift(suffix):
    g, kp, des = {}, {}, {}
    for angle in ["Frontal", "Left_Oblique", "Right_Oblique"]:
        g[angle] = vl.load_gray(f"{SUBJECT}_{angle}_{suffix}.jpg", vl.LONG_ASIFT)
        kp[angle], des[angle] = vl.asift_detect(g[angle])
    out = {}
    for a, b in PAIRS[:2]:          # L<->R est structurellement vide
        pa, pb = vl.match(des[a], des[b], kp[a], kp[b], ratio=0.80, flann=True)
        ia, ib, _ = vl.geometric_filter(pa, pb)
        n_skin = int(vl.skin_mask(ia, g[a].shape[0]).sum()) if len(ia) else 0
        out[(a, b)] = (len(pa), len(ia), n_skin)
    return out


def main():
    use_asift = "--asift" in sys.argv

    print("=" * 78)
    print("SIFT  (côté long %d px, ratio de Lowe 0.78, RANSAC F 3.0 px)" % vl.LONG_SIFT)
    print("=" * 78)
    for name, suffix in SETS.items():
        res = bench_sift(suffix)
        line = " | ".join(
            f"{a[0]}-{b[0]}: brut={v[0]:4d} inl={v[1]:3d} peau={v[2]:3d}"
            for (a, b), v in res.items())
        print(f"{name:20s} {line}")

    if use_asift:
        print()
        print("=" * 78)
        print("ASIFT (côté long %d px, ratio 0.80) — lent" % vl.LONG_ASIFT)
        print("=" * 78)
        for name, suffix in SETS.items():
            res = bench_asift(suffix)
            line = " | ".join(
                f"{a[0]}-{b[0]}: brut={v[0]:4d} inl={v[1]:3d} peau={v[2]:3d}"
                for (a, b), v in res.items())
            print(f"{name:20s} {line}")

    print()
    print("=" * 78)
    print("Photométrie (région cutanée)")
    print("=" * 78)
    for name, suffix in SETS.items():
        for angle in ["Frontal", "Left_Oblique"]:
            m = vl.photometry(f"{SUBJECT}_{angle}_{suffix}.jpg")
            print(f"{name:20s} {angle:14s} "
                  f"speculaire={m['specular_pct']:5.2f} %   "
                  f"texture={m['texture_lap_std']:6.1f}")


if __name__ == "__main__":
    main()

"""
selftest.py — validation du pipeline de comparaison sur deformation SIMULEE.

Ne necessite aucune acquisition Dx : on part d'une session reelle, on applique
une deformation d'amplitude CONNUE, et on verifie que le pipeline la retrouve.

    python selftest.py <sujet>_D0.npz

Verifie trois choses :
  1. une deformation nulle donne une mesure nulle (pas de biais)
  2. une deformation connue est retrouvee proportionnellement
  3. le test a blanc fournit un plancher de bruit coherent
"""
import sys, numpy as np
import config as C, visia_compare as cmp

def main():
    d = np.load(sys.argv[1], allow_pickle=True)
    X0 = d["X"].astype(float); pix = d["pix"]
    c = X0.mean(0)
    r2 = (X0[:, 0] - c[0]) ** 2 + (X0[:, 1] - c[1]) ** 2
    rad = 0.025 * np.median(X0[:, 2])
    roi = r2 < (1.5 * rad) ** 2
    stable = r2 > (4.0 * rad) ** 2
    print(f"{len(X0)} points | ROI {roi.sum()} | stable {stable.sum()}")
    if roi.sum() < C.MIN_ROI_POINTS or stable.sum() < C.MIN_STABLE_POINTS:
        raise SystemExit("couverture insuffisante pour l'autotest")

    print("\n--- 1. deformation nulle (controle de biais)")
    m, _, _ = cmp.compare(X0, X0.copy(), roi, stable)
    print(f"    mesure : {m['ratio_relief_moyen']:+.6f}  (attendu 0)")

    print("\n--- 2. deformations connues")
    print("    amplitude imposee | mesure | rapport")
    base = None
    for amp_frac in [0.001, 0.002, 0.004, 0.008]:
        amp = amp_frac * np.median(X0[:, 2])
        Xd = X0.copy()
        Xd[:, 2] -= amp * np.exp(-r2 / (2 * rad ** 2))
        m, _, _ = cmp.compare(X0, Xd, roi, stable)
        val = m["ratio_relief_moyen"]
        if base is None:
            base = val / amp_frac
        print(f"    {amp_frac*100:6.2f} %          | {val:+.5f} | {val/(base*amp_frac):5.3f}")

    print("\n--- 3. test a blanc (plancher de bruit)")
    nt = cmp.null_test(X0, X0.copy(), stable)
    if nt:
        print(f"    ecart-type {nt['ecart_type']:.6f} | "
              f"seuil 2 sigma {nt['seuil_detection_2sigma']:.6f}")
    print("\nAutotest termine.")

if __name__ == "__main__":
    main()

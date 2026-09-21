"""
export_pointcloud.py — export du nuage 3D d'une session au format PLY.

OBJET
    Produit un nuage colore, lisible dans MeshLab ou CloudCompare.

UNITES — POINT IMPORTANT
    Les coordonnees sont en UNITES ARBITRAIRES : la profondeur mediane est
    normalisee a 1. La focale est connue (50 mm verifie), donc la FORME est
    euclidienne, mais le FACTEUR D'ECHELLE GLOBAL ne l'est pas.
    -> ne jamais convertir en mm sans avoir mesure la distance de travail.

TROUS
    Les zones non reconstruites (nez, paupieres, cavites orbitaires, sourcils)
    apparaissent comme des trous. Elles correspondent aux regions occluses ou a
    trop faible texture. AUCUNE INTERPOLATION n'y est appliquee : une surface
    incomplete mais mesuree est defendable, une surface complete partiellement
    hallucinee ne l'est pas (hypothese H4).

USAGE
    python export_pointcloud.py --session sess_D0.npz --frontal frontal_D0.png \
        --out nuage_D0.ply
"""
import argparse, numpy as np, cv2


def write_ply(path, X, C, comment=""):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        for line in comment.split("\n"):
            if line:
                f.write(f"comment {line}\n")
        f.write(f"element vertex {len(X)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(X, C):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="percentile de rejet en profondeur (haut et bas)")
    a = ap.parse_args()
    im = cv2.imread(a.frontal)
    h, w = im.shape[:2]
    d = np.load(a.session)
    X, pix, src = d["X"].astype(float), d["pix"], d["src"]
    ok = np.isfinite(X).all(1) & (X[:, 2] > 0)
    X, pix, src = X[ok], pix[ok], src[ok]
    lo, hi = np.percentile(X[:, 2], [a.clip, 100 - a.clip])
    k = (X[:, 2] > lo) & (X[:, 2] < hi)
    X, pix, src = X[k], pix[k], src[k]
    Xn = X / np.median(X[:, 2])
    Xn = Xn - Xn.mean(0)
    Xn[:, 1] *= -1                       # +Y vers le haut de l'image
    C = np.array([im[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))][::-1]
                  for x, y in pix])
    write_ply(a.out, Xn, C,
              f"{len(Xn)} points | unites ARBITRAIRES (profondeur mediane = 1)\n"
              f"focale 50 mm f/13 verifiee | facteur d'echelle global INCONNU\n"
              f"paire L: {int((src=='L').sum())} | paire R: {int((src=='R').sum())}\n"
              f"trous = zones non mesurees, aucune interpolation")
    print(f"{len(Xn)} points -> {a.out}")
    print(f"  paire L {int((src=='L').sum())} | paire R {int((src=='R').sum())}")
    print(f"  etendue X {np.ptp(Xn[:,0]):.4f} | Y {np.ptp(Xn[:,1]):.4f} | Z {np.ptp(Xn[:,2]):.4f}")
    print("  RAPPEL : unites arbitraires, ne pas convertir en mm")

if __name__ == "__main__":
    main()

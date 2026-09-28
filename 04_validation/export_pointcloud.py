"""
export_pointcloud.py — export du nuage 3D d'une session au format PLY.

OBJET
    Produit un nuage colore, lisible dans MeshLab ou CloudCompare.

DEUX ENTREES POSSIBLES
    --session  sortie de run_session.py : des points 3D deja triangules, pour
               UNE modalite. Porte le residu epipolaire et, en RoMa v2,
               l'overlap et l'ecart-type de localisation.
    --fusion   sortie de fuse_modalities.py (ou de outlier_filter.py) : une
               CARTE de profondeur Z(y,x) issue de plusieurs modalites. Les
               points 3D sont retroprojetes ici meme, X = Z * K^-1 [x, y, 1].
               Le champ scalaire est alors `nch`, le nombre de modalites ayant
               mesure chaque pixel — une redondance, pas une precision.

    Passer par la fusion donne la COUVERTURE des quatre modalites (74,6 % du
    masque contre 51,9 % pour le standard seul), pas une meilleure precision :
    les erreurs entre modalites sont correlees. Voir fuse_modalities.py.

    Ordre recommande :
        fuse_modalities.py  ->  outlier_filter.py  ->  export_pointcloud.py
    outlier_filter rejette 12 a 13 % de points aberrants ; l'ignorer laisse
    dans le nuage les pics des bords de paupieres et d'ailes du nez.

REPERE — CORRIGE LE 2026-09-22
    L'ancienne version negriait Y seul pour mettre le haut de l'image vers le
    haut. Cette transformation a un determinant de -1 : c'est une SYMETRIE.
    Les visualiseurs regardant par defaut le long de -Z, le nuage etait vu de
    l'arriere et le relief apparaissait INVERSE (nez rentrant, "masque creux").
    Le defaut est maintenant `--frame viewer`, qui negrie Y et Z : rotation
    pure, relief correct. `--frame legacy` reproduit l'ancien comportement.

    Un nuage produit avant cette date est donc a regenerer si son relief
    semblait faux.

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

IMAGE FRONTALE
    `--frontal` accepte le jpg NATIF : il est ramene a la resolution de travail
    par la meme convention que visia_core.load(), puis la coherence avec le
    champ `shape` de la session est verifiee. Donner une image a une autre
    resolution sans redimensionnement echantillonnerait les couleurs au mauvais
    endroit, silencieusement.

CHAMPS SCALAIRES
    Le PLY porte le residu epipolaire de chaque point (`conf`), et, pour une
    session reconstruite avec RoMa v2, l'overlap (`cert`) et l'ecart-type de
    localisation predit (`sigma`). Dans CloudCompare ils apparaissent comme
    scalar fields : colorer le nuage par `sigma` montre directement ou la
    mesure est contrainte et ou elle ne l'est pas.

USAGE
    # une modalite
    python export_pointcloud.py --session alban_D0.npz \
        --frontal /donnees/visia/alban_D0_Frontal_Standard_1.jpg \
        --out nuage_D0.ply

    # les quatre modalites fusionnees, filtrees
    python fuse_modalities.py --subject alban --session D0 --rig rig.json \
        --img-dir /donnees/visia --roma-dir .
    python outlier_filter.py --fusion fusion_alban_D0.npz
    python export_pointcloud.py --fusion fusion_alban_D0_filtre.npz \
        --frontal /donnees/visia/alban_D0_Frontal_Standard_1.jpg \
        --out nuage_D0_fusion.ply
"""
import argparse, os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "00_pipeline"))
import config as C                      # noqa: E402


SCALAR_LABELS = {"conf": "residu epipolaire (px)",
                 "cert": "overlap RoMa [0,1]",
                 "sigma": "ecart-type de localisation (px)",
                 "nch": "nombre de modalites ayant mesure le pixel"}


def write_ply(path, X, C_rgb, scalars=None, comment="", binary=True):
    """PLY colore, avec champs scalaires optionnels.

    Binaire par defaut : sur ~780 000 points l'ecriture ASCII prend des
    dizaines de secondes et produit un fichier cinq fois plus gros. MeshLab et
    CloudCompare lisent les deux.
    """
    scalars = scalars or {}
    n = len(X)
    head = ["ply",
            "format " + ("binary_little_endian 1.0" if binary else "ascii 1.0")]
    for line in comment.split("\n"):
        if line:
            head.append(f"comment {line}")
    for k in scalars:
        head.append(f"comment scalar {k} = {SCALAR_LABELS.get(k, k)}")
    head += [f"element vertex {n}",
             "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green", "property uchar blue"]
    head += [f"property float {k}" for k in scalars]
    head.append("end_header")
    header = ("\n".join(head) + "\n").encode("ascii")

    if binary:
        dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        dt += [(k, "<f4") for k in scalars]
        arr = np.empty(n, dtype=dt)
        arr["x"], arr["y"], arr["z"] = X[:, 0], X[:, 1], X[:, 2]
        arr["red"], arr["green"], arr["blue"] = C_rgb[:, 0], C_rgb[:, 1], C_rgb[:, 2]
        for k, v in scalars.items():
            arr[k] = np.nan_to_num(np.asarray(v, np.float32),
                                   nan=-1.0, posinf=-1.0, neginf=-1.0)
        with open(path, "wb") as f:
            f.write(header)
            arr.tofile(f)
    else:
        cols = [X, C_rgb.astype(int)] + [np.asarray(v, float)[:, None]
                                         for v in scalars.values()]
        M = np.hstack(cols)
        fmt = "%.6f %.6f %.6f %d %d %d" + " %.4f" * len(scalars)
        with open(path, "wb") as f:
            f.write(header)
            np.savetxt(f, M, fmt=fmt)


def load_frontal(path, shape):
    """Charge l'image frontale et la ramene a la resolution de travail."""
    im = cv2.imread(path)
    if im is None:
        raise SystemExit(f"image frontale illisible : {path}")
    h, w = shape
    if im.shape[:2] != (h, w):
        s = C.WORK_LONG / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)),
                        interpolation=cv2.INTER_AREA)
        print(f"  image frontale ramenee a {im.shape[1]}x{im.shape[0]} "
              f"(cote long {C.WORK_LONG})")
    if im.shape[:2] != (h, w):
        raise SystemExit(
            f"image frontale {im.shape[1]}x{im.shape[0]} incompatible avec la "
            f"session ({w}x{h}). Verifier que c'est bien la vue frontale de "
            f"cette session et que config.WORK_LONG n'a pas change.")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None,
                    help="sortie de run_session.py (points 3D)")
    ap.add_argument("--fusion", default=None,
                    help="sortie de fuse_modalities.py ou outlier_filter.py "
                         "(carte de profondeur)")
    ap.add_argument("--frontal", required=True,
                    help="vue frontale de CETTE session (jpg natif accepte)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="percentile de rejet en profondeur (haut et bas)")
    ap.add_argument("--pair", choices=["L", "R"], default=None,
                    help="n'exporter qu'une branche stereo")
    ap.add_argument("--max-epi", type=float, default=None,
                    help="rejet sur le residu epipolaire (px)")
    ap.add_argument("--max-sigma", type=float, default=None,
                    help="rejet sur l'ecart-type predit (sessions RoMa v2)")
    ap.add_argument("--frame", choices=["viewer", "legacy"], default="viewer",
                    help="'viewer' (defaut) : X droite, Y haut, Z vers le "
                         "spectateur — rotation pure, relief correct. "
                         "'legacy' : ancien comportement (Y negrie seul), qui "
                         "est une SYMETRIE et inverse le relief.")
    ap.add_argument("--ascii", action="store_true",
                    help="PLY ASCII au lieu de binaire (5x plus gros, bien plus lent)")
    ap.add_argument("--no-scalars", action="store_true")
    a = ap.parse_args()
    if bool(a.session) == bool(a.fusion):
        raise SystemExit("donner --session OU --fusion, pas les deux ni aucun")

    if a.fusion:
        d = np.load(a.fusion, allow_pickle=True)
        Zmap = d["Z"].astype(np.float64)
        h, w = Zmap.shape
        f = float(d["focal"])
        nch = d["nch"] if "nch" in d.files else None
        mods = list(d["modalites"]) if "modalites" in d.files else None
        print(f"fusion     : {os.path.basename(a.fusion)}")
        print(f"carte {w}x{h} | focale {f:.1f} px"
              + (f" | modalites : {', '.join(str(m) for m in mods)}" if mods else ""))
        if "keep" in d.files:
            print(f"   filtre par outlier_filter.py : "
                  f"{int(d['rej'].sum())} points rejetes")
        else:
            print("   NON FILTRE — passer par outlier_filter.py avant toute mesure "
                  "(12 a 13 % d'aberrants)")
        ys, xs = np.nonzero(np.isfinite(Zmap))
        Zv = Zmap[ys, xs]
        # retroprojection : X = Z * K^-1 [x, y, 1]
        X = np.column_stack([(xs - w / 2) / f * Zv, (ys - h / 2) / f * Zv, Zv])
        pix = np.column_stack([xs, ys]).astype(np.float64)
        src = np.full(len(X), "F")
        conf = cert = sigma = None
        nchv = nch[ys, xs].astype(float) if nch is not None else None
        stamp = "fusion multi-modalites"
        im = load_frontal(a.frontal, (h, w))
        n0 = len(X)
        lo, hi = np.percentile(X[:, 2], [a.clip, 100 - a.clip])
        k = (X[:, 2] > lo) & (X[:, 2] < hi)
        X, pix, src = X[k], pix[k], src[k]
        if nchv is not None:
            nchv = nchv[k]
        Xn = X / np.median(X[:, 2])
        Xn = Xn - Xn.mean(0)
        if a.frame == "legacy":
            Xn[:, 1] *= -1
            print("  repere 'legacy' : symetrie (determinant -1), relief inverse")
        else:
            Xn[:, 1] *= -1
            Xn[:, 2] *= -1
            print("  repere 'viewer' : X droite, Y haut, Z vers le spectateur "
                  "(rotation pure)")
        xi = np.clip(pix[:, 0].astype(int), 0, w - 1)
        yi = np.clip(pix[:, 1].astype(int), 0, h - 1)
        C_rgb = im[yi, xi][:, ::-1]
        scalars = {} if a.no_scalars or nchv is None else {"nch": nchv}
        write_ply(a.out, Xn.astype(np.float32), C_rgb, scalars,
                  comment=(f"{len(Xn)} points | {stamp}\n"
                           f"unites ARBITRAIRES (profondeur mediane = 1)\n"
                           f"focale 50 mm f/13 verifiee | facteur d'echelle global INCONNU\n"
                           f"nch = nombre de modalites ayant mesure le pixel\n"
                           f"trous = zones non mesurees, aucune interpolation"),
                  binary=not a.ascii)
        print(f"\n{len(Xn)} points retenus sur {n0} -> {a.out} "
              f"({os.path.getsize(a.out)/1e6:.1f} Mo, "
              f"{'ASCII' if a.ascii else 'binaire'})")
        print(f"  etendue X {np.ptp(Xn[:,0]):.4f} | Y {np.ptp(Xn[:,1]):.4f} | "
              f"Z {np.ptp(Xn[:,2]):.4f}")
        if nchv is not None:
            for k2 in range(1, 5):
                n_ = int((nchv == k2).sum())
                if n_:
                    print(f"  {k2} modalite(s) : {n_:8d} points "
                          f"({100*n_/len(nchv):.1f} %)")
        print("  RAPPEL : unites arbitraires, ne pas convertir en mm")
        return

    d = np.load(a.session, allow_pickle=True)
    stamp = str(d["calib"]) if "calib" in d.files else "(non estampille)"
    h, w = (int(v) for v in d["shape"])
    print(f"session    : {os.path.basename(a.session)}")
    print(f"chaine     : {stamp}")
    if not stamp.startswith(C.CALIB_VERSION):
        print(f"  ATTENTION : stamp different de config.CALIB_VERSION "
              f"({C.CALIB_VERSION})")
    im = load_frontal(a.frontal, (h, w))

    X, pix, src = d["X"].astype(float), d["pix"], d["src"]
    conf = d["conf"].astype(float) if "conf" in d.files else None
    cert = d["cert"].astype(float) if "cert" in d.files else None
    sigma = d["sigma"].astype(float) if "sigma" in d.files else None

    def apply(mask):
        nonlocal X, pix, src, conf, cert, sigma
        X, pix, src = X[mask], pix[mask], src[mask]
        conf = conf[mask] if conf is not None else None
        cert = cert[mask] if cert is not None else None
        sigma = sigma[mask] if sigma is not None else None

    n0 = len(X)
    apply(np.isfinite(X).all(1) & (X[:, 2] > 0))
    if a.pair:
        apply(src == a.pair)
    if a.max_epi is not None and conf is not None:
        apply(conf < a.max_epi)
    if a.max_sigma is not None:
        if sigma is None:
            print("  --max-sigma ignore : la session ne porte pas de champ sigma "
                  "(reconstruite sans RoMa v2)")
        else:
            apply(sigma < a.max_sigma)
    if not len(X):
        raise SystemExit("aucun point ne subsiste apres filtrage")
    lo, hi = np.percentile(X[:, 2], [a.clip, 100 - a.clip])
    apply((X[:, 2] > lo) & (X[:, 2] < hi))

    Xn = X / np.median(X[:, 2])
    Xn = Xn - Xn.mean(0)
    # Repere camera OpenCV : X droite, Y BAS, Z avant (droitier).
    # Negrier Y SEUL a un determinant de -1 : c'est une symetrie, pas une
    # rotation. Le nuage est alors vu de l'arriere par les visualiseurs, qui
    # regardent par defaut le long de -Z, et le relief s'inverse — le nez
    # rentre au lieu de sortir ("masque creux"). Negrier Y ET Z redonne une
    # rotation pure : X droite, Y haut, Z vers le spectateur.
    if a.frame == "legacy":
        Xn[:, 1] *= -1
        print("  repere 'legacy' : symetrie (determinant -1), relief inverse")
    else:
        Xn[:, 1] *= -1
        Xn[:, 2] *= -1
        print("  repere 'viewer' : X droite, Y haut, Z vers le spectateur "
              "(rotation pure)")

    xi = np.clip(pix[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pix[:, 1].astype(int), 0, h - 1)
    C_rgb = im[yi, xi][:, ::-1]          # BGR -> RGB

    scalars = {}
    if not a.no_scalars:
        if conf is not None:
            scalars["conf"] = conf
        if cert is not None:
            scalars["cert"] = cert
        if sigma is not None:
            scalars["sigma"] = sigma

    write_ply(a.out, Xn.astype(np.float32), C_rgb, scalars,
              comment=(f"{len(Xn)} points | chaine {stamp}\n"
                       f"unites ARBITRAIRES (profondeur mediane = 1)\n"
                       f"focale 50 mm f/13 verifiee | facteur d'echelle global INCONNU\n"
                       f"paire L: {int((src=='L').sum())} | paire R: {int((src=='R').sum())}\n"
                       f"trous = zones non mesurees, aucune interpolation"),
              binary=not a.ascii)

    print(f"\n{len(Xn)} points retenus sur {n0} -> {a.out} "
          f"({os.path.getsize(a.out)/1e6:.1f} Mo, "
          f"{'ASCII' if a.ascii else 'binaire'})")
    print(f"  paire L {int((src=='L').sum())} | paire R {int((src=='R').sum())}")
    print(f"  etendue X {np.ptp(Xn[:,0]):.4f} | Y {np.ptp(Xn[:,1]):.4f} | "
          f"Z {np.ptp(Xn[:,2]):.4f}")
    if scalars:
        print(f"  champs scalaires : {', '.join(scalars)}")
        for k, v in scalars.items():
            print(f"    {k:6s} mediane {np.nanmedian(v):.3f} | "
                  f"90e centile {np.nanpercentile(v, 90):.3f}")
    print("  RAPPEL : unites arbitraires, ne pas convertir en mm")


if __name__ == "__main__":
    main()

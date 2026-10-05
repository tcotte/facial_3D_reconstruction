"""
fuse_modalities.py — fusion des quatre modalites d'eclairage.

OBJET
    Reconstruire une chaine 3D COMPLETE par modalite, puis fusionner les cartes
    de profondeur. Les quatre modalites partagent le meme point de vue (1,5 px
    de decalage sur 2400, soit 0,06 % du champ), donc le meme rig s'applique et
    les nuages sont directement superposables.

CE QUI MARCHE, CE QUI NE MARCHE PAS
    Apparier ENTRE modalites differentes (frontal standard <-> oblique raked)
    echoue : chaque modalite porte une information dependante du point de vue.
    Ce qui marche est de reconstruire une chaine independante PAR modalite, puis
    de fusionner les resultats 3D.

RESULTAT — la couverture depend fortement du sujet
    modalite            sujet 1   sujet 2
    standard             51,9 %    32,9 %
    cross-polarise       45,5 %    41,8 %
    raked                41,9 %    27,0 %
    parallele-polarise   34,6 %    22,4 %
    UNION                74,6 %    64,4 %

    Le classement S'INVERSE selon le type de peau : sur un visage mature et
    texture le standard domine, sur un visage jeune et lisse c'est le
    cross-polarise. Le CP est la seule modalite stable entre sujets.

AVEC ROMA v2 (--roma-dir)
    Chaque modalite est densifiee par son propre champ RoMa v2 exporte, au lieu
    du flot optique DIS. Meme rig gele, meme triangulation, meme seuillage.
    Le residu epipolaire est calcule a partir du rig, donc independant de
    l'appariement.

    Le rapport de reduction de bruit ci-dessous a ete mesure AVEC LE FLOT
    OPTIQUE. Il est desormais RE-MESURE a chaque execution : ne pas supposer
    qu'il vaut encore 1,004 avec un autre moteur de densification. C'est le
    chiffre a regarder pour savoir si la fusion vous achete de la precision ou
    seulement de la couverture.

CE QUE LA FUSION N'APPORTE PAS
    Le moyennage des chaines ne reduit PAS le bruit : les erreurs sont
    correlees (rapport observe 1,004 contre 0,707 attendu si independantes).
    La fusion apporte de la COUVERTURE et de la REDONDANCE DE CONTROLE, pas de
    la precision.

USAGE
    python fuse_modalities.py --subject alban --session D0 --rig rig.json
"""
import argparse
import json
import os
import cv2
import numpy as np
from face_mask import face_mask

MODALITES = [("Standard 1", "std"), ("Cross-Polarized", "cp"),
             ("Raked", "raked"), ("Parallel-Polarized", "pp")]


def triangulate(K, R, t, p1, p2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (X[:3] / X[3]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--dense-dir", default=".",
                    help="dossier des sorties de densification (obx, oby, epi)")
    ap.add_argument("--img-dir", default=".")
    ap.add_argument("--long", type=int, default=2000)
    ap.add_argument("--focal-rel", type=float, default=1.42222)
    ap.add_argument("--epi-max", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--roma-dir", default=None,
                    help="dossier des champs RoMa v2 exportes "
                         "(roma2_<sujet>_<session>_<modalite>.npz). Remplace la "
                         "densification par flot optique.")
    ap.add_argument("--roma-cert", type=float, default=0.05)
    ap.add_argument("--epi-auto", type=float, default=None,
                    help="seuil epipolaire exprime en MULTIPLE du plancher "
                         "mesure sur chaque champ, au lieu d'une valeur en "
                         "pixels. 2 a 3 est raisonnable. Rend le filtrage "
                         "comparable entre sujets : un seuil fixe de 1 px vaut "
                         "5x le plancher sur un sujet et 0,6x sur un autre, "
                         "ou il coupe au milieu du bruit irreductible.")
    ap.add_argument("--texture-min", type=float, default=5.0,
                    help="seuil de texture locale (etait code en dur a 5.0)")
    ap.add_argument("--roma-fill", action="store_true",
                    help="interpole le champ entre les points exportes, comme "
                         "run_session.py --roma-fill. Sans cette option, un "
                         "export au pas 2 plafonne a 25 % des pixels du masque.")
    a = ap.parse_args()

    champs = {}
    if a.roma_dir:
        import roma_field as rfmod
        for mod, pfx in MODALITES:
            for nom in (mod, mod.replace("_", " "), mod.replace(" ", "_")):
                p = os.path.join(a.roma_dir,
                                 f"roma2_{a.subject}_{a.session}_{nom}.npz")
                if os.path.exists(p):
                    champs[pfx] = rfmod.load(p)
                    break
        if not champs:
            raise SystemExit(f"aucun champ RoMa trouve dans {a.roma_dir}")
        # la resolution est imposee par les champs, pas par --long
        gh, gw = next(iter(champs.values())).H, next(iter(champs.values())).W
        for pfx, f_ in champs.items():
            if (f_.H, f_.W) != (gh, gw):
                raise SystemExit(f"les champs n'ont pas tous la meme grille "
                                 f"({pfx} : {f_.W}x{f_.H} au lieu de {gw}x{gh})")
        a.long = max(gh, gw)
        print(f"densification : champs RoMa v2 ({len(champs)} modalites) "
              f"sur une grille {gw}x{gh}")
        for pfx, f_ in champs.items():
            print(f"   {pfx:6s} <- {os.path.basename(f_.path)}")

    A = cv2.imread(f"{a.img_dir}/{a.subject}_{a.session}_Frontal_Standard 1.jpg")
    h0, w0 = A.shape[:2]
    s = a.long / max(h0, w0)
    h, w = int(h0 * s), int(w0 * s)
    A = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    if champs and (h, w) != (gh, gw):
        raise SystemExit(
            f"l'image frontale redimensionnee fait {w}x{h} alors que les champs "
            f"RoMa sont sur une grille {gw}x{gh}. Verifier que les jpg sont bien "
            f"ceux qui ont servi a l'export.")
    face = face_mask(A)[0]
    g = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    tex = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (15, 15))

    f = a.focal_rel * max(w, h)
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], float)
    D = np.zeros(4)
    und = lambda p: cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64),
                                        K, D, P=K).reshape(-1, 2)
    rig = json.load(open(a.rig))
    POSE = {k: (np.array(v["R"]), np.array(v["t"])) for k, v in rig["poses"].items()}

    def chain(pfx):
        Z = np.full((h, w), np.nan, np.float32)
        for tag in "LR":
            if pfx in champs:
                f_ = champs[pfx]
                if tag not in f_.pairs:
                    continue
                seuil = a.epi_max
                if a.epi_auto:
                    sol = f_.plancher(tag, cert_min=max(a.roma_cert, 0.5))
                    if sol:
                        seuil = a.epi_auto * sol
                        print(f"      {pfx}/{tag} : plancher {sol:.2f} px "
                              f"-> seuil {seuil:.2f} px")
                    else:
                        print(f"      {pfx}/{tag} : plancher non mesurable, "
                              f"seuil fixe {seuil} px")
                # residu epipolaire issu du RIG : independant de l'appariement
                obx, oby, epi, _ = f_.densify_like(
                    tag, K, *POSE[tag], epi_from="rig", cert_min=a.roma_cert,
                    fill=a.roma_fill)
            else:
                obx = np.load(f"{a.dense_dir}/{a.session}_{pfx}_obx_{tag}.npy")
                oby = np.load(f"{a.dense_dir}/{a.session}_{pfx}_oby_{tag}.npy")
                epi = np.load(f"{a.dense_dir}/{a.session}_{pfx}_epi_{tag}.npy")
            seuil_eff = seuil if (pfx in champs) else a.epi_max
            hi = face & (~np.isnan(obx)) & (epi < seuil_eff) & (tex > a.texture_min)
            ys, xs = np.nonzero(hi)
            p1 = np.column_stack([xs, ys]).astype(np.float32)
            p2 = np.column_stack([obx[ys, xs], oby[ys, xs]]).astype(np.float32)
            X = triangulate(K, *POSE[tag], und(p1), und(p2))
            m = np.isfinite(X).all(1) & (X[:, 2] > 0)
            Z[ys[m], xs[m]] = X[m, 2]
        return Z

    if a.epi_auto:
        print(f"\nseuil epipolaire AUTOMATIQUE : {a.epi_auto} x le plancher de "
              f"chaque champ")
        print(f"   Le plancher est le residu de la meilleure F possible sur ce "
              f"champ :")
        print(f"   aucun rig ne fait mieux. Un seuil sous le plancher ne filtre "
              f"plus,")
        print(f"   il tire au sort. Valeur inscrite dans le fichier de sortie.")
    seuil_txt = (f"{a.epi_auto} x le plancher de chaque champ" if a.epi_auto
                 else f"{a.epi_max} px (fixe)")
    print(f"\nseuils appliques : residu < {seuil_txt} | texture > {a.texture_min}"
          f" | interpolation du champ {'OUI' if a.roma_fill else 'NON'}")
    print(f"   Ces seuils sont INDEPENDANTS de ceux de run_session.py. Pour que le")
    print(f"   nuage fusionne soit comparable a un nuage mono-modalite, il faut les")
    print(f"   aligner : --epi-max, --texture-min et --roma-fill ici correspondent a")
    print(f"   --epi-high, --texture-min et --roma-fill la-bas.")
    print(f"   Le masque facial differe aussi : face_mask.py exclut la charte "
          f"ColorChecker,")
    print(f"   visia_core.face_mask non. La couverture des deux chaines ne se "
          f"compare donc")
    print(f"   qu'a seuils ET masque identiques.\n")

    Zs, noms = [], []
    Zref = None
    for mod, pfx in MODALITES:
        try:
            Zi = chain(pfx)
        except (FileNotFoundError, KeyError):
            print(f"  {mod:20s} : donnees absentes, ignoree")
            continue
        if Zref is None:
            Zref = Zi
        else:
            # recalage d'echelle sur la zone commune avec la chaine de reference
            both = np.isfinite(Zref) & np.isfinite(Zi)
            if both.sum() > 1000:
                u, v = Zi[both], Zref[both]
                Zi = (Zi - u.mean()) * (v.std() / max(u.std(), 1e-9)) + v.mean()
        cov = 100 * np.isfinite(Zi).sum() / face.sum()
        print(f"  {mod:20s} : couverture {cov:5.1f} % du masque")
        Zs.append(Zi)
        noms.append(mod)

    Z = np.array(Zs)
    nch = np.sum(np.isfinite(Z), axis=0)
    Zf = np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=0)
    couv = [100*np.isfinite(z).sum()/face.sum() for z in Zs]
    union = 100*np.isfinite(Zf).sum()/face.sum()
    print(f"\nUNION : {union:5.1f} % du masque")
    if couv and union < max(couv) - 0.01:
        print(f"   ANOMALIE : l'union ({union:.1f} %) est INFERIEURE a la meilleure")
        print(f"   modalite seule ({max(couv):.1f} %). L'union ne peut pas perdre de")
        print(f"   pixels : verifier le recalage d'echelle entre chaines.")
    print("redondance :")
    for k in range(len(Zs) + 1):
        print(f"  {k} modalite(s) : {100*((nch==k)&face).sum()/face.sum():5.1f} %")
    # --- la fusion achete-t-elle de la PRECISION, ou seulement de la couverture ?
    # On mesure la rugosite locale (ecart a une mediane 9x9) sur le MEME jeu de
    # pixels, pour une modalite seule puis pour la moyenne. Si les erreurs
    # etaient independantes, la moyenne de k mesures serait 1/sqrt(k) fois
    # moins rugueuse. Un rapport proche de 1 signifie des erreurs correlees :
    # la fusion n'apporte alors que de la couverture.
    # --- la fusion achete-t-elle de la PRECISION, ou seulement de la couverture ?
    # On compare la rugosite locale d'une modalite seule a celle de la moyenne.
    # Si les erreurs etaient independantes, la moyenne de k mesures serait
    # 1/sqrt(k) fois moins rugueuse. C'est le MEILLEUR cas : des erreurs
    # correlees font moins bien, jamais mieux.
    #
    # La moyenne locale est calculee en ignorant les trous (normalisation par
    # le nombre de voisins valides) et non en les bouchant : une modalite seule
    # ayant bien plus de trous que la moyenne, le bouchage creait une fausse
    # rugosite et faisait tomber le rapport SOUS 1/sqrt(k), ce qui est
    # impossible. Toutes les cartes sont ensuite evaluees sur le MEME jeu de
    # pixels.
    WIN = 9
    MIN_VOISINS = 0.60

    def local_et_zone(M):
        v = np.isfinite(M)
        num = cv2.blur(np.where(v, M, 0.0).astype(np.float64), (WIN, WIN))
        den = cv2.blur(v.astype(np.float64), (WIN, WIN))
        loc = num / np.maximum(den, 1e-12)
        return loc, v & (den > MIN_VOISINS)

    locs, zone = [], (nch >= 1) & face
    for z in list(Zs) + [Zf]:
        loc, zo = local_et_zone(z)
        locs.append(loc)
        zone &= zo
    if int(zone.sum()) > 5000 and len(Zs) >= 2:
        r = [float(np.std((z - l)[zone])) for z, l in zip(list(Zs) + [Zf], locs)]
        r_seule, r_fusion = float(np.median(r[:-1])), r[-1]
        rapport = r_fusion / max(r_seule, 1e-30)
        attendu = 1.0 / np.sqrt(len(Zs))
        print(f"\nreduction de bruit par la fusion, sur {int(zone.sum())} pixels")
        print(f"ou les {len(Zs)} modalites et la moyenne ont assez de voisins :")
        print(f"   rugosite d'une modalite seule (mediane) : {r_seule:.3e}")
        print(f"   rugosite de la moyenne                  : {r_fusion:.3e}")
        print(f"   RAPPORT OBSERVE {rapport:.3f}  "
              f"contre {attendu:.3f} attendu si les erreurs etaient independantes")
        if rapport < attendu * 0.9:
            print("   -> SOUS l'optimum theorique. Des erreurs independantes")
            print("      donnent 1/sqrt(k) au mieux : un rapport inferieur signale")
            print("      un biais de mesure, pas un resultat a publier.")
        elif rapport > 0.9:
            print("   -> erreurs CORRELEES : la fusion apporte de la couverture")
            print("      et de la redondance de controle, PAS de la precision.")
        else:
            print("   -> reduction de bruit reelle, entre l'optimum et l'absence")
            print("      de gain. A documenter : c'est un changement par rapport")
            print("      au 1,004 mesure au flot optique.")
    else:
        print("\nreduction de bruit : zone commune trop petite pour conclure")

    out = a.out or f"fusion_{a.subject}_{a.session}.npz"
    np.savez(out, Z=Zf, Zall=Z, nch=nch, face=face, modalites=noms, focal=f)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

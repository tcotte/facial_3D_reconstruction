"""
coverage_budget.py — ou passent les points ?

OBJET
    run_session.py annonce un pourcentage de "mesure" sans dire ce qui a
    elimine le reste. Ce script rejoue exactement le meme filtrage et en donne
    le budget, critere par critere, puis balaie les seuils pour chiffrer ce que
    chacun couterait ou rapporterait.

    Il distingue surtout deux familles de trous, qui n'appellent pas du tout
    les memes remedes :

      TROUS DE SEUIL       le champ a une valeur, elle est simplement rejetee.
                           -> desserrer le seuil, ou densifier le champ (tuilage)
      TROUS STRUCTURELS    le champ n'a AUCUNE valeur : zone occluse dans
                           l'oblique, ou trop lisse pour etre appariee.
                           -> aucun seuil n'y changera rien. Il faut une autre
                              modalite d'eclairage (fuse_modalities.py) ou une
                              vue intermediaire.

    Le README annonce le nez, les paupieres et les cavites orbitaires comme
    chroniquement absents : ce sont des trous structurels.

LE PLAFOND DU PAS D'EXPORT
    Un champ exporte au pas N ne peut renseigner qu'un pixel sur N^2. Au pas 2,
    le plafond est de 25 % du masque facial, quels que soient les seuils. Le
    script l'affiche explicitement pour eviter de lire 6 % comme un echec alors
    que c'est 26 % du disponible.

USAGE
    python coverage_budget.py --subject alban --session D0 --rig rig.json \
        --roma roma2_alban_D0.npz --out budget_D0
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "00_pipeline"))
import config as C                      # noqa: E402
import roma_field as rfmod              # noqa: E402
import visia_core as vc                 # noqa: E402


def pct(n, d):
    return 100.0 * n / max(d, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--rig", default="rig.json")
    ap.add_argument("--roma", required=True)
    ap.add_argument("--out", default="budget")
    ap.add_argument("--cert", type=float, default=0.05)
    ap.add_argument("--reduire", type=int, default=3)
    a = ap.parse_args()
    print(C.summary())

    rig = vc.load_rig(a.rig)
    field = rfmod.load(a.roma)
    A = vc.load(vc.img_path(a.subject, a.session, "F"))
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    h, w = gA.shape
    field.check(a.subject, a.session, (h, w))
    K = vc.K_matrix(w, h)
    fmask = vc.face_mask(gA)
    tex = cv2.blur(np.abs(cv2.Laplacian(gA, cv2.CV_32F, ksize=3)), (15, 15))
    nf = int(fmask.sum())

    # --- le masque coupe-t-il du visage ? -----------------------------------
    # face_mask borne la peau a la bande [SKIN_TOP, SKIN_BOTTOM] de l'image.
    # Un menton, un front ou des cheveux tombant hors de cette bande sont
    # exclus PAR LE MASQUE : RoMa n'y a jamais ete interroge. Confondre cela
    # avec un echec d'appariement fait chercher au mauvais endroit.
    peau = (gA > 45).astype(np.uint8)
    peau = cv2.morphologyEx(peau, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    peau = cv2.morphologyEx(peau, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    peau = peau.astype(bool)
    yb0, yb1 = int(C.SKIN_TOP * h), int(C.SKIN_BOTTOM * h)
    coupe_haut = int(peau[:yb0].sum())
    coupe_bas = int(peau[yb1:].sum())
    print(f"\nbande de peau            lignes {yb0} a {yb1} sur {h} "
          f"(SKIN_TOP={C.SKIN_TOP}, SKIN_BOTTOM={C.SKIN_BOTTOM})")
    print(f"   peau au-dessus        {coupe_haut:>10d} pixels  "
          f"{pct(coupe_haut, peau.sum()):.1f} % de la peau detectee")
    print(f"   peau en-dessous       {coupe_bas:>10d} pixels  "
          f"{pct(coupe_bas, peau.sum()):.1f} % de la peau detectee")
    if coupe_bas > 0.03 * peau.sum():
        print(f"   ATTENTION : la bande coupe le BAS du visage — menton, machoire.")
        print(f"   Ces pixels ne sont pas mal apparies, ils ne sont PAS DEMANDES.")
        print(f"   config.SKIN_BOTTOM est une constante gelee : ne pas la modifier")
        print(f"   en cours d'etude. Pour verifier, comparer a la photo :")
        print(f"   la ligne {yb1} doit passer SOUS le menton.")
    if coupe_haut > 0.20 * peau.sum():
        print(f"   NOTE : la bande coupe le haut — front, cheveux. Souvent voulu.")

    # --- le seuil de peau est-il adapte a CE sujet ? ------------------------
    # `gray > 45` separe le visage du fond par la luminance. C'est un seuil
    # FIXE : sur un phototype plus fonce, ou sous un eclairage plus faible, une
    # part du visage passe dessous et sort du masque AVANT tout appariement.
    # Une couverture qui s'effondre en changeant de sujet commence souvent ici.
    print(f"\nsensibilite du masque au seuil de luminance (actuel : 45)")
    ref = None
    for seuil in (25, 35, 45, 55, 65):
        m2 = (gA > seuil).astype(np.uint8)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        n2 = int((m2.astype(bool)[yb0:yb1]).sum())
        if seuil == 45:
            ref = n2
        marque = "  <- actuel" if seuil == 45 else ""
        print(f"   > {seuil:3d} : {n2:>10d} pixels dans la bande"
              + (f"  ({100*n2/ref-100:+.1f} %)" if ref and seuil != 45 else "")
              + marque)
    m25 = (gA > 25).astype(np.uint8)
    m25 = cv2.morphologyEx(m25, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m25 = cv2.morphologyEx(m25, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    gain = int((m25.astype(bool)[yb0:yb1]).sum()) - ref
    if ref and gain > 0.10 * ref:
        print(f"   ATTENTION : abaisser le seuil a 25 ajouterait {100*gain/ref:.0f} %")
        print(f"   de surface. Le masque exclut donc une part importante du")
        print(f"   visage par simple luminance — phototype fonce, ou sous-exposition.")
        print(f"   Re-exporter avec roma_v2_export.py --seuil-peau 25.")

    step = field.step
    plafond = pct(nf / (step * step), nf)
    print(f"\nmasque facial            {nf:>10d} pixels")
    print(f"pas d'export             {step}  -> plafond theorique "
          f"{plafond:.1f} % du masque")
    print(f"seuils                   EPI_HIGH={C.EPI_HIGH} px | "
          f"TEXTURE_MIN={C.TEXTURE_MIN}")

    # la GRILLE reellement exportee : hors d'elle, rien n'est "manquant",
    # c'est simplement non echantillonne. Confondre les deux ferait passer le
    # pas d'export pour un probleme d'appariement.
    grille = np.zeros(h * w, bool)
    grille[field.idx] = True
    grille = grille.reshape(h, w) & fmask
    hors_grille = int((fmask & ~grille).sum())
    print(f"grille exportee          {int(grille.sum()):>10d} pixels  "
          f"{pct(grille.sum(), nf):.1f} % du masque")
    print(f"hors grille (pas {step})      {hors_grille:>10d} pixels  "
          f"{pct(hors_grille, nf):.1f} % — non echantillonne, pas manquant")

    # trace de la bande sur la photo, pour verifier a l'oeil
    bande = (A * 0.55).astype(np.uint8)
    bande[peau] = (0.6 * bande[peau] + 0.4 * np.array([90, 220, 90])).astype(np.uint8)
    cv2.line(bande, (0, yb0), (w - 1, yb0), (60, 60, 255), 9)
    cv2.line(bande, (0, yb1), (w - 1, yb1), (60, 60, 255), 9)
    cv2.imwrite(f"{a.out}_bande.png",
                cv2.resize(bande, (w // a.reduire, h // a.reduire),
                           interpolation=cv2.INTER_AREA))
    print(f"\n   {a.out}_bande.png   <- les deux lignes rouges bornent le masque")

    couv_union = np.zeros((h, w), bool)
    dispo_union = np.zeros((h, w), bool)
    par_paire = {}
    raisons = np.zeros((h, w), np.uint8)   # 0 hors masque, 1 sans champ, 2 epi, 3 texture, 4 retenu

    for k in field.pairs:
        if k not in rig:
            continue
        obx, oby, epi, wm = field.densify_like(
            k, K, *rig[k], epi_from="rig", cert_min=a.cert)
        dispo = grille & wm
        ok_epi = dispo & (epi < C.EPI_HIGH)
        ok_tex = dispo & (tex > C.TEXTURE_MIN)
        retenu = ok_epi & ok_tex

        print(f"\n=== paire F<->{k} " + "=" * 52)
        print(f"{'etape':38s} {'pixels':>10s} {'% masque':>10s} {'% dispo':>9s}")
        print(f"{'masque facial':38s} {nf:10d} {100.0:9.1f}% {'':9s}")
        print(f"{'grille exportee (pas '+str(step)+')':38s} "
              f"{int(grille.sum()):10d} {pct(grille.sum(), nf):9.1f}% {'':9s}")
        print(f"{'  champ renseigne':38s} "
              f"{int(dispo.sum()):10d} {pct(dispo.sum(), nf):9.1f}% "
              f"{pct(dispo.sum(), grille.sum()):8.1f}%")
        print(f"{'  et residu < EPI_HIGH':38s} {int(ok_epi.sum()):10d} "
              f"{pct(ok_epi.sum(), nf):9.1f}% {pct(ok_epi.sum(), dispo.sum()):8.1f}%")
        print(f"{'  et texture > TEXTURE_MIN':38s} {int(retenu.sum()):10d} "
              f"{pct(retenu.sum(), nf):9.1f}% {pct(retenu.sum(), dispo.sum()):8.1f}%")

        perdu_champ = int((grille & ~wm).sum())
        perdu_epi = int((dispo & ~ok_epi).sum())
        perdu_tex = int((ok_epi & ~ok_tex).sum())
        print(f"\n   pertes, par cause :")
        print(f"      pas apparie (STRUCTUREL)  {perdu_champ:10d}  "
              f"{pct(perdu_champ, grille.sum()):5.1f} % de la grille")
        print(f"      residu epipolaire         {perdu_epi:10d}  "
              f"{pct(perdu_epi, grille.sum()):5.1f} % de la grille")
        print(f"      texture insuffisante      {perdu_tex:10d}  "
              f"{pct(perdu_tex, grille.sum()):5.1f} % de la grille")

        e = epi[dispo]
        e = e[np.isfinite(e)]
        if len(e):
            print(f"\n   ce que couterait ou rapporterait le seuil epipolaire :")
            for s in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
                n2 = int((dispo & (epi < s) & (tex > C.TEXTURE_MIN)).sum())
                print(f"      EPI_HIGH = {s:4.1f} px -> {n2:9d} pixels  "
                      f"{pct(n2, grille.sum()):5.1f} % de la grille"
                      + ("   <== actuel" if abs(s - C.EPI_HIGH) < 1e-9 else ""))

        t = tex[dispo & (epi < C.EPI_HIGH)]
        if len(t):
            print(f"\n   et le seuil de texture :")
            for s in (0.0, 2.0, 5.0, 8.0):
                n2 = int((dispo & (epi < C.EPI_HIGH) & (tex > s)).sum())
                print(f"      TEXTURE_MIN = {s:4.1f} -> {n2:9d} pixels  "
                      f"{pct(n2, grille.sum()):5.1f} % de la grille"
                      + ("   <== actuel" if abs(s - C.TEXTURE_MIN) < 1e-9 else ""))

        couv_union |= retenu
        dispo_union |= dispo
        par_paire[k] = dispo
        raisons[grille & ~wm & (raisons == 0)] = 1
        raisons[dispo & ~ok_epi & (raisons <= 1)] = 2
        raisons[ok_epi & ~ok_tex & (raisons <= 2)] = 3
        raisons[retenu] = 4

    print(f"\n=== les deux paires reunies " + "=" * 44)
    print(f"champ renseigne  {int(dispo_union.sum()):10d}  "
          f"{pct(dispo_union.sum(), grille.sum()):5.1f} % de la grille")
    print(f"points retenus   {int(couv_union.sum()):10d}  "
          f"{pct(couv_union.sum(), grille.sum()):5.1f} % de la grille  "
          f"({pct(couv_union.sum(), dispo_union.sum()):.0f} % du disponible)")

    # --- qui couvre quoi : chaque joue ne repose que sur UNE paire ----------
    if len(par_paire) == 2:
        L, R = par_paire.get("L"), par_paire.get("R")
        seulL = L & ~R & grille
        seulR = R & ~L & grille
        deux = L & R & grille
        aucune = ~L & ~R & grille
        print(f"\n--- repartition entre les deux paires stereo ---")
        for nom, m2 in (("L seule", seulL), ("R seule", seulR),
                        ("les deux", deux), ("aucune", aucune)):
            print(f"   {nom:10s} {int(m2.sum()):10d}  {pct(m2.sum(), grille.sum()):5.1f} %"
                  f" de la grille")
        print("   Une zone occluse dans une oblique doit etre couverte par")
        print("   L'AUTRE paire. Si elle apparait en 'aucune', ce n'est plus")
        print("   une occlusion simple : les deux vues la perdent.")
        vis2 = (A * 0.4).astype(np.uint8)
        for m2, c in ((seulL, (90, 200, 90)), (seulR, (200, 150, 70)),
                      (deux, (240, 240, 240)), (aucune, (60, 60, 220))):
            vis2[m2] = (0.3 * vis2[m2] + 0.7 * np.array(c)).astype(np.uint8)
        pth = f"{a.out}_paires.png"
        cv2.imwrite(pth, cv2.resize(vis2, (w // a.reduire, h // a.reduire),
                                    interpolation=cv2.INTER_AREA))
        print(f"\n   {pth}")
        print("   vert = L seule | orange = R seule | blanc = les deux | "
              "bleu = AUCUNE")

    struct = int((grille & ~dispo_union).sum())
    print(f"\n--- ou agir, par ordre de rendement ---")
    if hors_grille:
        print(f"1. PAS D'EXPORT : {pct(hors_grille, nf):.0f} % du masque n'est meme pas")
        print(f"   echantillonne. --step 1 multiplie les points par {step*step}, "
              f"sans rien changer d'autre.")
    perdu_seuil = int((dispo_union & ~couv_union).sum())
    print(f"2. SEUIL / RESOLUTION : {perdu_seuil} pixels apparies puis rejetes, "
          f"{pct(perdu_seuil, grille.sum()):.0f} % de la grille.")
    print(f"   Le tableau ci-dessus chiffre ce que chaque seuil rendrait. Mais si")
    print(f"   le residu depasse le seuil parce que le champ est grossier, la")
    print(f"   vraie reponse est le TUILAGE (roma_v2_export.py --mode tiled),")
    print(f"   qui gagne un facteur ~4 sur la resolution du champ.")
    print(f"3. TROUS STRUCTURELS : {struct} pixels, "
          f"{pct(struct, grille.sum()):.0f} % de la grille, sur lesquels aucune")
    print(f"   paire n'apparie. Aucun seuil n'y changera rien :")
    print(f"   - fusionner les quatre modalites (00_pipeline/fuse_modalities.py)")
    print(f"   - vues intermediaires a +-22 deg (protocole, cf. README)")

    couleurs = np.array([[0, 0, 0], [40, 40, 200], [0, 165, 255],
                         [0, 220, 220], [90, 220, 90]], np.uint8)
    vis = (A * 0.4).astype(np.uint8)
    m = raisons > 0
    vis[m] = (0.35 * vis[m] + 0.65 * couleurs[raisons[m]]).astype(np.uint8)
    p = f"{a.out}_causes.png"
    cv2.imwrite(p, cv2.resize(vis, (w // a.reduire, h // a.reduire),
                              interpolation=cv2.INTER_AREA))
    print(f"\ncarte des causes -> {p}")
    print("   vert = retenu | cyan = rejete par la texture | orange = par le")
    print("   residu epipolaire | rouge = aucun appariement (structurel)")


if __name__ == "__main__":
    main()

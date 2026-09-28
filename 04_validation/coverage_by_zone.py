"""
coverage_by_zone.py — quelle modalite couvre quelle region du visage ?

OBJET
    "Ca ne marche pas sur le nez et sur l'exterieur des joues" n'est pas un
    seul probleme mais deux, de causes differentes, et chacun a sa modalite.
    Ce script croise la couverture par REGION et par MODALITE a partir du
    fichier de fusion, qui conserve `Zall` : une carte de profondeur par
    modalite.

LES DEUX ECHECS TYPIQUES, ET POURQUOI ILS DIFFERENT
    Aretes et pointe du NEZ — reflet speculaire.
        C'est la zone la plus brillante du visage. Un reflet speculaire est
        DEPENDANT DU POINT DE VUE : il ne se trouve pas au meme endroit de la
        peau dans la vue frontale et dans l'oblique. L'appariement dense le
        suit et produit une correspondance fausse, ou echoue. Le
        cross-polarise supprime la composante speculaire : c'est la modalite
        faite pour cette zone.
        S'y ajoute un fort raccourcissement : l'arete est tres inclinee par
        rapport a l'oblique, donc comprimee dans l'image.

    EXTERIEUR des JOUES — absence de texture, et angle rasant.
        Peau lisse, faiblement texturee : le filtre TEXTURE_MIN y coupe
        beaucoup, et un champ dense n'a rien a quoi s'accrocher. L'eclairage
        rasant (raked) est concu pour creer cette texture.
        S'y ajoute que chaque joue n'est reconstruite que par UNE SEULE paire
        stereo (les deux obliques ne se recouvrent que sur 1,9 % du visage) :
        le bord externe est a la limite d'incidence de cette paire unique.

    Les deux points figurent en tete des ameliorations de protocole du README :
    vues intermediaires a +/-22 deg pour le recouvrement, projection d'un
    motif de speckle pour la texture. Ce ne sont pas des reglages logiciels.

DECOUPAGE
    Les regions sont definies GEOMETRIQUEMENT a partir de la boite englobante
    du masque facial, sans detecteur de points caracteristiques. C'est
    approximatif et assume : la carte _zones.png permet de verifier que le
    decoupage tombe juste sur VOTRE sujet avant d'interpreter les chiffres.

USAGE
    python coverage_by_zone.py --fusion fusion_alban_D0.npz \
        --frontal /donnees/visia/alban_D0_Frontal_Standard_1.jpg --out zones_D0
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

# (nom, y0, y1, |x - cx| min, |x - cx| max) en fractions de la boite du visage.
# L'affectation est EXCLUSIVE et suit cet ordre : un pixel va a la premiere
# region qui le contient. Les regions etroites et specifiques passent donc en
# premier, sinon la colonne du nez serait comptee deux fois, avec les orbites.
ZONES = [
    ("nez (centre)",     0.33, 0.62, 0.00, 0.13),
    ("front",            0.00, 0.22, 0.00, 1.00),
    ("orbites",          0.22, 0.40, 0.00, 1.00),
    ("joues internes",   0.40, 0.72, 0.13, 0.45),
    ("joues externes",   0.40, 0.72, 0.45, 1.00),
    ("bouche / menton",  0.72, 1.00, 0.00, 0.45),
    ("machoires",        0.72, 1.00, 0.45, 1.00),
]


def zone_maps(face):
    ys, xs = np.nonzero(face)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    hh, ww = max(y1 - y0, 1), max(x1 - x0, 1)
    cx = (x0 + x1) / 2
    YY, XX = np.mgrid[0:face.shape[0], 0:face.shape[1]]
    fy = (YY - y0) / hh
    fx = np.abs(XX - cx) / (ww / 2)
    out = {}
    pris = np.zeros_like(face)
    for nom, a0, a1, r0, r1 in ZONES:
        z = face & (fy >= a0) & (fy < a1) & (fx >= r0) & (fx < r1) & ~pris
        out[nom] = z
        pris |= z
    reste = face & ~pris
    if reste.sum():
        out["(non classe)"] = reste
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", required=True,
                    help="sortie de fuse_modalities.py (doit contenir Zall)")
    ap.add_argument("--frontal", default=None)
    ap.add_argument("--out", default="zones")
    ap.add_argument("--reduire", type=int, default=3)
    a = ap.parse_args()

    d = np.load(a.fusion, allow_pickle=True)
    if "Zall" not in d.files:
        raise SystemExit(
            f"{os.path.basename(a.fusion)} ne contient pas Zall. Donner le fichier "
            f"de fuse_modalities.py, pas celui de outlier_filter.py.")
    Zall = d["Zall"]
    face = d["face"].astype(bool)
    mods = [str(m) for m in d["modalites"]] if "modalites" in d.files \
        else [f"modalite {i}" for i in range(len(Zall))]
    Zf = d["Z"] if "Z" in d.files else np.nanmean(Zall, axis=0)
    h, w = face.shape
    print(f"{os.path.basename(a.fusion)} | carte {w}x{h} | {len(mods)} modalites")

    zones = zone_maps(face)
    larg = max(len(n) for n in zones) + 1
    ent = f"{'region':{larg}s}" + "".join(f"{m[:12]:>14s}" for m in mods) + f"{'UNION':>9s}"
    print("\ncouverture par region, en % des pixels de la region\n" + ent)
    print("-" * len(ent))
    faibles = []
    for nom, z in zones.items():
        n = int(z.sum())
        if n == 0:
            continue
        cov = [100 * np.isfinite(Zall[i][z]).sum() / n for i in range(len(mods))]
        uni = 100 * np.isfinite(Zf[z]).sum() / n
        best = int(np.argmax(cov))
        ligne = f"{nom:{larg}s}"
        for i, c in enumerate(cov):
            ligne += f"{c:13.1f}%" if i != best else f"{c:12.1f}%*"
        print(ligne + f"{uni:8.1f}%")
        if uni < 50:
            faibles.append((nom, uni, mods[best], cov[best]))
    print(f"\n* = meilleure modalite de la region")

    if faibles:
        print("\nregions sous 50 % de couverture :")
        for nom, uni, bm, bc in faibles:
            print(f"   {nom:{larg}s} union {uni:5.1f} %  | meilleure : {bm} ({bc:.1f} %)")
        print("\n   Si la meilleure modalite d'une region est nettement au-dessus")
        print("   des autres, c'est une piste : cross-polarise contre les reflets")
        print("   (nez), raked contre l'absence de texture (joues). Si TOUTES les")
        print("   modalites echouent ensemble, la cause est geometrique — occlusion")
        print("   ou angle rasant — et aucun eclairage n'y changera rien.")

    # --- cartes -------------------------------------------------------------
    r = a.reduire
    vis = np.zeros((h, w, 3), np.uint8)
    if a.frontal:
        im = cv2.imread(a.frontal)
        if im is not None and im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        if im is not None:
            vis = (im * 0.45).astype(np.uint8)
    couleurs = [(80, 80, 230), (80, 200, 230), (90, 210, 110), (230, 170, 80),
                (210, 110, 210), (230, 230, 110), (150, 150, 150)]
    for (nom, z), c in zip(zones.items(), couleurs):
        vis[z] = (0.5 * vis[z] + 0.5 * np.array(c)).astype(np.uint8)
    cv2.imwrite(f"{a.out}_zones.png",
                cv2.resize(vis, (w // r, h // r), interpolation=cv2.INTER_AREA))
    print(f"\n   {a.out}_zones.png   <- VERIFIER que le decoupage tombe juste")

    for i, m in enumerate(mods):
        c = np.zeros((h, w, 3), np.uint8)
        c[face] = (40, 40, 40)
        ok = face & np.isfinite(Zall[i])
        c[ok] = (90, 220, 90)
        nom = f"{a.out}_couverture_{m.replace(' ', '_').replace('/', '_')}.png"
        cv2.imwrite(nom, cv2.resize(c, (w // r, h // r), interpolation=cv2.INTER_AREA))
        print(f"   {nom}")


if __name__ == "__main__":
    main()

"""
run_subject.py — un nouveau sujet, de bout en bout.

OBJET
    Enchainer, dans le bon ordre et avec les bons reglages, tout ce qu'il faut
    faire d'un jeu de champs RoMa exportes jusqu'au nuage de points. Trois
    sujets ont montre que chacun echoue pour une raison DIFFERENTE, et que la
    cause se trouve toujours a une etape qu'on a sautee.

    Le script n'implemente rien : il appelle les scripts existants, affiche ce
    qu'ils disent, et S'ARRETE au premier controle qui echoue plutot que de
    produire un nuage dont on ne saura pas quoi penser.

CE QU'IL FAIT, ET POURQUOI DANS CET ORDRE
    1. calibration sur CE sujet        run_calibrate_roma.py
       Le rig n'est pas transferable d'un sujet a l'autre : mesure sur deux
       sujets, 69,07 deg contre 72,07 deg entre obliques. Utiliser le rig d'un
       autre sujet donne un residu epipolaire de l'ordre de 10 px et vide le
       nuage. C'est l'erreur la plus couteuse et la moins visible.
    2. controle du rig                 rig_check.py --roma
       Donne aussi le PLANCHER : le residu de la meilleure F possible. Un
       plancher eleve signale une scene non rigide entre les trois prises.
    3. budget de couverture            coverage_budget.py
       Separe ce qui manque par seuil de ce qui manque structurellement.
    4. fusion des modalites            fuse_modalities.py --epi-auto
       Seuil relatif au plancher : un seuil en pixels fixes n'a pas le meme
       sens d'un sujet a l'autre.
    5. rejet des aberrants             outlier_filter.py --par-zone
    6. nuage de points                 export_pointcloud.py --fusion
    7. controle visuel                 check_session.py --fusion

USAGE
    python run_subject.py --subject tristan --session D0 \
        --img-dir /donnees/visia --roma-dir . --out-dir resultats_tristan

    # en reutilisant un rig deja calcule pour CE sujet
    python run_subject.py ... --rig rig_tristan.json
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
MODALITES = ["Standard 1", "Cross-Polarized", "Raked", "Parallel-Polarized"]


def champ(roma_dir, sujet, session, modalite):
    """Cherche le champ, avec ou sans souligne dans le nom de modalite."""
    for nom in (modalite, modalite.replace("_", " "), modalite.replace(" ", "_")):
        p = os.path.join(roma_dir, f"roma2_{sujet}_{session}_{nom}.npz")
        if os.path.exists(p):
            return p
    return None


def etape(n, titre, cmd, obligatoire=True):
    print(f"\n{'=' * 74}\n{n}. {titre}\n{'=' * 74}")
    print("   " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]) + "\n")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n--- etape {n} : code de sortie {r.returncode} ---")
        if obligatoire:
            print("Arret. Corriger ce point avant de continuer : la suite de la")
            print("chaine produirait un resultat qu'on ne saurait pas interpreter.")
            sys.exit(r.returncode)
        print("Etape non bloquante, on continue.")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--roma-dir", default=".")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rig", default=None,
                    help="rig deja calcule POUR CE SUJET. Sinon il est estime.")
    ap.add_argument("--epi-auto", type=float, default=2.5,
                    help="seuil epipolaire, en multiples du plancher mesure")
    ap.add_argument("--texture-min", type=float, default=2.0)
    ap.add_argument("--reference", default="Standard_1",
                    help="modalite servant a la calibration et au diagnostic")
    ap.add_argument("--sauter-diagnostic", action="store_true",
                    help="ne pas lancer coverage_budget (plus rapide)")
    a = ap.parse_args()

    out = a.out_dir or f"resultats_{a.subject}_{a.session}"
    os.makedirs(out, exist_ok=True)
    py = sys.executable

    dispo = {m: champ(a.roma_dir, a.subject, a.session, m) for m in MODALITES}
    dispo = {m: p for m, p in dispo.items() if p}
    if not dispo:
        motif = os.path.join(a.roma_dir, f"roma2_{a.subject}_{a.session}_*.npz")
        raise SystemExit(
            f"aucun champ trouve pour {a.subject}/{a.session}.\n"
            f"  cherche : {motif}\n"
            f"  presents : {', '.join(os.path.basename(x) for x in glob.glob(os.path.join(a.roma_dir, '*.npz'))[:8]) or 'aucun .npz'}")
    print(f"sujet {a.subject} / session {a.session}")
    print(f"{len(dispo)} modalite(s) trouvee(s) :")
    for m, p in dispo.items():
        print(f"   {m:22s} {os.path.basename(p)}")

    ref = dispo.get(a.reference) or next(iter(dispo.values()))
    if a.reference not in dispo:
        print(f"\n   modalite de reference '{a.reference}' absente, "
              f"utilisation de {os.path.basename(ref)}")

    # 1. calibration
    rig = a.rig
    if rig:
        print(f"\nrig fourni : {rig}")
        print("   ATTENTION : verifier qu'il a bien ete estime sur CE sujet.")
        print("   Un rig d'un autre sujet donne un residu de l'ordre de 10 px.")
    else:
        rig = os.path.join(out, f"rig_{a.subject}_{a.session}.json")
        etape(1, "calibration du rig SUR CE SUJET",
              [py, os.path.join(ICI, "run_calibrate_roma.py"),
               "--roma", ref, "--out", rig])

    # 2. controle du rig + plancher
    etape(2, "controle du rig et mesure du plancher",
          [py, os.path.join(RACINE, "02_calibration", "rig_check.py"),
           "--rig", rig, "--roma", ref,
           "--carte", os.path.join(out, "residus")], obligatoire=False)

    # 3. budget de couverture
    if not a.sauter_diagnostic:
        etape(3, "budget de couverture (modalite de reference)",
              [py, os.path.join(RACINE, "04_validation", "coverage_budget.py"),
               "--subject", a.subject, "--session", a.session, "--rig", rig,
               "--roma", ref, "--out", os.path.join(out, "budget")],
              obligatoire=False)

    # 4. fusion
    fusion = os.path.join(out, f"fusion_{a.subject}_{a.session}.npz")
    etape(4, f"fusion des {len(dispo)} modalites",
          [py, os.path.join(ICI, "fuse_modalities.py"),
           "--subject", a.subject, "--session", a.session, "--rig", rig,
           "--img-dir", a.img_dir, "--roma-dir", a.roma_dir,
           "--epi-auto", str(a.epi_auto), "--texture-min", str(a.texture_min),
           "--roma-fill", "--out", fusion])

    # 5. rejet des aberrants
    filtre = fusion.replace(".npz", "_filtre.npz")
    etape(5, "rejet des points aberrants",
          [py, os.path.join(ICI, "outlier_filter.py"),
           "--fusion", fusion, "--out", filtre, "--par-zone"])

    # 6. nuage
    frontal = os.path.join(a.img_dir,
                           f"{a.subject}_{a.session}_Frontal_{a.reference}.jpg")
    if not os.path.exists(frontal):
        alt = glob.glob(os.path.join(a.img_dir,
                                     f"{a.subject}_{a.session}_Frontal_*.jpg"))
        if not alt:
            raise SystemExit(f"vue frontale introuvable dans {a.img_dir}")
        frontal = alt[0]
    ply = os.path.join(out, f"nuage_{a.subject}_{a.session}.ply")
    etape(6, "nuage de points",
          [py, os.path.join(RACINE, "04_validation", "export_pointcloud.py"),
           "--fusion", fusion, "--frontal", frontal, "--out", ply, "--clip", "0"])

    # 7. controle visuel
    etape(7, "controle visuel de la reconstruction",
          [py, os.path.join(RACINE, "04_validation", "check_session.py"),
           "--fusion", filtre, "--frontal", frontal,
           "--out", os.path.join(out, "check")], obligatoire=False)

    print(f"\n{'=' * 74}\nTermine. Tout est dans {out}/\n{'=' * 74}")
    print(f"   nuage        : {os.path.basename(ply)}")
    print(f"   rig          : {os.path.basename(rig)}")
    print("\nA regarder, dans cet ordre :")
    print("   check_relief.png      un visage doit y etre reconnaissable")
    print("   budget_causes.png     ou partent les points, et pourquoi")
    print("   residus_residu_*.png  un amas coherent = mouvement non rigide")
    print("   budget_bande.png      la bande coupe-t-elle le menton ou le front ?")


if __name__ == "__main__":
    main()

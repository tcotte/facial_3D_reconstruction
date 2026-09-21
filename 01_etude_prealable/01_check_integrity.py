"""
01_check_integrity.py — contrôle d'intégrité des exports VISIA-CR.

CONTEXTE
    Au cours de la collecte, 6 fichiers sur 12 sont arrivés entièrement blancs
    (moyenne 255,0 / écart-type 0,0), tous d'une taille strictement identique
    de 251 258 octets. Il ne s'agit pas d'un aléa de transfert mais d'un fichier
    de substitution produit par la chaîne d'export.

    À exécuter SYSTÉMATIQUEMENT après chaque export, avant toute analyse.
    Dans une étude à grand effectif, des acquisitions vides passeraient sinon
    inaperçues et corrompraient silencieusement le jeu de données.

USAGE
    python 01_check_integrity.py
"""

import os
import visia_lib as vl

ANGLES = ["Frontal", "Left_Oblique", "Right_Oblique"]
MODALITIES = ["Standard_1", "Cross-Polarized", "Parallel-Polarized", "Raked"]
SUBJECT = "alban_D0"


def main():
    print(f"Contrôle d'intégrité — dossier : {vl.IMG_DIR}\n")
    bad = []
    for angle in ANGLES:
        for mod in MODALITIES:
            fname = f"{SUBJECT}_{angle}_{mod}.jpg"
            ok, msg = vl.check_integrity(fname)
            flag = "  " if ok else "!!"
            print(f"{flag} {fname:52s} {msg}")
            if not ok:
                bad.append(fname)
    print()
    if bad:
        print(f"ÉCHEC : {len(bad)} fichier(s) invalide(s). Relancer l'export.")
        for f in bad:
            print(f"   - {f}")
    else:
        print("Tous les fichiers sont valides.")


if __name__ == "__main__":
    main()

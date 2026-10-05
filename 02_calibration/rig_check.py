"""
rig_check.py — le rig est-il physiquement plausible ?

OBJET
    Un rig estime par essential matrix peut converger vers une solution
    degeneree sans rien signaler : `recoverPose` renvoie toujours une pose.
    Ce script confronte le rig a ce que l'on SAIT du banc VISIA, et dit
    lequel des controles echoue.

CONTROLES
    1. Angle de rotation        doit etre voisin de ANGLE_L_DEG / ANGLE_R_DEG.
    2. NATURE de la rotation    le banc fait tourner la camera HORIZONTALEMENT :
                                la rotation doit etre un LACET, autour de l'axe
                                vertical Y. Une rotation autour de l'axe optique
                                (roulis) est mecaniquement impossible et trahit
                                un appariement trop pauvre.
    3. Direction de la base     doit etre a dominante horizontale.
    4. Conditionnement          angle entre la base et l'axe optique. En dessous
                                de ~20 deg la triangulation est fragile : les
                                deux vues regardent presque le long de la meme
                                ligne, la profondeur n'est plus contrainte.
    5. Ecart entre obliques     l'angle L->R doit valoir a peu pres la somme
                                des deux angles obliques. Les deux poses etant
                                estimees separement, leur composition exacte
                                n'est vraie que si elles tournent bien autour du
                                MEME axe vertical.
    5 bis. TEST D'ORBITE        le plus discriminant, et il ne demande aucune
                                correspondance. Le banc fait tourner la camera
                                autour du sujet : le mouvement (R, t) doit donc
                                admettre un axe fixe, vertical, passant devant
                                l'objectif. On resout (R - I) P = -t : le point
                                obtenu doit etre lateralement centre (x ~ 0) et
                                a une distance z > 0 qui est la DISTANCE DE
                                TRAVAIL, exprimee en unites de |t|. Les deux
                                branches doivent s'accorder sur cette distance.
                                Deux poses estimees independamment qui
                                convergent vers le meme centre d'orbite ne
                                peuvent pas etre fausses ensemble par hasard —
                                et cela valide l'echelle relative L/R SANS
                                passer par les pistes 3 vues.
    6. (option --roma) residu epipolaire du rig sur un champ RoMa exporte.
       C'est le controle decisif : si ce residu est grand alors que
       l'appariement est bon, c'est le rig qui est faux, et run_session.py
       rejettera presque tous les points au seuil EPI_HIGH.

USAGE
    python rig_check.py --rig rig.json
    python rig_check.py --rig rig.json --roma roma2_alban_D0_Standard_1.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "00_pipeline"))
import config as C                      # noqa: E402

OK, WARN, BAD = "  ok  ", " tiede", " FAUX "


def axis_angle(R):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, np.argmin(np.abs(w - 1))])
    ax = ax / np.linalg.norm(ax)
    if ax[1] < 0:
        ax = -ax
    return ang, ax


def verdict(cond_ok, cond_warn=False):
    """cond_warn est la condition RELACHEE : elle n'est consultee que si la
    condition stricte echoue."""
    if cond_ok:
        return OK
    if cond_warn:
        return WARN
    return BAD


def check_pose(k, R, t, attendu):
    print(f"\n=== pose {k} " + "=" * 58)
    ang, ax = axis_angle(R)
    lacet, tangage, roulis = abs(ax[1]), abs(ax[0]), abs(ax[2])
    Cc = -R.T @ np.asarray(t, float).ravel()
    Cn = Cc / (np.linalg.norm(Cc) + 1e-12)
    base_h = abs(Cn[0])
    par = np.degrees(np.arccos(np.clip(abs(Cn @ np.array([0, 0, 1.0])), -1, 1)))

    d_ang = abs(ang - attendu)
    v1 = verdict(d_ang < 8, d_ang < 15)
    print(f"[{v1}] angle de rotation      {ang:6.2f}°   attendu ~{attendu}°  "
          f"(ecart {ang-attendu:+.1f}°)")

    v2 = verdict(lacet > 0.80, lacet > 0.60)
    print(f"[{v2}] nature de la rotation  lacet {100*lacet:3.0f} % | "
          f"tangage {100*tangage:3.0f} % | roulis {100*roulis:3.0f} %")
    if lacet <= 0.60:
        print("         -> le banc tourne la camera HORIZONTALEMENT. Une rotation")
        print("            dominee par le roulis est mecaniquement impossible :")
        print("            l'estimation de la matrice essentielle a echoue.")

    v3 = verdict(base_h > 0.60, base_h > 0.35)
    print(f"[{v3}] direction de la base   horizontale {100*base_h:3.0f} % | "
          f"avant-arriere {100*abs(Cn[2]):3.0f} %")

    v4 = verdict(par > 25, par > 15)
    print(f"[{v4}] conditionnement        base a {par:5.1f}° de l'axe optique")
    if par <= 15:
        print("         -> les deux vues regardent presque le long de la meme")
        print("            ligne : la profondeur n'est pas contrainte.")

    # coherence interne : tourner vers +X impose de se deplacer vers -X
    coh = np.sign(R[0, 2]) == -np.sign(np.asarray(t, float).ravel()[0])
    v5 = verdict(bool(coh))
    print(f"[{v5}] sens rotation / base  "
          + ("coherents" if coh else "INCOHERENTS : la camera tourne du mauvais cote"))

    # test d'orbite : point fixe du mouvement, (R - I) P = -t
    Pf, *_ = np.linalg.lstsq(R - np.eye(3), -np.asarray(t, float).ravel(), rcond=None)
    tn = np.linalg.norm(np.asarray(t, float))
    lat = abs(Pf[0]) / max(abs(Pf[2]), 1e-9)
    v6 = verdict(Pf[2] > 0 and lat < 0.08, Pf[2] > 0 and lat < 0.20)
    print(f"[{v6}] axe d'orbite           passe par x={Pf[0]:+.4f}, z={Pf[2]:+.4f}  "
          f"(decentrage {100*lat:.1f} %)")
    if Pf[2] > 0:
        print(f"         distance de travail = {Pf[2]/tn:.4f} x la base F<->{k} "
              f"(soit {Pf[2]:.4f} dans l'unite commune du rig)")
    else:
        print("         -> axe d'orbite DERRIERE l'objectif : mouvement non physique")
    return ang, ax, Cn, [v1, v2, v3, v4, v5, v6], Pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="rig.json")
    ap.add_argument("--roma", default=None,
                    help="champ RoMa exporte, pour mesurer le residu epipolaire du rig")
    ap.add_argument("--compare", default=None,
                    help="second rig.json : compare les deux poses. Sert a "
                         "verifier si le rig est bien une propriete du BANC, "
                         "comme le suppose le protocole, ou s'il depend du "
                         "sujet et de son positionnement.")
    ap.add_argument("--carte", default=None,
                    help="prefixe de sortie pour la carte du residu epipolaire "
                         "(avec --roma). Montre OU se concentre l'erreur : un "
                         "amas anatomique coherent trahit un mouvement non "
                         "rigide entre les trois prises, un semis disperse du "
                         "bruit d'appariement.")
    ap.add_argument("--cert", type=float, default=0.5,
                    help="seuil d'overlap pour le controle epipolaire")
    a = ap.parse_args()

    with open(a.rig, encoding="utf-8") as f:
        rig = json.load(f)
    print(f"rig        : {os.path.basename(a.rig)}")
    print(f"stamp      : {rig.get('calib_version')}"
          + ("" if rig.get("calib_version") == C.CALIB_VERSION
             else f"   ATTENTION : config = {C.CALIB_VERSION}"))
    print(f"note       : {rig.get('note', '')}")
    print(f"focale     : {rig.get('focal_rel')} x cote long | k1 = {rig.get('k1')}")

    attendu = {"L": C.ANGLE_L_DEG, "R": C.ANGLE_R_DEG}
    poses, flags, orbite = {}, [], {}
    for k in ("L", "R"):
        if k not in rig["poses"]:
            print(f"\npose {k} absente du rig")
            continue
        R = np.array(rig["poses"][k]["R"], float)
        t = np.array(rig["poses"][k]["t"], float)
        poses[k] = (R, t)
        ang, ax, Cn, v, Pf = check_pose(k, R, t, attendu[k])
        flags += v
        orbite[k] = Pf

    if len(poses) == 2:
        dR = poses["R"][0] @ poses["L"][0].T
        a_lr = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        att = C.ANGLE_L_DEG + C.ANGLE_R_DEG
        v = verdict(abs(a_lr - att) < 12, abs(a_lr - att) < 20)
        print(f"\n[{v}] angle entre les deux obliques  {a_lr:6.2f}°   attendu ~{att:.0f}°")
        flags.append(v)
        a_sum = (np.degrees(np.arccos(np.clip((np.trace(poses["L"][0])-1)/2, -1, 1)))
                 + np.degrees(np.arccos(np.clip((np.trace(poses["R"][0])-1)/2, -1, 1))))
        v = verdict(abs(a_lr - a_sum) < 2.0, abs(a_lr - a_sum) < 5.0)
        print(f"[{v}] composition des deux poses     {a_lr:6.2f}° contre "
              f"{a_sum:.2f}° (somme)   ecart {a_lr-a_sum:+.2f}°")
        print("         les deux poses tournent-elles autour du meme axe vertical ?")
        flags.append(v)

    if len(orbite) == 2:
        dL, dR = orbite["L"][2], orbite["R"][2]
        if dL > 0 and dR > 0:
            ec = abs(dL - dR) / ((dL + dR) / 2)
            v = verdict(ec < 0.03, ec < 0.08)
            print(f"\n[{v}] accord des deux orbites        L {dL:.4f} | R {dR:.4f}   "
                  f"ecart {100*ec:.2f} %")
            print("         valide l'echelle relative L/R SANS passer par les pistes 3 vues")
            flags.append(v)
            if v is OK:
                print(f"\n  Distance de travail = {(dL+dR)/2:.3f} x la base F<->L.")
                print("  Mesurer physiquement l'UNE de ces deux longueurs donnerait le")
                print("  facteur d'echelle global, aujourd'hui inconnu (cf. README).")
            else:
                print("  -> les deux branches ne decrivent pas la meme orbite : au moins")
                print("     une pose est fausse, ou l'echelle relative L/R l'est.")

    # --- comparaison a un second rig ----------------------------------------
    if a.compare:
        with open(a.compare, encoding="utf-8") as f:
            rig2 = json.load(f)
        print(f"\n=== comparaison avec {os.path.basename(a.compare)} " + "=" * 20)
        print(f"   note : {rig2.get('note', '')}")
        ecarts = []
        for k in ("L", "R"):
            if k not in rig["poses"] or k not in rig2["poses"]:
                continue
            R1 = np.array(rig["poses"][k]["R"], float)
            R2 = np.array(rig2["poses"][k]["R"], float)
            t1 = np.array(rig["poses"][k]["t"], float)
            t2 = np.array(rig2["poses"][k]["t"], float)
            dR = np.degrees(np.arccos(np.clip((np.trace(R2 @ R1.T) - 1) / 2, -1, 1)))
            ct = np.degrees(np.arccos(np.clip(
                abs(t1 @ t2) / (np.linalg.norm(t1) * np.linalg.norm(t2)), -1, 1)))
            a1 = np.degrees(np.arccos(np.clip((np.trace(R1) - 1) / 2, -1, 1)))
            a2 = np.degrees(np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)))
            v = verdict(dR < 1.0, dR < 3.0)
            print(f"   [{v}] pose {k} : {a1:.2f}° contre {a2:.2f}°  | "
                  f"ecart de rotation {dR:.2f}° | de base {ct:.2f}°")
            ecarts.append(dR)
        if ecarts and max(ecarts) > 3.0:
            print("\n   Les deux rigs different de plus de 3°. Si chacun a ete estime")
            print("   sur un SUJET different, cela contredit l'hypothese fondatrice du")
            print("   protocole : \"les poses de camera sont une propriete du banc, pas")
            print("   du sujet\". Un rig gele sur un sujet ne vaudrait alors que pour lui,")
            print("   et les quatre conditions de validite du README sont a revoir.")
            print("   Verifier si le VISIA fait tourner la CAMERA ou le SUJET : dans le")
            print("   second cas, la pose relative depend du positionnement de la tete.")
        flags += [verdict(max(ecarts) < 1.0, max(ecarts) < 3.0)] if ecarts else []

    # --- controle epipolaire sur un champ RoMa ------------------------------
    if a.roma:
        import roma_field as rf
        field = rf.load(a.roma)
        print(f"\n=== residu epipolaire du rig sur {os.path.basename(a.roma)} "
              + "=" * 12)
        H, W = field.H, field.W
        K = np.array([[C.focal_px(max(W, H)), 0, W / 2],
                      [0, C.focal_px(max(W, H)), H / 2], [0, 0, 1]], float)
        for k in field.pairs:
            if k not in poses:
                continue
            xy = field._d[f"{k}_xy"].astype(np.float64)
            cert = field._d[f"{k}_cert"].astype(np.float64)
            ys, xs = field.idx // W, field.idx % W
            m = np.isfinite(xy[:, 0]) & (cert > a.cert)
            if m.sum() < 100:
                print(f"  paire {k} : trop peu de points au-dessus de cert={a.cert}")
                continue
            F_rig = rf.fundamental_from_rig(K, *poses[k])
            e = rf.epipolar_residual(F_rig, xs[m].astype(float), ys[m].astype(float),
                                     xy[m, 0], xy[m, 1])
            sous = 100.0 * (e < C.EPI_HIGH).mean()
            v = verdict(sous > 40, sous > 10)
            print(f"  [{v}] paire {k} : mediane {np.median(e):8.2f} px | "
                  f"90e centile {np.percentile(e,90):8.2f} px | "
                  f"{sous:5.1f} % sous EPI_HIGH={C.EPI_HIGH}")
            flags.append(v)
            # F re-estimee sur le champ : borne de ce que l'appariement permet
            import cv2
            sel = np.nonzero(m)[0]
            if len(sel) > 20000:
                sel = sel[np.linspace(0, len(sel) - 1, 20000).astype(int)]
            Fm, _ = cv2.findFundamentalMat(
                np.column_stack([xs[sel], ys[sel]]).astype(np.float64),
                xy[sel], cv2.USAC_MAGSAC, 1.0, 0.9999, 100000)
            if Fm is not None:
                Fm = Fm[:3] / (np.linalg.norm(Fm[:3]) + 1e-30)
                e2 = rf.epipolar_residual(Fm, xs[m].astype(float), ys[m].astype(float),
                                          xy[m, 0], xy[m, 1])
                print(f"  {'':6s}    a titre de reference, F re-estimee sur le champ : "
                      f"mediane {np.median(e2):.2f} px | "
                      f"{100.0*(e2 < C.EPI_HIGH).mean():.1f} % sous EPI_HIGH")
                if np.median(e) > 5 * max(np.median(e2), 0.05):
                    print(f"  {'':6s}    -> l'appariement est bon, C'EST LE RIG QUI EST FAUX.")
                elif np.median(e2) > 0.6:
                    print(f"  {'':6s}    -> le PLANCHER lui-meme est eleve : meme la meilleure")
                    print(f"  {'':6s}       F possible laisse {np.median(e2):.2f} px. Le champ n'est")
                    print(f"  {'':6s}       pas coherent avec UNE geometrie epipolaire unique,")
                    print(f"  {'':6s}       ce qui arrive quand la scene n'est pas rigide entre")
                    print(f"  {'':6s}       les trois prises (expression, tete qui bouge).")
                    print(f"  {'':6s}       Aucun rig ne rattrape cela. Voir --carte.")

                if a.carte:
                    import cv2
                    res = np.full(H * W, np.nan, np.float32)
                    res[field.idx[m]] = e2.astype(np.float32)
                    res = res.reshape(H, W)
                    ok2 = np.isfinite(res)
                    v = np.clip(res / 5.0, 0, 1)
                    img = cv2.applyColorMap((255 * np.nan_to_num(v)).astype(np.uint8),
                                            cv2.COLORMAP_TURBO)
                    img[~ok2] = (25, 25, 25)
                    pth = f"{a.carte}_residu_{k}.png"
                    cv2.imwrite(pth, cv2.resize(img, (W // 3, H // 3),
                                                interpolation=cv2.INTER_AREA))
                    print(f"  {'':6s}    carte -> {pth}  (bleu 0 px, rouge >= 5 px)")
                    print(f"  {'':6s}    Un amas COHERENT (bouche, machoire, oeil) = mouvement")
                    print(f"  {'':6s}    non rigide. Un semis disperse = bruit d'appariement.")

    nb_bad = flags.count(BAD)
    nb_warn = flags.count(WARN)
    print("\n" + "=" * 70)
    if nb_bad:
        print(f"{nb_bad} controle(s) en echec, {nb_warn} limite(s).")
        print("Ce rig ne decrit pas le banc. Le reconstruire :")
        print("   python 00_pipeline/run_calibrate_roma.py --roma <champ.npz> --out rig.json")
    elif nb_warn:
        print(f"Aucun echec, mais {nb_warn} controle(s) limite(s) : a surveiller.")
    else:
        print("Tous les controles passent.")
    sys.exit(2 if nb_bad else 0)


if __name__ == "__main__":
    main()

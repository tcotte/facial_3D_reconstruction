"""
run_calibrate_roma.py — estime le rig a partir du champ RoMa v2 exporte.

POURQUOI CE SCRIPT EXISTE
    run_calibrate.py estime les poses sur une passe grossiere DISK+LightGlue
    ou chaque tuile frontale pleine resolution est appariee a l'oblique ENTIERE,
    ramenee a 640 px — soit un facteur d'echelle de l'ordre de 6 en plus des
    32 deg de rotation. Quand cette passe donne peu d'appariements,
    findEssentialMat converge vers une solution degeneree SANS RIEN SIGNALER :
    recoverPose renvoie toujours une pose. Le symptome visible est le message
    "trop peu de points communs aux 3 vues", mais le mal est plus profond : les
    DEUX poses sont alors fausses, pas seulement l'echelle relative.

    Le champ RoMa, lui, donne pour CHAQUE pixel frontal son correspondant dans
    L et dans R. Le chainage 3 vues, qui est precisement ce que run_calibrate
    peine a obtenir, est ici exact par construction.

CE QUE CE SCRIPT AJOUTE PAR RAPPORT A run_calibrate.py
    - selection par SIGMA : RoMa v2 predit l'ecart-type de localisation de
      chaque correspondance. On calibre sur les points les mieux localises, pas
      sur un echantillon quelconque ;
    - repartition spatiale imposee (un point par cellule) : sans cela
      l'estimation se concentre sur les zones texturees et la pose penche ;
    - MAGSAC plutot que RANSAC ;
    - controles de plausibilite du resultat, AVANT d'ecrire le rig.

RESERVE
    L'etude a conclu qu'un champ dense localise moins bien qu'un detecteur de
    points pour la calibration. Cette conclusion reposait en partie sur une
    lecture du champ au plus proche voisin, corrigee depuis (cf.
    03_appariement/README_ROMA_V2.md, section 6). Elle reste a reexaminer.
    En attendant : un rig legerement moins precis vaut mieux qu'un rig faux, et
    les controles ci-dessous disent lequel des deux on a.

GEL
    Le rig produit remplace le precedent. Toutes les sessions deja
    reconstruites doivent etre retraitees : les poses de camera conditionnent
    la totalite des valeurs.

USAGE
    python run_calibrate_roma.py --roma roma2_alban_D0_Standard_1.npz --out rig.json
    python ../02_calibration/rig_check.py --rig rig.json --roma roma2_alban_D0_Standard_1.npz
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

import config as C
import roma_field as rf
import visia_core as vc


def spread_select(xs, ys, score, W, H, cell, cap):
    """Un point par cellule — le mieux note — puis plafonnement.

    Sans repartition imposee, l'estimation se concentre sur les zones les plus
    texturees (barbe, sourcils) et la pose penche vers ces regions.
    """
    key = (ys // cell).astype(np.int64) * (W // cell + 1) + (xs // cell).astype(np.int64)
    order = np.lexsort((-score, key))
    key_s = key[order]
    first = np.ones(len(order), bool)
    first[1:] = key_s[1:] != key_s[:-1]
    sel = order[first]
    if len(sel) > cap:
        sel = sel[np.linspace(0, len(sel) - 1, cap).astype(int)]
    return sel


def _pose_vers_params(R, t):
    rvec = cv2.Rodrigues(np.asarray(R, float))[0].ravel()
    t = np.asarray(t, float).ravel()
    t = t / (np.linalg.norm(t) + 1e-30)
    # direction de translation en coordonnees spheriques : 2 parametres, pas 3.
    # La geometrie epipolaire ne depend pas de |t|, et parametrer t par un
    # vecteur libre laisserait un degre de liberte non contraint.
    return np.concatenate([rvec, [np.arccos(np.clip(t[2], -1, 1)),
                                  np.arctan2(t[1], t[0])]])


def _params_vers_pose(x):
    R = cv2.Rodrigues(x[:3])[0]
    th, ph = x[3], x[4]
    t = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return R, t


def _residus_epipolaires(R, t, K, p1, p2):
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Ki = np.linalg.inv(K)
    F = Ki.T @ (tx @ R) @ Ki
    F = F / (np.linalg.norm(F) + 1e-30)
    h1 = np.column_stack([p1, np.ones(len(p1))])
    h2 = np.column_stack([p2, np.ones(len(p2))])
    l2 = h1 @ F.T
    l1 = h2 @ F
    num = np.abs(np.sum(l2 * h2, axis=1))
    d2 = num / np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + 1e-12)
    d1 = num / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-12)
    return 0.5 * (d1 + d2)


def raffiner(K, R, t, p1, p2, poids=None, f_scale=3.0):
    """Ajustement non lineaire de la pose sur le residu epipolaire.

    recoverPose donne une solution algebrique correcte mais pas optimale :
    elle minimise un critere sur la matrice essentielle, pas la distance
    geometrique aux droites epipolaires. Sur un sujet reel, l'ecart entre le
    residu du rig et le PLANCHER du champ atteignait 1,3 px — du signal
    recuperable, pas du bruit.

    Perte robuste (soft_l1) : le visage n'est pas parfaitement rigide entre
    les trois prises, et une bouche ou un oeil qui a bouge ne doit pas tirer
    la pose. Ponderation par 1/sigma quand RoMa fournit l'ecart-type.
    """
    from scipy.optimize import least_squares

    w = np.ones(len(p1)) if poids is None else np.clip(poids, 1e-3, None)

    def cout(x):
        Rr, tr = _params_vers_pose(x)
        return _residus_epipolaires(Rr, tr, K, p1, p2) * w

    x0 = _pose_vers_params(R, t)
    sol = least_squares(cout, x0, loss="soft_l1", f_scale=f_scale,
                        max_nfev=200, xtol=1e-12, ftol=1e-12)
    Rr, tr = _params_vers_pose(sol.x)
    # on rend t a la norme d'origine : l'echelle est fixee ailleurs
    return Rr, tr * np.linalg.norm(t)


def estimate(K, p1, p2, thresh):
    u1, u2 = vc.undistort(p1.astype(np.float32), K), vc.undistort(p2.astype(np.float32), K)
    E, mask = cv2.findEssentialMat(u1, u2, K, method=cv2.USAC_MAGSAC,
                                   prob=0.9999, threshold=thresh)
    if E is None:
        raise SystemExit("findEssentialMat a echoue")
    if E.shape[0] > 3:                      # plusieurs solutions candidates
        E = E[:3]
    n, R, t, mask2 = cv2.recoverPose(E, u1, u2, K, mask=mask.copy())
    inl = mask.ravel().astype(bool)
    return R, t.ravel(), inl, int(n)


def describe(k, R, t, attendu):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, np.argmin(np.abs(w - 1))])
    ax = ax / np.linalg.norm(ax)
    if ax[1] < 0:
        ax = -ax
    Cn = -R.T @ t
    Cn = Cn / (np.linalg.norm(Cn) + 1e-12)
    par = np.degrees(np.arccos(np.clip(abs(Cn @ np.array([0, 0, 1.0])), -1, 1)))
    print(f"   angle {ang:.2f}° (attendu ~{attendu}°) | lacet {100*abs(ax[1]):.0f} % "
          f"| roulis {100*abs(ax[2]):.0f} %")
    print(f"   base horizontale {100*abs(Cn[0]):.0f} % | a {par:.1f}° de l'axe optique")
    souci = []
    if abs(ang - attendu) > 15:
        souci.append(f"angle a {abs(ang-attendu):.0f}° de l'attendu")
    if abs(ax[1]) < 0.60:
        souci.append("rotation non dominee par le lacet")
    if abs(Cn[0]) < 0.35:
        souci.append("base peu horizontale")
    if par < 15:
        souci.append("base trop proche de l'axe optique")
    return ang, souci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roma", required=True, help="champ RoMa v2 exporte (.npz)")
    ap.add_argument("--out", default="rig.json")
    ap.add_argument("--cert", type=float, default=0.6,
                    help="seuil d'overlap pour la selection de calibration")
    ap.add_argument("--sigma-max", type=float, default=None,
                    help="ecart-type max (px). Defaut : 40e centile des points retenus")
    ap.add_argument("--cell", type=int, default=24,
                    help="cote de cellule (px) pour la repartition spatiale")
    ap.add_argument("--cap", type=int, default=8000, help="points max par paire")
    ap.add_argument("--thresh", type=float, default=1.0,
                    help="seuil MAGSAC sur la matrice essentielle (px)")
    ap.add_argument("--sans-raffinement", action="store_true",
                    help="s'arreter a recoverPose, sans ajustement non lineaire")
    ap.add_argument("--raffinement-cap", type=int, default=40000,
                    help="points utilises pour l'ajustement non lineaire")
    ap.add_argument("--force", action="store_true",
                    help="ecrire le rig meme si les controles echouent")
    a = ap.parse_args()
    print(C.summary())

    field = rf.load(a.roma)
    print()
    field.report()
    H, W = field.H, field.W
    K = vc.K_matrix(W, H)
    ys, xs = (field.idx // W).astype(np.float64), (field.idx % W).astype(np.float64)

    attendu = {"L": C.ANGLE_L_DEG, "R": C.ANGLE_R_DEG}
    poses, inliers, soucis = {}, {}, {}
    for k in field.pairs:
        print(f"\n-- paire F<->{k}")
        xy = field._d[f"{k}_xy"].astype(np.float64)
        cert = field._d[f"{k}_cert"].astype(np.float64)
        sig = field._d[f"{k}_sigma"].astype(np.float64)[:, 0]
        good = np.isfinite(xy[:, 0]) & (cert > a.cert)
        if good.sum() < 500:
            print(f"   seulement {int(good.sum())} points au-dessus de cert={a.cert} "
                  f"— baisser --cert")
            continue
        smax = a.sigma_max if a.sigma_max is not None else np.nanpercentile(sig[good], 40)
        good &= np.isfinite(sig) & (sig <= smax)
        print(f"   {int(good.sum())} candidats (cert > {a.cert}, sigma <= {smax:.2f} px)")

        gi = np.nonzero(good)[0]
        sel = gi[spread_select(xs[gi], ys[gi], -sig[gi], W, H, a.cell, a.cap)]
        p1 = np.column_stack([xs[sel], ys[sel]])
        p2 = xy[sel]
        print(f"   {len(sel)} points retenus apres repartition spatiale "
              f"(1 par cellule de {a.cell} px)")
        if len(sel) < 200:
            print(f"   ATTENTION : peu de points pour estimer une pose. Reduire "
                  f"--cell (actuellement {a.cell}) ou --cert (actuellement {a.cert}).")

        R, t, inl, npos = estimate(K, p1, p2, a.thresh)
        print(f"   MAGSAC : {int(inl.sum())}/{len(sel)} inliers "
              f"({100*inl.mean():.0f} %) | {npos} points devant les deux cameras")

        # --- ajustement non lineaire sur la geometrie, pas sur l'algebre ----
        if not a.sans_raffinement:
            gi2 = np.nonzero(good)[0]
            if len(gi2) > a.raffinement_cap:
                gi2 = gi2[np.linspace(0, len(gi2) - 1, a.raffinement_cap).astype(int)]
            q1 = np.column_stack([xs[gi2], ys[gi2]])
            q2 = xy[gi2]
            sig_q = np.where(np.isfinite(sig[gi2]) & (sig[gi2] > 0.05),
                             sig[gi2], np.nanmedian(sig[good]))
            avant = float(np.median(_residus_epipolaires(R, t, K, q1, q2)))
            sol_champ = field.plancher(k, cert_min=max(a.cert, 0.5)) or 1.0
            R, t = raffiner(K, R, t, q1, q2, poids=1.0 / sig_q,
                            f_scale=max(2.0 * sol_champ, 1.0))
            apres = float(np.median(_residus_epipolaires(R, t, K, q1, q2)))
            print(f"   ajustement non lineaire sur {len(gi2)} points : "
                  f"residu median {avant:.2f} -> {apres:.2f} px "
                  f"(plancher du champ {sol_champ:.2f} px)")
            if apres > 1.5 * sol_champ:
                print(f"      reste {apres - sol_champ:.2f} px au-dessus du plancher :")
                print(f"      la pose n'explique pas tout, regarder la carte des")
                print(f"      residus (rig_check.py --carte) avant d'aller plus loin.")
        ang, souci = describe(k, R, t, attendu[k])
        poses[k] = (R, t)
        inliers[k] = (p1[inl], p2[inl])
        soucis[k] = souci
        for s in souci:
            print(f"   PROBLEME : {s}")

    if not poses:
        raise SystemExit("aucune pose estimee")

    # --- echelle relative des deux branches ---------------------------------
    if len(poses) == 2:
        print("\n-- echelle relative des deux branches")
        xyL = field._d["L_xy"].astype(np.float64)
        xyR = field._d["R_xy"].astype(np.float64)
        certL = field._d["L_cert"].astype(np.float64)
        certR = field._d["R_cert"].astype(np.float64)
        both = (np.isfinite(xyL[:, 0]) & np.isfinite(xyR[:, 0])
                & (certL > a.cert) & (certR > a.cert))
        print(f"   {int(both.sum())} pixels frontaux vus dans L ET dans R "
              f"(chainage exact, pas de mise en correspondance a faire)")
        if both.sum() < 200:
            print("   ATTENTION : chainage trop faible, echelle relative peu fiable")
            s = 1.0
        else:
            bi = np.nonzero(both)[0]
            if len(bi) > 50000:
                bi = bi[np.linspace(0, len(bi) - 1, 50000).astype(int)]
            pf = np.column_stack([xs[bi], ys[bi]])
            uf = vc.undistort(pf.astype(np.float32), K)
            XL = vc.triangulate(K, *poses["L"], uf,
                                vc.undistort(xyL[bi].astype(np.float32), K))
            XR = vc.triangulate(K, *poses["R"], uf,
                                vc.undistort(xyR[bi].astype(np.float32), K))
            ok = (XL[:, 2] > 0) & (XR[:, 2] > 0) & np.isfinite(XL).all(1) & np.isfinite(XR).all(1)
            r = XL[ok, 2] / XR[ok, 2]
            s = float(np.median(r))
            disp = float(np.percentile(r, 75) - np.percentile(r, 25)) / max(abs(s), 1e-9)
            print(f"   facteur d'echelle L/R = {s:.4f} sur {int(ok.sum())} points "
                  f"| dispersion interquartile {100*disp:.1f} %")
            if disp > 0.15:
                print("   ATTENTION : dispersion elevee — les deux branches ne "
                      "s'accordent pas, au moins une pose est douteuse")
        poses["R"] = (poses["R"][0], poses["R"][1] * s)

    # --- controle final : relief du visage ----------------------------------
    print("\n-- controle de plausibilite du relief")
    for k, (p1, p2) in inliers.items():
        X = vc.triangulate(K, *poses[k], vc.undistort(p1.astype(np.float32), K),
                           vc.undistort(p2.astype(np.float32), K))
        ok = np.isfinite(X).all(1) & (X[:, 2] > 0)
        if ok.sum() < 50:
            print(f"   paire {k} : trop peu de points positifs — pose invalide")
            soucis[k].append("relief non mesurable")
            continue
        Z = X[ok, 2]
        etendue = (np.percentile(Z, 97) - np.percentile(Z, 3)) / np.median(Z)
        print(f"   paire {k} : etendue en profondeur {100*etendue:.1f} % de la "
              f"distance de travail (attendu 3-4 %)")
        if not (0.005 < etendue < 0.15):
            soucis[k].append(f"relief invraisemblable ({100*etendue:.1f} %)")
            print(f"      PROBLEME : hors de toute plage plausible pour un visage")

    tous = [s for v in soucis.values() for s in v]
    print("\n" + "=" * 70)
    if tous and not a.force:
        print(f"{len(tous)} probleme(s) detecte(s) :")
        for s in tous:
            print(f"   - {s}")
        print("\nRig NON ecrit. Pistes, dans l'ordre :")
        print("   1. verifier que le champ exporte est bon :")
        print("      python 00_pipeline/roma_field.py --info <champ.npz>")
        print("   2. relacher la selection : --cert 0.4")
        print("   3. elargir le seuil MAGSAC : --thresh 2.0")
        print("   4. re-exporter avec tuilage (roma_v2_export.py --mode tiled)")
        print("   5. --force pour ecrire malgre tout (a vos risques)")
        raise SystemExit(2)

    vc.save_rig(a.out, poses,
                note=f"estime sur le champ RoMa v2 {os.path.basename(a.roma)} "
                     f"({field.meta['subject']}/{field.meta['session']}/"
                     f"{field.meta['modality']}, mode {field.meta['mode']})")
    print(f"rig -> {a.out}")
    print("Ce rig REMPLACE le precedent : toutes les sessions deja reconstruites")
    print("doivent etre retraitees avant toute comparaison.")
    print(f"\nControle : python 02_calibration/rig_check.py --rig {a.out} --roma {a.roma}")


if __name__ == "__main__":
    main()

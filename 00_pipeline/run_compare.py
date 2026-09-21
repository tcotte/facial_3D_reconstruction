"""
run_compare.py — compare deux sessions reconstruites (D0 vs Dx).

    python run_compare.py <sujet>_D0.npz <sujet>_Dx.npz --roi roi.png --stable stable.png

ROI et ZONE STABLE
------------------
Deux masques PNG (blanc = actif) definis UNE FOIS sur la vue frontale de D0,
a la resolution de travail. La zone stable doit etre anatomiquement non
affectee par le traitement ; la ROI est la zone evaluee. Elles ne doivent pas
se chevaucher.

MISE EN CORRESPONDANCE DES DEUX SESSIONS
----------------------------------------
Le sujet est repositionne entre les sessions : les deux vues frontales ne se
superposent pas. On les apparie donc (meme modalite, faible ecart de point de
vue) pour transferer la ROI et associer les points 3D deux a deux.

SORTIES
-------
- carte de deviation signee
- metriques SANS DIMENSION (voir visia_compare.py)
- test a blanc donnant le seuil de detection empirique

AUCUNE VALEUR N'EST METRIQUE. Ne jamais convertir en mm ou mm^3.
"""
import argparse, json
import numpy as np, cv2
from scipy.spatial import cKDTree
import config as C, visia_core as vc, visia_compare as cmp


def load_session(npz):
    d = np.load(npz, allow_pickle=True)
    if str(d["calib"]) != C.CALIB_VERSION:
        raise RuntimeError(f"{npz} : calibration {d['calib']} != {C.CALIB_VERSION}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("test")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--ref-session", required=True)
    ap.add_argument("--test-session", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--stable", required=True)
    ap.add_argument("--out", default="comparaison")
    a = ap.parse_args()
    print(C.summary())

    d0, dx = load_session(a.ref), load_session(a.test)
    h, w = d0["shape"]

    # --- appariement inter-session sur la vue frontale
    A0 = vc.load(vc.img_path(a.subject, a.ref_session, "F"))
    Ax = vc.load(vc.img_path(a.subject, a.test_session, "F"))
    matcher = vc.Matcher()
    r = vc.match_pair(matcher, A0, Ax, guide=None)
    if r is None:
        raise SystemExit("appariement inter-session echoue")
    q0, qx, _ = r
    print(f"appariement inter-session : {len(q0)} points")

    # transfert : pour chaque point 3D de D0, trouver son homologue dans Dx
    # via le champ de correspondance frontal->frontal (interpole localement)
    from scipy.interpolate import griddata
    tgt = griddata(q0, qx, d0["pix"], method="linear")
    ok = np.isfinite(tgt).all(1)
    print(f"points transferables : {ok.sum()}/{len(tgt)}")

    tree = cKDTree(dx["pix"])
    dist, idx = tree.query(tgt[ok], k=1)
    paired = dist < 3.0
    print(f"points apparies entre sessions : {paired.sum()}")

    i0 = np.nonzero(ok)[0][paired]
    ix = idx[paired]
    X0 = d0["X"][i0]; Xx = dx["X"][ix]; pix0 = d0["pix"][i0]

    # --- masques ROI / zone stable, lus sur la vue frontale de reference
    def read_mask(p):
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise SystemExit(f"masque illisible : {p}")
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m > 127
    roi_img, stab_img = read_mask(a.roi), read_mask(a.stable)
    if (roi_img & stab_img).any():
        print("ATTENTION : ROI et zone stable se chevauchent — le recalage sera biaise")
    xi = np.clip(pix0[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pix0[:, 1].astype(int), 0, h - 1)
    roi = roi_img[yi, xi]; stable = stab_img[yi, xi]
    print(f"points en ROI : {roi.sum()} | en zone stable : {stable.sum()}")

    # --- comparaison
    metrics, dev, Xa = cmp.compare(X0, Xx, roi, stable)
    nt = cmp.null_test(X0, Xx, stable)
    metrics["test_a_blanc"] = nt

    print("\n=== RESULTATS (sans dimension — NON metriques) ===")
    print(f"recalage : echelle {metrics['recalage']['scale']:.4f}, "
          f"residu median {metrics['recalage']['residu_median']:.5f}")
    print(f"variation relative de relief (moyenne) : {metrics['ratio_relief_moyen']:+.4f}")
    print(f"variation relative de volume           : {metrics['variation_volume_relative']:+.4f}")
    print(f"controle zone stable (doit etre ~0)    : {metrics['controle_zone_stable_moyen']:+.4f}")
    if nt:
        print(f"seuil de detection empirique (2 sigma) : {nt['seuil_detection_2sigma']:.4f}")
        if abs(metrics['ratio_relief_moyen']) < nt['seuil_detection_2sigma']:
            print(">>> VARIATION NON SIGNIFICATIVE : sous le plancher de bruit.")
        else:
            print(">>> variation superieure au plancher de bruit.")

    img, vmax, m = cmp.deviation_map((h, w), pix0[roi], dev[roi])
    base = (A0 * 0.4).astype(np.uint8)
    base[m] = img[m]
    cv2.imwrite(a.out + "_deviation.png", base)
    with open(a.out + "_metriques.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\ncarte -> {a.out}_deviation.png   (echelle +/-{vmax:.5f} u.a.)")
    print(f"metriques -> {a.out}_metriques.json")

if __name__ == "__main__":
    main()

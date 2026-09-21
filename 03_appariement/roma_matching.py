"""
roma_matching.py — appariement dense RoMa, deux strategies.

OBJET
    RoMa produit un CHAMP de correspondance dense plutot qu'un ensemble de
    points. Deux usages sont implementes :
      --mode sample : echantillonnage stochastique (usage nominal de RoMa)
      --mode grid   : lecture du champ a une GRILLE FIXE de points frontaux

CONCLUSION DE L'EVALUATION (voir rapport, § 9.6)
    Densification   : RoMa est SUPERIEUR (9873/13484 appariements contre
                      3616/3770, et 5 a 6 fois plus rapide).
    Calibration     : RoMa est INFERIEUR. En mode 'sample', deux appels
                      independants ne tirent pas les memes points frontaux, ce
                      qui detruit le chainage 3 vues (50 pistes contre 272-580).
                      Le mode 'grid' restaure le chainage mais le filtrage
                      epipolaire rejette 93 % des pistes : la precision de
                      localisation d'un champ dense reste inferieure a celle
                      d'un detecteur de points.

    -> LightGlue pour la CALIBRATION, RoMa pour la DENSIFICATION.

INSTALLATION
    pip install git+https://github.com/Parskatt/RoMa.git
    Sans GPU, seul `tiny_roma_v1_outdoor` est utilisable (le modele complet
    repose sur un backbone DINOv2). Le chargement de tiny_roma passe par
    torch.hub : en environnement non interactif, pre-remplir le cache hub.

USAGE
    python roma_matching.py --subject alban --session D0 --pair L --mode grid
"""
import argparse, os, time, numpy as np, cv2, torch

IMG_DIR = r"C:\donnees\visia"
ANGLES = {"F": "Frontal", "L": "Left_Oblique", "R": "Right_Oblique"}
MODALITY = "Standard_1"
TILE, STRIDE = 640, 512


def get_model(full=False):
    if full and torch.cuda.is_available():
        from romatch import roma_outdoor
        print("[RoMa] modele complet, GPU")
        return roma_outdoor(device="cuda"), "cuda"
    from romatch import tiny_roma_v1_outdoor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RoMa] tiny_roma, {dev}")
    return tiny_roma_v1_outdoor(device=dev), dev


def load(subject, session, angle, long_side):
    p = os.path.join(IMG_DIR, f"{subject}_{session}_{ANGLES[angle]}_{MODALITY}.jpg")
    im = cv2.imread(p)
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--session", required=True)
    ap.add_argument("--pair", choices=["L", "R"], required=True)
    ap.add_argument("--mode", choices=["sample", "grid"], default="grid")
    ap.add_argument("--guide", required=True, help="npz de correspondance grossiere")
    ap.add_argument("--long", type=int, default=4000)
    ap.add_argument("--step", type=int, default=6, help="pas de la grille (mode grid)")
    ap.add_argument("--cert", type=float, default=0.05, help="seuil de certitude")
    ap.add_argument("--full", action="store_true", help="modele complet (GPU requis)")
    a = ap.parse_args()
    model, dev = get_model(a.full)
    A = load(a.subject, a.session, "F", a.long)
    B = load(a.subject, a.session, a.pair, a.long)
    HA, WA = A.shape[:2]; HB, WB = B.shape[:2]
    face = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY) > 45
    g = np.load(a.guide)
    obx, oby, sc = g["obx"], g["oby"], float(g["scale"])

    if a.mode == "grid":
        gy, gx = np.mgrid[0:HA:a.step, 0:WA:a.step]
        keep = face[gy, gx] & (gy > 0.28 * HA) & (gy < 0.88 * HA)
        GX, GY = gx[keep].astype(float), gy[keep].astype(float)
        OUT = np.full((len(GX), 2), np.nan); CERT = np.zeros(len(GX), np.float32)
        print(f"grille frontale : {len(GX)} points")
    else:
        PA, PB = [], []

    t0 = time.time(); n = 0
    for y0 in range(int(0.28 * HA), int(0.88 * HA) - TILE, STRIDE):
        for x0 in range(0, WA - TILE, STRIDE):
            if face[y0:y0 + TILE, x0:x0 + TILE].mean() < 0.45:
                continue
            gy0, gy1 = int(y0 / sc), int((y0 + TILE) / sc)
            gx0, gx1 = int(x0 / sc), int((x0 + TILE) / sc)
            ox, oy = obx[gy0:gy1, gx0:gx1], oby[gy0:gy1, gx0:gx1]
            m = ~np.isnan(ox)
            if m.sum() < 200:
                continue
            bx0 = max(0, int(np.nanpercentile(ox[m], 2) * sc) - 80)
            bx1 = min(WB, int(np.nanpercentile(ox[m], 98) * sc) + 80)
            by0 = max(0, int(np.nanpercentile(oy[m], 2) * sc) - 80)
            by1 = min(HB, int(np.nanpercentile(oy[m], 98) * sc) + 80)
            if bx1 - bx0 < 64 or by1 - by0 < 64:
                continue
            ca, cb = A[y0:y0 + TILE, x0:x0 + TILE], B[by0:by1, bx0:bx1]
            cv2.imwrite("_ra.png", ca); cv2.imwrite("_rb.png", cb)
            try:
                warp, cert = model.match("_ra.png", "_rb.png")
            except Exception:
                continue
            n += 1
            if a.mode == "sample":
                mk = model.sample(warp, cert, num=2000)
                matches = mk[0] if isinstance(mk, tuple) else mk
                kA, kB = model.to_pixel_coordinates(matches, ca.shape[0], ca.shape[1],
                                                    cb.shape[0], cb.shape[1])
                PA.append(kA.detach().cpu().numpy() + np.array([x0, y0]))
                PB.append(kB.detach().cpu().numpy() + np.array([bx0, by0]))
            else:
                Wn = warp.detach().cpu().numpy(); Cn = cert.detach().cpu().numpy()
                hh, ww = Wn.shape[:2]
                sel = (GX >= x0) & (GX < x0 + TILE) & (GY >= y0) & (GY < y0 + TILE)
                if sel.sum() == 0:
                    continue
                u = (GX[sel] - x0) / TILE * (ww - 1); v = (GY[sel] - y0) / TILE * (hh - 1)
                ui = np.clip(np.round(u).astype(int), 0, ww - 1)
                vi = np.clip(np.round(v).astype(int), 0, hh - 1)
                c = Cn[vi, ui]
                bxp = (Wn[vi, ui, 2] + 1) / 2 * (cb.shape[1] - 1) + bx0
                byp = (Wn[vi, ui, 3] + 1) / 2 * (cb.shape[0] - 1) + by0
                idxs = np.nonzero(sel)[0]
                better = c > CERT[idxs]
                CERT[idxs[better]] = c[better]
                OUT[idxs[better], 0] = bxp[better]; OUT[idxs[better], 1] = byp[better]

    if a.mode == "sample":
        PA, PB = np.vstack(PA), np.vstack(PB)
    else:
        ok = np.isfinite(OUT[:, 0]) & (CERT > a.cert)
        PA, PB = np.column_stack([GX, GY])[ok], OUT[ok]
        print(f"points retenus (certitude > {a.cert}) : {ok.sum()}")
    F, mask = cv2.findFundamentalMat(PA, PB, cv2.FM_RANSAC, 3.0, 0.999, maxIters=50000)
    inl = mask.ravel().astype(bool)
    yn = PA[inl][:, 1] / HA
    print(f"{n} tuiles | {len(PA)} candidats | {inl.sum()} inliers | "
          f"{int(((yn>0.30)&(yn<0.86)).sum())} sur la peau | {time.time()-t0:.0f}s")
    out = f"roma_{a.mode}_{a.session}_{a.pair}.npz"
    np.savez(out, pa=PA[inl], pb=PB[inl])
    print(f"-> {out}")

if __name__ == "__main__":
    main()

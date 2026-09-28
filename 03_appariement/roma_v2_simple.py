"""
roma_v2_simple.py — export RoMa v2 SANS TUILAGE, un appel par paire.

Version minimale de roma_v2_export.py : une seule passe pleine image par paire,
exactement le schema

    model = RoMaV2(); model.apply_setting("precise")
    preds = model.match(im1_path, im2_path)

Le fichier produit est au MEME format que roma_v2_export.py : il se relit tel
quel avec 00_pipeline/roma_field.py et run_session.py --roma.

CE QU'IL FAUT SAVOIR AVANT DE L'UTILISER
    RoMa redimensionne les deux images a un carre fixe (1280x1280 en reglage
    'precise'). Le champ de correspondance a donc 1280 cellules de cote, alors
    que l'image de travail du pipeline en a 4000 : une cellule de champ couvre
    environ 3 pixels de travail. Sans tuilage, la position d'un appariement ne
    peut pas etre plus fine que cela.

    Consequence concrete : le residu epipolaire sera plus eleve qu'avec le
    tuilage, et le seuil EPI_HIGH = 1,0 px du pipeline retiendra moins de
    points. Ce n'est pas un defaut de RoMa, c'est la resolution du champ.
    Commencez par ici — c'est rapide et ca suffit a valider la chaine — puis
    passez a roma_v2_export.py --mode tiled si la couverture est insuffisante.

POURQUOI LES CHEMINS SONT PASSES DIRECTEMENT A match()
    Comme dans votre code : model.match() fait lui-meme le redimensionnement.
    Le warp est rendu en coordonnees NORMALISEES [-1, 1], independantes de la
    resolution d'entree — on peut donc lui donner le jpg natif 8000 px et
    convertir ensuite vers les coordonnees de l'image de travail a 4000 px. La
    ligne `Image.open(...).resize((W, H))` de la demo ne sert qu'a
    l'affichage ; elle est inutile ici.

CONVERSION DE COORDONNEES — le detail qui coute cher si on l'oublie
    RoMa : centre du pixel i sur N -> 2*(i+0.5)/N - 1   (align_corners=False)
    OpenCV : centre du pixel i -> i.0
    D'ou le decalage de 0,5 px applique dans les deux sens ci-dessous. Le champ
    est lu par interpolation BILINEAIRE, jamais au plus proche voisin : sur un
    champ 1280 lu depuis une image a 4000 px, l'arrondi coute a lui seul
    1,42 px d'erreur mediane, pour un seuil de 1,0 px.

USAGE
    # convention de nommage VISIA
    python roma_v2_simple.py --img-dir /data/visia --subject alban --session D0

    # ou chemins explicites
    python roma_v2_simple.py --front F.jpg --right R.jpg --left L.jpg \
        --subject alban --session D0 --out roma2_alban_D0.npz

    # cote pipeline, apres telechargement
    python 00_pipeline/roma_field.py --info roma2_alban_D0_Standard_1.npz
    python 00_pipeline/run_session.py alban D0 --rig rig.json --step 1 \
        --roma roma2_alban_D0_Standard_1.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

ANGLES = {"F": "Frontal", "L": "Left_Oblique", "R": "Right_Oblique"}
SKIN_TOP, SKIN_BOTTOM = 0.28, 0.88
EXPORT_FORMAT = "visia-roma2-field/1"       # lu par 00_pipeline/roma_field.py


def sha256(path, blocks=64):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(blocks * 1024))
        if size > 2 * blocks * 1024:
            f.seek(-blocks * 1024, os.SEEK_END)
            h.update(f.read())
    h.update(str(size).encode())
    return h.hexdigest()[:32]


def load(path, long_side):
    """Identique a visia_core.load() : memes coordonnees des deux cotes."""
    im = cv2.imread(path)
    if im is None:
        raise IOError(path)
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def face_mask(gray, margin=40):
    """Identique a visia_core.face_mask(), dilate : le filtrage definitif reste
    cote pipeline, la ou les constantes sont gelees."""
    m = (gray > 45).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    if margin > 0:
        m = cv2.dilate(m, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    band = np.zeros_like(m)
    h = gray.shape[0]
    band[max(0, int(SKIN_TOP * h) - margin):min(h, int(SKIN_BOTTOM * h) + margin), :] = 1
    return (m & band).astype(bool)


def read_field(preds, model, gx, gy, WA, HA, WB, HB):
    """Lit le champ aux points frontaux (gx, gy). Voir l'en-tete pour les
    conventions. Retourne (xy, cert, sigma1, sigma2, theta)."""
    import torch
    import torch.nn.functional as TF

    warp = preds["warp_AB"]                   # (1, Hf, Wf, 2) normalise dans B
    over = preds["overlap_AB"]                # (1, Hf, Wf, 1) dans [0, 1]
    prec = preds["precision_AB"]              # (1, Hf, Wf, 2, 2) px^-2 a (Hf, Wf)
    Hf, Wf = warp.shape[1], warp.shape[2]
    dev = warp.device

    un = 2.0 * (gx + 0.5) / WA - 1.0          # pixel OpenCV -> normalise
    vn = 2.0 * (gy + 0.5) / HA - 1.0
    grid = torch.from_numpy(np.stack([un, vn], -1).astype(np.float32))[None, :, None].to(dev)

    def gs(t):
        o = TF.grid_sample(t.permute(0, 3, 1, 2).float(), grid,
                           mode="bilinear", align_corners=False, padding_mode="border")
        return o[0, :, :, 0].T.contiguous()

    w = gs(warp)
    c = gs(over)[:, 0]
    P = gs(prec.reshape(1, Hf, Wf, 4)).reshape(-1, 2, 2)
    P = model.prec_map_coordinates(P, H_in=Hf, W_in=Wf, H_out=HB, W_out=WB)

    # Deux invalidations indispensables :
    #  (a) point frontal hors du domaine interpolable du champ (|n| > 1 - 1/N) :
    #      grid_sample y extrapole et rend une correspondance plausible mais
    #      fausse — jusqu'a plusieurs centaines de pixels d'erreur ;
    #  (b) correspondance tombant hors de l'image B.
    qx = torch.from_numpy(un.astype(np.float32)).to(dev)
    qy = torch.from_numpy(vn.astype(np.float32)).to(dev)
    valid = ((qx.abs() <= 1 - 1.0 / Wf) & (qy.abs() <= 1 - 1.0 / Hf)
             & (w[:, 0].abs() < 1.0) & (w[:, 1].abs() < 1.0))
    c = torch.where(valid, c, torch.zeros_like(c))

    # covariance = inverse de la precision -> ecarts-types en pixels de B
    P = 0.5 * (P.double() + P.double().transpose(-1, -2))
    ev, evec = torch.linalg.eigh(P)
    sig = ev.clamp_min(1e-12).rsqrt()
    theta = torch.atan2(evec[:, 1, 0], evec[:, 0, 0])

    wn = w.detach().cpu().numpy().astype(np.float64)
    bad = ~valid.detach().cpu().numpy()
    xB = (wn[:, 0] + 1.0) / 2.0 * WB - 0.5    # normalise -> pixel OpenCV
    yB = (wn[:, 1] + 1.0) / 2.0 * HB - 0.5
    xB[bad] = np.nan
    yB[bad] = np.nan
    return (np.stack([xB, yB], -1).astype(np.float32),
            c.detach().cpu().numpy().astype(np.float32),
            sig[:, 0].cpu().numpy().astype(np.float32),     # grand axe
            sig[:, 1].cpu().numpy().astype(np.float32),     # petit axe
            theta.cpu().numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--img-dir", default=None,
                    help="dossier VISIA ; ignore si --front/--left/--right sont donnes")
    ap.add_argument("--modality", default="Standard_1")
    ap.add_argument("--front"); ap.add_argument("--left"); ap.add_argument("--right")
    ap.add_argument("--out", default=None)
    ap.add_argument("--long", type=int, default=4000,
                    help="cote long de l'image de travail ; DOIT valoir config.WORK_LONG")
    ap.add_argument("--step", type=int, default=2,
                    help="pas de la grille frontale exportee. Sans tuilage, le champ "
                         "fait 1280 px de cote : descendre sous 2 n'apporte rien.")
    ap.add_argument("--setting", default="precise",
                    choices=["precise", "base", "fast", "turbo"])
    ap.add_argument("--cert-min", type=float, default=0.02)
    ap.add_argument("--mask-margin", type=int, default=40)
    a = ap.parse_args()

    # --- chemins ------------------------------------------------------------
    def vpath(k):
        return os.path.join(a.img_dir,
                            f"{a.subject}_{a.session}_{ANGLES[k]}_{a.modality}.jpg")
    paths = {}
    explicit = {"F": a.front, "L": a.left, "R": a.right}
    for k, p in explicit.items():
        if p:
            paths[k] = p
        elif a.img_dir and os.path.exists(vpath(k)):
            paths[k] = vpath(k)
    if "F" not in paths:
        raise SystemExit("vue frontale introuvable : donner --front ou --img-dir")
    pairs = "".join(k for k in "LR" if k in paths)
    if not pairs:
        raise SystemExit("aucune vue oblique : donner --left et/ou --right")

    # --- modele -------------------------------------------------------------
    import torch
    from romav2 import RoMaV2
    torch.set_float32_matmul_precision("highest")      # exige par RoMaV2.forward
    if not torch.cuda.is_available():
        print("ATTENTION : aucun GPU detecte, le reglage 'precise' sera tres lent.")
    model = RoMaV2()
    model.apply_setting(a.setting)
    print(f"[RoMa v2] reglage '{a.setting}' | champ {model.H_hr}x{model.W_hr} | "
          f"bidirectionnel={model.bidirectional}")

    # --- grille frontale ----------------------------------------------------
    A = load(paths["F"], a.long)
    HA, WA = A.shape[:2]
    fm = face_mask(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), a.mask_margin)
    gy, gx = np.mgrid[0:HA:a.step, 0:WA:a.step]
    keep = fm[gy, gx]
    GX = gx[keep].astype(np.float64)
    GY = gy[keep].astype(np.float64)
    idx = (gy[keep].astype(np.int64) * WA + gx[keep].astype(np.int64)).astype(np.int32)
    print(f"image de travail {WA}x{HA} | grille frontale {len(GX)} points (pas {a.step})")
    print(f"resolution du champ : ~{WA/model.W_hr:.1f} px de travail par cellule")

    # --- une passe par paire ------------------------------------------------
    out = {"idx": idx}
    stats = {}
    for k in pairs:
        B = load(paths[k], a.long)
        HB, WB = B.shape[:2]
        print(f"\n-- paire F<->{k}")
        t0 = time.time()
        preds = model.match(paths["F"], paths[k])      # chemins natifs, comme dans votre code
        xy, c, s1, s2, th = read_field(preds, model, GX, GY, WA, HA, WB, HB)
        del preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        good = np.isfinite(xy[:, 0]) & (c > a.cert_min)
        xy[~good] = np.nan
        out[f"{k}_xy"] = xy
        out[f"{k}_cert"] = c.astype(np.float16)
        out[f"{k}_sigma"] = np.stack([s1, s2], -1).astype(np.float16)
        out[f"{k}_theta"] = th.astype(np.float16)
        out[f"{k}_spread"] = np.full(len(idx), np.nan, np.float16)   # pas de tuiles
        out[f"{k}_src"] = np.zeros(len(idx), np.int8)                # 0 = pleine image
        out[f"{k}_shape"] = np.array([HB, WB], np.int32)
        stats[k] = {
            "points": int(good.sum()),
            "couverture_grille_%": round(100.0 * good.sum() / len(GX), 1),
            "cert_mediane": round(float(np.median(c[good])), 3) if good.any() else None,
            "sigma_median_px": round(float(np.nanmedian(s1[good])), 2) if good.any() else None,
            "spread_median_px": None,
            "tuiles": 0,
        }
        print(f"   {stats[k]['points']}/{len(GX)} points "
              f"({stats[k]['couverture_grille_%']} %) | cert med "
              f"{stats[k]['cert_mediane']} | sigma med "
              f"{stats[k]['sigma_median_px']} px | {time.time()-t0:.1f}s")
        del B

    meta = {
        "format": EXPORT_FORMAT,
        "subject": a.subject, "session": a.session, "modality": a.modality,
        "pairs": pairs,
        "work_long": a.long, "step": a.step, "mode": "full", "setting": a.setting,
        "tile": None, "stride": None, "margin": None,
        "cert_min": a.cert_min, "mask_margin": a.mask_margin,
        "grid_shape": [int(HA), int(WA)],
        "coord_convention": "OpenCV pixel centers (integer), bilinear field readout",
        "model": {"name": "RoMa v2", "setting": a.setting,
                  "H_lr": model.H_lr, "W_lr": model.W_lr,
                  "H_hr": model.H_hr, "W_hr": model.W_hr,
                  "bidirectional": bool(model.bidirectional)},
        "sources": {k: {"file": os.path.basename(p), "sha": sha256(p),
                        "bytes": os.path.getsize(p)} for k, p in paths.items()},
        "env": {"host": platform.node(), "python": sys.version.split()[0],
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "stats": stats,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "export sans tuilage : resolution du champ limitee a "
                f"{model.W_hr} px, soit ~{WA/model.W_hr:.1f} px de travail par cellule",
    }
    out["meta"] = np.array(json.dumps(meta, indent=1))

    dest = a.out or f"roma2_{a.subject}_{a.session}_{a.modality}.npz"
    np.savez_compressed(dest, **out)
    print(f"\n-> {dest}  ({os.path.getsize(dest)/1e6:.1f} Mo)")
    print("Telecharger, puis cote pipeline :")
    print(f"   python 00_pipeline/roma_field.py --info {dest}")
    print(f"   python 00_pipeline/run_session.py {a.subject} {a.session} "
          f"--rig rig.json --step 1 --roma {dest}")


if __name__ == "__main__":
    main()

"""
roma_v2_export.py — EXPORT des correspondances denses RoMa v2 depuis la machine GPU.

OBJET
    Ce script tourne sur la machine distante (H100). Il produit UN SEUL fichier
    .npz par session, autosuffisant et telechargeable, contenant le champ de
    correspondance dense F->L et F->R echantillonne sur la GRILLE DE PIXELS DE
    LA VUE FRONTALE.

    C'est le format qui compte : en lisant le champ aux MEMES points frontaux
    pour les deux paires, le chainage 3 vues est conserve par construction.
    C'est precisement ce que le mode 'sample' de RoMa v1 detruisait (50 pistes
    contre 272-580, cf. roma_matching.py).

CE QUI EST EXPORTE, ET POURQUOI
    xy      position correspondante dans l'oblique, en pixels OpenCV de l'image
            de travail (cote long = --long). C'est la donnee utile.
    cert    probabilite de recouvrement RoMa ('overlap'), dans [0, 1].
    sigma   ecart-type de localisation, en PIXELS, selon les deux axes propres
            de la covariance predite par RoMa v2. Nouveaute v2 : le modele
            predit une matrice de precision par pixel. C'est une incertitude
            ANISOTROPE et calibree, exploitable pour ponderer la triangulation
            et pour construire une carte de confiance bien meilleure qu'un
            simple seuil binaire.
    theta   orientation (rad) du grand axe de cette covariance.
    spread  desaccord entre tuiles qui voient le meme pixel (px), NaN si une
            seule tuile. Estimateur d'erreur INDEPENDANT du modele.

    Le residu epipolaire n'est PAS calcule ici : il doit etre evalue avec le
    rig GELE, cote pipeline, pour rester une validation geometrique
    independante de l'appariement. Voir 00_pipeline/roma_field.py.

CONVENTIONS — a ne pas modifier sans changer aussi roma_field.py
    * Image de travail : redimensionnement au cote long --long par
      cv2.resize(INTER_AREA), identique a visia_core.load().
    * Coordonnees pixel : convention OpenCV, le centre du pixel (i, j) est a
      (j, i) exactement. RoMa travaille en coordonnees normalisees [-1, 1] avec
      align_corners=False : le centre du pixel i est a 2*(i+0.5)/N - 1. La
      conversion applique donc bien le decalage de 0,5 px.
    * Le champ est lu par interpolation BILINEAIRE (grid_sample), pas au plus
      proche voisin. L'ancien roma_matching.py arrondissait a l'entier de la
      grille du champ, ce qui quantifiait la position a plusieurs pixels de
      l'image de travail : c'est une cause probable des 93 % de rejet
      epipolaire constates en mode 'grid'.

INSTALLATION SUR LA MACHINE DISTANTE
    pip install romav2 opencv-python-headless numpy torch
    Les poids (~1 Go) sont telecharges au premier appel par torch.hub.

USAGE
    python roma_v2_export.py --subject alban --session D0 \
        --img-dir /data/visia --out roma2_alban_D0.npz

    # toutes les modalites d'un coup (pour fuse_modalities.py)
    for m in Standard_1 Cross-Polarized Raked Parallel-Polarized; do
        python roma_v2_export.py --subject alban --session D0 --modality $m \
            --out roma2_alban_D0_$m.npz
    done

LICENCE — POINT D'ATTENTION
    RoMa v2 est en MIT, MAIS son descripteur est DINOv3, sous licence
    PROPRE a Meta (pas Apache 2.0, contrairement a DINOv2 depuis 2024). Le
    README du projet classe deja RoMa comme "evalue, non retenu" pour des
    raisons de licence : l'arrivee de DINOv3 ne leve pas ce point, elle le
    deplace. Validation juridique requise avant tout usage commercial.
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

# torch n'est importe qu'a l'execution : les fonctions de conversion de
# coordonnees ci-dessous doivent rester importables (et testables) sur une
# machine sans torch, typiquement le poste ou tourne le pipeline.

ANGLES = {"F": "Frontal", "L": "Left Oblique", "R": "Right Oblique"}
BLANK_SIZE = 251258          # fichier blanc de substitution produit par VISIA
SKIN_TOP, SKIN_BOTTOM = 0.2, 0.88
EXPORT_FORMAT = "visia-roma2-field/1"


# ---------------------------------------------------------------------------
# Entrees
# ---------------------------------------------------------------------------
def img_path(img_dir, subject, session, angle, modality):
    return os.path.join(img_dir, f"{subject}_{session}_{ANGLES[angle]}_{modality}.jpg")


def sha256(path, blocks=64):
    """Empreinte partielle (debut + fin) : suffit a detecter un fichier different
    sans relire 40 Mo, et sera reverifiee cote pipeline."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(blocks * 1024))
        if size > 2 * blocks * 1024:
            f.seek(-blocks * 1024, os.SEEK_END)
            h.update(f.read())
    h.update(str(size).encode())
    return h.hexdigest()[:32]


def check_file(path):
    if not os.path.exists(path):
        return False, "absent"
    if os.path.getsize(path) == BLANK_SIZE:
        return False, "fichier blanc (bug export VISIA)"
    im = cv2.imread(path)
    if im is None:
        return False, "illisible"
    if cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std() < 1.0:
        return False, "image uniforme"
    return True, "ok"


def load(path, long_side):
    """Identique a visia_core.load() : meme redimensionnement, donc memes
    coordonnees pixel des deux cotes de la chaine."""
    im = cv2.imread(path)
    if im is None:
        raise IOError(path)
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def face_mask(gray, margin=0, top=None, bottom=None):
    """Masque facial identique a visia_core.face_mask(), eventuellement dilate.

    On exporte volontairement un peu PLUS large que le masque du pipeline : le
    filtrage definitif doit rester cote pipeline, ou les constantes sont gelees.
    """
    m = (gray > 45).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    if margin > 0:
        m = cv2.dilate(m, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    band = np.zeros_like(m)
    h = gray.shape[0]
    t = SKIN_TOP if top is None else top
    b = SKIN_BOTTOM if bottom is None else bottom
    y0 = max(0, int(t * h) - margin)
    y1 = min(h, int(b * h) + margin)
    band[y0:y1, :] = 1
    return (m & band).astype(bool)


# ---------------------------------------------------------------------------
# Conversions de coordonnees
# ---------------------------------------------------------------------------
def px_to_norm(x, y, W, H):
    """Pixel OpenCV (centre du pixel i a i.0) -> coordonnee normalisee RoMa."""
    return 2.0 * (x + 0.5) / W - 1.0, 2.0 * (y + 0.5) / H - 1.0


def norm_to_px(u, v, W, H):
    """Coordonnee normalisee RoMa -> pixel OpenCV."""
    return (u + 1.0) / 2.0 * W - 0.5, (v + 1.0) / 2.0 * H - 0.5


def _eig2x2(P, eps=1e-12):
    """Decomposition analytique d'un lot de matrices 2x2 SYMETRIQUES.

    torch.linalg.eigh passe par cuSOLVER, qui echoue avec
    CUSOLVER_STATUS_INVALID_VALUE sur les tres grands lots — et un export au
    pas 1 sur un visage en produit plusieurs millions — comme a la moindre
    valeur non finie en entree.

    Pour du 2x2 symetrique la solution est analytique : ni cuSOLVER, ni limite
    de lot, et une entree non finie ressort en NaN au lieu de lever une
    exception. Valide contre numpy.linalg.eigh sur matrices definies positives,
    nulles, isotropes, diagonales, quasi singulieres et indefinies.

    Retourne (sigma_grand_axe, sigma_petit_axe, angle_du_grand_axe).
    """
    import torch

    a, b, c = P[:, 0, 0], P[:, 0, 1], P[:, 1, 1]
    tr = a + c
    det = a * c - b * b
    r = torch.sqrt(torch.clamp(((a - c) / 2) ** 2 + b * b, min=0.0))
    lmax = tr / 2 + r
    # det / lmax evite l'annulation catastrophique de tr/2 - r quand la petite
    # valeur propre est tres inferieure a la grande.
    sur = lmax.abs() > eps
    lmin = torch.where(sur, det / torch.where(sur, lmax, torch.ones_like(lmax)),
                       tr / 2 - r)
    # Le vecteur propre se calcule avec la valeur propre BRUTE : l'ecretage
    # ci-dessous ne sert qu'aux ecarts-types, l'appliquer ici fausserait la
    # direction sur les matrices mal conditionnees.
    hb = b.abs() > eps
    un, zero = torch.ones_like(a), torch.zeros_like(a)
    vx = torch.where(hb, b, torch.where(a <= c, un, zero))
    vy = torch.where(hb, lmin - a, torch.where(a <= c, zero, un))
    # petite valeur propre <-> grand ecart-type
    return (torch.clamp(lmin, min=eps).rsqrt(),
            torch.clamp(lmax, min=eps).rsqrt(),
            torch.atan2(vy, vx))


def sample_field(preds, gx, gy, WA, HA, WB, HB, model, device):
    """Lit le champ dense aux points frontaux (gx, gy) par interpolation bilineaire.

    Retourne (xy_B, cert, sigma1, sigma2, theta) :
        xy_B    (N,2) pixels OpenCV dans l'image B
        cert    (N,)  overlap dans [0,1]
        sigma1  (N,)  ecart-type px selon le grand axe
        sigma2  (N,)  ecart-type px selon le petit axe
        theta   (N,)  orientation rad du grand axe
    """
    import torch
    import torch.nn.functional as TF

    warp = preds["warp_AB"]                       # (1, Hf, Wf, 2), normalise dans B
    over = preds["overlap_AB"]                    # (1, Hf, Wf, 1)
    prec = preds["precision_AB"]                  # (1, Hf, Wf, 2, 2), px^-2 a la
    Hf, Wf = warp.shape[1], warp.shape[2]         #   resolution de travail du modele

    un, vn = px_to_norm(gx, gy, WA, HA)
    grid = torch.from_numpy(np.stack([un, vn], axis=-1).astype(np.float32))
    grid = grid[None, :, None, :].to(device)      # (1, N, 1, 2)

    def gs(t, C):
        out = TF.grid_sample(t.permute(0, 3, 1, 2).float(), grid,
                             mode="bilinear", align_corners=False,
                             padding_mode="border")
        return out[0, :, :, 0].T.contiguous()     # (N, C)

    w = gs(warp, 2)
    c = gs(over, 1)[:, 0]

    # Deux invalidations indispensables, sans lesquelles des points faux
    # passent pour valides :
    #  (a) point frontal hors du domaine interpolable du champ. Avec
    #      align_corners=False le champ n'est defini que pour |n| <= 1 - 1/N ;
    #      au-dela grid_sample extrapole (ou renvoie 0 en padding 'zeros'), ce
    #      qui produit une correspondance plausible mais totalement fausse.
    #      RoMa applique la meme condition dans sample().
    #  (b) correspondance tombant hors de l'image B.
    qx = torch.from_numpy(np.asarray(un, np.float32)).to(device)
    qy = torch.from_numpy(np.asarray(vn, np.float32)).to(device)
    inside_A = (qx.abs() <= 1 - 1.0 / Wf) & (qy.abs() <= 1 - 1.0 / Hf)
    inside_B = (w[:, 0].abs() < 1.0) & (w[:, 1].abs() < 1.0)
    valid = inside_A & inside_B
    c = torch.where(valid, c, torch.zeros_like(c))

    # precision : exprimee en px^-2 a la resolution interne du modele (Hf, Wf).
    # On la ramene a la resolution de l'image B de travail.
    P = prec.reshape(1, Hf, Wf, 4)
    P = gs(P, 4).reshape(-1, 2, 2)
    P = model.prec_map_coordinates(P, H_in=Hf, W_in=Wf, H_out=HB, W_out=WB)

    # covariance = inverse de la precision ; valeurs propres -> ecarts-types
    P = P.double()
    P = 0.5 * (P + P.transpose(-1, -2))                      # symetrisation
    sigma_big, sigma_small, theta = _eig2x2(P)

    wn = w.detach().cpu().numpy().astype(np.float64)
    xB, yB = norm_to_px(wn[:, 0], wn[:, 1], WB, HB)
    bad = ~valid.detach().cpu().numpy()
    xB[bad] = np.nan
    yB[bad] = np.nan
    return (np.stack([xB, yB], axis=-1).astype(np.float32),
            c.detach().cpu().numpy().astype(np.float32),
            sigma_big.cpu().numpy().astype(np.float32),
            sigma_small.cpu().numpy().astype(np.float32),
            theta.cpu().numpy().astype(np.float32))


# ---------------------------------------------------------------------------
# Appariement d'une paire
# ---------------------------------------------------------------------------
def to_rgb(bgr):
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def tile_origins(extent, tile, stride):
    """Positions de tuiles couvrant [0, extent] jusqu'au bord inclus."""
    if extent <= tile:
        return [0]
    pos = list(range(0, extent - tile + 1, stride))
    if pos[-1] != extent - tile:
        pos.append(extent - tile)
    return pos


def match_pair(model, device, A, B, GX, GY, args):
    """Passe grossiere pleine image, puis (mode tiled) passe par tuiles guidee.

    Le champ RoMa est produit a la resolution carree interne du modele
    (1280x1280 en reglage 'precise'). Sur une image de travail a 4000 px de
    cote long, cela sous-echantillonne d'un facteur ~3 : le tuilage restitue la
    resolution native, chaque tuile etant traitee a la pleine resolution du
    modele.

    Les deux passes sont conservees SEPAREMENT :
      - la passe pleine image sert de guide, et de repli la ou aucune tuile
        n'a produit de resultat ;
      - `spread` ne mesure que le desaccord ENTRE TUILES, jamais tuile contre
        passe grossiere (qui refleterait la difference de resolution, pas une
        erreur d'appariement).
    `src` vaut 0 (passe pleine image) ou 1 (tuiles).
    """
    import torch

    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]
    N = len(GX)

    # --- passe 1 : pleine image ---------------------------------------------
    t0 = time.time()
    preds = model.match(to_rgb(A), to_rgb(B))
    xy0, c0, s10, s20, th0 = sample_field(preds, GX, GY, WA, HA, WB, HB, model, device)
    del preds
    if device == "cuda":
        torch.cuda.empty_cache()
    ok0 = np.isfinite(xy0[:, 0]) & (c0 > args.cert_min)
    xy0[~ok0] = np.nan
    c0 = np.where(ok0, c0, 0.0).astype(np.float32)
    print(f"    passe pleine image : {int(ok0.sum())}/{N} points "
          f"(cert > {args.cert_min}) en {time.time()-t0:.1f}s")

    if args.mode == "full":
        return (xy0, c0, s10, s20, th0,
                np.full(N, np.nan, np.float32), np.zeros(N, np.int8), 0)

    # --- passe 2 : tuiles guidees par la passe 1 -----------------------------
    XY = np.full((N, 2), np.nan, np.float32)
    CERT = np.zeros(N, np.float32)
    S1 = np.full(N, np.nan, np.float32)
    S2 = np.full(N, np.nan, np.float32)
    TH = np.full(N, np.nan, np.float32)
    SPREAD = np.full(N, np.nan, np.float32)
    NVOTE = np.zeros(N, np.int16)

    T = args.tile
    origins = [(x0, y0) for y0 in tile_origins(HA, T, args.stride)
               for x0 in tile_origins(WA, T, args.stride)]
    ntile = 0
    t0 = time.time()
    print(f"    {len(origins)} positions de tuile ({args.tile} px, pas {args.stride})")
    for ti, (x0, y0) in enumerate(origins):
        x1, y1 = min(WA, x0 + T), min(HA, y0 + T)
        sel = (GX >= x0) & (GX < x1) & (GY >= y0) & (GY < y1)
        if sel.sum() < args.min_pts:
            continue
        g = xy0[sel]
        gok = np.isfinite(g[:, 0])
        if gok.sum() < args.min_pts:
            continue
        bx0 = max(0, int(np.percentile(g[gok, 0], 2)) - args.margin)
        bx1 = min(WB, int(np.percentile(g[gok, 0], 98)) + args.margin)
        by0 = max(0, int(np.percentile(g[gok, 1], 2)) - args.margin)
        by1 = min(HB, int(np.percentile(g[gok, 1], 98)) + args.margin)
        if bx1 - bx0 < 64 or by1 - by0 < 64:
            continue

        ca = A[y0:y1, x0:x1]
        cb = B[by0:by1, bx0:bx1]
        try:
            preds = model.match(to_rgb(ca), to_rgb(cb))
        except Exception as e:
            print(f"    tuile ({x0},{y0}) ignoree : {type(e).__name__}")
            continue
        ntile += 1
        el = time.time() - t0
        print(f"      tuile {ntile:3d} en ({x0:5d},{y0:5d}) | {int(sel.sum()):7d} points "
              f"| fenetre {bx1-bx0}x{by1-by0} | {el:5.1f}s", flush=True)

        idxs = np.nonzero(sel)[0]
        xy, c, s1, s2, th = sample_field(
            preds, GX[sel] - x0, GY[sel] - y0,
            ca.shape[1], ca.shape[0], cb.shape[1], cb.shape[0], model, device)
        del preds
        if device == "cuda" and ti % 8 == 7:
            torch.cuda.empty_cache()
        xy[:, 0] += bx0
        xy[:, 1] += by0

        valid = np.isfinite(xy[:, 0]) & (c > args.cert_min)
        # desaccord avec une AUTRE tuile ayant deja vote sur le meme pixel
        seen = valid & (NVOTE[idxs] > 0)
        if seen.any():
            prev = XY[idxs[seen]]
            d = np.hypot(xy[seen, 0] - prev[:, 0], xy[seen, 1] - prev[:, 1])
            tgt = idxs[seen]
            SPREAD[tgt] = np.fmax(np.where(np.isnan(SPREAD[tgt]), 0.0, SPREAD[tgt]), d)
        NVOTE[idxs[valid]] += 1

        better = valid & (c > CERT[idxs])
        tgt = idxs[better]
        XY[tgt] = xy[better]
        CERT[tgt] = c[better]
        S1[tgt], S2[tgt], TH[tgt] = s1[better], s2[better], th[better]

    SRC = np.ones(N, np.int8)
    hole = ~np.isfinite(XY[:, 0]) & np.isfinite(xy0[:, 0])
    XY[hole] = xy0[hole]
    CERT[hole] = c0[hole]
    S1[hole], S2[hole], TH[hole] = s10[hole], s20[hole], th0[hole]
    SRC[hole] = 0
    SRC[~np.isfinite(XY[:, 0])] = -1

    print(f"    passe par tuiles : {ntile}/{len(origins)} tuiles en {time.time()-t0:.1f}s | "
          f"{int(np.isfinite(XY[:,0]).sum())}/{N} points retenus "
          f"(dont {int(hole.sum())} repris de la passe pleine image)")
    return XY, CERT, S1, S2, TH, SPREAD, SRC, ntile


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--img-dir", default=os.environ.get("VISIA_IMG_DIR", "."))
    ap.add_argument("--modality", default="Standard_1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--long", type=int, default=4000,
                    help="cote long de l'image de travail ; DOIT valoir config.WORK_LONG")
    ap.add_argument("--step", type=int, default=2,
                    help="pas de la grille frontale exportee (1 = tous les pixels)")
    ap.add_argument("--mode", choices=["full", "tiled"], default="tiled")
    ap.add_argument("--setting", default="precise",
                    choices=["precise", "base", "fast", "turbo"])
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--margin", type=int, default=96,
                    help="marge (px) autour de la fenetre predite dans l'oblique")
    ap.add_argument("--min-pts", type=int, default=200)
    ap.add_argument("--hr", type=int, default=None,
                    help="force la resolution HAUTE du modele (defaut 1280 en "
                         "reglage 'precise'). Doit etre un multiple de 4 : les "
                         "raffineurs travaillent a des patchs 4, 2 et 1. "
                         "NON VALIDE au-dela de 1280, valeur d'entrainement — "
                         "mesurer le residu avant de conclure.")
    ap.add_argument("--lr", type=int, default=None,
                    help="force la resolution BASSE du modele (defaut 800). "
                         "Doit etre un multiple de 16, taille de patch de DINOv3.")
    ap.add_argument("--cert-min", type=float, default=0.02,
                    help="seuil d'overlap en-dessous duquel le point n'est pas exporte")
    ap.add_argument("--skin-top", type=float, default=None,
                    help="borne HAUTE de la bande exportee, en fraction de la "
                         "hauteur d'image (defaut config.SKIN_TOP = 0.28). "
                         "La baisser exporte le front. C'est ici qu'il faut "
                         "agir : le pipeline ne peut pas reconstruire ce qui "
                         "n'a pas ete exporte.")
    ap.add_argument("--skin-bottom", type=float, default=None,
                    help="borne BASSE (defaut config.SKIN_BOTTOM = 0.88). "
                         "La monter exporte le menton et la machoire.")
    ap.add_argument("--mask-margin", type=int, default=40,
                    help="dilatation du masque facial a l'export")
    ap.add_argument("--pairs", default="LR")
    a = ap.parse_args()

    import torch
    from romav2 import RoMaV2
    torch.set_float32_matmul_precision("highest")      # exige par RoMaV2.forward
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ATTENTION : aucun GPU detecte, le reglage 'precise' sera tres lent.")

    print(f"[RoMa v2] chargement du modele (reglage '{a.setting}') sur {device}")
    model = RoMaV2()
    model.apply_setting(a.setting)
    if a.hr is not None:
        if a.hr % 4:
            raise SystemExit(f"--hr {a.hr} n'est pas multiple de 4 "
                             f"(patchs des raffineurs : 4, 2, 1)")
        model.H_hr = model.W_hr = a.hr
    if a.lr is not None:
        if a.lr % 16:
            raise SystemExit(f"--lr {a.lr} n'est pas multiple de 16 "
                             f"(taille de patch de DINOv3)")
        model.H_lr = model.W_lr = a.lr
    if a.hr is not None or a.lr is not None:
        print(f"[RoMa v2] resolutions FORCEES : lr={model.H_lr} hr={model.H_hr}")
        print("   Le modele a ete entraine a 1280 : au-dela, le gain n'est pas")
        print("   garanti. Comparer le residu epipolaire avant et apres.")
    print(f"[RoMa v2] resolution interne lr={model.H_lr}x{model.W_lr} "
          f"hr={model.H_hr}x{model.W_hr} bidirectionnel={model.bidirectional}")

    paths = {k: img_path(a.img_dir, a.subject, a.session, k, a.modality)
             for k in "F" + a.pairs}
    for k, p in paths.items():
        ok, msg = check_file(p)
        print(f"  {k}: {os.path.basename(p)} -> {msg}")
        if not ok:
            raise SystemExit("fichier invalide — corriger l'export VISIA avant de continuer")

    A = load(paths["F"], a.long)
    HA, WA = A.shape[:2]
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    fm = face_mask(gA, margin=a.mask_margin,
               top=a.skin_top, bottom=a.skin_bottom)

    gy, gx = np.mgrid[0:HA:a.step, 0:WA:a.step]
    keep = fm[gy, gx]
    GX = gx[keep].astype(np.float64)
    GY = gy[keep].astype(np.float64)
    idx = (gy[keep].astype(np.int64) * WA + gx[keep].astype(np.int64)).astype(np.int32)
    octets = len(GX) * (4 + 2 * (8 + 2 + 4 + 2 + 2 + 1))
    print(f"\ngrille frontale : {len(GX)} points "
          f"(image {WA}x{HA}, pas {a.step}, masque dilate de {a.mask_margin} px)")
    print(f"taille du fichier attendue : ~{octets/1e6:.0f} Mo avant compression"
          + ("   (--step 2 le diviserait par 4)" if a.step == 1 else ""))
    if a.mode == "tiled":
        print(f"champ a {model.W_hr or model.W_lr} px par tuile de {a.tile} px, soit "
              f"{a.tile/(model.W_hr or model.W_lr):.2f} px de travail par cellule "
              f"(contre {max(WA,HA)/(model.W_hr or model.W_lr):.1f} sans tuilage)")

    out = {"idx": idx}
    stats = {}
    for k in a.pairs:
        B = load(paths[k], a.long)
        print(f"\n-- paire F<->{k}  ({B.shape[1]}x{B.shape[0]})")
        XY, CERT, S1, S2, TH, SPREAD, SRC, nt = match_pair(model, device, A, B, GX, GY, a)
        good = np.isfinite(XY[:, 0])
        out[f"{k}_xy"] = XY
        out[f"{k}_cert"] = CERT.astype(np.float16)
        out[f"{k}_sigma"] = np.stack([S1, S2], -1).astype(np.float16)
        out[f"{k}_theta"] = TH.astype(np.float16)
        out[f"{k}_spread"] = SPREAD.astype(np.float16)
        out[f"{k}_src"] = SRC
        out[f"{k}_shape"] = np.array([B.shape[0], B.shape[1]], np.int32)
        stats[k] = {
            "points": int(good.sum()),
            "couverture_grille_%": round(100.0 * good.sum() / len(GX), 1),
            "cert_mediane": round(float(np.median(CERT[good])), 3) if good.any() else None,
            "sigma_median_px": round(float(np.nanmedian(S1[good])), 2) if good.any() else None,
            "spread_median_px": (round(float(np.nanmedian(SPREAD[good])), 2)
                                 if np.isfinite(SPREAD[good]).any() else None),
            "tuiles": nt,
            "points_repris_passe_grossiere": int((SRC == 0).sum()),
        }
        print(f"    cert mediane {stats[k]['cert_mediane']} | "
              f"sigma median {stats[k]['sigma_median_px']} px | "
              f"desaccord inter-tuiles median {stats[k]['spread_median_px']} px")
        del B

    meta = {
        "format": EXPORT_FORMAT,
        "subject": a.subject, "session": a.session, "modality": a.modality,
        "pairs": a.pairs,
        "work_long": a.long, "step": a.step, "mode": a.mode, "setting": a.setting,
        "tile": a.tile, "stride": a.stride, "margin": a.margin,
        "cert_min": a.cert_min, "mask_margin": a.mask_margin,
        "skin_top": a.skin_top if a.skin_top is not None else SKIN_TOP,
        "skin_bottom": a.skin_bottom if a.skin_bottom is not None else SKIN_BOTTOM,
        "grid_shape": [int(HA), int(WA)],
        "coord_convention": "OpenCV pixel centers (integer), bilinear field readout",
        "loader": "cv2-array",
        "model": {
            "name": "RoMa v2", "setting": a.setting,
            "hr_force": a.hr, "lr_force": a.lr,
            "H_lr": model.H_lr, "W_lr": model.W_lr,
            "H_hr": model.H_hr, "W_hr": model.W_hr,
            "bidirectional": bool(model.bidirectional),
        },
        "sources": {k: {"file": os.path.basename(p), "sha": sha256(p),
                        "bytes": os.path.getsize(p)} for k, p in paths.items()},
        "env": {"host": platform.node(), "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "stats": stats,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out["meta"] = np.array(json.dumps(meta, indent=1))

    dest = a.out or f"roma2_{a.subject}_{a.session}_{a.modality}.npz"
    np.savez_compressed(dest, **out)
    mb = os.path.getsize(dest) / 1e6
    print(f"\n-> {dest}  ({mb:.1f} Mo)")
    print("Telecharger ce fichier, puis cote pipeline :")
    print(f"   python run_session.py {a.subject} {a.session} --rig rig.json --roma {dest}")


if __name__ == "__main__":
    main()

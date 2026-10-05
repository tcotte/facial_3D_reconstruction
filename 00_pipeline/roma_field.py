"""
roma_field.py — relecture du champ RoMa v2 exporte, cote pipeline.

OBJET
    Transforme le fichier produit sur la machine GPU par
    03_appariement/roma_v2_export.py en la MEME structure que renvoie
    visia_core.densify() : (obx, oby, epi, wmask). C'est un remplacement
    direct de l'etape de densification, le reste de la chaine est inchange.

CE QUI CHANGE, CE QUI NE CHANGE PAS
    Change     : la source des correspondances denses. A la place du
                 pre-recalage Delaunay + flot optique DIS, on lit le champ
                 RoMa v2 exporte.
    Ne change pas : le rig GELE, la focale, la distorsion, le point principal,
                 les seuils EPI_HIGH / EPI_MED / TEXTURE_MIN, le masque
                 facial, la triangulation, le filtrage des aberrants.

    Le rig reste estime par DISK + LightGlue (run_calibrate.py). C'est
    deliberé : l'evaluation menee dans 03_appariement/roma_matching.py a
    montre qu'un champ dense localise moins bien qu'un detecteur de points
    pour la calibration. RoMa densifie, LightGlue calibre.

LE RESIDU EPIPOLAIRE RESTE UNE VALIDATION INDEPENDANTE
    Par defaut, la matrice fondamentale utilisee pour calculer `epi` est
    DERIVEE DU RIG GELE :

        F = K^-T [t]x R K^-1

    Elle ne doit donc rien a RoMa. C'est plus fort que la chaine actuelle, qui
    re-estime F a chaque session sur sa propre passe grossiere : le residu y
    est en partie auto-valide. Ici, un residu eleve signale reellement quelque
    chose — derive du banc, ou mouvement du sujet entre les trois prises.

    `--epi-from field` re-estime F sur le champ lui-meme (comportement proche
    de l'ancien), utile uniquement en diagnostic comparatif.

ECHANTILLONNAGE
    L'export se fait sur une grille frontale de pas `step`. Par defaut ce
    module NE COMBLE PAS les pixels intermediaires : les points reconstruits
    sont exactement ceux qui ont ete apparies. L'option `fill=True` interpole
    le champ (bilineaire) sur tous les pixels du masque. C'est defendable — le
    champ RoMa est de toute facon continu et produit en interne a 1280 px —
    mais cela gonfle le nombre de points sans ajouter d'information
    independante. A ne pas utiliser pour un comptage d'appariements.

STAMP DE VERSION
    Une session reconstruite avec RoMa porte le stamp CALIB_VERSION + '+roma2'.
    run_compare.py refuse alors de la comparer a une session reconstruite
    autrement — c'est voulu : changer la densification change les valeurs, et
    les regles de l'etude interdisent de melanger deux chaines de traitement.

USAGE
    # inspecter un fichier telecharge
    python roma_field.py --info roma2_alban_D0_Standard_1.npz

    # dans le pipeline
    python run_session.py alban D0 --rig rig.json --step 1 \
        --roma roma2_alban_D0_Standard_1.npz
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

import config as C

EXPECTED_FORMAT = "visia-roma2-field/1"
ROMA_STAMP = "+roma2"


# ---------------------------------------------------------------------------
def fundamental_from_rig(K, R, t):
    """F telle que x_oblique^T F x_frontal = 0, derivee du rig GELE.

    Meme convention que visia_core.triangulate : P1 = K[I|0], P2 = K[R|t].
    """
    t = np.asarray(t, float).ravel()
    tx = np.array([[0, -t[2], t[1]],
                   [t[2], 0, -t[0]],
                   [-t[1], t[0], 0]], float)
    E = tx @ np.asarray(R, float)
    Ki = np.linalg.inv(np.asarray(K, float))
    F = Ki.T @ E @ Ki
    return F / (np.linalg.norm(F) + 1e-30)


def epipolar_residual(F, x1, y1, x2, y2):
    """Distance point-droite symetrisee, en pixels."""
    p1 = np.column_stack([x1, y1, np.ones_like(x1)])
    p2 = np.column_stack([x2, y2, np.ones_like(x2)])
    l2 = p1 @ F.T                       # droite epipolaire dans l'image 2
    l1 = p2 @ F                         # droite epipolaire dans l'image 1
    num = np.abs(np.sum(l2 * p2, axis=1))
    d2 = num / np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + 1e-12)
    d1 = num / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-12)
    return 0.5 * (d1 + d2)


# ---------------------------------------------------------------------------
class RomaField:
    """Champ RoMa v2 exporte pour une session (une modalite, une ou deux paires)."""

    TYPES = [
        (("Z", "Zall", "nch", "face"), "une FUSION multi-modalites "
         "(sortie de fuse_modalities.py)"),
        (("Z", "keep", "rej", "face"), "une fusion FILTREE "
         "(sortie de outlier_filter.py)"),
        (("pix", "X", "conf", "src"), "une SESSION reconstruite "
         "(sortie de run_session.py)"),
    ]

    def __init__(self, path):
        self.path = path
        d = np.load(path, allow_pickle=False)
        if "meta" not in d.files:
            quoi = None
            for cles, nom in self.TYPES:
                if all(c in d.files for c in cles):
                    quoi = nom
                    break
            msg = [f"{os.path.basename(path)} n'est pas un champ RoMa exporte : "
                   f"la cle 'meta' est absente."]
            if quoi:
                msg.append(f"Ce fichier est {quoi}.")
            else:
                msg.append(f"Cles presentes : {', '.join(sorted(d.files)[:10])}")
            msg.append("Attendu ici : un fichier roma2_<sujet>_<session>_"
                       "<modalite>.npz produit par 03_appariement/"
                       "roma_v2_simple.py ou roma_v2_export.py.")
            if quoi and "FUSION" in quoi.upper():
                msg.append("Pour analyser une fusion, voir plutot "
                           "04_validation/coverage_by_zone.py --fusion "
                           "ou export_pointcloud.py --fusion.")
            raise RuntimeError("\n  ".join(msg))
        self.meta = json.loads(str(d["meta"]))
        if self.meta.get("format") != EXPECTED_FORMAT:
            raise RuntimeError(
                f"{path} : format '{self.meta.get('format')}' inattendu "
                f"(attendu '{EXPECTED_FORMAT}'). Re-exporter avec la version "
                f"courante de roma_v2_export.py.")
        self.idx = d["idx"].astype(np.int64)
        self.H, self.W = (int(v) for v in self.meta["grid_shape"])
        self.step = int(self.meta["step"])
        self.pairs = str(self.meta["pairs"])
        self._d = {k: d[k] for k in d.files if k != "meta"}

    # -- controles ----------------------------------------------------------
    def check(self, subject, session, shape, modality=None, strict=False):
        """Verifie que le fichier correspond bien a ce que le pipeline va traiter."""
        problems, warnings = [], []
        m = self.meta
        if m["subject"] != subject or m["session"] != session:
            problems.append(f"fichier pour {m['subject']}/{m['session']}, "
                            f"pipeline sur {subject}/{session}")
        if int(m["work_long"]) != int(C.WORK_LONG):
            problems.append(f"cote long {m['work_long']} != config.WORK_LONG "
                            f"{C.WORK_LONG} : les coordonnees ne correspondent pas")
        if tuple(int(v) for v in m["grid_shape"]) != tuple(int(v) for v in shape):
            problems.append(f"image frontale {tuple(m['grid_shape'])} != "
                            f"{tuple(shape)} localement")
        mod = modality or C.MODALITY_GEOM
        if m["modality"] != mod:
            warnings.append(f"modalite {m['modality']} != {mod}")
        if m.get("loader") != "cv2-array":
            warnings.append(
                "champ exporte AVANT le correctif d'orientation (loader non "
                "renseigne). Si le residu epipolaire est de l'ordre de la "
                "centaine de pixels, lancer "
                "04_validation/diagnose_field_frame.py")
        if problems:
            raise RuntimeError(f"{os.path.basename(self.path)} incompatible :\n  - "
                               + "\n  - ".join(problems))
        for w in warnings:
            print(f"   ATTENTION : {w}")
        if strict:
            self.check_sources(subject, session, mod)
        return True

    def check_sources(self, subject, session, modality):
        """Compare l'empreinte des jpg utilises a distance et de ceux presents ici."""
        import visia_core as vc
        for k, info in self.meta["sources"].items():
            p = vc.img_path(subject, session, k, modality)
            if not os.path.exists(p):
                print(f"   ATTENTION : {os.path.basename(p)} absent localement")
                continue
            if os.path.getsize(p) != info["bytes"]:
                raise RuntimeError(
                    f"{os.path.basename(p)} : taille {os.path.getsize(p)} != "
                    f"{info['bytes']} sur la machine GPU. Les images ne sont pas "
                    f"les memes, le champ ne s'applique pas a celles-ci.")

    # -- reconstruction des cartes -----------------------------------------
    def _scatter(self, values, fill_value=np.nan):
        """Points exportes -> carte pleine resolution (NaN ailleurs)."""
        n = values.shape[1] if values.ndim == 2 else 1
        out = np.full((self.H * self.W, n), fill_value, np.float32)
        out[self.idx] = values.reshape(-1, n).astype(np.float32)
        return out.reshape(self.H, self.W, n) if n > 1 else out.reshape(self.H, self.W)

    def _subgrid(self, values):
        """Points exportes -> grille de pas `step` (pour interpolation)."""
        Hs = (self.H + self.step - 1) // self.step
        Ws = (self.W + self.step - 1) // self.step
        n = values.shape[1] if values.ndim == 2 else 1
        out = np.full((Hs * Ws, n), np.nan, np.float32)
        ys, xs = self.idx // self.W, self.idx % self.W
        out[(ys // self.step) * Ws + (xs // self.step)] = values.reshape(-1, n)
        return out.reshape(Hs, Ws, n)

    def _upsample(self, values):
        """Interpolation bilineaire du champ sur tous les pixels, trous exclus.

        cv2.remap avec les coordonnees explicites x/step : la mise a l'echelle
        implicite de cv2.resize introduirait un decalage systematique de
        (0.5/step - 0.5) pixel de sous-grille.
        """
        sub = self._subgrid(values)
        Hs, Ws, n = sub.shape
        valid = np.isfinite(sub[..., 0]).astype(np.float32)
        filled = np.where(np.isfinite(sub), sub, 0.0).astype(np.float32)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        mx, my = xx / self.step, yy / self.step
        num = cv2.remap(filled, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        den = cv2.remap(valid, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if num.ndim == 2:
            num = num[..., None]
        # 0.999 : on refuse tout pixel dont le support d'interpolation touche un
        # trou, pour ne pas faire deborder le champ hors du domaine apparie.
        ok = den > 0.999
        out = np.full((self.H, self.W, n), np.nan, np.float32)
        out[ok] = num[ok] / den[ok][:, None]
        return out if n > 1 else out[..., 0]

    # -- interface pipeline -------------------------------------------------
    def densify_like(self, pair, K, R, t, fill=False, epi_from="rig",
                     cert_min=0.05, sigma_max=None, spread_max=None):
        """Equivalent de visia_core.densify() -> (obx, oby, epi, wmask).

        cert_min    seuil sur l'overlap RoMa (defaut 0.05, comme le seuil de
                    certitude retenu dans roma_matching.py)
        sigma_max   rejet sur l'ecart-type de localisation predit (px), None = inactif
        spread_max  rejet sur le desaccord inter-tuiles (px), None = inactif
        """
        if pair not in self.pairs:
            raise KeyError(f"paire {pair} absente de {os.path.basename(self.path)} "
                           f"(contient '{self.pairs}')")
        xy = self._d[f"{pair}_xy"].astype(np.float32)
        cert = self._d[f"{pair}_cert"].astype(np.float32)
        sig = self._d[f"{pair}_sigma"].astype(np.float32)[:, 0]
        spread = self._d[f"{pair}_spread"].astype(np.float32)

        keep = np.isfinite(xy[:, 0]) & (cert > cert_min)
        if sigma_max is not None:
            keep &= np.isfinite(sig) & (sig < sigma_max)
        if spread_max is not None:
            keep &= ~(np.isfinite(spread) & (spread > spread_max))
        xy = np.where(keep[:, None], xy, np.nan)

        if fill:
            ob = self._upsample(xy)
        else:
            ob = self._scatter(xy)
        obx, oby = ob[..., 0].copy(), ob[..., 1].copy()

        ok = np.isfinite(obx) & np.isfinite(oby)
        ys, xs = np.nonzero(ok)
        epi = np.full((self.H, self.W), np.inf, np.float32)
        if len(ys):
            if epi_from == "rig":
                F = fundamental_from_rig(K, R, t)
            elif epi_from == "field":
                Fm, _ = cv2.findFundamentalMat(
                    np.column_stack([xs, ys]).astype(np.float64),
                    np.column_stack([obx[ys, xs], oby[ys, xs]]).astype(np.float64),
                    cv2.FM_RANSAC, C.RANSAC_F_THRESH, 0.999, maxIters=50000)
                if Fm is None:
                    raise RuntimeError("estimation de F sur le champ RoMa echouee")
                F = Fm[:3] / (np.linalg.norm(Fm[:3]) + 1e-30)
            else:
                raise ValueError(epi_from)
            epi[ys, xs] = epipolar_residual(
                F, xs.astype(float), ys.astype(float),
                obx[ys, xs].astype(float), oby[ys, xs].astype(float)).astype(np.float32)
        return obx, oby, epi, ok

    def plancher(self, pair, cert_min=0.5, nmax=40000):
        """Residu median de la MEILLEURE matrice fondamentale possible.

        On re-estime F sur le champ lui-meme : aucun rig ne peut faire mieux,
        c'est la coherence interne des correspondances. Un plancher eleve
        signifie que le champ n'est pas explicable par UNE geometrie
        epipolaire — scene non rigide entre les trois prises.

        Sert a exprimer le seuil epipolaire RELATIVEMENT au bruit du sujet,
        au lieu d'une constante en pixels qui n'a pas le meme sens d'un sujet
        a l'autre : mesure sur deux sujets, 0,20 px contre 1,70 px.
        """
        xy = self._d[f"{pair}_xy"].astype(np.float64)
        cert = self._d[f"{pair}_cert"].astype(np.float64)
        m = np.isfinite(xy[:, 0]) & (cert > cert_min)
        if m.sum() < 500:
            return None
        ys, xs = self.idx // self.W, self.idx % self.W
        sel = np.nonzero(m)[0]
        if len(sel) > nmax:
            sel = sel[np.linspace(0, len(sel) - 1, nmax).astype(int)]
        F, _ = cv2.findFundamentalMat(
            np.column_stack([xs[sel], ys[sel]]).astype(np.float64),
            xy[sel], cv2.USAC_MAGSAC, 1.0, 0.9999, 100000)
        if F is None:
            return None
        F = F[:3] / (np.linalg.norm(F[:3]) + 1e-30)
        e = epipolar_residual(F, xs[m].astype(float), ys[m].astype(float),
                              xy[m, 0], xy[m, 1])
        e = e[np.isfinite(e)]
        return float(np.median(e)) if len(e) else None

    def sigma_map(self, pair, fill=False):
        s = self._d[f"{pair}_sigma"].astype(np.float32)
        return self._upsample(s) if fill else self._scatter(s)

    def cert_map(self, pair, fill=False):
        c = self._d[f"{pair}_cert"].astype(np.float32)
        return self._upsample(c) if fill else self._scatter(c)

    # -- rapport ------------------------------------------------------------
    def report(self):
        m = self.meta
        print(f"fichier    : {os.path.basename(self.path)} "
              f"({os.path.getsize(self.path)/1e6:.1f} Mo)")
        print(f"sujet      : {m['subject']} / {m['session']} / {m['modality']}")
        print(f"exporte le : {m['date']}  sur {m['env']['host']} "
              f"({m['env'].get('gpu') or 'CPU'})")
        print(f"modele     : {m['model']['name']} reglage '{m['setting']}' "
              f"hr={m['model']['H_hr']}x{m['model']['W_hr']} "
              f"bidirectionnel={m['model']['bidirectional']}")
        print(f"grille     : {self.W}x{self.H}, pas {self.step}, "
              f"{len(self.idx)} points exportes")
        if m.get("tile"):
            print(f"mode       : {m['mode']} (tuile {m['tile']}, pas {m['stride']}, "
                  f"marge {m['margin']})")
        else:
            px = self.W / max(1, int(m["model"]["W_hr"] or m["model"]["W_lr"]))
            print(f"mode       : {m['mode']} (sans tuilage — champ "
                  f"{m['model']['W_hr'] or m['model']['W_lr']} px, soit ~{px:.1f} px "
                  f"de travail par cellule)")
        print(f"compatible : work_long={m['work_long']} "
              f"(config={C.WORK_LONG}) {'OK' if int(m['work_long'])==C.WORK_LONG else 'INCOMPATIBLE'}")
        for k, s in m["stats"].items():
            print(f"  paire F<->{k} : {s['points']} points "
                  f"({s['couverture_grille_%']} % de la grille) | "
                  f"cert med {s['cert_mediane']} | sigma med {s['sigma_median_px']} px"
                  + (f" | desaccord inter-tuiles med {s['spread_median_px']} px"
                     if s.get("spread_median_px") is not None else ""))
        print("\nRappel : une session reconstruite avec ce champ porte le stamp "
              f"'{C.CALIB_VERSION}{ROMA_STAMP}' et ne peut pas etre comparee "
              "a une session reconstruite autrement.")


def load(path):
    return RomaField(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True, help="fichier .npz exporte")
    a = ap.parse_args()
    RomaField(a.info).report()

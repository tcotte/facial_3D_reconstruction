"""
visia_core.py — briques communes : E/S, appariement, densification, reconstruction.

Chaine par session :
    images (3 angles) -> appariement par tuiles -> pre-recalage par morceaux
    -> flot optique residuel -> validation epipolaire -> triangulation
    -> nuage 3D + masque de confiance

Le "rig" (poses relatives des 3 positions de camera) est une propriete du banc,
pas du sujet : il est estime une fois (run_calibrate.py) puis GELE et reutilise
pour toutes les sessions. Cela supprime une source majeure de variabilite
inter-session.
"""

import os, json
import numpy as np
import cv2
from scipy.spatial import Delaunay
import config as C

# --------------------------------------------------------------------------
# Entrees / sorties
# --------------------------------------------------------------------------
BLANK_SIZE = 251258     # fichier blanc de substitution produit par l'export VISIA

def img_path(subject, session, angle_key, modality=None):
    modality = modality or C.MODALITY_GEOM
    return os.path.join(C.IMG_DIR,
                        f"{subject}_{session}_{C.ANGLES[angle_key]}_{modality}.jpg")

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

def load(path, long_side=None):
    long_side = long_side or C.WORK_LONG
    im = cv2.imread(path)
    if im is None:
        raise IOError(path)
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def face_mask(gray):
    m = (gray > 45).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    band = np.zeros_like(m)
    h = gray.shape[0]
    band[int(C.SKIN_TOP * h):int(C.SKIN_BOTTOM * h), :] = 1
    return (m & band).astype(bool)

# --------------------------------------------------------------------------
# Appariement
# --------------------------------------------------------------------------
class Matcher:
    """DISK + LightGlue si disponibles, repli ASIFT sinon.

    ASIFT reste utilisable mais donne environ 5x moins d'appariements ; le
    repli est signale explicitement pour que ce ne soit jamais silencieux.
    """
    def __init__(self):
        self.backend = None
        try:
            import torch, kornia as K, kornia.feature as KF
            self.torch, self.K, self.KF = torch, K, KF
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.disk = KF.DISK.from_pretrained("depth").eval().to(self.dev)
            self.lg = KF.LightGlueMatcher("disk").eval().to(self.dev)
            self.backend = "disk+lightglue"
        except Exception as e:
            print(f"[Matcher] LightGlue indisponible ({type(e).__name__}) -> repli ASIFT")
            self.sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
            self.backend = "asift"
        print(f"[Matcher] backend = {self.backend}")

    def _prep(self, crop, cap=640):
        h, w = crop.shape[:2]
        s = min(1.0, cap / max(h, w))
        nh = max(16, int(round(h * s / 16)) * 16)
        nw = max(16, int(round(w * s / 16)) * 16)
        r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        t = self.K.color.bgr_to_rgb(self.K.image_to_tensor(r, False).float() / 255.)
        return t.to(self.dev), (w / nw, h / nh)

    def match_crops(self, ca, cb):
        if self.backend == "asift":
            return self._asift(ca, cb)
        ta, sa = self._prep(ca)
        tb, sb = self._prep(cb)
        with self.torch.inference_mode():
            fa = self.disk(ta, C.MAX_FEATURES, pad_if_not_divisible=True)[0]
            fb = self.disk(tb, C.MAX_FEATURES, pad_if_not_divisible=True)[0]
            if len(fa.keypoints) < 8 or len(fb.keypoints) < 8:
                return None
            la = self.KF.laf_from_center_scale_ori(fa.keypoints[None])
            lb = self.KF.laf_from_center_scale_ori(fb.keypoints[None])
            _, idx = self.lg(fa.descriptors, fb.descriptors, la, lb)
        if len(idx) < 8:
            return None
        i = idx.cpu().numpy()
        p0 = fa.keypoints.cpu().numpy()[i[:, 0]] * np.array(sa)
        p1 = fb.keypoints.cpu().numpy()[i[:, 1]] * np.array(sb)
        return p0, p1

    def _asift(self, ca, cb):
        ga = cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY)
        ka, da = self.sift.detectAndCompute(ga, None)
        kb, db = self.sift.detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [x for x, y in m if x.distance < 0.8 * y.distance]
        if len(good) < 8:
            return None
        return (np.float32([ka[x.queryIdx].pt for x in good]),
                np.float32([kb[x.trainIdx].pt for x in good]))


def match_pair(matcher, A, B, guide=None):
    """Appariement par tuiles. `guide` = (obx, oby, scale) issu d'une passe
    grossiere, pour predire la fenetre correspondante dans B."""
    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]
    fm = face_mask(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY))
    PA, PB = [], []
    T, S = C.TILE_SIZE, C.TILE_STRIDE
    for y0 in range(int(C.SKIN_TOP * HA), int(C.SKIN_BOTTOM * HA) - T, S):
        for x0 in range(0, WA - T, S):
            if fm[y0:y0 + T, x0:x0 + T].mean() < 0.40:
                continue
            if guide is None:
                cb = B                      # premiere passe : image entiere
                off = (0, 0)
            else:
                obx, oby, sc = guide
                gy0, gy1 = int(y0 / sc), int((y0 + T) / sc)
                gx0, gx1 = int(x0 / sc), int((x0 + T) / sc)
                ox, oy = obx[gy0:gy1, gx0:gx1], oby[gy0:gy1, gx0:gx1]
                ok = ~np.isnan(ox)
                if ok.sum() < 200:
                    continue
                bx0 = max(0, int(np.nanpercentile(ox[ok], 2) * sc) - 80)
                bx1 = min(WB, int(np.nanpercentile(ox[ok], 98) * sc) + 80)
                by0 = max(0, int(np.nanpercentile(oy[ok], 2) * sc) - 80)
                by1 = min(HB, int(np.nanpercentile(oy[ok], 98) * sc) + 80)
                if bx1 - bx0 < 64 or by1 - by0 < 64:
                    continue
                cb = B[by0:by1, bx0:bx1]
                off = (bx0, by0)
            r = matcher.match_crops(A[y0:y0 + T, x0:x0 + T], cb)
            if r is None:
                continue
            p0, p1 = r
            PA.append(p0 + np.array([x0, y0]))
            PB.append(p1 + np.array(off))
    if not PA:
        return None
    PA, PB = np.vstack(PA), np.vstack(PB)
    F, mask = cv2.findFundamentalMat(PA, PB, cv2.FM_RANSAC,
                                     C.RANSAC_F_THRESH, 0.999, maxIters=50000)
    if mask is None:
        return None
    inl = mask.ravel().astype(bool)
    return PA[inl], PB[inl], F

# --------------------------------------------------------------------------
# Densification
# --------------------------------------------------------------------------
def densify(A, B, pa, pb, F):
    """Pre-recalage par morceaux (Delaunay) + flot optique residuel.

    Retourne (obx, oby, epi) : pour chaque pixel du frontal, la position
    correspondante dans l'oblique et le residu epipolaire associe.
    Le residu epipolaire est une validation GEOMETRIQUE INDEPENDANTE du flot
    optique qui a produit la correspondance : c'est ce qui fonde la confiance.
    """
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(2.0, (8, 8))
    gAe, gBe = cl.apply(gA), cl.apply(gB)
    h, w = gA.shape
    tri = Delaunay(pa)

    warp = np.zeros_like(gBe)
    wmask = np.zeros((h, w), np.uint8)
    for s in tri.simplices:
        src, dst = pb[s].astype(np.float32), pa[s].astype(np.float32)
        M = cv2.getAffineTransform(src, dst)
        r = cv2.boundingRect(dst)
        if r[2] <= 0 or r[3] <= 0:
            continue
        msk = np.zeros((r[3], r[2]), np.uint8)
        cv2.fillConvexPoly(msk, (dst - np.array([r[0], r[1]], np.float32)).astype(np.int32), 255)
        piece = cv2.warpAffine(gBe, M, (w, h))[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
        if piece.size == 0:
            continue
        roi = warp[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
        roi[msk > 0] = piece[msk > 0]
        wmask[r[1]:r[1] + r[3], r[0]:r[0] + r[2]][msk > 0] = 1

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    flow = dis.calc(gAe, warp, None)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.column_stack([(xx + flow[..., 0]).ravel(), (yy + flow[..., 1]).ravel()])
    simp = tri.find_simplex(pts)
    obx = np.full((h, w), np.nan, np.float32)
    oby = np.full((h, w), np.nan, np.float32)
    for si in np.unique(simp[simp >= 0]):
        sel = simp == si
        src = pa[tri.simplices[si]].astype(np.float32)
        dst = pb[tri.simplices[si]].astype(np.float32)
        M = cv2.getAffineTransform(src, dst)
        Q = (M[:, :2] @ pts[sel].T + M[:, 2:3]).T
        obx.ravel()[sel] = Q[:, 0]
        oby.ravel()[sel] = Q[:, 1]

    ok = ~np.isnan(obx)
    epi = np.full((h, w), np.inf, np.float32)
    p1 = np.column_stack([xx[ok], yy[ok], np.ones(ok.sum(), np.float32)])
    p2 = np.column_stack([obx[ok], oby[ok], np.ones(ok.sum(), np.float32)])
    l = (F @ p1.T).T
    epi[ok] = np.abs(np.sum(l * p2, axis=1)) / np.sqrt(l[:, 0] ** 2 + l[:, 1] ** 2 + 1e-12)
    return obx, oby, epi, wmask.astype(bool)

# --------------------------------------------------------------------------
# Geometrie
# --------------------------------------------------------------------------
def K_matrix(w, h):
    f = C.focal_px(max(w, h))
    cx, cy = C.principal_point(w, h)
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)

def undistort(p, K):
    d = np.array([C.K1, 0, 0, 0], float)
    return cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64), K, d, P=K).reshape(-1, 2)

def estimate_pose(pa, pb, K):
    ua, ub = undistort(pa, K), undistort(pb, K)
    E, _ = cv2.findEssentialMat(ua, ub, K, cv2.RANSAC, 0.999, C.RANSAC_E_THRESH)
    _, R, t, _ = cv2.recoverPose(E, ua, ub, K)
    return R, t.ravel()

def triangulate(K, R, t, p1, p2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (X[:3] / X[3]).T

def save_rig(path, poses, note=""):
    d = {"calib_version": C.CALIB_VERSION, "focal_rel": C.FOCAL_REL, "k1": C.K1,
         "note": note,
         "poses": {k: {"R": v[0].tolist(), "t": v[1].tolist()} for k, v in poses.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def load_rig(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d["calib_version"] != C.CALIB_VERSION:
        raise RuntimeError(
            f"Version de calibration incompatible : rig={d['calib_version']} "
            f"vs config={C.CALIB_VERSION}. Ne jamais melanger deux versions.")
    return {k: (np.array(v["R"]), np.array(v["t"])) for k, v in d["poses"].items()}

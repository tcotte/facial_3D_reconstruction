"""
tiled_matching.py — appariement par tuiles (DISK + LightGlue) haute resolution.

OBJET
    L'appariement pleine image sature la memoire au-dela de ~1300 px. Le
    decoupage en tuiles permet de travailler a 4000 px effectifs : chaque tuile
    du frontal est appariee a la FENETRE CORRESPONDANTE de l'oblique, predite
    par une passe grossiere prealable.

GAIN MESURE (appariements sur la peau, F<->L / F<->R)
    SIFT                             17 /   26
    ASIFT                           435 /  206
    DISK+LightGlue 1280 px          768 /  779
    DISK+LightGlue par tuiles      3616 / 3770     <- facteur 140-210 vs SIFT

    C'est le facteur limitant reel de toute la chaine : ni les images ni
    l'eclairage, mais la capacite du moteur a absorber 45 deg de rotation.

USAGE
    # passe 1 : grossiere, pleine image
    python tiled_matching.py --subject alban --session D0 --pair L --coarse
    # passe 2 : par tuiles, guidee par la passe 1
    python tiled_matching.py --subject alban --session D0 --pair L --guide coarse_L.npz
"""
import argparse, os, time, numpy as np, cv2

IMG_DIR = r"C:\donnees\visia"
ANGLES = {"F": "Frontal", "L": "Left_Oblique", "R": "Right_Oblique"}
MODALITY = "Standard_1"
TILE, STRIDE, MAXF = 640, 512, 2048
SKIN_TOP, SKIN_BOTTOM = 0.28, 0.88


class Matcher:
    def __init__(self):
        import torch, kornia as K, kornia.feature as KF
        self.torch, self.K, self.KF = torch, K, KF
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Matcher] peripherique = {self.dev}")
        self.disk = KF.DISK.from_pretrained("depth").eval().to(self.dev)
        self.lg = KF.LightGlueMatcher("disk").eval().to(self.dev)

    def _prep(self, crop, cap=640):
        h, w = crop.shape[:2]
        s = min(1.0, cap / max(h, w))
        nh = max(16, int(round(h * s / 16)) * 16)
        nw = max(16, int(round(w * s / 16)) * 16)
        r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        t = self.K.color.bgr_to_rgb(self.K.image_to_tensor(r, False).float() / 255.)
        return t.to(self.dev), (w / nw, h / nh)

    def match(self, ca, cb):
        ta, sa = self._prep(ca); tb, sb = self._prep(cb)
        with self.torch.inference_mode():
            fa = self.disk(ta, MAXF, pad_if_not_divisible=True)[0]
            fb = self.disk(tb, MAXF, pad_if_not_divisible=True)[0]
            if len(fa.keypoints) < 8 or len(fb.keypoints) < 8:
                return None
            la = self.KF.laf_from_center_scale_ori(fa.keypoints[None])
            lb = self.KF.laf_from_center_scale_ori(fb.keypoints[None])
            _, idx = self.lg(fa.descriptors, fb.descriptors, la, lb)
        if len(idx) < 8:
            return None
        i = idx.cpu().numpy()
        return (fa.keypoints.cpu().numpy()[i[:, 0]] * np.array(sa),
                fb.keypoints.cpu().numpy()[i[:, 1]] * np.array(sb))


def load(subject, session, angle, long_side):
    p = os.path.join(IMG_DIR, f"{subject}_{session}_{ANGLES[angle]}_{MODALITY}.jpg")
    im = cv2.imread(p)
    if im is None:
        raise IOError(p)
    h, w = im.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--session", required=True)
    ap.add_argument("--pair", choices=["L", "R"], required=True)
    ap.add_argument("--long", type=int, default=4000)
    ap.add_argument("--coarse", action="store_true", help="passe grossiere pleine image")
    ap.add_argument("--guide", help="npz de la passe grossiere (obx, oby, scale)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = Matcher()
    long_side = 1280 if a.coarse else a.long
    A = load(a.subject, a.session, "F", long_side)
    B = load(a.subject, a.session, a.pair, long_side)
    t0 = time.time()

    if a.coarse:
        r = m.match(A, B)
        PA, PB = r[0], r[1]
    else:
        g = np.load(a.guide)
        obx, oby, sc = g["obx"], g["oby"], float(g["scale"])
        HA, WA = A.shape[:2]; HB, WB = B.shape[:2]
        face = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY) > 45
        PA, PB, n = [], [], 0
        for y0 in range(int(SKIN_TOP * HA), int(SKIN_BOTTOM * HA) - TILE, STRIDE):
            for x0 in range(0, WA - TILE, STRIDE):
                if face[y0:y0 + TILE, x0:x0 + TILE].mean() < 0.40:
                    continue
                gy0, gy1 = int(y0 / sc), int((y0 + TILE) / sc)
                gx0, gx1 = int(x0 / sc), int((x0 + TILE) / sc)
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
                r = m.match(A[y0:y0 + TILE, x0:x0 + TILE], B[by0:by1, bx0:bx1])
                n += 1
                if r is None:
                    continue
                PA.append(r[0] + np.array([x0, y0])); PB.append(r[1] + np.array([bx0, by0]))
        print(f"{n} tuiles traitees")
        PA, PB = np.vstack(PA), np.vstack(PB)

    F, mask = cv2.findFundamentalMat(PA, PB, cv2.FM_RANSAC, 3.0, 0.999, maxIters=50000)
    inl = mask.ravel().astype(bool)
    yn = PA[inl][:, 1] / A.shape[0]
    n_skin = int(((yn > 0.30) & (yn < 0.86)).sum())
    print(f"{len(PA)} bruts | {inl.sum()} inliers RANSAC | {n_skin} sur la peau "
          f"| {time.time()-t0:.0f}s")
    out = a.out or f"match_{a.session}_{a.pair}{'_coarse' if a.coarse else ''}.npz"
    np.savez(out, pa=PA[inl], pb=PB[inl], F=F, long=long_side)
    print(f"-> {out}")

if __name__ == "__main__":
    main()

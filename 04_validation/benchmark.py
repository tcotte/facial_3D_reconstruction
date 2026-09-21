"""
benchmark.py — mesure des temps de calcul et de la memoire par etape.

MESURES DE REFERENCE (Intel Xeon 2,10 GHz, 1 coeur, 4 Go, SANS GPU)
    lecture + reduction d'une image 42 Mpx          2,0 s     300 Mo
    masque facial (1 image)                         1,1 s     300 Mo
    chargement des modeles DISK + LightGlue         3,7 s     660 Mo
    appariement pleine image 1280 px (1 paire)     68,9 s   3 334 Mo
    appariement d'une tuile 640 px                  6,6 s   3 334 Mo
    densification (1 paire)                         1,9 s     395 Mo
    triangulation de 400 000 points                11,8 s     300 Mo
    fusion et filtrage (1 session)                  7,2 s     300 Mo
    cartes qualite / redondance                     1,6 s     300 Mo
    export PLY (200 000 points)                     1,9 s     300 Mo

    reconstruction d'une session (4 modalites)     ~10 min
    comparaison complete (2 sessions + liaison)    ~28 min

TROIS ENSEIGNEMENTS
    - l'appariement represente 90 % du temps ; densifier une paire prend 1,9 s
      contre 69 s pour l'apparier
    - c'est le seul poste qui sature la memoire (3,3 Go sur 4), ce qui a impose
      le tuilage
    - sur GPU, DISK et LightGlue sont typiquement 20 a 50 fois plus rapides :
      la reconstruction d'une session passerait sous la minute. Les 4 modalites
      et les 2 paires etant independantes, une parallelisation sur 8 coeurs
      diviserait aussi le temps par 8 sans GPU.

USAGE
    python benchmark.py --frontal img_F.jpg --oblique img_L.jpg
"""
import argparse
import resource
import time
import cv2
import numpy as np


def pic_mo():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def chrono(nom, fn):
    t0 = time.time()
    fn()
    print(f"{nom:44s} {time.time()-t0:7.1f} s   pic RAM {pic_mo():6.0f} Mo", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--oblique", required=True)
    ap.add_argument("--long", type=int, default=2000)
    a = ap.parse_args()

    chrono("lecture + reduction d'une image", lambda: cv2.resize(
        cv2.imread(a.frontal), (1333, 2000), interpolation=cv2.INTER_AREA))

    A = cv2.resize(cv2.imread(a.frontal), (1333, 2000), interpolation=cv2.INTER_AREA)
    B = cv2.resize(cv2.imread(a.oblique), (1333, 2000), interpolation=cv2.INTER_AREA)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_pipeline"))
    from face_mask import face_mask
    chrono("masque facial (1 image)", lambda: face_mask(A))

    try:
        import torch, kornia as K, kornia.feature as KF
        t0 = time.time()
        disk = KF.DISK.from_pretrained("depth").eval()
        lg = KF.LightGlueMatcher("disk").eval()
        print(f"{'chargement DISK + LightGlue':44s} {time.time()-t0:7.1f} s   "
              f"pic RAM {pic_mo():6.0f} Mo")
        print(f"{'peripherique':44s} {'cuda' if torch.cuda.is_available() else 'cpu':>9s}")

        def prep(c, cap):
            hh, ww = c.shape[:2]
            sc = min(1.0, cap / max(hh, ww))
            nh = max(16, int(round(hh*sc/16))*16)
            nw = max(16, int(round(ww*sc/16))*16)
            r = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
            return K.color.bgr_to_rgb(K.image_to_tensor(r, False).float()/255.)

        def appar():
            ta, tb = prep(A, 1280), prep(B, 1280)
            with torch.inference_mode():
                fa = disk(ta, 6144, pad_if_not_divisible=True)[0]
                fb = disk(tb, 6144, pad_if_not_divisible=True)[0]
                la = KF.laf_from_center_scale_ori(fa.keypoints[None])
                lb = KF.laf_from_center_scale_ori(fb.keypoints[None])
                lg(fa.descriptors, fb.descriptors, la, lb)
        chrono("appariement pleine image 1280 px (1 paire)", appar)
    except ImportError:
        print("torch/kornia absents : etape d'appariement non mesuree")

    ga, gb = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    chrono("flot optique DIS (1 paire)", lambda: dis.calc(ga, gb, None))

    n = 400000
    K3 = np.array([[2844.4, 0, 666.5], [0, 2844.4, 1000], [0, 0, 1]])
    p1 = np.random.rand(n, 2) * np.array([1333, 2000])
    p2 = p1 + np.random.randn(n, 2) * 3
    P1 = K3 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K3 @ np.hstack([np.eye(3), np.array([[1.], [0], [0]])])
    chrono("triangulation de 400 000 points",
           lambda: cv2.triangulatePoints(P1, P2, p1.T, p2.T))


if __name__ == "__main__":
    main()

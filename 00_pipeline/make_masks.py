"""
make_masks.py — trace interactivement la ROI et la zone stable sur la vue frontale.

    python make_masks.py <sujet> <session>

Clic gauche : ajouter un sommet | 'n' : polygone suivant | 's' : enregistrer
'r' : recommencer | 'q' : quitter

Produit roi.png et stable.png a la resolution de travail.

CONSEIL : la zone stable doit etre anatomiquement non affectee par le
traitement (front, dos du nez selon le cas). Recaler sur une zone qui evolue
avec le traitement fausserait toute la mesure.
"""
import sys, numpy as np, cv2
import config as C, visia_core as vc

def draw(name, img):
    pts, polys = [], []
    disp = img.copy()
    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
            if len(pts) > 1:
                cv2.line(disp, pts[-2], pts[-1], (0, 255, 0), 2)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 900)
    cv2.setMouseCallback(name, on_mouse)
    while True:
        cv2.imshow(name, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('n') and len(pts) >= 3:
            polys.append(np.array(pts, np.int32)); pts = []
        elif k == ord('s'):
            if len(pts) >= 3: polys.append(np.array(pts, np.int32))
            break
        elif k == ord('r'):
            pts, polys, disp = [], [], img.copy()
        elif k == ord('q'):
            cv2.destroyAllWindows(); sys.exit(0)
    cv2.destroyWindow(name)
    m = np.zeros(img.shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [p], 255)
    return m

def main():
    subject, session = sys.argv[1], sys.argv[2]
    A = vc.load(vc.img_path(subject, session, "F"))
    print("1/2 — tracer la ROI (zone evaluee)")
    roi = draw("ROI", A)
    print("2/2 — tracer la ZONE STABLE (non affectee par le traitement)")
    stable = draw("ZONE STABLE", A)
    if (roi & stable).any():
        print("ATTENTION : les deux masques se chevauchent")
    cv2.imwrite("roi.png", roi); cv2.imwrite("stable.png", stable)
    print("-> roi.png, stable.png")

if __name__ == "__main__":
    main()

"""
sensor_check.py — coherence physique entre focale estimee et capteur.

OBJET
    Une focale estimee par auto-calibration n'a de sens que si elle correspond
    a une optique physiquement plausible. Ce controle a permis d'ECARTER
    l'hypothese d'une optique 24 mm, qui aurait conduit a appliquer des
    coefficients de distorsion totalement inadaptes.

RESULTAT OBTENU
    Fichiers VISIA 5335 x 8000 px contre 5464 x 8192 natif R5
    -> recadrage uniforme et centre de 2,3 % (confirme le point principal centre)
    Focale estimee 6250 px @ 4000 -> 56 mm -> distance de travail ~52 cm
    Hypothese 24 mm                        -> distance de travail ~22 cm (exclu)

USAGE
    python sensor_check.py --focal-px 6250 --work-long 4000
"""
import argparse, math

SENSOR_MM = (36.0, 24.0)          # Canon R5 plein format
SENSOR_PX = (8192, 5464)
FILE_PX   = (8000, 5335)          # fichiers exportes par le VISIA (cote long, cote court)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal-px", type=float, required=True,
                    help="focale estimee, en pixels, a la resolution de travail")
    ap.add_argument("--work-long", type=float, default=4000,
                    help="cote long de la resolution de travail")
    ap.add_argument("--face-height-cm", type=float, default=33.0,
                    help="hauteur de scene couverte verticalement, en cm")
    a = ap.parse_args()

    crop_long = FILE_PX[0] / SENSOR_PX[0]
    crop_short = FILE_PX[1] / SENSOR_PX[1]
    print(f"recadrage VISIA : {crop_long:.4f} (cote long), {crop_short:.4f} (cote court)")
    if abs(crop_long - crop_short) < 0.005:
        print("  -> recadrage UNIFORME et centre : le point principal reste au centre")
    else:
        print("  -> recadrage ASYMETRIQUE : le point principal est decale, a modeliser")

    pitch_native = SENSOR_MM[0] / SENSOR_PX[0]                 # mm/px capteur
    pitch_work = pitch_native * (FILE_PX[0] / a.work_long)     # mm/px a la resolution de travail
    print(f"pas pixel capteur      : {pitch_native*1000:.3f} um")
    print(f"pas pixel de travail   : {pitch_work*1000:.2f} um")

    f_mm = a.focal_px * pitch_work
    half = math.degrees(math.atan((a.work_long / 2) / a.focal_px))
    dist = (a.face_height_cm / 2) / math.tan(math.radians(half))
    print(f"\nfocale {a.focal_px:.0f} px -> {f_mm:.1f} mm")
    print(f"  demi-champ         : {half:.1f} deg")
    print(f"  distance de travail: {dist:.1f} cm")
    if dist < 30:
        print("  -> IMPLAUSIBLE pour un poste d'imagerie faciale")
    elif dist > 120:
        print("  -> distance inhabituellement grande, a verifier")
    else:
        print("  -> plausible")

if __name__ == "__main__":
    main()

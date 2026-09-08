import sys
from pathlib import Path

import cv2
import joblib

from detector.sliding_window import detect_on_image
from detector.nms import non_max_suppression

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_FIGURES = RACINE / "figures"


def charger_verites_image(stem):
    """Cherche le CSV d'annotation de l'image (dans train/ ou val/) et renvoie
    ses vraies boîtes au format [y, x, h, w]. Renvoie [] si introuvable."""
    for dossier in [RACINE / "data" / "train" / "labels",
                    RACINE / "data" / "val" / "labels"]:
        chemin = dossier / (stem + ".csv")
        if chemin.exists():
            boites = []
            for ligne in open(chemin, encoding="utf-8"):
                p = ligne.strip().split(",")
                if len(p) < 4:
                    continue
                try:
                    y, x, h, w = int(p[0]), int(p[1]), int(p[2]), int(p[3])
                except ValueError:
                    continue
                if h > 0 and w > 0:
                    boites.append((y, x, h, w))
            return boites
    return []


def visualiser(chemin_image, seuil_affichage=0.5, montrer_brutes=False,
               nb_brutes=8, stride=16, seuil_score=0.3):
    """Enregistre une image annotée : vraies boîtes (bleu), détections finales
    après NMS (vert + score), et éventuellement les meilleures détections
    AVANT NMS (rouge fin) pour visualiser la concurrence entre boîtes.

    chemin_image    : chemin de l'image à analyser
    seuil_affichage : on ne dessine que les détections finales au-dessus de ce score
    montrer_brutes  : si True, dessine aussi les nb_brutes meilleures boîtes avant NMS
    """
    chemin_image = Path(chemin_image)
    chemin_modele = RACINE / "modele_panneaux.joblib"
    if not chemin_modele.exists():
        print("Modèle introuvable. Lance d'abord : python predict_test.py retrain")
        return

    modele = joblib.load(chemin_modele)
    modele.n_jobs = -1  # prédiction sur tous les cœurs

    img = cv2.imread(str(chemin_image))
    if img is None:
        print(f"Image illisible : {chemin_image}")
        return

    dessin = img.copy()

    # --- Vraies boîtes (bleu) ---
    verites = charger_verites_image(chemin_image.stem)
    for (y, x, h, w) in verites:
        cv2.rectangle(dessin, (x, y), (x + w, y + h), (255, 0, 0), 2)  # bleu (BGR)

    # --- Détection ---
    brutes = detect_on_image(img, modele, stride=stride, score_threshold=seuil_score)
    finales = non_max_suppression(brutes, iou_threshold=0.3)

    # --- Meilleures boîtes AVANT NMS (rouge fin), pour voir la concurrence ---
    if montrer_brutes and len(brutes) > 0:
        ordre = brutes[:, 4].argsort()[::-1]      # tri par score décroissant
        for det in brutes[ordre][:nb_brutes]:
            y, x, h, w, score = det
            cv2.rectangle(dessin, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
            cv2.putText(dessin, f"{score:.2f}", (int(x), int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # --- Détections finales (vert) ---
    for det in finales:
        y, x, h, w, score = det
        if score < seuil_affichage:
            continue
        cv2.rectangle(dessin, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
        cv2.putText(dessin, f"{score:.2f}", (int(x), int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    DOSSIER_FIGURES.mkdir(exist_ok=True)
    sortie = DOSSIER_FIGURES / f"{chemin_image.stem}_resultat.png"
    cv2.imwrite(str(sortie), dessin)

    print(f"Image enregistrée : {sortie}")
    print("Légende -> Bleu : vraies boîtes | Vert : détections finales (NMS) | "
          "Rouge : meilleures boîtes avant NMS." if montrer_brutes else
          "Légende -> Bleu : vraies boîtes | Vert : détections finales (NMS).")


def _est_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage : python visualiser.py <chemin_image> [seuil] [brutes]")
        print("  <chemin_image> : image à analyser")
        print("  [seuil]        : score minimal des boîtes vertes affichées (def. 0.5)")
        print("  [brutes]       : ajoute le mot 'brutes' pour voir les boîtes avant NMS")
        print("Exemple : python visualiser.py ../data/val/images/pos/0034.jpg 0.4 brutes")
        return

    chemin = sys.argv[1]
    seuil = next((float(a) for a in sys.argv[2:] if _est_float(a)), 0.5)
    montrer_brutes = "brutes" in sys.argv[2:]
    visualiser(chemin, seuil_affichage=seuil, montrer_brutes=montrer_brutes)


if __name__ == "__main__":
    main()

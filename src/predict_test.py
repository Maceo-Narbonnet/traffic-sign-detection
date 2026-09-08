import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import joblib

from evaluation import entrainer_modele
from detector.sliding_window import detect_on_image
from detector.nms import non_max_suppression


def predire_sur_test(modele, dossier_test, fichier_sortie, seuil_score=0.3, stride=16,
                     afficher=False, enregistrer_images=True, seuil_affichage=0.7):
    """Détecte les panneaux sur toutes les images de test et écrit le CSV de
    soumission.

    Format de chaque ligne (une ligne par détection) :
        Num img, Coin h-g y, Coin h-g x, Hauteur, Largeur, Score

    modele       : classifieur entraîné
    dossier_test : dossier 'data/test'
    fichier_sortie : chemin du CSV à écrire
    seuil_score  : on ne garde que les détections au-dessus de ce score (pour le CSV)
    stride       : pas de la fenêtre glissante
    afficher     : si True, montre chaque image dans une fenêtre
    enregistrer_images : si True, enregistre chaque image de test annotée dans le
                         dossier 'images_results' (nom horodaté)
    seuil_affichage : pour le DESSIN seulement, on ne trace que les détections
                      au-dessus de ce score (sinon l'image est illisible)
    """
    # Prédiction parallélisée sur tous les cœurs du CPU
    modele.n_jobs = -1

    # Dossier de sortie des images annotées + horodatage commun à ce lancement
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier_images = Path(fichier_sortie).parent / "images_results"
    if enregistrer_images:
        dossier_images.mkdir(exist_ok=True)
        print(f"Images annotées enregistrées dans : {dossier_images}")

    chemins = sorted(Path(dossier_test).glob("*.jpg"))
    print(f"Détection sur {len(chemins)} image(s) de test...")

    lignes = []
    for chemin in chemins:
        img = cv2.imread(str(chemin))
        if img is None:
            print(f"Image illisible, ignorée : {chemin.name}")
            continue

        # Numéro de l'image à partir du nom de fichier : "0296" -> 296.
        # Si le nom n'est pas un nombre, on garde le nom tel quel.
        try:
            num_img = int(chemin.stem)
        except ValueError:
            num_img = chemin.stem

        # Détection multi-échelle puis NMS
        brutes = detect_on_image(img, modele, stride=stride, score_threshold=seuil_score)
        finales = non_max_suppression(brutes, iou_threshold=0.3)

        # detect_on_image renvoie [y, x, h, w, score] -> on écrit dans cet ordre
        for det in finales:
            y, x, h, w, score = det
            lignes.append([num_img, int(y), int(x), int(h), int(w), float(score)])

        print(f"  {chemin.name} : {len(finales)} détection(s)")

        # On dessine seulement les détections les plus sûres pour que l'image
        # reste lisible (le CSV, lui, garde toutes les détections).
        if afficher or enregistrer_images:
            image_dessin = img.copy()
            for det in finales:
                y, x, h, w, score = det
                if score < seuil_affichage:
                    continue
                cv2.rectangle(image_dessin, (int(x), int(y)), (int(x + w), int(y + h)),
                              (0, 255, 0), 3)
                cv2.putText(image_dessin, f"{score:.2f}", (int(x), int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Enregistrement dans images_results, nom horodaté
            if enregistrer_images:
                sortie_img = dossier_images / f"{chemin.stem}_{horodatage}.png"
                cv2.imwrite(str(sortie_img), image_dessin)

            # Affichage interactif si demandé
            if afficher:
                cv2.imshow(f"Detections (score >= {seuil_affichage}) - {chemin.name}", image_dessin)
                print("    [Affichage] Appuie sur une touche pour passer a l'image suivante.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # Écriture du CSV global (une ligne par détection, sans ligne d'en-tête,
    # comme les fichiers d'annotation du projet).
    with open(fichier_sortie, "w", newline="", encoding="utf-8") as f:
        writeur = csv.writer(f)
        writeur.writerows(lignes)

    print(f"\n{len(lignes)} détection(s) écrite(s) dans : {fichier_sortie}")


def main():
    # 'data' est à la racine du projet (un niveau au-dessus de 'src').
    base = Path(__file__).resolve().parent.parent / "data"
    racine = base.parent
    chemin_modele = racine / "modele_panneaux.joblib"
    fichier_sortie = racine / "detection.csv"

    # Options en ligne de commande (dans n'importe quel ordre) :
    #   retrain -> force le réentraînement (utile après avoir modifié le code)
    #   show    -> affiche aussi chaque image dans une fenêtre
    #   un nombre -> seuil d'affichage des boîtes dessinées (def. 0.7)
    #               ex. "python predict_test.py show 0.4" pour voir plus de boîtes
    # Dans tous les cas, chaque image de test annotée est enregistrée dans le
    # dossier 'images_results' (nom horodaté).
    args = sys.argv[1:]
    forcer_entrainement = "retrain" in args
    afficher = "show" in args
    seuil_aff = next((float(a) for a in args if _est_float(a)), 0.7)

    if chemin_modele.exists() and not forcer_entrainement:
        print(f"Chargement du modèle existant : {chemin_modele.name}")
        modele = joblib.load(chemin_modele)
    else:
        print("Entraînement du modèle (avec toutes les améliorations)...")
        modele = entrainer_modele(base / "train")
        joblib.dump(modele, chemin_modele)
        print(f"Modèle sauvegardé dans : {chemin_modele.name}")

    predire_sur_test(modele, base / "test", fichier_sortie,
                     seuil_score=0.6, afficher=afficher, seuil_affichage=seuil_aff)


def _est_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()

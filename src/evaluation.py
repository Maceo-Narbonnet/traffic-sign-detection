import csv
import sys
import time
from pathlib import Path

import numpy as np
import cv2

from data_load import load_pos_data, load_neg_data, crop_and_resize_pannels
from features import extract_hog_features, appliquer_distorsion_fisheye
from training import (
    extract_negative_samples,
    extract_hard_negatives,
    train_panel_classifier,
)
from detector.sliding_window import detect_on_image
from detector.nms import non_max_suppression
from metrics import evaluer_detection


def entrainer_modele(base_dir, target_size=(64, 64), neg_par_image=10, hard_neg=True,
                     augment_fisheye=False, modele_type="rf"):
    """Entraîne le classifieur UNIQUEMENT sur les images du dossier d'entraînement.

    base_dir        : dossier 'data/train'
    hard_neg        : si True, mine les faux positifs et réentraîne dessus
    augment_fisheye : si True, ajoute une version fisheye de chaque panneau positif
    modele_type     : "rf" (Random Forest) ou "svm" (SVM linéaire calibré)
    Renvoie le modèle entraîné.
    """
    chemin_pos = base_dir / "images" / "pos"
    chemin_neg = base_dir / "images" / "neg"
    chemin_labels = base_dir / "labels"

    # --- Positifs : on découpe les panneaux et on calcule leurs descripteurs ---
    images_pos, boites = load_pos_data(chemin_pos, chemin_labels)
    panneaux = crop_and_resize_pannels(images_pos, boites, target_size=target_size)

    # Augmentation fisheye : on ajoute une copie déformée de chaque panneau, pour
    # que le modèle reconnaisse aussi les panneaux distordus (images test fisheye).
    if augment_fisheye:
        panneaux_fisheye = [appliquer_distorsion_fisheye(p) for p in panneaux]
        panneaux = panneaux + panneaux_fisheye
        print(f"Augmentation fisheye : {len(panneaux_fisheye)} panneaux déformés ajoutés.")

    X_pos = extract_hog_features(panneaux)
    y_pos = np.ones(len(X_pos))  # étiquette 1 = panneau

    # --- Négatifs : morceaux de fond pris au hasard dans les images neg ---
    images_neg = load_neg_data(chemin_neg)
    fonds = extract_negative_samples(images_neg, samples_per_image=neg_par_image, target_size=target_size)
    X_neg = extract_hog_features(fonds)
    y_neg = np.zeros(len(X_neg))  # étiquette 0 = fond

    # --- Fusion et premier entraînement ---
    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    modele = train_panel_classifier(X, y, modele_type=modele_type)

    # --- Hard negative mining : on apprend au modèle à rejeter ses faux positifs ---
    if hard_neg:
        print("\n--- Hard negative mining (multi-échelle) ---")
        durs = extract_hard_negatives(images_neg, modele, target_size=target_size)
        if durs:
            X_dur = extract_hog_features(durs)
            y_dur = np.zeros(len(X_dur))  # ce sont des négatifs (étiquette 0)
            X = np.vstack((X, X_dur))
            y = np.concatenate((y, y_dur))
            print(f"Réentraînement avec {len(X_dur)} hard negatives ajoutés...")
            modele = train_panel_classifier(X, y, modele_type=modele_type)

    return modele


def charger_verites_terrain(dossier_csv):
    """Lit les CSV d'annotation et renvoie, pour chaque image, ses vraies boîtes
    au format [y, x, h, w] (l'ordre direct du CSV, sans transformation).

    Renvoie un dictionnaire : { 'nom_image' : tableau (M, 4) }
    """
    verites = {}
    for chemin_csv in sorted(Path(dossier_csv).glob("*.csv")):
        boites = []
        with open(chemin_csv, "r", encoding="utf-8") as f:
            for ligne in csv.reader(f):
                if not ligne or len(ligne) < 4:
                    continue
                try:
                    y = int(ligne[0])
                    x = int(ligne[1])
                    h = int(ligne[2])
                    w = int(ligne[3])
                except ValueError:
                    continue
                if h <= 0 or w <= 0:   # on ignore les boîtes vides (ex : 0309)
                    continue
                boites.append([y, x, h, w])
        verites[chemin_csv.stem] = np.array(boites)
    return verites


def evaluer_sur_validation(modele, base_val, seuil_score=0.3, iou_seuil=0.5, stride=16, max_images=None):
    """Lance la détection sur toutes les images de validation et calcule AP / F1.

    modele     : classifieur entraîné
    base_val   : dossier 'data/val'
    seuil_score: seuil bas pour garder beaucoup de détections (courbe AP complète)
    iou_seuil  : recouvrement minimal pour qu'une détection compte comme correcte
    stride     : pas de la fenêtre glissante (plus grand = plus rapide, moins précis)
    max_images : si fourni, n'évalue que sur les N premières images (itération rapide)
    """
    chemin_pos = base_val / "images" / "pos"
    chemin_labels = base_val / "labels"

    # On parallélise la prédiction du Random Forest sur tous les cœurs du CPU
    # (le modèle a été entraîné avec n_jobs=1, on l'accélère juste pour la détection).
    modele.n_jobs = -1

    verites = charger_verites_terrain(chemin_labels)

    predictions_par_image = []
    verites_par_image = []

    chemins_images = sorted(chemin_pos.glob("*.jpg"))
    if max_images is not None:
        chemins_images = chemins_images[:max_images]
    print(f"\nDétection sur {len(chemins_images)} images de validation...")

    for i, chemin_img in enumerate(chemins_images):
        img = cv2.imread(str(chemin_img))
        if img is None:
            print(f"Image illisible, ignorée : {chemin_img.name}")
            continue

        # Détection multi-échelle puis NMS
        brutes = detect_on_image(img, modele, stride=stride, score_threshold=seuil_score)
        finales = non_max_suppression(brutes, iou_threshold=0.3)

        predictions_par_image.append(finales)
        verites_par_image.append(verites.get(chemin_img.stem, np.empty((0, 4))))

        print(f"  [{i + 1}/{len(chemins_images)}] {chemin_img.name} : "
              f"{len(finales)} détection(s) pour {len(verites.get(chemin_img.stem, []))} vrai(s) panneau(x)")

    # Calcul des métriques globales
    return evaluer_detection(predictions_par_image, verites_par_image, iou_seuil=iou_seuil)


def main():
    # Le dossier 'data' est à la racine du projet (un niveau au-dessus de 'src').
    # On le calcule à partir de l'emplacement de ce fichier : le script marche
    # alors quel que soit le dossier depuis lequel on le lance.
    base = Path(__file__).resolve().parent.parent / "data"

    # Options en ligne de commande (dans n'importe quel ordre) :
    #   un nombre  -> n'évalue que sur les N premières images (itération rapide)
    #   "svm"      -> utilise un SVM linéaire au lieu du Random Forest (par défaut)
    # Exemples : "python evaluation.py 25"      "python evaluation.py 25 svm"
    args = sys.argv[1:]
    max_images = next((int(a) for a in args if a.isdigit()), None)
    modele_type = "svm" if "svm" in args else "rf"

    print(f"=== Entraînement sur le set d'entraînement (modèle : {modele_type}) ===")
    debut = time.time()
    modele = entrainer_modele(base / "train", modele_type=modele_type)
    print(f"Entraînement terminé en {time.time() - debut:.1f} s")

    print("\n=== Évaluation sur le set de validation ===")
    debut = time.time()
    evaluer_sur_validation(modele, base / "val", max_images=max_images)
    print(f"Évaluation terminée en {time.time() - debut:.1f} s")


if __name__ == "__main__":
    main()

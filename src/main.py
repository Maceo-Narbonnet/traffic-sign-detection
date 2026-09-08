import os
from pathlib import Path  # pour une meilleure gestion des chemins d'accès
import numpy as np

from data_load import (
    load_pos_data,
    load_neg_data,
    crop_and_resize_pannels,
)  # importation des fonctions de chargement de données

from features import extract_hog_features  # Importation de ta fonction HOG

from training import (
    extract_negative_samples,
    extract_hard_negatives,
    train_panel_classifier,
)

from detector.sliding_window import detect_on_image
from detector.nms import non_max_suppression


def main():
    print("Début du pipeline de détection des panneaux")

    # Définition des chemins d'accès aux données
    BASE_DIR = Path("data")  # répertoire de base pour les données
    print(BASE_DIR)  # Affiche le chemin de base pour vérifier qu'il est correct

    IMAGE_POS_PATH = BASE_DIR / "train" / "images" / "pos"  # répertoire des images positives
    NEG_IMAGE_PATH = BASE_DIR / "train" / "images" / "neg"  # répertoire des images négatives
    CSV_POS_PATH = BASE_DIR / "train" / "labels"  # chemin vers le fichier CSV des bounding boxes

    TARGET_SIZE = (64, 64)  # taille cible pour le redimensionnement des panneaux

    # --- 1. CHARGEMENT ET TRAITEMENT DES DONNÉES POSITIVES ---
    print("\n--- Étape 1 : Traitement des données positives ---")
    if not CSV_POS_PATH.exists():
        print(f"Erreur : le fichier {CSV_POS_PATH} n'existe pas.")
        return
    
    pos_images, bounding_boxes = load_pos_data(IMAGE_POS_PATH, CSV_POS_PATH)

    print("Extraction et normalisation des panneaux positifs...")
    resized_pos_images = crop_and_resize_pannels(pos_images, bounding_boxes, target_size=TARGET_SIZE)

    print("Extraction des descripteurs HOG pour les panneaux positifs...")
    X_pos = extract_hog_features(resized_pos_images)
    y_pos = np.ones(len(X_pos))  # Label 1 pour la classe "Panneau"


    # --- 2. CHARGEMENT ET TRAITEMENT DES DONNÉES NÉGATIVES ---
    print("\n--- Étape 2 : Traitement des données négatives ---")
    # Génération des morceaux de fond de taille 64x64 (2 par image pour équilibrer avec les ~312 panneaux)
    neg_images = load_neg_data(NEG_IMAGE_PATH)
    resized_neg_images = extract_negative_samples(neg_images, samples_per_image=10, target_size=TARGET_SIZE)

    print("Extraction des descripteurs HOG pour les images négatives...")
    X_neg = extract_hog_features(resized_neg_images)
    y_neg = np.zeros(len(X_neg))  # Label 0 pour la classe "Négatif/Fond"


    # --- 3. SYNTHÈSE DES DONNÉES DU DATASET ---
    print("\n--- Étape 3 : Résumé du Dataset ---")
    print(f"Forme de la matrice HOG positive X_pos : {X_pos.shape}")
    print(f"Forme de la matrice HOG négative X_neg : {X_neg.shape}")

    # Fusion finale pour préparer l'entrée du Random Forest
    X_final = np.vstack((X_pos, X_neg))
    y_final = np.concatenate((y_pos, y_neg))

    print(f"\nDataset final prêt pour le Random Forest !")
    print(f"Matrice globale X_final : {X_final.shape}")
    print(f"Vecteur global y_final  : {y_final.shape}")
    print("Pipeline de détection des panneaux terminé")

    # --- 4. ENTRAÎNEMENT ET ÉVALUATION DU RANDOM FOREST ---
    print("\n--- Étape 4 : Entraînement du classifieur Random Forest ---")
    
    # Appel de la fonction d'entraînement
    rf_model = train_panel_classifier(X_final, y_final, test_size=0.2)
    
    print("\n[INFO] Le modèle Random Forest a été entraîné et évalué avec succès.")


    # --- 4.5 HARD NEGATIVE MINING ---
    print("\n--- Étape 4.5 : Hard Negative Mining ---")
    hard_neg_images = extract_hard_negatives(neg_images, rf_model, target_size=TARGET_SIZE)

    if hard_neg_images:
        X_hard_neg = extract_hog_features(hard_neg_images)
        y_hard_neg = np.zeros(len(X_hard_neg))

        X_final = np.vstack((X_final, X_hard_neg))
        y_final = np.concatenate((y_final, y_hard_neg))

        print(f"Dataset enrichi : {X_final.shape[0]} échantillons ({len(X_hard_neg)} hard negatives ajoutés)")
        print("Réentraînement du modèle avec les hard negatives...")
        rf_model = train_panel_classifier(X_final, y_final, test_size=0.2)
        print("\n[INFO] Modèle réentraîné avec succès.")


    # --- 5. DÉTECTION SUR UNE IMAGE DE TEST ---
    print("\n--- Étape 5 : Détection par fenêtre glissante ---")
    
    IMAGE_TEST_PATH = BASE_DIR / "test" / "0034.jpg" #raccourci_test
    
    if not IMAGE_TEST_PATH.exists():
        print(f"[Alerte] L'image de test {IMAGE_TEST_PATH} n'existe pas. Étape de détection sautée.")
        return

    # Chargement de l'image avec OpenCV
    import cv2
    img_test = cv2.imread(str(IMAGE_TEST_PATH))
    img_brute = img_test.copy()
    img_nms   = img_test.copy()

    # Détection multi-échelle (pyramide gaussienne + fenêtre glissante)
    print(f"Analyse de l'image {IMAGE_TEST_PATH.name} avec la fenêtre glissante multi-échelle...")
    raw_detections = detect_on_image(img_test, rf_model)
    # raw_detections : (N, 5) — colonnes [y, x, h, w, score]
    print(f"{len(raw_detections)} fenêtres positives avant NMS.")

    # Application de la NMS (format [y, x, h, w, score])
    print("Application de la NMS...")
    final_detections = non_max_suppression(raw_detections, iou_threshold=0.3)
    print(f"{len(final_detections)} détection(s) finale(s) après NMS.")

    # --- 6. VISUALISATION DES RÉSULTATS ---
    # Boîtes brutes (rouge)
    for det in raw_detections:
        y, x, h, w, _ = det
        cv2.rectangle(img_brute, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

    # Boîtes finales après NMS (vert)
    for det in final_detections:
        y, x, h, w, score = det
        cv2.rectangle(img_nms, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
        cv2.putText(img_nms, f"Panneau {score:.2f}", (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print("\n[INFO] Affichage des résultats. Appuie sur une touche pour fermer.")
    cv2.imshow("1. Detections Brutes (Rouge)", img_brute)
    cv2.imshow("2. Apres NMS (Vert)", img_nms)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
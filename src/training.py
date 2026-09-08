import random
import numpy as np
import cv2

#import pour le random forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog


def extract_negative_samples(neg_image_list, samples_per_image = 2, target_size=(128,128)):
    """Extraction de samples négatifs à partir des images négatives"""

    negative_samples = []
    target_w, target_h = target_size

    for img in neg_image_list :
        if img is None :
            print(f"Erreur de chargement de l'image (FONCTION EXTRACT)")
            continue

        img_h, img_w = img.shape[:2]
        if img_w < target_w or img_h < target_h :
            print(f"Erreur d'extraction dû aux dimensions de l'image (FONCTION EXTRACT)")
            continue

        for _ in range(samples_per_image) :
            xmin = random.randint(0, img_w - target_w)
            ymin = random.randint(0, img_h - target_h)

            crop_neg = img[ymin:ymin + target_h, xmin:xmin + target_w]
            negative_samples.append(crop_neg)

    print(f"Extraction de {len(negative_samples)} samples négatifs à partir de {len(neg_image_list)} images négatives.")
    return negative_samples


def extract_hard_negatives(neg_image_list, model, target_size=(64, 64),
                           crops_per_image=50, tailles=(64, 96, 128, 200, 300)):
    """Hard negative mining MULTI-ÉCHELLE, par crops aléatoires (rapide).

    Pour chaque image négative, on tire des crops carrés de tailles variées
    (pour couvrir les faux positifs petits ET grands), on les ramène à
    target_size, puis on les classe tous en une fois. On garde ceux que le
    modèle prend à tort pour des panneaux : ces faux positifs deviennent de
    nouveaux exemples négatifs d'entraînement.

    crops_per_image : nombre de crops tirés par image négative
    tailles         : tailles de crop possibles (en pixels), choisies au hasard
    """

    from features import extract_hog_features

    target_w, target_h = target_size
    all_crops = []

    # Étape 1 : extraire des crops de tailles variées depuis les images négatives
    for img in neg_image_list:
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        for _ in range(crops_per_image):
            # On ne garde que les tailles qui tiennent dans l'image
            tailles_possibles = [t for t in tailles if t <= img_w and t <= img_h]
            if not tailles_possibles:
                continue
            cote = random.choice(tailles_possibles)
            x = random.randint(0, img_w - cote)
            y = random.randint(0, img_h - cote)
            crop = img[y:y + cote, x:x + cote]
            # On ramène le crop à la taille cible, comme on le fait pour un positif
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
            all_crops.append(crop)

    if not all_crops:
        print("Aucun crop extrait pour le hard negative mining.")
        return []

    print(f"{len(all_crops)} crops multi-échelle extraits depuis {len(neg_image_list)} images négatives.")

    # Étape 2 : calcul HOG en batch sur tous les crops
    X = extract_hog_features(all_crops)

    # Étape 3 : prédiction en batch — on garde uniquement les faux positifs
    predictions = model.predict(X)
    hard_neg_crops = [all_crops[i] for i in range(len(all_crops)) if predictions[i] == 1]

    print(f"{len(hard_neg_crops)} hard negatives extraits (faux positifs du modèle).")
    return hard_neg_crops



def extract_hard_negatives_positifs(pos_image_list, bbox_list, model, target_size=(64, 64),
                                    crops_per_image=40, tailles=(64, 96, 128, 200, 300),
                                    iou_max=0.2):
    """Hard negative mining CIBLÉ sur les images POSITIVES.

    On cherche des fenêtres MAL CADRÉES (qui ne tombent pas bien sur un panneau)
    que le modèle prend quand même pour des panneaux, et on les ajoute comme
    négatifs. Cela apprend au classifieur à ne PAS donner une forte confiance à
    une sous-région d'un panneau (ex. un coin de panneau triangulaire) ni à une
    fenêtre qui déborde -> le panneau bien cadré redevient le plus confiant.

    bbox_list : pour chaque image, ses vraies boîtes au format [xmin, ymin, xmax, ymax]
                (celui renvoyé par load_pos_data)
    iou_max   : une fenêtre n'est candidate que si son IoU avec TOUTES les vraies
                boîtes est < iou_max. On le garde BAS (0.2) : ainsi on ne capture
                que des fenêtres clairement mal cadrées (coins, fragments, fond).
                Les fenêtres "presque correctes" (IoU 0.2 à 0.5) sont ambiguës et
                volontairement IGNORÉES : les étiqueter négatif ferait chuter le
                rappel (on apprendrait à rejeter ce qui ressemble à un panneau).
    """
    from features import extract_hog_features
    from detector.nms import iou

    target_w, target_h = target_size
    candidats = []

    for img, bboxes_xyxy in zip(pos_image_list, bbox_list):
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Vraies boîtes converties en [y, x, h, w] (format attendu par iou)
        verites = [[ymin, xmin, ymax - ymin, xmax - xmin]
                   for (xmin, ymin, xmax, ymax) in bboxes_xyxy]

        positions = []  # liste de (x, y, cote) des fenêtres candidates

        # 1) crops aléatoires multi-échelle un peu partout dans l'image
        for _ in range(crops_per_image):
            tailles_possibles = [t for t in tailles if t <= img_w and t <= img_h]
            if not tailles_possibles:
                continue
            cote = random.choice(tailles_possibles)
            x = random.randint(0, img_w - cote)
            y = random.randint(0, img_h - cote)
            positions.append((x, y, cote))

        # 2) sous-fenêtres À L'INTÉRIEUR de chaque panneau (coins, fragments) :
        #    ce sont les morceaux de panneau qu'on veut apprendre à rejeter
        for (xmin, ymin, xmax, ymax) in bboxes_xyxy:
            cote_base = min(xmax - xmin, ymax - ymin)
            if cote_base < 16:
                continue
            for _ in range(4):
                cote = int(cote_base * random.uniform(0.25, 0.45))
                if cote < 8:
                    continue
                x = random.randint(xmin, max(xmin, xmax - cote))
                y = random.randint(ymin, max(ymin, ymax - cote))
                positions.append((x, y, cote))

        # On garde les fenêtres MAL cadrées (IoU < iou_max avec toutes les vérités)
        for (x, y, cote) in positions:
            crop_box = [y, x, cote, cote]
            meilleure_iou = 0.0
            for v in verites:
                meilleure_iou = max(meilleure_iou, iou(crop_box, v))
            if meilleure_iou < iou_max:
                crop = img[y:y + cote, x:x + cote]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
                candidats.append(crop)

    if not candidats:
        print("Aucun candidat hard negative sur les images positives.")
        return []

    print(f"{len(candidats)} fenêtres mal cadrées candidates sur les images positives.")

    # Classement en batch : on ne garde que celles que le modèle croit être des panneaux
    X = extract_hog_features(candidats)
    predictions = model.predict(X)
    durs = [candidats[i] for i in range(len(candidats)) if predictions[i] == 1]

    print(f"{len(durs)} hard negatives 'partiels' extraits (fragments pris pour des panneaux).")
    return durs


def train_panel_classifier(X_data, Y_data, test_size=0.2, modele_type="rf"):
    """Entraînement du classifieur à partir des données d'entraînement.

    X_data      : matrice des caractéristiques (HOG features)
    Y_data      : vecteur des étiquettes (1 pour les panneaux, 0 pour les négatifs)
    modele_type : "rf"  -> Random Forest
                  "svm" -> SVM linéaire (standardisé + calibré en probabilités,
                           pour rester comparable au Random Forest)
    """

    print("--- Début de l'entraînement du classifieur ---")

    #Départage des données en ensembles d'entraînement (80%) et de test (20%)
    #stratify sert à s'assurer qu'il y a une répartition équilibrée des classes dans les ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=42, stratify=Y_data)

    print(f"Ensemble d'entraînement : {len(X_train)} échantillons, Ensemble de test : {len(X_test)} échantillons")

    #Initialisation du classifieur selon le type demandé
    if modele_type == "svm":
        # SVM linéaire (HOG + SVM = détecteur de Dalal-Triggs). On standardise
        # les descripteurs (les SVM y sont sensibles) puis on calibre la sortie
        # en probabilités, pour avoir un score dans [0, 1] comme le Random Forest.
        classifieur = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3),
        )
        print("Entraînement du classifieur SVM linéaire...")
    else:
        # n_estimators = nombre d'arbres, random_state pour la reproductibilité
        classifieur = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        print("Entraînement du classifieur Random Forest...")

    classifieur.fit(X_train, Y_train)
    print("Entraînement terminé !")

    y_pred = classifieur.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Précision du classifieur sur l'ensemble de test : {accuracy:.2%}")

    print("Rapport de classification :")
    print(classification_report(Y_test, y_pred, target_names=['Négatif(0)', 'Panneau(1)']))

    print("Matrice de confusion :")
    print(confusion_matrix(Y_test, y_pred, labels=[0, 1]))

    return classifieur




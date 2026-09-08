import numpy as np

# On réutilise la fonction iou() déjà écrite pour la NMS, au lieu de la réécrire.
# iou() prend deux boîtes au format [y, x, h, w] et renvoie un nombre entre 0 et 1.
from detector.nms import iou


def evaluer_image(predictions, verites, iou_seuil=0.5):
    """Compare les prédictions d'UNE image avec ses vraies boîtes (vérité terrain).

    predictions : tableau (N, 5) -> colonnes [y, x, h, w, score]
    verites     : tableau (M, 4) -> colonnes [y, x, h, w]
    iou_seuil   : un panneau est considéré "bien détecté" si IoU >= ce seuil

    Retourne :
        resultats  : liste de couples (score, est_vrai_positif) pour chaque prédiction
                     est_vrai_positif vaut 1 (vrai positif) ou 0 (faux positif)
        nb_verites : nombre de vrais panneaux dans l'image (sert au calcul du rappel)
    """
    resultats = []
    nb_verites = len(verites)

    # On trie les prédictions par score décroissant : on traite d'abord les
    # détections les plus sûres (logique standard en détection).
    if len(predictions) > 0:
        ordre = np.argsort(predictions[:, 4])[::-1]
        predictions = predictions[ordre]

    # On retient les vraies boîtes déjà associées à une détection, pour qu'une
    # même vérité ne soit pas comptée deux fois.
    verite_deja_prise = [False] * nb_verites

    for pred in predictions:
        score = pred[4]
        meilleure_iou = 0.0
        meilleur_indice = -1

        # On cherche la vraie boîte qui recouvre le mieux cette prédiction.
        for i in range(nb_verites):
            if verite_deja_prise[i]:
                continue
            recouvrement = iou(pred[:4], verites[i][:4])
            if recouvrement > meilleure_iou:
                meilleure_iou = recouvrement
                meilleur_indice = i

        # Recouvrement suffisant et vérité encore libre -> vrai positif.
        # Sinon -> faux positif (le détecteur a vu un panneau là où il n'y en a pas).
        if meilleure_iou >= iou_seuil and meilleur_indice != -1:
            resultats.append((score, 1))
            verite_deja_prise[meilleur_indice] = True
        else:
            resultats.append((score, 0))

    return resultats, nb_verites


def courbe_precision_rappel(resultats_globaux, nb_verites_total):
    """Construit la courbe précision/rappel à partir de TOUTES les détections
    de TOUTES les images.

    resultats_globaux : liste de couples (score, est_vrai_positif)
    nb_verites_total  : nombre total de vrais panneaux (toutes images confondues)

    Retourne deux tableaux numpy : precisions, rappels
    (un point par détection, du score le plus élevé au plus bas).
    """
    if len(resultats_globaux) == 0 or nb_verites_total == 0:
        return np.array([]), np.array([])

    # On classe toutes les détections par score décroissant.
    resultats_tries = sorted(resultats_globaux, key=lambda couple: couple[0], reverse=True)

    vp = 0  # vrais positifs cumulés
    fp = 0  # faux positifs cumulés
    precisions = []
    rappels = []

    for score, est_vp in resultats_tries:
        if est_vp == 1:
            vp += 1
        else:
            fp += 1

        precision = vp / (vp + fp)        # % de corrects parmi ce qu'on a prédit
        rappel = vp / nb_verites_total    # % de vrais panneaux retrouvés
        precisions.append(precision)
        rappels.append(rappel)

    return np.array(precisions), np.array(rappels)


def average_precision(precisions, rappels):
    """Average Precision (AP) = aire sous la courbe précision/rappel (AUC).

    Méthode : on additionne l'aire des rectangles entre chaque niveau de rappel.
    On applique d'abord le "lissage" standard : à chaque point, on garde la
    meilleure précision atteinte plus à droite. Cela évite que les dents de
    scie de la courbe ne fassent sous-estimer l'aire (convention type VOC).
    """
    if len(precisions) == 0:
        return 0.0

    # On ajoute un point de départ à rappel = 0 pour bien fermer l'aire à gauche.
    rappels = np.concatenate(([0.0], rappels))
    precisions = np.concatenate(([precisions[0]], precisions))

    # Lissage : on parcourt de droite à gauche et on garde le maximum de précision.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Aire = somme des (variation de rappel) x precision.
    ap = 0.0
    for i in range(1, len(rappels)):
        ap += (rappels[i] - rappels[i - 1]) * precisions[i]

    return ap


def meilleur_f1(precisions, rappels):
    """Meilleur score F1 le long de la courbe précision/rappel.

    F1 = 2 * P * R / (P + R) : moyenne harmonique de la précision et du rappel.
    Le F1 n'est élevé que si la précision ET le rappel sont bons en même temps.
    On renvoie la valeur maximale rencontrée sur la courbe.
    """
    if len(precisions) == 0:
        return 0.0

    meilleur = 0.0
    for p, r in zip(precisions, rappels):
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > meilleur:
            meilleur = f1
    return meilleur


def precision_rappel_f1(precisions, rappels):
    """Renvoie la précision, le rappel et le F1 au POINT qui maximise le F1.

    Une précision et un rappel uniques n'existent qu'à un seuil de score donné.
    On choisit le seuil qui donne le meilleur compromis (F1 maximal) le long de
    la courbe, et on renvoie les trois valeurs scalaires correspondantes :
    ce sont les chiffres à reporter pour noter l'algorithme.
    """
    if len(precisions) == 0:
        return 0.0, 0.0, 0.0

    meilleur_indice = 0
    meilleur_f1_val = -1.0
    for k in range(len(precisions)):
        p = precisions[k]
        r = rappels[k]
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        if f1 > meilleur_f1_val:
            meilleur_f1_val = f1
            meilleur_indice = k

    return precisions[meilleur_indice], rappels[meilleur_indice], meilleur_f1_val


def evaluer_detection(predictions_par_image, verites_par_image, iou_seuil=0.5):
    """Évalue la détection sur un ensemble d'images, et affiche les résultats.

    predictions_par_image : liste de tableaux (N, 5) -> [y, x, h, w, score]
    verites_par_image     : liste de tableaux (M, 4) -> [y, x, h, w]
    Les deux listes sont dans le même ordre (même indice = même image).

    Retourne : ap, f1, precisions, rappels
    """
    resultats_globaux = []
    nb_verites_total = 0

    # On accumule les résultats image par image dans une seule grande liste.
    for predictions, verites in zip(predictions_par_image, verites_par_image):
        resultats, nb_verites = evaluer_image(predictions, verites, iou_seuil)
        resultats_globaux.extend(resultats)
        nb_verites_total += nb_verites

    precisions, rappels = courbe_precision_rappel(resultats_globaux, nb_verites_total)
    ap = average_precision(precisions, rappels)
    precision, rappel, f1 = precision_rappel_f1(precisions, rappels)

    print(f"--- Évaluation de la détection (IoU >= {iou_seuil}) ---")
    print(f"Nombre total de vrais panneaux : {nb_verites_total}")
    print(f"Nombre total de détections     : {len(resultats_globaux)}")
    print(f"Average Precision (AP / AUC)   : {ap:.3f}")
    print(f"Précision (au meilleur F1)     : {precision:.3f}")
    print(f"Rappel    (au meilleur F1)     : {rappel:.3f}")
    print(f"Meilleur score F1              : {f1:.3f}")

    return ap, precision, rappel, f1, precisions, rappels

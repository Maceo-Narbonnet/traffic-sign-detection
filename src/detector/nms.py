import numpy as np

def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calcule l'Intersection over Union entre deux bounding boxes.

    Entrée : deux tableaux [y, x, h, w]
    Sortie : float entre 0.0 (aucun chevauchement) et 1.0 (identiques)
    """
    # Décomposer les coordonnées
    y_a, x_a, h_a, w_a = box_a
    y_b, x_b, h_b, w_b = box_b

    # Calculer les coins de chaque rectangle
    # "top" = coin haut,  "bot" = coin bas
    top_a, bot_a = y_a, y_a + h_a
    lft_a, rgt_a = x_a, x_a + w_a

    top_b, bot_b = y_b, y_b + h_b
    lft_b, rgt_b = x_b, x_b + w_b

    # Intersection : le rectangle commun aux deux
    # Si les rectangles ne se touchent pas, ces valeurs seront négatives
    inter_top = max(top_a, top_b)
    inter_lft = max(lft_a, lft_b)
    inter_bot = min(bot_a, bot_b)
    inter_rgt = min(rgt_a, rgt_b)

    inter_h = max(0, inter_bot - inter_top)
    inter_w = max(0, inter_rgt - inter_lft)
    inter_area = inter_h * inter_w

    # Union : tout ce que couvrent les deux rectangles ensemble
    area_a    = h_a * w_a
    area_b    = h_b * w_b
    union_area = area_a + area_b - inter_area  # on soustrait pour ne pas compter deux fois l'intersection

    if union_area == 0:
        return 0.0

    return inter_area / union_area



def intersection_sur_min(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Aire d'intersection divisée par l'aire de la PLUS PETITE des deux boîtes.

    Proche de 1 quand une petite boîte est presque entièrement contenue dans une
    plus grande. Utile car l'IoU classique ne détecte PAS ce cas : une petite
    boîte dans une grande a un IoU très faible (l'union est énorme).

    Entrée : deux tableaux [y, x, h, w]
    """
    y_a, x_a, h_a, w_a = box_a
    y_b, x_b, h_b, w_b = box_b

    inter_top = max(y_a, y_b)
    inter_lft = max(x_a, x_b)
    inter_bot = min(y_a + h_a, y_b + h_b)
    inter_rgt = min(x_a + w_a, x_b + w_b)

    inter_h = max(0, inter_bot - inter_top)
    inter_w = max(0, inter_rgt - inter_lft)
    inter_area = inter_h * inter_w

    aire_min = min(h_a * w_a, h_b * w_b)
    if aire_min == 0:
        return 0.0

    return inter_area / aire_min


def non_max_suppression(boxes: np.ndarray,
                        iou_threshold: float = 0.3,
                        contenu_threshold: float = 0.6) -> np.ndarray:
    """
    Filtre les bounding boxes redondantes.

    Entrée  : (N, 5) — colonnes : y, x, h, w, score
    Sortie  : (M, 5) — avec M <= N, les meilleures bbox sans doublons

    Algorithme :
        1. Trier toutes les bbox par score décroissant
        2. Prendre la meilleure (score le plus élevé) → elle est acceptée
        3. Supprimer toutes les bbox qui SOIT ont un IoU élevé avec elle
           (iou_threshold), SOIT sont largement contenues dans/contiennent elle
           (contenu_threshold, via intersection sur la plus petite boîte)
        4. Recommencer sur les bbox restantes jusqu'à ce qu'il n'en reste plus

    Le critère "contenance" rattrape le cas qu'IoU rate : une petite boîte
    imbriquée dans un grand panneau (IoU faible mais clairement un doublon).
    """
    if len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    # Étape 1 — trier par score décroissant (colonne 4 = score)
    order = np.argsort(boxes[:, 4])[::-1]
    boxes = boxes[order]

    kept = []      # indices des bbox qu'on garde
    rejected = set()  # indices des bbox qu'on supprime

    for i in range(len(boxes)):

        if i in rejected:
            continue

        # Cette bbox est la meilleure parmi les restantes → on la garde
        kept.append(i)

        # Comparer avec toutes les bbox suivantes (de score inférieur)
        for j in range(i + 1, len(boxes)):

            if j in rejected:
                continue

            overlap = iou(boxes[i, :4], boxes[j, :4])
            contenu = intersection_sur_min(boxes[i, :4], boxes[j, :4])

            # On supprime boxes[j] (qui a un score inférieur à boxes[i]) si elle
            # chevauche fortement boxes[i] (IoU) OU si l'une est largement
            # imbriquée dans l'autre (contenance). Comme boxes[i] a le meilleur
            # score, on garde toujours la boîte la plus confiante du groupe.
            if overlap > iou_threshold or contenu > contenu_threshold:
                rejected.add(j)

    return boxes[kept]
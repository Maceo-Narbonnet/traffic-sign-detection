import cv2
import numpy as np
from skimage.transform import pyramid_gaussian
from features import extract_hog_features, masque_couleur_panneau
from config import PATCH_SIZE, STRIDE, PYRAMID_DOWNSCALE, PYRAMID_MIN_SIZE, SCORE_THRESHOLD


def detect_on_image(
    image: np.ndarray,
    clf,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    pyramid_downscale: float = PYRAMID_DOWNSCALE,
    pyramid_min_size: int = PYRAMID_MIN_SIZE,
    score_threshold: float = SCORE_THRESHOLD,
    seuil_couleur: float = 0.05,
) -> np.ndarray:
    """
    Lance le détecteur multi-échelle sur une image complète.

    Inputs
    ----------
    image : np.ndarray
        Image RGB (H, W, 3) en uint8.
    clf : sklearn estimator
        Classifieur entraîné (SVM, RF ou AdaBoost).
    patch_size : int
        Taille de la fenêtre carrée (doit correspondre à l'entraînement).
    stride : int
        Pas du décalage entre les fenêtres en pixels (à l'échelle réduite).
    pyramid_downscale : float
        Facteur de réduction entre chaque niveau de la pyramide.
    pyramid_min_size : int
        On arrête la pyramide quand l'image devient plus petite que ça.
    score_threshold : float
        On ignore les fenêtres dont le score est en dessous de ce seuil.
    seuil_couleur : float
        Pré-filtre couleur : on ne classe une fenêtre que si au moins cette
        fraction de ses pixels est de "couleur panneau" (rouge/bleu/jaune saturé).
        Mettre 0 pour désactiver le filtre.

    Outputs
    --------
    np.ndarray de forme (N, 5) : (y, x, h, w, score)
        Coordonnées dans l'espace de l'image ORIGINALE.
    """
    detections = []

    for scale, resized in _iter_pyramid(image, pyramid_downscale, pyramid_min_size):

        h_img, w_img = resized.shape[:2]

        if h_img < patch_size or w_img < patch_size:
            break

        # Pré-filtre couleur : on calcule le masque "couleur panneau" du niveau,
        # puis son image intégrale, pour compter en O(1) les pixels colorés de
        # chaque fenêtre. On ne garde que les fenêtres assez colorées.
        masque = masque_couleur_panneau(resized)

        # Garde-fou nuit / éclairage étrange : si l'image n'a presque aucune
        # couleur de panneau (cas des photos sombres), on DÉSACTIVE le filtre pour
        # ce niveau (seuil ramené à 0 = on classe tout), pour ne pas rater les
        # panneaux faute de couleur détectable.
        seuil_effectif = seuil_couleur if masque.mean() >= 0.002 else 0.0

        integrale = cv2.integral(masque)  # forme (H+1, W+1)
        aire_fenetre = patch_size * patch_size

        coords = []
        for y in range(0, h_img - patch_size + 1, stride):
            for x in range(0, w_img - patch_size + 1, stride):
                # somme des pixels colorés dans la fenêtre (formule de l'image intégrale)
                total = (integrale[y + patch_size, x + patch_size]
                         - integrale[y, x + patch_size]
                         - integrale[y + patch_size, x]
                         + integrale[y, x])
                if total / aire_fenetre >= seuil_effectif:
                    coords.append((y, x))
        if not coords:
            continue

        # Extraction HOG en batch (un seul appel pour tout le niveau)
        patches = [resized[y : y + patch_size, x : x + patch_size] for y, x in coords]
        feats   = extract_hog_features(patches)          # (N, n_features)
        scores  = _get_scores_batch(clf, feats)          # (N,)

        for (y, x), score in zip(coords, scores):
            if score < score_threshold:
                continue

            detections.append([
                int(y / scale),
                int(x / scale),
                int(patch_size / scale),
                int(patch_size / scale),
                score,
            ])

    if len(detections) == 0:
        return np.empty((0, 5), dtype=np.float32)

    return np.array(detections, dtype=np.float32)


def _iter_pyramid(image, downscale, min_size):
    """
    Génère les niveaux de la pyramide gaussienne avec leur facteur d'échelle.

    À chaque niveau, l'image est réduite d'un facteur `downscale`.
    On s'arrête quand l'image devient plus petite que `min_size`.

    Yields : (scale, resized_uint8)
        scale        — rapport entre la taille réduite et la taille originale
        resized_uint8 — image uint8, prête pour HOG
    """
    scale = 1.0

    for resized in pyramid_gaussian(image, downscale=downscale, channel_axis=-1):

        h, w = resized.shape[:2]

        if h < min_size or w < min_size:
            break

        # pyramid_gaussian renvoie des floats [0.0, 1.0]
        # → il faut repasser en uint8 [0, 255] pour HOG
        resized_uint8 = (resized * 255).astype(np.uint8)

        yield scale, resized_uint8

        scale /= downscale  # on met à jour le scale APRÈS le yield


def _get_scores_batch(clf, feats: np.ndarray) -> np.ndarray:
    """
    Retourne un vecteur de scores pour un batch de features.

    - SVM linéaire / RBF : decision_function (non borné, mais monotone)
    - RandomForest / AdaBoost : predict_proba[:, 1] (probabilité classe positive)
    """
    if hasattr(clf, "decision_function"):
        return clf.decision_function(feats)
    else:
        return clf.predict_proba(feats)[:, 1]
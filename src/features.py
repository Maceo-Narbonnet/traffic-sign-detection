import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern


def _extract_hsv_histogram(img_bgr, bins_h=36, bins_s=32, bins_v=16):
    """Histogramme HSV normalisé : capture la couleur dominante du patch.
    H (teinte) : discrimine rouge/bleu/jaune — canaux clés pour les panneaux.
    S (saturation) : distingue couleurs vives du fond neutre/gris.
    V (valeur) : information de luminosité, moins discriminante."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins_h], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins_s], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins_v], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v])
    norm = hist.sum()
    if norm > 0:
        hist /= norm
    return hist


def _extract_lbp_histogram(gray, P=8, R=1):
    """Histogramme de texture LBP (Local Binary Pattern).

    Le LBP code, pour chaque pixel, le motif de ses P voisins sur un rayon R
    (plus clair / plus sombre). La variante 'uniform' regroupe ces motifs en
    P+2 catégories. On renvoie l'histogramme normalisé de ces catégories, qui
    décrit la TEXTURE locale (lisse vs irrégulière)."""
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_hog_features(panels_list):
    """Extraction des descripteurs : HOG (forme) + histogramme HSV (couleur) +
    histogramme LBP (texture). Les trois vecteurs sont concaténés pour former le
    descripteur final donné au classifieur."""

    features_list = []

    for panel in panels_list:

        if len(panel.shape) == 3:
            gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        else:
            gray_panel = panel

        hog_feat = hog(
            gray_panel,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
        )

        if len(panel.shape) == 3:
            color_feat = _extract_hsv_histogram(panel)
        else:
            color_feat = np.zeros(36 + 32 + 16)

        texture_feat = _extract_lbp_histogram(gray_panel)

        features_list.append(np.concatenate([hog_feat, color_feat, texture_feat]))

    return np.array(features_list)


def masque_couleur_panneau(image_bgr, seuil_s=45, seuil_v=30):
    """Masque binaire (0/1) des pixels de COULEUR PANNEAU : rouge, bleu ou jaune.

    Sert au PRÉ-FILTRAGE des régions : on ne lancera le classifieur que là où il y
    a assez de pixels de couleur de panneau (cf. detect_on_image). Cela écarte les
    fenêtres sur fond gris (bâtiments, route, ciel) avant même la classification.

    Seuils VOLONTAIREMENT permissifs (seuil_s/seuil_v bas) : certaines photos de
    test sont de nuit ou ont un éclairage étrange, où les couleurs sont sombres et
    peu saturées. Trop restreindre ferait rater ces panneaux.

    Rappels OpenCV : H ∈ [0,179], S ∈ [0,255], V ∈ [0,255]. Le rouge "enveloppe"
    les deux extrémités de la teinte (proche de 0 et proche de 179).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Assez coloré (saturé) et pas trop sombre
    sature = (s >= seuil_s) & (v >= seuil_v)

    rouge = (h <= 10) | (h >= 170)
    bleu = (h >= 100) & (h <= 130)
    jaune = (h >= 18) & (h <= 35)

    masque = sature & (rouge | bleu | jaune)
    return masque.astype(np.uint8)


def appliquer_distorsion_fisheye(image, force=0.2):
    """Simule une distorsion fisheye (barrel) sur une image, pour l'AUGMENTATION
    de données. Les bords sont 'gonflés' comme avec un objectif grand-angle, ce
    qui apprend au modèle à reconnaître les panneaux même déformés (le jeu de
    test contiendra des images fisheye).

    force : intensité de la déformation (0 = aucune ; 0.2 = légère, réaliste).
            Une valeur trop forte (0.4) déforme trop les crops et dégrade la
            détection sur les images normales (régression mesurée).
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    rmax2 = cx * cx + cy * cy  # rayon² maximal, pour normaliser la déformation

    # Pour chaque pixel de sortie, on calcule la position d'origine d'où il vient.
    # Le déplacement radial est proportionnel à r² (effet barrel typique du fisheye).
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    dx = j - cx
    dy = i - cy
    facteur = 1 + force * (dx * dx + dy * dy) / rmax2
    map_x = (cx + dx * facteur).astype(np.float32)
    map_y = (cy + dy * facteur).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def undistort_fisheye(image):
    """Correction de la distorsion fisheye"""

    height, width = image.shape[:2]

    focal_length = width / 2.0
    center_x = width / 2.0
    center_y = height / 2.0

    K = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    D = np.array([-0.2, 0.1, 0.0, 0.0], dtype=np.float32) #coefficients de distorsion fisheye (k1, k2, p1, p2)

    #np.eye(3) pour la matrice de rectification (identité dans ce cas), balance=0.0 pour conserver le champ de vision d'origine
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=0.0)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2)

    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_image
import csv 
import cv2
from pathlib import Path

def load_pos_data(image_pos_path, csv_path):
    """Charge les images positives avec les bounding box associées"""

    images_list = []
    bounding_boxes_list = []

    image_path = Path(image_pos_path) #nécessaire pour bonne gestion des chemins d'accès

    image_paths = sorted(p for p in image_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg']) #filtre les fichiers d'images dans l'ordre

    for img_path in image_paths:

        csv_filename = img_path.stem + ".csv" #le nom du fichier CSV doit correspondre au nom de l'image (sans extension)
        current_csv_path = csv_path / csv_filename
        if not current_csv_path.exists():
            print(f"Erreur : le fichier {current_csv_path} n'existe pas pour l'image {img_path}.")
            continue


        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Erreur de chargement de l'image : {img_path}")
            continue

        with open(current_csv_path, "r", encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            img_bboxes = []

            for row in csv_reader:
                
                if not row or len(row) < 4:
                    print(f"Erreur de format dans le fichier {current_csv_path} : ligne vide ou nombre de colonnes insuffisant.")
                    continue

                try :

                    ymin = int(row[0])
                    xmin = int(row[1])
                    hauteur = int(row[2])
                    largeur = int(row[3])

                    xmax = xmin + largeur
                    ymax = ymin + hauteur

                    bbox = [xmin, ymin, xmax, ymax]

                    img_bboxes.append(bbox)
                    

                except ValueError:
                    print(f"Erreur de conversion dans le fichier {current_csv_path} : valeurs non entières dans la ligne {row}.")
                    continue
            
            if img_bboxes:
                images_list.append(img)
                bounding_boxes_list.append(img_bboxes)

    total_bboxes = sum(len(bboxes) for bboxes in bounding_boxes_list)
    print(f"Chargement réussi : {len(images_list)} images et {total_bboxes} bounding boxes récupérées.")
    return images_list, bounding_boxes_list

def load_neg_data(image_neg_path):
    """Charge les images négatives"""
    
    images_list = []

    image_path = Path(image_neg_path) #nécessaire pour bonne gestion des chemins d'accès

    image_paths = sorted(p for p in image_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg']) #filtre les fichiers d'images dans l'ordre

    for img_path in image_paths:
        img = cv2.imread(str(img_path))

        if img is not None:
            images_list.append(img)
        else :
            print(f"Erreur de chargement de l'image (FONCTION LOAD_NEG_DATA)")

    print(f"Chargement réussi : {len(images_list)} images récupérées.")

    return images_list



def crop_and_resize_pannels(images_list, bbox_list, target_size=(128,128)):
    """Découpe les panneaux en CARRÉ puis les redimensionne à une taille fixe.

    Pour rendre la boîte carrée, on agrandit le côté le plus court en prenant du
    VRAI contexte autour du panneau dans l'image (et non du noir). Cela évite que
    le modèle apprenne à reconnaître les bandes noires du remplissage, qui
    n'existent jamais dans les fenêtres glissantes réelles."""

    processed_images = []
    target_width, target_height = target_size
    for idx, (img, bboxes) in enumerate(zip(images_list, bbox_list)): #la fonction zip permet de parcourir les 2 listes en même temps
        img_h, img_w = img.shape[:2]
        for bbox in bboxes :
            xmin, ymin, xmax, ymax = bbox

            # On borne d'abord la boîte aux limites de l'image
            x_min_clipped = max(0, min(xmin, img_w))
            y_min_clipped = max(0, min(ymin, img_h))
            x_max_clipped = max(0, min(xmax, img_w))
            y_max_clipped = max(0, min(ymax, img_h))

            largeur = x_max_clipped - x_min_clipped
            hauteur = y_max_clipped - y_min_clipped

            if largeur <= 0 or hauteur <= 0:
                print(f"Erreur de découpage : bbox {bbox} vide après ajustement. Image index : {idx}.")
                continue

            # Côté du carré = la plus grande dimension du panneau.
            # On centre ce carré sur le centre du panneau.
            cote = max(largeur, hauteur)
            centre_x = (x_min_clipped + x_max_clipped) // 2
            centre_y = (y_min_clipped + y_max_clipped) // 2

            x1 = centre_x - cote // 2
            y1 = centre_y - cote // 2
            x2 = x1 + cote
            y2 = y1 + cote

            # Si le carré dépasse un bord, on le DÉCALE (au lieu de le rétrécir)
            # pour qu'il reste carré et rempli de vrai contenu.
            if x1 < 0:
                x2 -= x1; x1 = 0
            if y1 < 0:
                y2 -= y1; y1 = 0
            if x2 > img_w:
                x1 -= (x2 - img_w); x2 = img_w
            if y2 > img_h:
                y1 -= (y2 - img_h); y2 = img_h

            # Sécurité : si le panneau est plus grand que l'image (rare), on borne.
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w, x2); y2 = min(img_h, y2)

            panel_crop = img[y1:y2, x1:x2]

            if panel_crop.size == 0:
                print(f"Erreur de découpage : bbox {bbox} hors des limites. Image index : {idx}.")
                continue

            # Redimensionnement à la taille cible. INTER_AREA est adapté à la
            # réduction de taille (rééchantillonnage tenant compte des voisins).
            panel_resized = cv2.resize(panel_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

            processed_images.append(panel_resized)

    print(f"Traitement réussi : {len(processed_images)} panneaux traités.")

    return processed_images
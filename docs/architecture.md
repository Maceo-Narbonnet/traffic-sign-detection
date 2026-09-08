# Déroulement du code — Détection de panneaux (Groupe C)

Documentation technique du fonctionnement du code : enchaînement des étapes et
rôle de **chaque fonction**. Le niveau de détail est proportionnel à l'importance
de la fonction (les fonctions clés sont développées, les utilitaires résumés).

---

## 1. Vue d'ensemble

Le projet a **deux points d'entrée** (à lancer depuis le dossier `src/`) :

- **`evaluation.py`** — entraîne un modèle et **mesure** ses performances (AP, F1…)
  sur le jeu de validation `data/val`.
- **`predict_test.py`** — entraîne (ou recharge) le modèle, **prédit** sur
  `data/test`, écrit le CSV de soumission et enregistre les images annotées.

Un troisième script, **`visualiser.py`**, sert à produire des images d'analyse sur
une image précise (vérité + détections).

Chaîne de traitement (du début à la fin) :

```
images + annotations
        │
        ▼
 préparation des exemples      (data_load.py : load_pos_data, crop_and_resize_pannels,
        │                       load_neg_data ; training.py : extract_negative_samples)
        ▼
 descripteurs HOG + HSV        (features.py : extract_hog_features)
        │
        ▼
 entraînement Random Forest    (training.py : train_panel_classifier)
        │
        ▼
 hard negative mining          (training.py : extract_hard_negatives) → réentraînement
        │
        ▼
 détection multi-échelle       (detector/sliding_window.py : detect_on_image)
        │
        ▼
 suppression des doublons       (detector/nms.py : non_max_suppression)
        │
        ▼
 mesure (AP, F1) OU sortie CSV  (metrics.py : evaluer_detection / predict_test.py)
```

---

## 2. Convention de format des boîtes (IMPORTANT)

Deux formats coexistent dans le code, source d'erreurs si on les confond :

| Format | Où il est utilisé |
|---|---|
| `[xmin, ymin, xmax, ymax]` | sortie de `load_pos_data`, entrée de `crop_and_resize_pannels` et de `extract_hard_negatives_positifs` |
| `[y, x, h, w]` | partout ailleurs : `detect_on_image`, `iou`, `non_max_suppression`, `metrics.py`, `charger_verites_terrain` |

Le CSV d'annotation est, lui, au format `ymin, xmin, hauteur, largeur, difficulté`
(donc directement `[y, x, h, w]` pour les 4 premières colonnes).

---

## 3. Les fichiers et leurs fonctions

### 3.1 `config.py`
Aucune fonction : des constantes et le **figeage du hasard**.
- `np.random.seed(42)` et `random.seed(42)` : exécutés à l'import → résultats
  **reproductibles** (échantillonnage des négatifs, hard negative mining).
- Constantes : `PATCH_SIZE=64`, `STRIDE=5`, `PYRAMID_DOWNSCALE=1.25`,
  `PYRAMID_MIN_SIZE=64`, `SCORE_THRESHOLD=0.8`.
- ⚠️ `STRIDE` et `SCORE_THRESHOLD` ne sont que les **valeurs par défaut** de
  `detect_on_image`. Les scripts `evaluation.py`/`predict_test.py` les **écrasent**
  (stride=16, score=0.3). Seul le vieux `main.py` utilise les valeurs de config.

### 3.2 `data_load.py`

**`load_pos_data(image_pos_path, csv_path)`** — *clé*
Parcourt les images positives (triées), lit le CSV associé à chacune, convertit
chaque ligne `ymin,xmin,h,l` en `[xmin, ymin, xmax, ymax]`. Ne garde une image que
si elle a au moins une boîte valide.
- **Entrée** : chemins des dossiers images pos et labels.
- **Sortie** : `(images_list, bounding_boxes_list)` — liste d'images BGR et, pour
  chacune, sa liste de boîtes `[xmin,ymin,xmax,ymax]`.

**`load_neg_data(image_neg_path)`** — *simple*
Charge toutes les images négatives (sans panneau).
- **Sortie** : liste d'images BGR.

**`crop_and_resize_pannels(images_list, bbox_list, target_size=(128,128))`** — *clé*
Découpe chaque panneau en **carré** puis le redimensionne à `target_size`
(appelé avec `(64,64)`). Pour rendre carré : côté = plus grande dimension de la
boîte, carré centré sur le panneau, **décalé** s'il dépasse un bord (jamais de
bandes noires → on prend du vrai contexte). Redimensionnement `INTER_AREA`.
- **Sortie** : liste d'imagettes carrées `target_size` (les exemples positifs).
- *Détail important* : c'est ce « vrai contexte au lieu du noir » qui a fait passer
  l'AP de 0,095 à 0,173.

### 3.3 `features.py`

**`extract_hog_features(panels_list)`** — *clé*
Pour chaque imagette : calcule le **HOG** sur la version niveaux de gris
(9 orientations, cellules 8×8, blocs 2×2, L2-Hys) = **forme**, l'**histogramme HSV**
= **couleur**, et l'**histogramme LBP** = **texture**, puis **concatène** les trois.
Si l'imagette est en niveaux de gris, la partie couleur est mise à zéro.
- **Entrée** : liste d'imagettes (BGR de préférence).
- **Sortie** : matrice `(N, 1858)` — un vecteur descripteur par imagette
  (1764 HOG + 84 HSV + 10 LBP).

**`_extract_hsv_histogram(img_bgr, bins_h=36, bins_s=32, bins_v=16)`** — *interne*
Convertit en HSV et calcule un histogramme normalisé des 3 canaux (teinte,
saturation, valeur), concaténé en un vecteur de 84 valeurs.

**`_extract_lbp_histogram(gray, P=8, R=1)`** — *interne (texture)*
Calcule le LBP (Local Binary Pattern, variante 'uniform') sur l'imagette en gris et
en renvoie l'histogramme normalisé (10 valeurs). Décrit la **texture** locale
(surface lisse de panneau vs fond texturé).

**`masque_couleur_panneau(image_bgr, seuil_s=80, seuil_v=40)`** — *clé (pré-filtre couleur)*
Renvoie un **masque binaire** (0/1) des pixels de couleur panneau (rouge / bleu /
jaune **saturés** et pas trop sombres), via une conversion HSV. Utilisé par
`detect_on_image` pour ne classer que les fenêtres assez colorées.
- ⚠️ À ne pas confondre avec l'**histogramme HSV** (une *feature* du classifieur) :
  ici c'est un **pré-filtre de régions**, en amont de la classification.

**`appliquer_distorsion_fisheye(image, force=0.2)`** — *présente, NON utilisée*
Simule une distorsion « grand-angle » (effet barrel) via `cv2.remap`. Servait à
l'augmentation fisheye, **désactivée** car elle dégradait le cas normal.

**`undistort_fisheye(image)`** — *présente, NON utilisée*
Tente de **corriger** une distorsion fisheye (`cv2.fisheye`). Non branchée (on ne
sait pas quelles images de test sont fisheye ni avec quels coefficients).

### 3.4 `training.py`

**`extract_negative_samples(neg_image_list, samples_per_image=2, target_size=(128,128))`** — *clé*
Tire `samples_per_image` morceaux **aléatoires** (à la taille cible) dans chaque
image négative. Appelé avec `samples_per_image=10`, `(64,64)`.
- **Sortie** : liste d'imagettes négatives (le « fond »).

**`extract_hard_negatives(neg_image_list, model, target_size=(64,64), crops_per_image=50, tailles=(64,96,128,200,300))`** — *clé*
**Hard negative mining multi-échelle.** Tire des morceaux de **tailles variées**
dans les images négatives, les ramène en 64×64, les décrit (HOG) et les classe en
**un seul batch** ; garde ceux que le modèle prend à tort pour des panneaux (faux
positifs). Rapide (pas de fenêtre glissante).
- **Sortie** : liste d'imagettes « pièges » à rajouter aux négatifs.

**`extract_hard_negatives_positifs(pos_image_list, bbox_list, model, target_size=(64,64), crops_per_image=40, tailles=..., iou_max=0.2)`** — *présente, NON appelée*
Hard negative mining **ciblé** : sur les images avec panneau, génère des fenêtres
aléatoires + des **sous-fenêtres de panneaux** (coins/fragments), ne garde que les
**mal cadrées** (IoU < 0,2 avec toute vérité) que le modèle confond avec un panneau.
Visait le problème des triangles ; essai **non concluant**, conservé pour mémoire.

**`train_panel_classifier(X_data, Y_data, test_size=0.2, modele_type="rf")`** — *clé*
Entraîne le classifieur.
1. `train_test_split` 80/20 **stratifié** (`random_state=42`). ⚠️ Ce split est **au
   niveau patch** et ne sert qu'à afficher l'accuracy / le rapport de
   classification (information, **pas** le score de détection).
2. Crée le modèle : `RandomForestClassifier(100, random_state=42)` si
   `modele_type="rf"` ; sinon un **SVM linéaire** standardisé + calibré
   (`make_pipeline(StandardScaler, CalibratedClassifierCV(LinearSVC))`).
3. `.fit`, puis affiche accuracy, rapport, matrice de confusion.
- **Sortie** : le modèle entraîné.

### 3.5 `detector/sliding_window.py`

**`detect_on_image(image, clf, patch_size=64, stride=STRIDE, pyramid_downscale=1.25, pyramid_min_size=64, score_threshold=SCORE_THRESHOLD)`** — *clé (cœur de la détection)*
Cherche les panneaux dans une image entière.
1. Parcourt les niveaux de la pyramide (`_iter_pyramid`).
2. À chaque niveau : calcule le **masque couleur** (`masque_couleur_panneau`) et son
   **image intégrale** (`cv2.integral`), puis ne garde que les positions de fenêtre
   dont la fraction de pixels colorés ≥ `seuil_couleur` (pré-filtre couleur).
3. Décrit ces fenêtres en batch (`extract_hog_features`) et les note
   (`_get_scores_batch`).
4. Garde les fenêtres de score ≥ `score_threshold`, ramène leurs coordonnées à
   l'image d'origine (division par `scale`).
- **Appelé avec** `stride=16`, `score_threshold=0.3`.
- **Sortie** : tableau `(N,5)` = `[y, x, h, w, score]` en coordonnées d'origine.

**`_iter_pyramid(image, downscale, min_size)`** — *interne, important*
Générateur : produit l'image réduite à chaque niveau (`pyramid_gaussian`, facteur
1,25), repassée en uint8, avec son facteur d'échelle `scale`. S'arrête quand un côté
passe sous `min_size`.

**`_get_scores_batch(clf, feats)`** — *interne, important*
Renvoie un score par fenêtre : `decision_function` si le modèle en a une, sinon
`predict_proba[:,1]`. **Pour le Random Forest → `predict_proba[:,1]`** (score ∈ [0,1]).

### 3.6 `detector/nms.py`

**`iou(box_a, box_b)`** — *clé (réutilisée partout)*
Calcule l'Intersection over Union de deux boîtes `[y,x,h,w]` (0 à 1).

**`non_max_suppression(boxes, iou_threshold=0.3, contenu_threshold=0.6)`** — *clé*
Supprime les détections en double. Trie par score décroissant, garde la meilleure,
rejette celles dont l'**IoU > 0,3** avec une boîte déjà gardée, et recommence.
- **Entrée/Sortie** : tableaux `(N,5)` / `(M,5)` `[y,x,h,w,score]`.
- *État actuel* : un critère de « contenance » (via `intersection_sur_min`) est
  présent dans le fichier mais **désactivé** ; la NMS se base sur l'IoU seul.

**`intersection_sur_min(box_a, box_b)`** — *présente, actuellement inactive*
Intersection divisée par l'aire de la **plus petite** boîte (≈1 si une petite boîte
est incluse dans une grande). Conçue pour rattraper les boîtes imbriquées que l'IoU
rate ; non utilisée tant que le critère de contenance est désactivé.

### 3.7 `metrics.py`

**`evaluer_detection(predictions_par_image, verites_par_image, iou_seuil=0.5)`** — *clé (fonction d'ensemble)*
Calcule et affiche AP, précision, rappel, F1 sur un ensemble d'images. Enchaîne les
fonctions ci-dessous.
- **Sortie** : `(ap, precision, rappel, f1, precisions, rappels)` — les deux
  derniers tableaux servent à tracer la courbe précision/rappel.

**`evaluer_image(predictions, verites, iou_seuil=0.5)`** — *clé*
Sur **une** image : trie les prédictions par score, apparie chacune à la vraie boîte
de meilleur IoU ; **vrai positif** si IoU ≥ 0,5 et vérité encore libre, sinon faux
positif (une vérité ne sert qu'une fois).
- **Sortie** : liste de `(score, est_vrai_positif)` + nombre de vraies boîtes.

**`courbe_precision_rappel(resultats_globaux, nb_verites_total)`** — *clé*
Trie toutes les détections par score décroissant et calcule précision et rappel
**cumulés** → deux tableaux (un point par détection).

**`average_precision(precisions, rappels)`** — *clé*
Aire sous la courbe précision/rappel (avec « enveloppe » à la PASCAL VOC). C'est
l'AP / AUC, la métrique principale.

**`precision_rappel_f1(precisions, rappels)`** — *clé*
Renvoie précision, rappel et F1 **au point qui maximise le F1** (les chiffres
scalaires à reporter).

**`meilleur_f1(precisions, rappels)`** — *présente, plus utilisée*
Renvoyait seulement le meilleur F1 ; remplacée par `precision_rappel_f1`.

### 3.8 `evaluation.py`

**`entrainer_modele(base_dir, target_size=(64,64), neg_par_image=10, hard_neg=True, augment_fisheye=False, modele_type="rf")`** — *clé (orchestration de l'entraînement)*
Enchaîne : `load_pos_data` → `crop_and_resize_pannels` → (`appliquer_distorsion_fisheye`
si activé, **off** par défaut) → `extract_hog_features` (positifs) ; `load_neg_data`
→ `extract_negative_samples` → `extract_hog_features` (négatifs) ; fusion →
`train_panel_classifier` ; puis `extract_hard_negatives` → réentraînement.
- **Sortie** : le modèle final.

**`charger_verites_terrain(dossier_csv)`** — *clé*
Lit tous les CSV d'un dossier et renvoie `{ nom_image : boîtes [y,x,h,w] }`
(ignore les boîtes de taille nulle).

**`evaluer_sur_validation(modele, base_val, seuil_score=0.3, iou_seuil=0.5, stride=16, max_images=None)`** — *clé*
Met `modele.n_jobs=-1`, charge les vérités, et pour chaque image de validation :
`detect_on_image` → `non_max_suppression` → stocke prédictions et vérités ; appelle
`evaluer_detection`. `max_images` limite à N images (essais rapides).

**`main()`** — *point d'entrée*
Lit les arguments (`N` images, `svm`), appelle `entrainer_modele` puis
`evaluer_sur_validation`. Chemins calculés par rapport à l'emplacement du fichier.

### 3.9 `predict_test.py`

**`predire_sur_test(modele, dossier_test, fichier_sortie, seuil_score=0.3, stride=16, afficher=False, enregistrer_images=True, seuil_affichage=0.7)`** — *clé*
Pour chaque image de `data/test` : `detect_on_image` → `non_max_suppression` →
écrit les lignes CSV `num, y, x, h, w, score` ; dessine les boîtes de score ≥
`seuil_affichage` et enregistre l'image dans `images_results/<num>_<horodatage>.png`
(`afficher=True` ouvre en plus une fenêtre).
- **Sortie** : fichier `predictions_test.csv`.

**`main()`** — *point d'entrée*
Si `modele_panneaux.joblib` existe et pas d'argument `retrain` → `joblib.load` ;
sinon `entrainer_modele` puis `joblib.dump`. Puis `predire_sur_test`. Arguments :
`retrain` (force l'entraînement), `show` (affiche les fenêtres).

### 3.10 `visualiser.py`

**`visualiser(chemin_image, seuil_affichage=0.5, montrer_brutes=False, nb_brutes=8, stride=16, seuil_score=0.3)`** — *utilitaire d'analyse*
Sur **une** image : recharge le modèle (`joblib`), dessine en **bleu** les vraies
boîtes (si CSV trouvé via `charger_verites_image`), en **vert** les détections
finales (après NMS) avec leur score, et en option en **rouge fin** les meilleures
détections **avant** NMS. Enregistre dans `figures/<nom>_resultat.png`.

**`charger_verites_image(stem)`** — *utilitaire*
Cherche le CSV de l'image dans `train/labels` puis `val/labels` et renvoie ses
boîtes `[y,x,h,w]`.

**`_est_float(s)`**, **`main()`** — *utilitaires* : analyse des arguments de ligne
de commande (chemin image, seuil, mot-clé `brutes`).

### 3.11 `main.py` — *script historique (legacy)*
Ancien pipeline qui entraîne puis détecte sur **une seule image codée en dur** et
affiche le résultat avec `imshow` (pas de CSV). Utilise les valeurs de `config.py`
(`STRIDE=5`, `SCORE_THRESHOLD=0.8`). Remplacé par `evaluation.py` et
`predict_test.py` ; gardé pour référence.

---

## 4. Déroulé d'une exécution (commandes)

### 4.1 Évaluer — `python evaluation.py [N] [svm]`
1. Import des modules → `config.py` fige le hasard.
2. `main()` lit `N` (nb d'images) et le modèle (`rf`/`svm`).
3. `entrainer_modele(data/train)` : positifs + négatifs → entraînement →
   hard negative mining → réentraînement.
4. `evaluer_sur_validation(data/val, max_images=N)` : détection + NMS sur chaque
   image, puis `evaluer_detection` → affiche **AP, précision, rappel, F1**.
- Exemples : `python evaluation.py 25` (rapide), `python evaluation.py` (68 images),
  `python evaluation.py 25 svm` (comparer le SVM).

### 4.2 Prédire sur le test — `python predict_test.py [retrain] [show]`
1. `main()` : recharge `modele_panneaux.joblib` ou entraîne + sauvegarde.
2. `predire_sur_test(data/test)` : détection + NMS → `predictions_test.csv` +
   images annotées dans `images_results/`.
- `retrain` force un nouvel entraînement ; `show` affiche les fenêtres.

### 4.3 Visualiser une image — `python visualiser.py <image> [seuil] [brutes]`
Recharge le modèle et enregistre l'image annotée (vérité + détections) dans
`figures/`. Le mot-clé `brutes` ajoute les meilleures boîtes avant NMS.

---

## 5. Paramètres effectifs

| Paramètre | Valeur réelle utilisée |
|---|---|
| Taille des imagettes | 64×64 px |
| Pas de la fenêtre | **16** px (config dit 5, écrasé à l'appel) |
| Pyramide | facteur 1,25 ; arrêt sous 64 px |
| Seuil de score (détection) | **0,3** (config dit 0,8, écrasé) |
| Seuil de pré-filtre couleur | 0,05 (fraction de pixels colorés ; 0 = désactivé) |
| Seuil d'IoU (NMS) | 0,3 |
| Seuil d'IoU (appariement éval) | 0,5 |
| Seuil d'affichage (images test) | 0,7 |
| Négatifs aléatoires / image | 10 |
| Hard negatives / image négative | 50, tailles {64,96,128,200,300} |
| Classifieur | Random Forest, 100 arbres, `random_state=42` |
| HOG | 9 orientations, cellules 8×8, blocs 2×2, L2-Hys |
| Couleur | histogramme HSV (36+32+16) |
| Texture | histogramme LBP (uniform, P=8, R=1 → 10 valeurs) |
| Taille du descripteur | 1858 (1764 HOG + 84 HSV + 10 LBP) |
| Graines aléatoires | NumPy + `random` = 42 |

---

## 6. Points de vigilance
- **`config.py` est partiellement trompeur** : `STRIDE` et `SCORE_THRESHOLD` n'y
  reflètent pas les valeurs réellement utilisées (16 et 0,3), seulement les défauts.
- **Le « 98 % » à l'entraînement** vient du split patch interne de
  `train_panel_classifier` : c'est de la classification d'imagettes, **pas** la
  performance de détection (l'AP, bien plus basse).
- **Deux formats de boîtes** cohabitent (voir section 2) : attention aux conversions.
- Fonctions **présentes mais non actives** dans le flux actuel :
  `appliquer_distorsion_fisheye`, `undistort_fisheye`,
  `extract_hard_negatives_positifs`, `meilleur_f1`, le critère de contenance de
  `non_max_suppression` (`intersection_sur_min`), et le script `main.py`.

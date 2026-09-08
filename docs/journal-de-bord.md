# JOURNAL DE BORD — PROJET SY32 P2026

> Détection de panneaux de signalisation (méthodes classiques, deep learning interdit)

```text
Ce fichier suit l'avancement : ce qu'on a fait, les résultats mesurés, et les
problèmes rencontrés. Mis à jour au fur et à mesure.
```

## 1. MISE EN PLACE DE L'ÉVALUATION

```text
PROBLÈME INITIAL
  Le pipeline mesurait seulement l'accuracy du classifieur au niveau "patch"
  (98,96 %), ce qui est TROMPEUR : ça ne dit rien sur la qualité de la
  détection sur image entière (le vrai objectif, et ce qui est noté).
  Aucune des métriques exigées par l'énoncé (IoU, précision/rappel, AP/AUC, F1)
  n'était calculée.

CE QU'ON A FAIT
  - Créé src/metrics.py : IoU (réutilise celui de detector/nms.py), courbe
    précision/rappel, Average Precision (AP/AUC), et précision + rappel + F1
    scalaires reportés au point qui maximise le F1 (pour noter l'algo simplement).
  - Créé un set de VALIDATION : on a déplacé 1 image sur 5 du dossier train vers
    un nouveau dossier data/val (273 images train / 68 images val), avec leurs
    annotations. Découpage PAR IMAGE (pas par patch) pour éviter la fuite de
    données. Vérifié : aucun chevauchement train/val.
  - Créé src/evaluation.py : entraîne sur data/train, détecte sur data/val,
    calcule l'AP et le F1.

  ATTENTION : avant le rendu final, il faudra REMETTRE les 68 images de val
  dans data/train (l'énoncé veut le dataset complet et corrigé).
```

## 2. RÉSULTATS MESURÉS (val = 68 images, IoU >= 0.5, stride=16, seuil_score=0.3)

```text
  Étape                              | AP (AUC) | F1     | Détections totales
  -----------------------------------|----------|--------|-------------------
  Baseline (code initial)            |  0,095   | 0,224  | 8 511
  + fix padding noir des positifs    |  0,173   | 0,272  | 19 341
  + hard negative mining multi-éch.  |  0,293   | 0,387  | 11 039
  + augmentation fisheye             |  (à vérifier : non-régression)

  (147 vrais panneaux au total dans le set de validation de 68 images.)
  Le hard-neg fait DEUX gains : AP 0,173 -> 0,293 (+69 %) ET détections
  19 341 -> 11 039 (-43 %, moins de faux positifs).

  --- Mesures rapides sur sous-ensemble de 25 images (47 panneaux) ---
  Sert au garde-fou anti-régression (même sous-ensemble = comparable entre elles,
  mais PAS comparable aux lignes 68 images ci-dessus) :
    hard-neg, sans fisheye (non figé) : AP=0,284  F1=0,429  dét=4985
    hard-neg, fisheye force=0,4       : AP=0,221  F1=0,343  dét=6198  <- RÉGRESSION
    hard-neg, fisheye force=0,2       : AP=0,219  F1=0,373  dét=5933  (idem, écarté)
    *** RÉFÉRENCE REPRODUCTIBLE (graine figée) : AP=0,279  F1=0,424  dét=2963 ***

  REPRODUCTIBILITÉ : on a constaté qu'un même code donnait des AP différentes
  (0,284 puis 0,232) car le module `random` de Python (échantillonnage des
  négatifs + hard negative mining) n'était pas figé : seul NumPy l'était. Ajout
  de random.seed(42) dans config.py -> runs désormais reproductibles (AP=0,279).
  Les anciens chiffres (0,095/0,173/0,293) venaient de runs non figés : la
  tendance reste valable (écarts >> bruit), mais pour le rapport on peut relancer
  les configs clés avec la graine si on veut des nombres 100 % reproductibles.

  --- NOUVELLE PHASE : base de données nettoyée + nouveau val (2026-06-13) ---
  La base a été re-téléchargée/nettoyée (annotations corrigées). Le set de
  validation a été refait : 329 images train / 12 images val (0302-0313),
  chevauchement train/val = 0 (vérifié). /!\ val petit (12 images, 33 panneaux)
  -> AP bruitée, ne détecte que les gros effets ; NON comparable aux chiffres
  ci-dessus (autres images). Nouveau baseline sur ces 12 images (modèle sur 329,
  sans filtre couleur) : AP=0,167  F1=0,273  précision=0,545  rappel=0,182  dét=1218.

  CONSTAT : force=0,4 déforme trop les crops 64x64 -> AP normale chute de 0,284
  à 0,221 (-22 %). On baisse force à 0,2 (distorsion légère, plus réaliste car
  dans une vraie photo fisheye un panneau occupe une petite zone donc subit une
  déformation locale douce). À revérifier sur les 25 mêmes images.

  CONSTAT CLÉ : 99 % d'accuracy patch -> seulement 0,095 d'AP au départ.
  Preuve que l'accuracy patch est trompeuse : les ~1 % d'erreur, multipliés par
  les milliers de fenêtres glissantes par image, donnent des milliers de faux
  positifs (sur-détection massive).

  --- PREMIÈRE SOUMISSION SUR LE SERVEUR UTC (2026-06-13, "test1_mac") ---
  Fichier detection.csv (385 ko, seuil_score=0,3). Scores du site :
    Précision = 0,82 %  |  Rappel = 30,97 %  |  F1 = 1,59 %  |  AUC = 18,59 %
    (rappel standard 37,72 / difficile 9,52 / fisheye 22,50)
  DIAGNOSTIC : rappel correct mais PRÉCISION minuscule -> ~99 fausses boîtes sur
  100. Cause = seuil_score laissé bas (0,3, utile pour la courbe AP) en soumission
  -> des milliers de fenêtres faiblement notées passent. C'est la version "image
  entière" du piège du début (bon classifieur noyé sous les faux positifs).
  CORRECTION : seuil_score=0,6 pour la soumission (predict_test.py, predire_sur_test)
  -> ne garde que les détections sûres. Précision attendue en forte hausse, rappel
  un peu plus bas = meilleur F1/AUC. Réglage de point de fonctionnement, sans
  réentraînement. À RESOUMETTRE pour mesurer le gain réel.
  Classement site à cette soumission : 5e/6 (par AUC).

  --- DEUXIÈME SOUMISSION ("test2", seuil_score=0,6, 2026-06-13 18:36) ---
  Précision = 15,16 %  |  Rappel = 26,87 %  |  F1 = 19,38 %  |  AUC = 23,68 %
  CSV passé de 385 ko à 18 ko. CONFIRMÉ : relever le seuil 0,3 -> 0,6 fait
  exploser la précision (0,82 -> 15,16 %, x18), avec un rappel quasi inchangé
  (30,97 -> 26,87 %). F1 x12 (1,59 -> 19,38) et AUC +5 pts (18,59 -> 23,68).
  => le défaut dominant était bien le seuil de soumission, pas le modèle.
  Piste pour gagner encore : tester seuil 0,5 (récupérer du rappel) et 0,7
  (plus de précision) pour trouver le meilleur compromis AUC/F1.

  --- FIGURES DU RAPPORT (src/figures.py, dossier figures/) ---
  Script qui CALCULE les figures : stats dataset (tailles, difficulté, formes)
  depuis les 662 boîtes annotées ; graphes de résultats (progression AP, RF vs SVM,
  ablation couleur/LBP, soumission UTC) ; courbe précision/rappel via une vraie éval
  ("python src/figures.py pr" -> AP=0,185 P=0,467 R=0,212 F1=0,292, cohérent journal).
  Exemples qualitatifs générés via visualiser.py : 0305 (réussite) et 0304 (échec =
  façade brique sur-détectée, illustre le 0,82 % de précision).
  Stats dataset mesurées : 341 images, 662 boîtes, 458 faciles / 204 difficiles,
  14 panneaux < 16 px, petit côté médian 79 px, ratio l/h médian 1,00 (quasi carrés).
  rapport.txt réécrit et à jour (2 phases, descripteur 1858, soumission UTC, toutes
  les figures intégrées). Reste : noms/date/contributions à remplir + compiler.
```

## 2bis. LES DEUX SEUILS (à ne pas confondre)

```text
  seuil_score : confiance du modèle (0 à 1) pour garder une détection.
                N'intervient PAS à l'entraînement, seulement à la détection.
                On le met bas (0,3) pour l'évaluation afin de tracer toute la
                courbe AP. Le modèle reste le même quel que soit ce seuil.

  iou_seuil   : recouvrement géométrique entre deux boîtes (0 à 1).
                - dans la NMS (0,3) : deux détections sont-elles des doublons ?
                - à l'évaluation (0,5) : une détection tombe-t-elle bien sur une
                  vraie boîte (= vrai positif) ?

  En résumé : seuil_score = "est-ce un panneau ?", iou_seuil = "les deux boîtes
  sont-elles au même endroit ?". Aucun des deux ne modifie l'entraînement.
```

## 3. AMÉLIORATIONS APPLIQUÉES (détail)

```text
3.1 FIX DU PADDING NOIR  (src/data_load.py, crop_and_resize_pannels)
  Problème : les panneaux non carrés étaient complétés avec des bandes noires
  pour devenir carrés. Le HOG apprenait ces faux bords noirs, qui n'existent
  jamais dans les fenêtres glissantes réelles -> décalage entraînement/réalité.
  Fix : on agrandit le côté court en prenant du VRAI contexte dans l'image
  (carré centré sur le panneau, décalé s'il dépasse un bord), au lieu de noir.
  Résultat : AP 0,095 -> 0,173 (+82 %).

3.5 HARD NEGATIVE MINING CIBLÉ SUR IMAGES POSITIVES  (src/training.py)
  PROBLÈME : panneaux TRIANGULAIRES. Le modèle donne plus de confiance à une
  petite bbox dans un COIN du triangle (deux arêtes franches = fort gradient HOG)
  qu'à la bbox cadrant tout le triangle. Du coup la NMS garde la petite bbox.
  Cause : le classifieur note la "texture panneau-like", pas la qualité du cadrage.
  FIX : extract_hard_negatives_positifs(). Sur les images AVEC panneau, on génère
  des fenêtres candidates (a) aléatoires multi-échelle, (b) sous-fenêtres À
  L'INTÉRIEUR de chaque panneau (40-70 % de sa taille = coins/fragments). On ne
  garde que les fenêtres MAL cadrées (IoU < 0,5 avec toutes les vraies boîtes,
  pour ne jamais étiqueter un vrai panneau en négatif), on les classe, et on
  garde celles que le modèle prend pour des panneaux -> négatifs "partiels".
  Effet attendu : la confiance des coins/fragments chute, le panneau entier
  redevient le plus confiant -> la NMS "garder le plus confiant" choisit bien.
  Rapide (crops aléatoires + un seul batch HOG/prédiction, pas de fenêtre glissante).

  RÉGLAGE IMPORTANT (bug rencontré) : avec iou_max=0.5, on étiquetait négatif des
  fenêtres PRESQUE correctes (IoU jusqu'à 0.5) + de gros fragments ressemblant à un
  panneau -> le rappel patch s'est effondré (0.88 -> 0.36) et PLUS RIEN n'était
  détecté. Correction : iou_max=0.2 (on ne capture que les fenêtres clairement mal
  cadrées ; les IoU 0.2-0.5 ambiguës sont ignorées) + sous-fenêtres plus petites
  (0.25-0.45 de la taille du panneau, = vrais coins). Le ciblage des coins de
  triangle est conservé (IoU ~0.04-0.16) sans tuer la détection.
  RÉSULTAT : même après réglage (iou_max=0.2), l'approche n'a PAS été concluante
  en test (détection toujours dégradée). Approche écartée pour l'instant : retour
  à la version de base du hard-neg (uniquement images négatives, AP 0.293).
  La fonction extract_hard_negatives_positifs reste dans training.py (non appelée)
  pour garder la trace. Le problème des triangles reste ouvert -> on tente d'abord
  un autre classifieur (voir 3.6).

3.6 COMPARAISON DE CLASSIFIEURS : RANDOM FOREST vs SVM  (src/training.py)
  Ajout du choix du modèle dans train_panel_classifier(modele_type="rf"|"svm").
  - "rf"  : Random Forest (100 arbres), notre modèle de référence.
  - "svm" : SVM linéaire (HOG + SVM = Dalal-Triggs), standardisé (StandardScaler)
            et calibré en probabilités (CalibratedClassifierCV) pour rester
            comparable au RF (score dans [0,1], mêmes seuils).
  Sélection en ligne de commande : "python evaluation.py 25 svm".
  But : comparer les AP/F1 des deux modèles (point de comparaison pour le rapport).
  RÉSULTAT (25 images de validation, mêmes images) :
    Random Forest : AP=0,284 / F1~0,43
    SVM linéaire  : AP=0,069 / F1=0,194 / rappel(détection)=0,149
  => Le RF est NETTEMENT meilleur ; le SVM linéaire classe moins bien HOG+HSV
     (le RF capte des interactions non linéaires). MODÈLE RETENU : Random Forest.
  NB : ne pas confondre le rappel PATCH (classification_report, ~0,88) avec le
  rappel de DÉTECTION (metrics.py, ici 0,149) : le patch est facile, la détection
  sur image entière est bien plus dure (cf. paradoxe accuracy patch vs AP).

3.8 AJOUT TEXTURE LBP  (src/features.py, extract_hog_features)
  IDÉE : le HOG décrit la forme/les contours, mais pas la "texture". On ajoute un
  histogramme LBP (Local Binary Pattern, variante uniform P=8/R=1 -> 10 valeurs)
  calculé sur l'imagette en gris, concaténé au descripteur. But : aider à séparer
  une surface lisse (panneau) d'un fond texturé (feuillage, brique).
  Descripteur : 1848 -> 1858 (1764 HOG + 84 HSV + 10 LBP). Réentraînement requis.
  RÉSULTAT (12 images, filtre couleur + LBP) vs filtre couleur seul :
    AP 0,169 -> 0,185 (+9 %) ; rappel 0,182 -> 0,212 ; F1 0,273 -> 0,292 ;
    précision 0,545 -> 0,467 (autre point de fonctionnement) ; dét 1042 -> 877.
    => AP ET rappel montent ensemble -> le LBP aide. RETENU.
  Progression sur ce val 12 images : 0,167 (nu) -> 0,169 (couleur) -> 0,185 (+LBP).

3.7 PRÉ-FILTRAGE COULEUR  (src/features.py + detect_on_image)
  IDÉE : les panneaux sont rouge/bleu/jaune saturés. On construit un masque HSV
  des pixels "couleur panneau" (masque_couleur_panneau), et dans detect_on_image
  on ne classe une fenêtre QUE si une fraction suffisante de ses pixels est colorée
  (seuil_couleur=0,05 par défaut ; 0 = désactivé). On compte vite via une image
  intégrale (cv2.integral) -> O(1) par fenêtre.
  ATTENTION à ne pas confondre avec l'histogramme HSV déjà présent : celui-ci est
  une FEATURE donnée au classifieur (par fenêtre) ; le filtre, lui, décide AVANT
  quelles fenêtres classer (pré-sélection de régions).
  Bénéfices attendus : moins de faux positifs (fenêtres sur gris écartées) +
  détection plus rapide. Risque : rater les panneaux peu colorés (blanc/noir) si
  le seuil est trop haut -> on le garde bas.

  RÉSULTAT (12 images, filtre seuils stricts S>=80/V>=40) vs baseline sans filtre :
    AP 0,167 -> 0,170 (stable) ; précision 0,545 -> 0,600 ; rappel 0,182 (inchangé) ;
    détections 1218 -> 643 (-47 %) ; temps 215 s -> 57 s (~4x plus rapide).
    => gain net : autant de panneaux trouvés, bien moins de faux positifs, bcp + rapide.

  PERMISSIVITÉ NUIT : seuils abaissés (S>=45, V>=30) car des photos de test sont de
  nuit / mal éclairées (couleurs sombres, peu saturées). + GARDE-FOU dans
  detect_on_image : si une image n'a quasiment aucune couleur (masque.mean()<0,2%),
  on DÉSACTIVE le filtre pour cette image (on classe tout) -> jamais 0 détection
  faute de couleur.

  RÉSULTAT version permissive (S>=45/V>=30 + garde-fou), 12 images :
    AP=0,169 ; précision=0,545 ; rappel=0,182 ; détections=1042 ; temps=94 s.
    -> AP/rappel inchangés ; filtre BEAUCOUP moins agressif que le strict
       (1042 vs 643 dét) : on garde la sécurité nuit mais on perd une partie du
       gain précision/vitesse. Choix assumé (priorité : ne rien rater de nuit).
    Réglage retenu : version permissive. (Compromis possible : S60/V35.)

3.4 NMS AVEC CONTENANCE  (src/detector/nms.py)
  OBSERVATION (inspection visuelle des prédictions sur image test) :
  - le modèle détecte des sous-régions À L'INTÉRIEUR d'un vrai panneau avec une
    forte confiance (~0,85), tandis que des panneaux bien cadrés mais plus loin
    sont moins sûrs (~0,70).
  - plusieurs boîtes restaient empilées sur un même panneau.
  CAUSE : la NMS classique supprime les doublons si IoU > seuil. Or une PETITE
  boîte dans une GRANDE a un IoU minuscule (ex. 64x64 dans 305x305 -> IoU~0,04),
  donc elle n'est PAS supprimée. Ces boîtes imbriquées comptent comme des FAUX
  POSITIFS dans l'AP (la 1re boîte sur un panneau = vrai positif, les autres =
  faux positifs) -> elles plombent la précision.
  FIX RETENU : dans la NMS, on supprime une boîte si IoU élevé OU si elle est
  largement imbriquée (intersection/aire de la plus petite > 0,6) avec une boîte
  MIEUX NOTÉE. On garde donc toujours la boîte la plus CONFIANTE du groupe.

  VARIANTES TESTÉES ET ÉCARTÉES :
    - "garder la plus grande boîte" : CATASTROPHE -> une grosse boîte parasite
      couvrant plusieurs panneaux les avalait tous (4 panneaux -> 1 boîte).
      Conclusion : aucune règle géométrique simple ne distingue "sous-région
      d'un panneau" de "grosse boîte parasite sur plusieurs panneaux".
    - "garder la plus confiante" (RETENU) : évite l'effondrement car une grosse
      boîte parasite est en général moins sûre qu'un vrai panneau, donc supprimée.
  RAISONNEMENT VALIDÉ : en théorie, la boîte qui encadre le MIEUX le panneau
  devrait avoir la plus forte confiance -> garder la plus confiante est la bonne
  approche. Le vrai problème n'est donc PAS la NMS mais le CLASSIFIEUR : il faut
  qu'il donne plus de confiance à ce qui ressemble vraiment à un panneau (boîte
  bien cadrée) qu'à une sous-région ou une boîte avec beaucoup d'arrière-plan.
  => piste prioritaire : calibrer/améliorer la confiance du classifieur
     (cadrage des positifs, hard negatives ciblés, filtre couleur). Voir section 6.
  Résultat : (à mesurer -> moins de doublons imbriqués -> AP attendue en hausse)

3.3 AUGMENTATION FISHEYE  (src/features.py + entrainer_modele)
  Contexte : le jeu de test contiendra des images fisheye, mais on ne peut pas
  détecter automatiquement quelles images le sont, ni connaître les coefficients
  de distorsion. Corriger les images au test est donc risqué (on déformerait les
  images normales). Choix retenu : AUGMENTATION (on simule du fisheye à
  l'entraînement) -> le modèle apprend à reconnaître les panneaux déformés.
  Implémentation : appliquer_distorsion_fisheye() (effet barrel via cv2.remap).
  On ajoute une copie déformée de chaque panneau positif (force=0.4).
  Pas de set val_fisheye (choix assumé). Garde-fou : l'AP sur val NORMAL ne doit
  pas chuter -> si elle reste stable, on a gagné le fisheye sans casser le reste.
  (La fonction undistort_fisheye reste dans le code mais n'est pas branchée.)
  Résultat : (à mesurer)

3.2 HARD NEGATIVE MINING MULTI-ÉCHELLE  (src/training.py)
  Problème : la version initiale ne tirait des crops qu'à une seule taille
  (64x64) -> ne corrigeait pas les faux positifs apparaissant à grande taille.
  Fix : crops aléatoires de tailles variées (64/96/128/200/300 px) ramenés à
  64x64, classés en un seul batch -> on garde les faux positifs et on réentraîne.
  Volontairement rapide (un seul batch HOG + prédiction), pas de fenêtre
  glissante complète sur les images négatives.
  Résultat : (en cours de mesure)
```

## 4. PROBLÈMES RENCONTRÉS

```text
  - LENTEUR de la détection : ~130 s/image avec stride=5. Réduit à ~13 s/image
    avec stride=16 + prédiction Random Forest parallélisée (n_jobs=-1).
    Le vrai goulot est la fenêtre glissante (HOG sur des milliers de fenêtres).
    L'entraînement, lui, est rapide (~15 s).

  - NMS en O(n²) : quand le modèle produit beaucoup de boîtes, la NMS ralentit
    fortement (run complet monté jusqu'à ~56 min). À optimiser si gênant.

  - SUR-DÉTECTION : beaucoup de détections par image (ex. 273 pour 1 panneau),
    car (a) le modèle a trop de faux positifs et (b) on garde un seuil de score
    bas (0,3) EXPRÈS pour tracer toute la courbe AP. Normal à ce stade.

  - CHEMIN : evaluation.py plantait lancé depuis src/ (Path("data") cherchait
    src/data). Corrigé : chemin calculé par rapport à l'emplacement du fichier.
```

## 5. PROBLÈMES DE QUALITÉ DU JEU DE DONNÉES (à corriger — exigé Phase 3)

```text
  - 0309.csv contient "396,659,0,0,1" : boîte de hauteur et largeur nulles
    (invalide). À supprimer/corriger.
  - 19 boîtes ont un petit côté < 16 px, que l'énoncé dit de NE PAS annoter.
  - Déséquilibre suspect des difficultés : 562 "difficiles" (niveau 1) pour
    seulement 211 "faciles" (niveau 0) -> probable sur-annotation en difficile.
  - La colonne difficulté (5e colonne) n'est pas exploitée. En PASCAL VOC, les
    objets "difficiles" sont ignorés dans le calcul de l'AP -> à envisager.
```

## 6. PISTES SUIVANTES

```text
  - Pré-filtrage couleur (HSV : rouge/bleu/jaune) pour proposer des régions ->
    moins de fenêtres, moins de faux positifs, plus robuste au fisheye.
  - Comparer SVM linéaire vs Random Forest.
  - Plafonner le nombre de détections gardées par image (par score).
  - [FAIT] CSV de soumission (src/predict_test.py) : Num img, y, x, H, L, Score
  - [FAIT] Sauvegarde du modèle (joblib) : modele_panneaux.joblib
  - [FAIT] requirements.txt (numpy, opencv-python, scikit-learn, scikit-image, joblib)
  - [FAIT] README.md propre (remplace le template GitLab)
  - [À FAIRE] base de données : data/ est dans .gitignore -> décider comment livrer
    le dataset corrigé (repo dataset séparé de l'UTC ?) + recombiner val dans train
  - [BROUILLON] rapport LaTeX rédigé (rapport.txt) -> à compléter (noms, date,
    contributions, figures) + compiler en PDF. Soumission CSV sur la page web UTC.
  - [À FAIRE] déclarer l'usage d'IA générative dans le rapport (obligatoire)
```

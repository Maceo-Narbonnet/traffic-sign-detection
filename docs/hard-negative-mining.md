# RAPPORT — IMPLÉMENTATION DU HARD NEGATIVE MINING

> Projet SY32 P2026 — Détection de panneaux de signalisation

## 1. CONTEXTE ET PROBLÈME INITIAL

```text
Lors des premiers tests de détection sur une image contenant un seul panneau,
le pipeline produisait :
  - 403 fenêtres positives avant NMS
  - 107 détections après NMS
  - Le vrai panneau était détecté, mais de nombreux faux positifs apparaissaient
    (ex : stores d'un immeuble confondus avec un panneau)

Cause racine : le modèle initial était entraîné avec seulement 2 crops aléatoires
par image négative (soit ~326 négatifs pour ~773 positifs). Ces crops ne
représentaient pas les structures visuellement trompeuses qu'une fenêtre glissante
rencontre dans une image réelle.
```

## 2. PRINCIPE DU HARD NEGATIVE MINING

```text
Le hard negative mining est une technique itérative standard dans les pipelines
de détection par fenêtre glissante :

  Étape 1 — Entraînement initial
      Entraîner le modèle avec les positifs (panneaux cropés) et des négatifs
      aléatoires (crops au hasard dans les images négatives).

  Étape 2 — Collecte des faux positifs
      Faire tourner le détecteur entraîné sur toutes les images négatives
      (qui ne contiennent aucun panneau réel). Toute fenêtre détectée comme
      "panneau" est donc un faux positif.

  Étape 3 — Enrichissement du dataset
      Ces faux positifs sont redimensionnés à la taille cible (64×64) et ajoutés
      au dataset d'entraînement avec le label 0 (négatif).

  Étape 4 — Réentraînement
      Le modèle est réentraîné sur le dataset enrichi. Il apprend maintenant à
      rejeter spécifiquement les structures qui l'avaient induit en erreur.
```

## 3. MODIFICATIONS DU CODE

```text
  3.1 Fichier : src/training.py
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Ajout de la fonction extract_hard_negatives(neg_image_list, model, target_size)

  Fonctionnement :
    - Pour chaque image négative, appel de detect_on_image() (fenêtre glissante
      multi-échelle avec pyramide gaussienne)
    - Chaque détection retournée est un faux positif (format [y, x, h, w, score])
    - La région correspondante est cropée depuis l'image originale, clippée aux
      bords de l'image, puis redimensionnée à target_size=(64, 64)
    - La fonction retourne la liste de tous ces crops

  Signature :
    def extract_hard_negatives(neg_image_list, model, target_size=(64, 64))
    -> list[np.ndarray]

  3.2 Fichier : src/main.py
  ~~~~~~~~~~~~~~~~~~~~~~~~~~
  Ajout de l'étape 4.5 entre l'entraînement initial (étape 4) et la détection
  sur image test (étape 5).

  Déroulement de l'étape 4.5 :
    1. Appel de extract_hard_negatives() sur les 163 images négatives
    2. Extraction des features HOG (1764 features par crop, identique à
       l'entraînement)
    3. Fusion avec le dataset existant (X_final, y_final)
    4. Réentraînement du Random Forest sur le dataset enrichi
    5. Le rf_model mis à jour est ensuite utilisé pour la détection (étape 5)

  Import ajouté dans main.py :
    from training import extract_hard_negatives
```

## 4. PARAMÈTRES CLÉS

```text
  - target_size       : (64, 64) — cohérent avec PATCH_SIZE et l'entraînement
  - score_threshold   : 0.8 (SCORE_THRESHOLD dans config.py) — seules les
                        fenêtres avec score > 0.8 sont collectées comme hard
                        negatives (on ne collecte que les "pires" faux positifs)
  - iou_threshold NMS : 0.3 — pour la détection finale
```

## 5. RÉSULTATS ATTENDUS

```text
  Avant hard negative mining :
    ~403 détections brutes, ~107 après NMS, nombreux faux positifs (stores, etc.)

  Après hard negative mining :
    Réduction significative des faux positifs sur les structures répétitives,
    le modèle ayant appris à les rejeter explicitement lors du réentraînement.
    Le vrai panneau reste détecté car il figure dans les positifs d'entraînement.
```

## 6. LIMITES ET PISTES D'AMÉLIORATION

```text
  - Une seule itération de hard negative mining est effectuée. Plusieurs
    itérations successives (miner → réentraîner → miner à nouveau) améliorent
    encore les résultats.
  - Les faux positifs sur les images POSITIVES (régions hors panneau) ne sont
    pas encore collectés. Les ajouter constituerait une amélioration notable.
  - La taille fixe de la fenêtre (64×64) limite la détection aux panneaux dont
    la taille à l'écran correspond à un niveau de la pyramide. Ajuster
    PYRAMID_DOWNSCALE ou le nombre de niveaux peut aider.
```

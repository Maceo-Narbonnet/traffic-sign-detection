"""Génère les figures du rapport dans le dossier 'figures/'.

Deux familles de figures :
  1) CALCULÉES À LA VOLÉE depuis les annotations (statistiques du jeu de données).
  2) GRAPHES DE RÉSULTATS construits à partir des chiffres mesurés et consignés
     dans journal_projet.txt (progression de l'AP, RF vs SVM, ablation, soumission
     UTC). Ces chiffres sont regroupés en constantes en haut du fichier : si une
     mesure change, on la met à jour ici à un seul endroit.

La courbe précision/rappel (figure_pr) nécessite une vraie évaluation sur le set
de validation : elle est donc séparée et ne se lance qu'avec l'argument 'pr'
(c'est l'étape lente, quelques minutes), pour ne pas ralentir le reste.

Utilisation :
    python src/figures.py          # toutes les figures rapides
    python src/figures.py pr       # en plus, la courbe précision/rappel (lent)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans fenêtre : on enregistre des fichiers
import matplotlib.pyplot as plt


RACINE = Path(__file__).resolve().parent.parent
DOSSIER_FIGURES = RACINE / "figures"
DOSSIER_LABELS = [RACINE / "data" / "train" / "labels",
                  RACINE / "data" / "val" / "labels"]

# Palette sobre, cohérente entre les figures
BLEU = "#2c6fbb"
VERT = "#2ca25f"
ROUGE = "#d7301f"
GRIS = "#9e9e9e"


# ----------------------------------------------------------------------
#  Chiffres mesurés (source : journal_projet.txt)
# ----------------------------------------------------------------------
# Phase 1 (ancien val = 68 images, 147 panneaux). Progression principale.
PROGRESSION = [
    ("Chaîne\nde base", 0.095),
    ("+ découpage\n(contexte)", 0.173),
    ("+ hard negative\nmining", 0.293),
]

# Random Forest vs SVM linéaire (mêmes 25 images).
RF_SVM = [("Random Forest", 0.284), ("SVM linéaire", 0.069)]

# Phase 2 (val nettoyé = 12 images, 33 panneaux). Ablation des descripteurs.
ABLATION = [
    ("HOG + HSV\n(nu)", 0.167),
    ("+ filtre\ncouleur", 0.169),
    ("+ texture\nLBP", 0.185),
]

# Soumissions UTC du 2026-06-13 : scores du site, en %. Avant/après le relèvement
# du seuil de score (0,3 -> 0,6), qui corrige l'excès de fausses détections.
SOUMISSION_METRIQUES = ["Précision", "Rappel", "F1", "AUC"]
SOUMISSION_AVANT = [0.82, 30.97, 1.59, 18.59]   # seuil 0,3 (test1)
SOUMISSION_APRES = [15.16, 26.87, 19.38, 23.68]  # seuil 0,6 (test2)


def _charger_boites():
    """Lit toutes les annotations (train + val) et renvoie un tableau (N, 5)
    de boîtes : colonnes y, x, hauteur, largeur, difficulté."""
    boites = []
    for dossier in DOSSIER_LABELS:
        if not dossier.exists():
            continue
        for chemin in sorted(dossier.glob("*.csv")):
            with open(chemin, newline="") as f:
                for ligne in csv.reader(f):
                    if not ligne:
                        continue
                    y, x, h, w, diff = (float(v) for v in ligne[:5])
                    boites.append([y, x, h, w, diff])
    return np.array(boites)


def figure_tailles(boites):
    """Histogramme de la taille des panneaux (plus petit côté), avec le seuil
    de 16 px (en dessous, l'énoncé dit de ne pas annoter)."""
    petit_cote = np.minimum(boites[:, 2], boites[:, 3])
    petit_cote = petit_cote[petit_cote > 0]  # on ignore les boîtes nulles

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(petit_cote, bins=40, color=BLEU, edgecolor="white")
    ax.axvline(16, color=ROUGE, linestyle="--", linewidth=1.6,
               label="seuil 16 px (énoncé)")
    ax.set_xlabel("Plus petit côté du panneau (px)")
    ax.set_ylabel("Nombre de panneaux")
    ax.set_title("Distribution de la taille des panneaux")
    ax.legend()
    _sauver(fig, "dataset_tailles.png")

    n_petits = int((petit_cote < 16).sum())
    print(f"  panneaux < 16 px : {n_petits} / {len(petit_cote)}")
    print(f"  taille médiane (petit côté) : {np.median(petit_cote):.0f} px")


def figure_difficulte(boites):
    """Répartition facile (0) / difficile (1) des panneaux annotés."""
    faciles = int((boites[:, 4] == 0).sum())
    difficiles = int((boites[:, 4] == 1).sum())

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.bar(["Faciles (0)", "Difficiles (1)"], [faciles, difficiles],
           color=[VERT, ROUGE], edgecolor="white", width=0.6)
    for i, v in enumerate([faciles, difficiles]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Nombre de panneaux")
    ax.set_title("Répartition facile / difficile")
    _sauver(fig, "dataset_difficulte.png")
    print(f"  faciles : {faciles}  |  difficiles : {difficiles}")


def figure_formes(boites):
    """Histogramme du rapport largeur/hauteur (forme des boîtes)."""
    h = boites[:, 2]
    w = boites[:, 3]
    garde = (h > 0) & (w > 0)
    ratio = w[garde] / h[garde]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(ratio, bins=40, color=BLEU, edgecolor="white")
    ax.axvline(1.0, color=GRIS, linestyle="--", linewidth=1.4, label="carré")
    ax.set_xlabel("Rapport largeur / hauteur")
    ax.set_ylabel("Nombre de panneaux")
    ax.set_title("Forme des panneaux (allongement)")
    ax.legend()
    _sauver(fig, "dataset_formes.png")
    print(f"  ratio l/h médian : {np.median(ratio):.2f}")


def figure_progression():
    """Barres : progression de l'AP au fil des grandes améliorations (phase 1)."""
    labels = [n for n, _ in PROGRESSION]
    valeurs = [v for _, v in PROGRESSION]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    barres = ax.bar(labels, valeurs, color=[GRIS, BLEU, VERT],
                    edgecolor="white", width=0.6)
    for b, v in zip(barres, valeurs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}".replace(".", ","),
                ha="center", va="bottom")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_ylim(0, max(valeurs) * 1.2)
    ax.set_title("Progression de l'AP (validation, 68 images)")
    _sauver(fig, "ap_progression.png")


def figure_rf_svm():
    """Barres : Random Forest contre SVM linéaire (mêmes 25 images)."""
    labels = [n for n, _ in RF_SVM]
    valeurs = [v for _, v in RF_SVM]

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    barres = ax.bar(labels, valeurs, color=[VERT, GRIS],
                    edgecolor="white", width=0.55)
    for b, v in zip(barres, valeurs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}".replace(".", ","),
                ha="center", va="bottom")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_ylim(0, max(valeurs) * 1.25)
    ax.set_title("Classifieur : RF vs SVM (25 images)")
    _sauver(fig, "rf_vs_svm.png")


def figure_ablation():
    """Barres : apport du filtre couleur puis du LBP (phase 2, val 12 images)."""
    labels = [n for n, _ in ABLATION]
    valeurs = [v for _, v in ABLATION]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    barres = ax.bar(labels, valeurs, color=[GRIS, BLEU, VERT],
                    edgecolor="white", width=0.6)
    for b, v in zip(barres, valeurs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}".replace(".", ","),
                ha="center", va="bottom")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_ylim(0, max(valeurs) * 1.2)
    ax.set_title("Apport couleur + texture (validation, 12 images)")
    _sauver(fig, "ablation_descripteurs.png")


def figure_soumission():
    """Barres groupées avant/après : effet du relèvement du seuil de score sur la
    soumission UTC. À seuil bas (0,3) la précision s'effondre ; à 0,6 elle remonte
    fortement sans presque perdre de rappel -> F1 et AUC bien meilleurs."""
    x = np.arange(len(SOUMISSION_METRIQUES))
    largeur = 0.38

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    b1 = ax.bar(x - largeur / 2, SOUMISSION_AVANT, largeur, label="seuil 0,3 (avant)",
                color=GRIS, edgecolor="white")
    b2 = ax.bar(x + largeur / 2, SOUMISSION_APRES, largeur, label="seuil 0,6 (après)",
                color=VERT, edgecolor="white")
    for barres in (b1, b2):
        for b in barres:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.1f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(SOUMISSION_METRIQUES)
    ax.set_ylabel("Score sur le site UTC (%)")
    ax.set_ylim(0, max(SOUMISSION_AVANT + SOUMISSION_APRES) * 1.2)
    ax.set_title("Soumission UTC : effet du seuil de score (0,3 → 0,6)")
    ax.legend()
    _sauver(fig, "soumission_utc.png")


def figure_pr():
    """Courbe précision/rappel sur le set de validation (étape LENTE).

    On entraîne le modèle (toutes améliorations) puis on réutilise directement
    evaluer_sur_validation, qui renvoie aussi les tableaux précision/rappel."""
    sys.path.insert(0, str(RACINE / "src"))
    from evaluation import entrainer_modele, evaluer_sur_validation

    base = RACINE / "data"
    print("  entraînement du modèle...")
    modele = entrainer_modele(base / "train")

    print("  évaluation sur le set de validation...")
    ap, prec, rap, f1, precisions, rappels = evaluer_sur_validation(modele, base / "val")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(rappels, precisions, color=BLEU, linewidth=2)
    ax.fill_between(rappels, precisions, alpha=0.15, color=BLEU)
    ax.scatter([rap], [prec], color=ROUGE, zorder=5,
               label=f"meilleur F1 = {f1:.3f}".replace(".", ","))
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Courbe précision/rappel  (AP = {ap:.3f})".replace(".", ","))
    ax.legend()
    _sauver(fig, "courbe_pr.png")
    print(f"  AP={ap:.3f}  P={prec:.3f}  R={rap:.3f}  F1={f1:.3f}")


def _sauver(fig, nom):
    fig.tight_layout()
    chemin = DOSSIER_FIGURES / nom
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    print(f"  -> {chemin.relative_to(RACINE)}")


def main():
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    args = sys.argv[1:]

    print("Statistiques du jeu de données :")
    boites = _charger_boites()
    print(f"  {len(boites)} boîtes annotées au total")
    figure_tailles(boites)
    figure_difficulte(boites)
    figure_formes(boites)

    print("Graphes de résultats :")
    figure_progression()
    figure_rf_svm()
    figure_ablation()
    figure_soumission()

    if "pr" in args:
        print("Courbe précision/rappel (évaluation complète, patiente) :")
        figure_pr()
    else:
        print("(Astuce : 'python src/figures.py pr' ajoute la courbe précision/rappel.)")

    print("Terminé.")


if __name__ == "__main__":
    main()

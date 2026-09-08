import random
import numpy as np

# On fige le hasard pour que les résultats soient REPRODUCTIBLES d'un run à
# l'autre : NumPy ET le module random de Python (utilisé pour l'échantillonnage
# des négatifs et le hard negative mining). Sans cela, le modèle s'entraîne sur
# des négatifs différents à chaque exécution et l'AP fluctue.
np.random.seed(42)
random.seed(42)

PATCH_SIZE= 64
STRIDE = 5
PYRAMID_DOWNSCALE = 1.25
PYRAMID_MIN_SIZE = 64
SCORE_THRESHOLD = 0.8


"""
Pipeline de prétraitement : standardisation -> SMOTE (entraînement seulement) -> modèle.

Le Pipeline garantit qu'en validation croisée, la standardisation et SMOTE sont appris
uniquement sur les plis d'entraînement : aucune fuite de données (Cours 1, slides 57-59 et 84-86).
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def build_pipeline(model, use_smote: bool = True) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=C.RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)

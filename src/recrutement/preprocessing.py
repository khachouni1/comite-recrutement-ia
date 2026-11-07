"""
Prétraitement sous forme de Pipeline scikit-learn / imbalanced-learn.

Pourquoi un Pipeline ? Parce que chaque étape (standardisation, SMOTE) est alors apprise
UNIQUEMENT sur les données d'entraînement de chaque pli de validation croisée :
c'est la garantie technique qu'il n'y a pas de fuite de données.

    Pipeline = [ prétraitement ]  ->  [ SMOTE (train seulement) ]  ->  [ modèle ]
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def build_preprocessor(features) -> ColumnTransformer:
    """
    - Variables numériques et ordinales : centrage-réduction (z = (x - μ) / σ).
    - Variable catégorielle (stratégie 1/2/3) : codage one-hot.
    - Variable binaire : laissée telle quelle.
    """
    num = [f for f in C.NUMERIC_FEATURES + C.ORDINAL_FEATURES if f in features]
    cat = [f for f in C.CATEGORICAL_FEATURES if f in features]
    binary = [f for f in C.BINARY_FEATURES if f in features]
    transformers = [("num", StandardScaler(), num)]
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if binary:
        transformers.append(("bin", "passthrough", binary))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def build_pipeline(model, features, use_smote: bool = True) -> Pipeline:
    """Pipeline complet : prétraitement -> (SMOTE) -> modèle."""
    steps = [("prep", build_preprocessor(features))]
    if use_smote:
        # SMOTE (Cours 1, slides 57-59) : appliqué au moment du fit uniquement,
        # jamais lors de la prédiction sur validation / test.
        steps.append(("smote", SMOTE(random_state=C.RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)

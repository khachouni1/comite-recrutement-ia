"""
Chargement des données et découpage Three-Way Holdout (Cours 1, slides 84-86).

Règle anti-fuite : le découpage est fait sur les données BRUTES, avant tout prétraitement.
Les transformations (standardisation, SMOTE...) sont ensuite apprises sur Train uniquement.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw() -> pd.DataFrame:
    """Charge le jeu de données brut et vérifie son schéma."""
    df = pd.read_csv(C.DATA_RAW)
    missing = set(C.ALL_FEATURES + [C.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier brut : {missing}")
    return df


def split_three_way(df: pd.DataFrame):
    """
    Train 70 % / Validation 15 % / Test 15 %, stratifié sur la cible
    (même proportion d'embauchés dans chaque sous-ensemble).
    """
    train_val, test = train_test_split(
        df, test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.RANDOM_STATE)
    train, val = train_test_split(
        train_val, test_size=C.VAL_SIZE / (1 - C.TEST_SIZE),
        stratify=train_val[C.TARGET], random_state=C.RANDOM_STATE)
    return train, val, test


def save_splits(train, val, test) -> None:
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(C.DATA_PROCESSED / f"{name}.csv", index=True, index_label="id")


def load_splits():
    """Recharge les trois sous-ensembles (l'identifiant d'origine est conservé en index)."""
    parts = []
    for name in ("train", "val", "test"):
        path = C.DATA_PROCESSED / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError("Découpage absent : exécutez d'abord  python scripts/01_eda.py")
        parts.append(pd.read_csv(path, index_col="id"))
    return tuple(parts)


def X_y(df: pd.DataFrame, features=None):
    features = features or C.MODEL_FEATURES
    return df[features], df[C.TARGET]

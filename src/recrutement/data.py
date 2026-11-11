"""
Chargement des paires (CV, offre, étiquette) et découpage Train / Validation / Test.

Particularité : un même CV ou une même offre apparaît dans plusieurs paires. Pour éviter toute fuite,
le découpage se fait PAR OFFRE (GroupShuffleSplit) : les offres du Test et de la Validation ne sont
jamais vues à l'entraînement. Le système est donc évalué sur des postes nouveaux, comme sur le site.
"""
import hashlib

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as C
from .texte import anonymiser, nettoyer


def _id(texte: str, prefixe: str) -> str:
    return prefixe + hashlib.md5(texte.encode("utf-8")).hexdigest()[:8]


def charger_brut() -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for nom in ("train", "test"):
        path = C.RAW_DIR / f"{nom}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} absent : voir data/README.md pour le téléchargement.")
        out.append(pd.read_csv(path))
    return tuple(out)


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage, anonymisation, identifiants de CV / d'offre, étiquette numérique ordinale."""
    df = df.drop_duplicates().copy()
    for brut, propre in (("resume_text", "cv"), ("job_description_text", "offre")):
        uniques = {t: anonymiser(nettoyer(t)) for t in df[brut].unique()}  # chaque texte traité une fois
        df[propre] = df[brut].map(uniques)
    df["cv_id"] = df["cv"].map(lambda t: _id(t, "CV_"))
    df["offre_id"] = df["offre"].map(lambda t: _id(t, "JOB_"))
    df["y"] = df["label"].map(C.LABEL_TO_INT)
    return df[["cv_id", "offre_id", "cv", "offre", "label", "y"]].reset_index(drop=True)


def decouper(train_officiel: pd.DataFrame, test_officiel: pd.DataFrame):
    """Test = test officiel ; Train / Val = découpage du train officiel PAR OFFRE (85 % / 15 %)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=C.VAL_SIZE, random_state=C.RANDOM_STATE)
    i_tr, i_va = next(gss.split(train_officiel, groups=train_officiel["offre_id"]))
    return train_officiel.iloc[i_tr].copy(), train_officiel.iloc[i_va].copy(), test_officiel.copy()


def sauvegarder(**parts) -> None:
    for nom, df in parts.items():
        df.to_pickle(C.DATA_PROCESSED / f"{nom}.pkl")


def charger(*noms):
    out = []
    for nom in noms:
        path = C.DATA_PROCESSED / f"{nom}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} absent : exécutez d'abord les étapes précédentes (scripts/run_all.py).")
        out.append(pd.read_pickle(path))
    return out[0] if len(out) == 1 else tuple(out)

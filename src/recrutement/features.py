"""
Caractéristiques d'une paire (CV, offre) : ce que chaque agent observe.

Pour chaque paire, on calcule des variables EXPLICABLES :
  - technique   : compétences de l'offre retrouvées dans le CV (couverture), nombre de compétences ;
  - expérience  : années détectées dans le CV, rapport aux années exigées par l'offre ;
  - formation   : niveau de diplôme du CV, écart avec le niveau exigé ;
  - soft skills : savoir-être demandés par l'offre et présents dans le CV ;
  - adéquation  : similarités textuelles (TF-IDF, LSA, embeddings).
"""
import numpy as np
import pandas as pd

from . import texte as T


def analyser_cv(cv: str) -> dict:
    return {"hard": T.competences(cv, "hard"), "soft": T.competences(cv, "soft"),
            "annees": T.annees_experience_cv(cv), "diplome": T.niveau_diplome(cv), "longueur": len(cv.split())}


def analyser_offre(offre: str) -> dict:
    return {"hard": T.competences(offre, "hard"), "soft": T.competences(offre, "soft"),
            "annees": T.annees_exigees_offre(offre), "diplome": T.niveau_diplome(offre, exigence=True),
            "longueur": len(offre.split())}


def comparer(a_cv: dict, a_of: dict) -> dict:
    """Variables d'une paire à partir des analyses du CV et de l'offre."""
    hard_c, soft_c = a_of["hard"] & a_cv["hard"], a_of["soft"] & a_cv["soft"]
    req_annees, req_dipl = a_of["annees"], a_of["diplome"]
    return {
        "couverture_hard": len(hard_c) / len(a_of["hard"]) if a_of["hard"] else 0.5,
        "n_hard_communs": len(hard_c), "n_hard_cv": len(a_cv["hard"]), "n_hard_offre": len(a_of["hard"]),
        "couverture_soft": len(soft_c) / len(a_of["soft"]) if a_of["soft"] else 0.5,
        "n_soft_communs": len(soft_c), "n_soft_cv": len(a_cv["soft"]),
        "annees_cv": a_cv["annees"],
        "ratio_annees": min(a_cv["annees"] / req_annees, 2.0) if req_annees else 1.0,
        "exigence_annees_connue": int(req_annees is not None),
        "diplome_cv": a_cv["diplome"],
        "ecart_diplome": (a_cv["diplome"] - req_dipl) if req_dipl else 0,
        "exigence_diplome_connue": int(req_dipl > 0),
        "longueur_cv": a_cv["longueur"], "longueur_offre": a_of["longueur"],
    }


def details(a_cv: dict, a_of: dict) -> dict:
    """Éléments lisibles pour justifier les avis des agents."""
    return {
        "hard_trouvees": sorted(a_of["hard"] & a_cv["hard"]), "hard_manquantes": sorted(a_of["hard"] - a_cv["hard"]),
        "hard_en_plus": sorted(a_cv["hard"] - a_of["hard"]),
        "soft_trouvees": sorted(a_of["soft"] & a_cv["soft"]), "soft_manquantes": sorted(a_of["soft"] - a_cv["soft"]),
        "annees_cv": a_cv["annees"], "annees_exigees": a_of["annees"],
        "diplome_cv": a_cv["diplome"], "diplome_exige": a_of["diplome"],
    }


def table_regles(paires: pd.DataFrame) -> pd.DataFrame:
    """Variables issues des règles et lexiques ; chaque CV et chaque offre n'est analysé qu'une fois."""
    a_cv = {i: analyser_cv(t) for i, t in paires.drop_duplicates("cv_id")[["cv_id", "cv"]].values}
    a_of = {i: analyser_offre(t) for i, t in paires.drop_duplicates("offre_id")[["offre_id", "offre"]].values}
    rows = [comparer(a_cv[c], a_of[o]) for c, o in paires[["cv_id", "offre_id"]].values]
    return pd.DataFrame(rows, index=paires.index)


def interactions_embeddings(cvs, offres, reps, index=None) -> pd.DataFrame:
    """
    Représentation riche d'une paire pour le modèle avancé :
    produit élément par élément (dimensions communes) et écart absolu (dimensions qui diffèrent)
    entre l'embedding du CV et celui de l'offre :  [u ⊙ v , |u − v|].
    """
    u, v = reps.embedding_vec(cvs), reps.embedding_vec(offres)
    d = u.shape[1]
    cols = [f"emb_prod_{i}" for i in range(d)] + [f"emb_diff_{i}" for i in range(d)]
    return pd.DataFrame(np.hstack([u * v, np.abs(u - v)]), columns=cols, index=index)


def colonnes_embeddings(feats: pd.DataFrame) -> list:
    return [c for c in feats.columns if c.startswith("emb_")]


def construire(paires: pd.DataFrame, reps) -> pd.DataFrame:
    """Table complète des variables : règles + similarités textuelles + interactions d'embeddings."""
    feats = table_regles(paires)
    for nom, valeurs in reps.similarites(paires["cv"], paires["offre"]).items():
        feats[nom] = valeurs
    emb = interactions_embeddings(paires["cv"], paires["offre"], reps, index=paires.index)
    return pd.concat([feats, emb], axis=1).astype(float)


def construire_une(cv: str, offre: str, reps) -> tuple[pd.DataFrame, dict]:
    """Variables + détails pour une seule paire (utilisé par le site web)."""
    a_cv, a_of = analyser_cv(cv), analyser_offre(offre)
    feats = comparer(a_cv, a_of)
    feats.update({k: float(v[0]) for k, v in reps.similarites([cv], [offre]).items()})
    out = pd.concat([pd.DataFrame([feats]), interactions_embeddings([cv], [offre], reps)], axis=1)
    return out.astype(float), details(a_cv, a_of)


def resume_extraction(feats: pd.DataFrame) -> pd.Series:
    """Taux de réussite de l'extraction, pour contrôler la qualité des règles."""
    return pd.Series({
        "offres avec compétences détectées": float((feats["n_hard_offre"] > 0).mean()),
        "offres avec années exigées": float(feats["exigence_annees_connue"].mean()),
        "offres avec diplôme exigé": float(feats["exigence_diplome_connue"].mean()),
        "CV avec années détectées": float((feats["annees_cv"] > 0).mean()),
        "CV avec diplôme détecté": float((feats["diplome_cv"] > 0).mean()),
    }).round(3)

"""
Apprentissage supervisé : prédire l'étiquette d'une paire (non adapté / potentiel / adapté).

Classification à 3 classes (Cours 1) :
  - baseline naïve (classe majoritaire) ;
  - baseline simple : régression softmax (logistique multinomiale, Cours 1 slides 77-82) ;
  - modèles avancés : arbre, forêt aléatoire, gradient boosting, XGBoost.

Protocole (Cours 1 slides 84-88) :
  1. hyperparamètres par validation croisée 5 plis sur Train, plis GROUPÉS PAR OFFRE ;
  2. choix du modèle sur Validation ;
  3. ré-entraînement sur Train + Validation ;
  4. une seule évaluation sur Test.

Score d'adéquation attendu (utilisé par l'agent prédictif) :  100 · (0.5 · P(potentiel) + 1 · P(adapté)).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocessing import build_pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def catalogue(riche: bool = False) -> dict:
    """
    riche=False : modèles comparés sur les 17 variables explicables.
    riche=True  : modèles adaptés aux 785 variables (avec interactions d'embeddings) :
                  régression softmax régularisée et XGBoost (sous-échantillonnage des colonnes).
    """
    rs = C.RANDOM_STATE
    if riche:
        cat = {"Régression softmax": (LogisticRegression(max_iter=3000, random_state=rs),
                                      {"model__C": [0.003, 0.01, 0.03]})}
        if XGBClassifier is not None:
            cat["XGBoost"] = (XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.5,
                                            eval_metric="mlogloss", random_state=rs, n_jobs=-1, verbosity=0),
                              {"model__max_depth": [3, 4], "model__learning_rate": [0.05, 0.1]})
        return cat
    cat = {
        "Baseline naïve (classe majoritaire)": (DummyClassifier(strategy="most_frequent"), {}),
        "Régression softmax (baseline)": (LogisticRegression(max_iter=3000, random_state=rs),
                                          {"model__C": [0.01, 0.1, 1, 10]}),
        "Arbre de décision": (DecisionTreeClassifier(random_state=rs),
                              {"model__max_depth": [4, 6, 10], "model__min_samples_leaf": [5, 20]}),
        "Forêt aléatoire": (RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
                            {"model__max_depth": [8, 14, None], "model__min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=rs),
                              {"model__n_estimators": [150, 300], "model__max_depth": [2, 3],
                               "model__learning_rate": [0.05, 0.1]}),
    }
    if XGBClassifier is not None:
        cat["XGBoost"] = (XGBClassifier(n_estimators=300, eval_metric="mlogloss", random_state=rs, n_jobs=-1,
                                        verbosity=0),
                          {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]})
    return cat


def score_attendu(proba: np.ndarray) -> np.ndarray:
    """Espérance de l'adéquation sur [0, 100] : non adapté = 0, potentiel = 50, adapté = 100."""
    return 100 * (0.5 * proba[:, 1] + proba[:, 2])


def metriques(y, proba) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "AUC macro (OvR)": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "AUC adéquat vs non": roc_auc_score(np.asarray(y) > 0, proba[:, 1] + proba[:, 2]),
        "F1 macro": f1_score(y, pred, average="macro"),
        "Accuracy": accuracy_score(y, pred),
        "Accuracy équilibrée": balanced_accuracy_score(y, pred),
        "Spearman (score attendu)": spearmanr(score_attendu(proba), y)[0],
    }


def regler_et_comparer(f_tr, y_tr, g_tr, f_va, y_va, jeux: dict) -> tuple[pd.DataFrame, dict]:
    """
    jeux = {nom du jeu de variables: liste de colonnes}. Chaque modèle est réglé par validation croisée
    groupée par offre sur Train, puis évalué sur Validation. Clé des modèles : « modèle — jeu ».
    """
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    rows, modeles = [], {}
    for nom_jeu, cols in jeux.items():
        for nom, (model, grille) in catalogue(riche=len(cols) > 50).items():
            pipe = build_pipeline(model, use_smote=not isinstance(model, DummyClassifier))
            search = GridSearchCV(pipe, grille or {"model__strategy": ["most_frequent"]},
                                  scoring="roc_auc_ovr", cv=cv, n_jobs=-1)
            search.fit(f_tr[cols], y_tr, groups=g_tr)
            m = metriques(y_va, search.best_estimator_.predict_proba(f_va[cols]))
            cle = f"{nom} — {nom_jeu}"
            rows.append({"Modèle": cle, "Algorithme": nom, "Variables": nom_jeu, "Nb variables": len(cols),
                         "AUC CV (Train)": search.best_score_, **{f"{k} (Val)": v for k, v in m.items()},
                         "Hyperparamètres": {k.replace("model__", ""): v for k, v in search.best_params_.items()}})
            modeles[cle] = search.best_estimator_
            print(f"   {cle:58s} AUC CV = {search.best_score_:.3f} | AUC Val = {m['AUC macro (OvR)']:.3f} | "
                  f"F1 macro Val = {m['F1 macro']:.3f}")
    # Critère de sélection : AUC en validation croisée groupée (5 plis, 238 offres) plutôt que l'AUC sur la
    # seule Validation (42 offres), trop bruitée pour départager des modèles proches. La Validation sert de contrôle.
    table = pd.DataFrame(rows).sort_values("AUC CV (Train)", ascending=False).reset_index(drop=True)
    return table, modeles


def reentrainer(pipe, X_tr, y_tr, X_va, y_va):
    final = clone(pipe)
    final.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
    return final

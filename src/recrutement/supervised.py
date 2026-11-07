"""
Apprentissage supervisé : classification binaire « embauché / rejeté » (Cours 1).

Protocole (Three-Way Holdout + validation croisée, Cours 1 slides 84-88) :
  1. Pour chaque modèle, hyperparamètres choisis par validation croisée stratifiée (5 plis) sur TRAIN.
  2. Comparaison des modèles sur VALIDATION -> choix du meilleur modèle.
  3. Ré-entraînement du modèle retenu sur TRAIN + VALIDATION.
  4. Évaluation finale, une seule fois, sur TEST.

Modèles comparés : une baseline naïve, une baseline simple (régression logistique)
et des modèles plus avancés (arbre, forêt aléatoire, gradient boosting, XGBoost).
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             balanced_accuracy_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocessing import build_pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost est optionnel
    XGBClassifier = None


def model_catalog() -> dict:
    """Modèles et grilles d'hyperparamètres (préfixe 'model__' = étape 'model' du pipeline)."""
    rs = C.RANDOM_STATE
    cat = {
        "Baseline naïve (classe majoritaire)": (DummyClassifier(strategy="most_frequent"), {}),
        "Régression logistique (baseline)": (
            LogisticRegression(max_iter=2000, random_state=rs),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "Arbre de décision": (
            DecisionTreeClassifier(random_state=rs),
            {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [1, 10]}),
        "Forêt aléatoire": (
            RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
            {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"model__n_estimators": [100, 200], "model__max_depth": [2, 3],
             "model__learning_rate": [0.05, 0.1]}),
    }
    if XGBClassifier is not None:
        cat["XGBoost"] = (
            XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=rs, n_jobs=-1, verbosity=0),
            {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1]})
    return cat


def metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        "Accuracy": accuracy_score(y_true, pred),
        "Accuracy équilibrée": balanced_accuracy_score(y_true, pred),
        "Précision": precision_score(y_true, pred, zero_division=0),
        "Rappel": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def tune_and_compare(X_train, y_train, X_val, y_val, features) -> tuple:
    """Étapes 1 et 2 : validation croisée sur Train puis comparaison sur Validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    rows, fitted = [], {}
    for name, (model, grid) in model_catalog().items():
        pipe = build_pipeline(model, features, use_smote=not isinstance(model, DummyClassifier))
        search = GridSearchCV(pipe, grid or {"model__strategy": ["most_frequent"]},
                              scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        proba_val = search.best_estimator_.predict_proba(X_val)[:, 1]
        m = metrics(y_val, proba_val)
        rows.append({"Modèle": name, "AUC CV (Train)": search.best_score_,
                     **{f"{k} (Val)": v for k, v in m.items()},
                     "Hyperparamètres": {k.replace("model__", ""): v for k, v in search.best_params_.items()}})
        fitted[name] = search.best_estimator_
        print(f"   {name:38s} AUC CV = {search.best_score_:.3f} | AUC Val = {m['ROC-AUC']:.3f} | "
              f"F1 Val = {m['F1']:.3f}")
    table = pd.DataFrame(rows).sort_values("ROC-AUC (Val)", ascending=False).reset_index(drop=True)
    return table, fitted


def refit_on_train_val(pipeline, X_train, y_train, X_val, y_val):
    """Étape 3 : ré-entraînement du modèle retenu sur Train + Validation."""
    final = clone(pipeline)
    final.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    return final

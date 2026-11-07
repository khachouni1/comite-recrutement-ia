"""
Interprétabilité et explicabilité (Cours 1, slides 51-54) — les trois méthodes citées par la consigne :

1. SHAP : explicabilité globale (importance moyenne |SHAP|) et locale (cascade pour un candidat).
2. Importance des variables par permutation : baisse de l'AUC quand on mélange une variable.
3. Analyse détaillée des erreurs : profil des faux positifs / faux négatifs, erreurs par groupe.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C
from .plots import save


# ------------------------------------------------------------------ SHAP
def shap_values(pipeline, X: pd.DataFrame):
    """Valeurs SHAP du modèle final, calculées dans l'espace des variables prétraitées."""
    import shap
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xt)
    else:
        explainer = shap.TreeExplainer(model)
    exp = explainer(Xt)
    if exp.values.ndim == 3:  # sortie (n, variables, classes) -> classe « embauché »
        exp = exp[:, :, 1]
    return exp, Xt


def plot_shap(exp, Xt, y: pd.Series, proba: np.ndarray) -> None:
    import shap
    shap.summary_plot(exp.values, Xt, show=False, plot_size=(9, 5.5))
    fig = plt.gcf()
    fig.suptitle("SHAP — explicabilité globale (Test) : impact de chaque variable", fontweight="bold")
    save(fig, "13_shap_global")

    pred = (proba >= 0.5).astype(int)
    cases = {
        "14_shap_local_vrai_positif": ("Candidat embauché, correctement prédit", (y.values == 1) & (pred == 1)),
        "15_shap_local_faux_negatif": ("Candidat embauché mais prédit rejeté (erreur)", (y.values == 1) & (pred == 0)),
    }
    for name, (title, mask) in cases.items():
        if not mask.any():
            continue
        i = int(np.where(mask)[0][0])
        shap.plots.waterfall(exp[i], show=False, max_display=10)
        fig = plt.gcf()
        fig.suptitle(f"SHAP local — {title} (id {Xt.index[i]}, p = {proba[i]:.2f})", fontweight="bold")
        save(fig, name)


# ------------------------------------------------------------------ permutation
def permutation_table(pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    res = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=30,
                                 random_state=C.RANDOM_STATE, n_jobs=-1)
    return pd.DataFrame({"variable": X.columns, "baisse_AUC_moyenne": res.importances_mean,
                         "ecart_type": res.importances_std}).sort_values("baisse_AUC_moyenne", ascending=False)


def plot_permutation(table: pd.DataFrame) -> None:
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([C.LABELS[v] for v in t["variable"]], t["baisse_AUC_moyenne"], xerr=t["ecart_type"], color="#16a085")
    ax.set_title("Importance par permutation (Test) : perte d'AUC si la variable est mélangée")
    ax.set_xlabel("Baisse moyenne de l'AUC")
    save(fig, "16_importance_permutation")


# ------------------------------------------------------------------ erreurs
def error_analysis(test: pd.DataFrame, proba: np.ndarray, threshold: float = 0.5) -> dict:
    df = test.copy()
    df["proba"] = proba
    pred = (proba >= threshold).astype(int)
    y = df[C.TARGET].values
    df["type"] = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1)],
                           ["Vrai positif", "Vrai négatif", "Faux positif"], "Faux négatif")
    profile = df.groupby("type")[C.ALL_FEATURES + ["proba"]].mean().round(2)
    profile.insert(0, "effectif", df["type"].value_counts())
    df["erreur"] = pred != y
    df["Genre"] = df["Gender"].map({0: "Femme", 1: "Homme"})
    df["Tranche_age"] = pd.cut(df["Age"], [0, 29, 39, 100], labels=["< 30", "30-39", "≥ 40"])
    by_group = {g: df.groupby(g, observed=True)["erreur"].mean().round(3).to_dict()
                for g in ("Genre", "Tranche_age", "RecruitmentStrategy")}
    return {"profil": profile, "taux_erreur_par_groupe": by_group, "detail": df}

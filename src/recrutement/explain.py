"""
Interprétabilité (Cours 1, slides 51-54) — les trois approches demandées par la consigne :

1. SHAP : contribution de chaque variable à la prédiction « adapté » (globale et locale).
2. Importance par permutation : perte d'AUC quand on mélange une variable.
3. Analyse des erreurs : matrice de confusion 3 × 3 et profil des candidats mal classés.

Les 768 dimensions d'embeddings n'ont pas de sens individuel : elles sont REGROUPÉES en un seul bloc
« interactions sémantiques » (somme des valeurs SHAP ; permutation simultanée de toutes les colonnes).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score

from . import config as C
from .plots import save

BLOC = "interactions sémantiques (embeddings)"


def _groupes(colonnes) -> dict:
    g = {c: [c] for c in colonnes if not c.startswith("emb_")}
    emb = [c for c in colonnes if c.startswith("emb_")]
    if emb:
        g[BLOC] = emb
    return g


def shap_adapte(pipe, X: pd.DataFrame, n_expl: int = 400):
    """Valeurs SHAP pour la classe « adapté », regroupées par variable / bloc."""
    import shap
    expl = X.sample(min(n_expl, len(X)), random_state=C.RANDOM_STATE)
    scaler, model = pipe.named_steps["scaler"], pipe.named_steps["model"]
    Xt = pd.DataFrame(scaler.transform(expl), columns=X.columns, index=expl.index)
    if hasattr(model, "coef_"):  # modèle linéaire : SHAP exact sur le logit de la classe « adapté »
        fond = pd.DataFrame(scaler.transform(X.sample(min(500, len(X)), random_state=0)), columns=X.columns)
        vals = shap.LinearExplainer((model.coef_[2], model.intercept_[2]), fond).shap_values(Xt)
    else:                        # modèle à arbres : TreeExplainer (exact et rapide)
        vals = np.asarray(shap.TreeExplainer(model).shap_values(Xt))
        vals = vals[:, :, 2] if vals.ndim == 3 else vals
    groupes = _groupes(X.columns)
    idx = {c: i for i, c in enumerate(X.columns)}
    valeurs = pd.DataFrame({g: vals[:, [idx[c] for c in cols]].sum(axis=1) for g, cols in groupes.items()},
                           index=expl.index)
    donnees = pd.DataFrame({g: expl[cols[0]] if len(cols) == 1 else expl["sim_embedding"]
                            for g, cols in groupes.items()}, index=expl.index)
    return valeurs, donnees


def plot_shap(valeurs: pd.DataFrame, donnees: pd.DataFrame, proba_adapte: pd.Series, y: pd.Series) -> None:
    import shap
    shap.summary_plot(valeurs.to_numpy(), donnees, show=False, plot_size=(9, 6), max_display=15)
    fig = plt.gcf()
    fig.suptitle("SHAP — impact des variables sur la prédiction « adapté » (Test)\n"
                 "(bloc embeddings coloré par la similarité sémantique)", fontweight="bold", fontsize=10)
    save(fig, "12_shap_global")
    cas = {"13_shap_local_adapte": ("candidat adapté, bien reconnu", (y == 2) & (proba_adapte > 0.5)),
           "14_shap_local_non_adapte": ("candidat non adapté, bien reconnu", (y == 0) & (proba_adapte < 0.15))}
    for nom, (titre, masque) in cas.items():
        ids = [i for i in valeurs.index if masque.get(i, False)]
        if ids:
            i = ids[0]
            e = shap.Explanation(values=valeurs.loc[i].to_numpy(), base_values=0.0,
                                 data=donnees.loc[i].to_numpy(), feature_names=list(valeurs.columns))
            shap.plots.waterfall(e, show=False, max_display=10)
            fig = plt.gcf()
            fig.suptitle(f"SHAP local — {titre}", fontweight="bold")
            save(fig, nom)


def permutation(pipe, X: pd.DataFrame, y: pd.Series, n_repeats: int = 8) -> pd.DataFrame:
    """Perte d'AUC macro quand on mélange une variable (ou tout le bloc d'embeddings d'un coup)."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    ref = roc_auc_score(y, pipe.predict_proba(X), multi_class="ovr")
    rows = []
    for g, cols in _groupes(X.columns).items():
        pertes = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            pertes.append(ref - roc_auc_score(y, pipe.predict_proba(Xp), multi_class="ovr"))
        rows.append({"variable": g, "baisse_AUC": np.mean(pertes), "ecart_type": np.std(pertes)})
    t = pd.DataFrame(rows).sort_values("baisse_AUC", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    r = t.head(15).iloc[::-1]
    ax.barh(r["variable"], r["baisse_AUC"], xerr=r["ecart_type"], color="#16a085")
    ax.set_title("Importance par permutation (Test) : perte d'AUC si la variable est mélangée")
    save(fig, "15_importance_permutation")
    return t


def erreurs(y: pd.Series, proba: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    pred = proba.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=[C.LABELS_FR[l] for l in C.LABELS],
                                            normalize="true", values_format=".0%", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Matrice de confusion normalisée (Test)")
    save(fig, "16_matrice_confusion")
    df = X.assign(reel=y.map(dict(enumerate(C.LABELS))).values, predit=[C.LABELS[p] for p in pred])
    df["type"] = np.where(df["reel"] == df["predit"], "correct", df["reel"] + " → " + df["predit"])
    profil = df.groupby("type")[["couverture_hard", "sim_embedding", "annees_cv", "ratio_annees", "diplome_cv"]].mean()
    profil.insert(0, "effectif", df["type"].value_counts())
    return profil.sort_values("effectif", ascending=False).round(3)

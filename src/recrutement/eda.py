"""
Analyse exploratoire des données (EDA) — phase obligatoire de la consigne.

1. Contrôle qualité (sur l'ensemble des données : aucun apprentissage, donc aucune fuite).
2. Statistiques descriptives, distributions, corrélations, tests statistiques
   (sur le TRAIN uniquement : le jeu de test ne doit pas influencer nos choix).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

from . import config as C
from .plots import save, COLOR_HIRED, COLOR_REJECTED


def data_quality(df: pd.DataFrame) -> dict:
    """Valeurs manquantes, doublons, plages de valeurs, équilibre des classes."""
    return {
        "n_lignes": int(len(df)),
        "n_colonnes": int(df.shape[1]),
        "valeurs_manquantes": int(df.isna().sum().sum()),
        "doublons": int(df.duplicated().sum()),
        "taux_embauche": round(float(df[C.TARGET].mean()), 4),
        "plages": {c: [float(df[c].min()), float(df[c].max())] for c in C.ALL_FEATURES},
    }


def descriptive_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Moyenne, écart-type, quartiles... par variable, et moyenne par classe."""
    desc = train[C.ALL_FEATURES].describe().T
    by_class = train.groupby(C.TARGET)[C.ALL_FEATURES].mean().T
    desc["moyenne_rejetés"] = by_class[0]
    desc["moyenne_embauchés"] = by_class[1]
    return desc.round(2)


def statistical_tests(train: pd.DataFrame) -> pd.DataFrame:
    """
    Lien entre chaque variable et la décision d'embauche :
      - variables quantitatives : test de Mann-Whitney (comparaison de deux distributions) ;
      - variables qualitatives : test du χ² d'indépendance.
    H0 : la variable n'est pas liée à la décision. On rejette H0 si p < 0.05.
    """
    rows = []
    hired, rejected = train[train[C.TARGET] == 1], train[train[C.TARGET] == 0]
    for f in C.NUMERIC_FEATURES + C.ORDINAL_FEATURES:
        stat, p = mannwhitneyu(hired[f], rejected[f])
        rows.append({"variable": f, "test": "Mann-Whitney", "p_value": p})
    for f in C.CATEGORICAL_FEATURES + C.BINARY_FEATURES:
        chi2, p, _, _ = chi2_contingency(pd.crosstab(train[f], train[C.TARGET]))
        rows.append({"variable": f, "test": "Chi2", "p_value": p})
    out = pd.DataFrame(rows)
    out["significatif_5%"] = out["p_value"] < 0.05
    return out.sort_values("p_value")


# ------------------------------------------------------------------ figures
def plot_target(train: pd.DataFrame) -> None:
    counts = train[C.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Rejeté (0)", "Embauché (1)"], counts.values, color=[COLOR_REJECTED, COLOR_HIRED])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v} ({v / counts.sum():.0%})", ha="center")
    ax.set_title("Distribution de la cible (Train) : classes déséquilibrées")
    ax.set_ylabel("Nombre de candidats")
    save(fig, "01_distribution_cible")


def plot_distributions(train: pd.DataFrame) -> None:
    feats = C.NUMERIC_FEATURES + C.ORDINAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.ravel(), feats):
        sns.histplot(data=train, x=f, hue=C.TARGET, stat="density", common_norm=False, kde=True,
                     palette=[COLOR_REJECTED, COLOR_HIRED], ax=ax, alpha=0.4)
        ax.set_title(C.LABELS[f])
        ax.set_xlabel("")
    fig.suptitle("Distribution des variables selon la décision (0 = rejeté, 1 = embauché)", fontweight="bold")
    save(fig, "02_distributions_par_classe")


def plot_correlations(train: pd.DataFrame) -> None:
    corr = train[C.ALL_FEATURES + [C.TARGET]].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=[C.LABELS[c] for c in corr.columns], yticklabels=[C.LABELS[c] for c in corr.index])
    ax.set_title("Matrice de corrélation de Spearman (Train)")
    save(fig, "03_correlations")


def plot_categorical_rates(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, f in zip(axes, ["EducationLevel", "RecruitmentStrategy", "PreviousCompanies"]):
        rates = train.groupby(f)[C.TARGET].agg(["mean", "count"])
        ax.bar(rates.index.astype(str), rates["mean"] * 100, color=COLOR_HIRED)
        for x, (m, n) in zip(rates.index.astype(str), rates.values):
            ax.text(x, m * 100 + 1.5, f"{m:.0%}\n(n={int(n)})", ha="center", fontsize=8)
        ax.set_title(f"Taux d'embauche par {C.LABELS[f].lower()}")
        ax.set_ylabel("Embauchés (%)")
        ax.set_ylim(0, 100)
    save(fig, "04_taux_embauche_par_modalite")


def plot_fairness(train: pd.DataFrame) -> None:
    """Taux d'embauche historiques par genre et par tranche d'âge (point de départ de l'audit éthique)."""
    data = train.assign(
        Genre=train["Gender"].map({0: "Femme", 1: "Homme"}),
        Tranche_age=pd.cut(train["Age"], [0, 29, 39, 100], labels=["< 30", "30-39", "≥ 40"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, f in zip(axes, ["Genre", "Tranche_age"]):
        rates = data.groupby(f, observed=True)[C.TARGET].mean() * 100
        ax.bar(rates.index.astype(str), rates.values, color="#8e44ad")
        for x, v in zip(rates.index.astype(str), rates.values):
            ax.text(x, v + 1, f"{v:.1f}%", ha="center")
        ax.set_title(f"Taux d'embauche historique par {f.replace('_', ' ').lower()}")
        ax.set_ylim(0, 60)
    save(fig, "05_equite_historique")


def run_all_figures(train: pd.DataFrame) -> None:
    plot_target(train)
    plot_distributions(train)
    plot_correlations(train)
    plot_categorical_rates(train)
    plot_fairness(train)

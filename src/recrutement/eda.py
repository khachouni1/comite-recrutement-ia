"""
Analyse exploratoire des paires (CV, offre) — sur le TRAIN uniquement.

- Structure des données : répartition des étiquettes, nombre de paires par CV et par offre.
- Longueur des textes.
- Qualité de l'extraction des critères (lexiques et règles).
- Lien de chaque variable avec l'étiquette : test de Kruskal-Wallis (comparaison de 3 groupes,
  H0 : même distribution pour « non adapté », « potentiel », « adapté »).
"""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal

from . import config as C
from .plots import PALETTE_LABELS, save


def structure(parts: dict) -> pd.DataFrame:
    rows = []
    for nom, df in parts.items():
        rows.append({"jeu": nom, "paires": len(df), "CV distincts": df["cv_id"].nunique(),
                     "offres distinctes": df["offre_id"].nunique(),
                     **{C.LABELS_FR[l]: f"{(df['label'] == l).mean():.1%}" for l in C.LABELS}})
    return pd.DataFrame(rows).set_index("jeu")


def tests_kruskal(feats: pd.DataFrame, y: pd.Series, variables) -> pd.DataFrame:
    rows = []
    for v in variables:
        groupes = [feats.loc[y == k, v] for k in range(3)]
        stat, p = kruskal(*groupes)
        rows.append({"variable": v, **{f"moyenne {C.LABELS_FR[l]}": g.mean() for l, g in zip(C.LABELS, groupes)},
                     "H": stat, "p_value": p})
    out = pd.DataFrame(rows).sort_values("H", ascending=False)
    out["significatif_5%"] = out["p_value"] < 0.05
    return out


def plot_etiquettes_longueurs(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    counts = train["label"].value_counts().reindex(C.LABELS)
    axes[0].bar([C.LABELS_FR[l] for l in C.LABELS], counts.values, color=[PALETTE_LABELS[l] for l in C.LABELS])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, f"{v} ({v / counts.sum():.0%})", ha="center")
    axes[0].set_title("Répartition des étiquettes (Train)")
    long = pd.DataFrame({"mots": pd.concat([train["cv"].str.split().str.len(), train["offre"].str.split().str.len()]),
                         "texte": ["CV"] * len(train) + ["Offre"] * len(train)})
    sns.histplot(long, x="mots", hue="texte", bins=60, ax=axes[1], element="step")
    axes[1].set_title("Longueur des textes (nombre de mots)")
    axes[1].set_xlim(0, 2000)
    par_offre = train.groupby("offre_id").size()
    axes[2].hist(par_offre, bins=30, color="#7f8c8d")
    axes[2].set_title(f"Candidats évalués par offre (médiane {par_offre.median():.0f})")
    axes[2].set_xlabel("Nombre de CV")
    save(fig, "01_etiquettes_longueurs")


def plot_competences_demandees(train: pd.DataFrame, analyses_offres: list) -> None:
    hard = Counter(s for a in analyses_offres for s in a["hard"]).most_common(20)
    soft = Counter(s for a in analyses_offres for s in a["soft"]).most_common(12)
    n = len(analyses_offres)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, data, titre, col in ((axes[0], hard, "Compétences techniques les plus demandées", "#2e86de"),
                                 (axes[1], soft, "Soft skills les plus demandés", "#27ae60")):
        noms, vals = zip(*data[::-1])
        ax.barh(noms, [v / n * 100 for v in vals], color=col)
        ax.set_title(f"{titre} (% des {n} offres du Train)")
        ax.set_xlabel("% des offres")
    save(fig, "02_competences_demandees")


def plot_variables_par_etiquette(feats: pd.DataFrame, y: pd.Series, variables) -> None:
    data = feats[variables].assign(etiquette=y.map(dict(enumerate(C.LABELS_FR[l] for l in C.LABELS))))
    n = len(variables)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7.5))
    for ax, v in zip(axes.ravel(), variables):
        sns.boxplot(data=data, x="etiquette", y=v, ax=ax, showfliers=False,
                    order=[C.LABELS_FR[l] for l in C.LABELS],
                    palette=[PALETTE_LABELS[l] for l in C.LABELS], hue="etiquette", legend=False)
        ax.set_xlabel("")
        ax.set_title(v)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Variables extraites selon l'étiquette (Train)", fontweight="bold")
    save(fig, "03_variables_par_etiquette")

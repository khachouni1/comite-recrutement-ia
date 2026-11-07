"""
Apprentissage non supervisé (Cours 2).

1. ACP — Analyse en Composantes Principales (slides 17-37)
2. Clustering — K-Means et CAH de Ward (slides 38-62), évalués par inertie intra-classe,
   silhouette, Davies-Bouldin et Calinski-Harabasz (slides 66-76), profilage (slide 77)
3. Détection d'anomalies — Distribution Gaussienne Multivariée (slides 78-85)

Toutes les méthodes sont ajustées sur le TRAIN uniquement.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import multivariate_normal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from . import config as C
from .plots import save, COLOR_HIRED, COLOR_REJECTED

# Variables décrivant le profil d'un candidat (hors attributs protégés et hors cible)
PROFILE_FEATURES = ["EducationLevel", "ExperienceYears", "PreviousCompanies", "DistanceFromCompany",
                    "InterviewScore", "SkillScore", "PersonalityScore"]


# =========================================================================== 1. ACP
def run_pca(train: pd.DataFrame, features=PROFILE_FEATURES) -> dict:
    """
    Étapes du cours (slide 33) : standardiser X, calculer S = XᵀX/(n-1),
    valeurs propres λk (variance de chaque axe), vecteurs propres Uk (directions).
    Critère de Kaiser : retenir les axes de valeur propre > 1.
    """
    scaler = StandardScaler().fit(train[features])
    Z = scaler.transform(train[features])
    pca = PCA(random_state=C.RANDOM_STATE).fit(Z)
    coords = pca.transform(Z)
    eigenvalues = pca.explained_variance_
    # Corrélation variable / axe = Uk * sqrt(λk)  (slide 32 : interprétation des facteurs)
    correlations = pd.DataFrame(pca.components_.T * np.sqrt(eigenvalues), index=features,
                                columns=[f"PC{k + 1}" for k in range(len(features))])
    table = pd.DataFrame({
        "valeur_propre": eigenvalues,
        "variance_expliquee_%": pca.explained_variance_ratio_ * 100,
        "variance_cumulee_%": np.cumsum(pca.explained_variance_ratio_) * 100,
    }, index=correlations.columns)
    return {"pca": pca, "scaler": scaler, "coords": coords, "table": table,
            "correlations": correlations, "n_kaiser": int((eigenvalues > 1).sum())}


def plot_pca(res: dict, y: pd.Series) -> None:
    table, corr, coords = res["table"], res["correlations"], res["coords"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]  # éboulis des valeurs propres
    ax.bar(table.index, table["variance_expliquee_%"], color="#2e86de", label="Variance expliquée")
    ax.plot(table.index, table["variance_cumulee_%"], "o-", color="#e67e22", label="Variance cumulée")
    ax2 = ax.twinx()
    ax2.plot(table.index, table["valeur_propre"], "s--", color="grey", label="Valeur propre")
    ax2.axhline(1, color="red", ls=":", label="Kaiser (λ = 1)")
    ax.set_title(f"Éboulis des valeurs propres — {res['n_kaiser']} axes retenus (Kaiser)")
    ax.set_ylabel("% de variance")
    ax2.set_ylabel("Valeur propre λ")
    ax.legend(loc="center right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(False)

    ax = axes[1]  # cercle des corrélations
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="grey"))
    for f in corr.index:
        x, yv = corr.loc[f, "PC1"], corr.loc[f, "PC2"]
        ax.arrow(0, 0, x, yv, head_width=0.03, color="#c0392b", length_includes_head=True)
        ax.text(x * 1.12, yv * 1.12, C.LABELS[f], ha="center", fontsize=8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title("Cercle des corrélations (PC1 × PC2)")
    ax.set_xlabel(f"PC1 ({table.iloc[0, 1]:.1f} %)")
    ax.set_ylabel(f"PC2 ({table.iloc[1, 1]:.1f} %)")

    ax = axes[2]  # projection des individus
    ax.scatter(coords[:, 0], coords[:, 1], c=np.where(y.values == 1, COLOR_HIRED, COLOR_REJECTED), s=10, alpha=0.5)
    ax.set_title("Projection des candidats (bleu = embauché, rouge = rejeté)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save(fig, "06_acp")


# =========================================================================== 2. Clustering
def evaluate_kmeans(Z: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    """Inertie intra-classe (coude), silhouette (↑), Davies-Bouldin (↓), Calinski-Harabasz (↑)."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(Z)
        rows.append({"k": k, "inertie_intra": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_),
                     "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                     "calinski_harabasz": calinski_harabasz_score(Z, km.labels_)})
    return pd.DataFrame(rows).set_index("k")


def run_clustering(train: pd.DataFrame, features=PROFILE_FEATURES) -> dict:
    Z = StandardScaler().fit_transform(train[features])
    scores = evaluate_kmeans(Z)
    # Règle de parcimonie : plus petit k dont la silhouette atteint 95 % du maximum
    # (évite de retenir un grand k pour un gain de silhouette négligeable).
    best_k = int(scores.index[scores["silhouette"] >= 0.95 * scores["silhouette"].max()].min())
    km = KMeans(n_clusters=best_k, n_init=10, random_state=C.RANDOM_STATE).fit(Z)
    cah = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(Z)
    agreement = pd.crosstab(pd.Series(km.labels_, name="K-Means"), pd.Series(cah.labels_, name="CAH"))
    ari = adjusted_rand_score(km.labels_, cah.labels_)
    # Profil statistique de chaque cluster (slide 77) : effectif, moyennes, taux d'embauche
    profile = train[features + [C.TARGET]].assign(cluster=km.labels_).groupby("cluster").mean()
    profile.insert(0, "effectif", np.bincount(km.labels_))
    return {"Z": Z, "scores": scores, "best_k": best_k, "kmeans_labels": km.labels_,
            "cah_labels": cah.labels_, "agreement": agreement, "ari": float(ari), "profile": profile.round(2)}


def plot_clustering(res: dict, pca_coords: np.ndarray) -> None:
    s = res["scores"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, col, better in zip(axes, ["inertie_intra", "silhouette", "davies_bouldin", "calinski_harabasz"],
                               ["coude", "max", "min", "max"]):
        ax.plot(s.index, s[col], "o-")
        ax.axvline(res["best_k"], color="red", ls=":")
        ax.set_title(f"{col.replace('_', ' ').title()} ({better})")
        ax.set_xlabel("k")
    fig.suptitle(f"Choix du nombre de clusters K-Means — k retenu = {res['best_k']} "
                 "(plus petit k à ≥ 95 % de la silhouette max)",
                 fontweight="bold")
    save(fig, "07_choix_k_clustering")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    linked = linkage(res["Z"], method="ward")
    dendrogram(linked, truncate_mode="lastp", p=30, ax=axes[0], color_threshold=None)
    axes[0].set_title("Dendrogramme de la CAH (critère de Ward)")
    axes[0].set_ylabel("Distance d'agrégation")
    sc = axes[1].scatter(pca_coords[:, 0], pca_coords[:, 1], c=res["kmeans_labels"], cmap="tab10", s=10, alpha=0.6)
    axes[1].legend(*sc.legend_elements(), title="Cluster", fontsize=8)
    axes[1].set_title("Clusters K-Means projetés sur le plan factoriel de l'ACP")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    save(fig, "08_clusters_cah_kmeans")


# =========================================================================== 3. Anomalies
class GaussianAnomalyDetector:
    """
    Détection d'anomalies par Distribution Gaussienne Multivariée (Cours 2, slides 82-85) :
      1. standardiser ;  2. estimer μ = 1/n Σ xi  et  Σ = 1/n Σ (xi-μ)(xi-μ)ᵀ  sur le Train ;
      3. calculer p(x) ;  4. anomalie si p(x) < ε  (ε = quantile bas des densités du Train).
    On travaille en log-densité pour éviter les valeurs numériquement nulles.
    """

    def __init__(self, features=None, contamination: float = C.ANOMALY_CONTAMINATION):
        self.features = list(features or ["Age"] + PROFILE_FEATURES)
        self.contamination = contamination

    def fit(self, df: pd.DataFrame) -> "GaussianAnomalyDetector":
        self.scaler_ = StandardScaler().fit(df[self.features])
        Z = self.scaler_.transform(df[self.features])
        self.mu_ = Z.mean(axis=0)
        self.sigma_ = np.cov(Z, rowvar=False, bias=True)
        self.dist_ = multivariate_normal(self.mu_, self.sigma_, allow_singular=True)
        self.log_epsilon_ = float(np.quantile(self.dist_.logpdf(Z), self.contamination))
        return self

    def log_density(self, df: pd.DataFrame) -> np.ndarray:
        return np.atleast_1d(self.dist_.logpdf(self.scaler_.transform(df[self.features])))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """True = anomalie."""
        return self.log_density(df) < self.log_epsilon_

    def zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler_.transform(df[self.features]), columns=self.features, index=df.index)


def plot_anomalies(det: GaussianAnomalyDetector, train: pd.DataFrame, test: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(det.log_density(train), bins=50, alpha=0.6, label="Train", density=True)
    ax.hist(det.log_density(test), bins=50, alpha=0.6, label="Test", density=True)
    ax.axvline(det.log_epsilon_, color="red", ls="--", label=f"Seuil log ε = {det.log_epsilon_:.1f}")
    ax.set_title("Log-densité gaussienne multivariée des candidats")
    ax.set_xlabel("log p(x)")
    ax.legend()
    save(fig, "09_anomalies_gaussiennes")

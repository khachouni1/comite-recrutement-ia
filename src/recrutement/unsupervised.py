"""
Apprentissage non supervisé sur les textes (Cours 2).

1. Réduction de dimension : LSA = SVD tronquée du TF-IDF (l'ACP appliquée au texte), éboulis de variance.
2. Clustering des CV : K-Means et CAH de Ward dans l'espace LSA, choix de k par inertie intra-classe,
   silhouette, Davies-Bouldin, Calinski-Harabasz ; profil des clusters (mots caractéristiques).
3. Détection d'anomalies : distribution gaussienne multivariée sur les variables d'un CV (slides 82-85).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import multivariate_normal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler, normalize

from . import config as C
from .plots import save


# =========================================================================== 1. LSA
def plot_lsa_variance(reps) -> pd.Series:
    var = pd.Series(reps.lsa.explained_variance_ratio_, index=range(1, C.LSA_COMPONENTS + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(var.index, var.values * 100, color="#2e86de", label="Variance expliquée par axe")
    ax2 = ax.twinx()
    ax2.plot(var.index, var.cumsum() * 100, color="#e67e22", label="Variance cumulée")
    ax2.grid(False)
    ax.set_title(f"LSA (SVD du TF-IDF) : {C.LSA_COMPONENTS} axes = {var.sum():.0%} de la variance")
    ax.set_xlabel("Axe (thème latent)")
    ax.set_ylabel("% par axe")
    ax2.set_ylabel("% cumulé")
    save(fig, "04_lsa_variance")
    return var


# =========================================================================== 2. clustering des CV
def clustering_cv(cv_lsa: np.ndarray, k_range=range(2, 13)) -> dict:
    Z = normalize(cv_lsa)  # norme 1 : la distance euclidienne devient équivalente au cosinus
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(Z)
        rows.append({"k": k, "inertie_intra": km.inertia_, "silhouette": silhouette_score(Z, km.labels_),
                     "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                     "calinski_harabasz": calinski_harabasz_score(Z, km.labels_)})
    scores = pd.DataFrame(rows).set_index("k")
    best_k = int(scores["silhouette"].idxmax())
    km = KMeans(n_clusters=best_k, n_init=10, random_state=C.RANDOM_STATE).fit(Z)
    cah = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(Z)
    return {"Z": Z, "scores": scores, "best_k": best_k, "kmeans": km, "labels": km.labels_,
            "ari_cah": float(adjusted_rand_score(km.labels_, cah.labels_))}


def mots_cles_clusters(res: dict, reps, n_mots: int = 8) -> dict:
    """Mots caractéristiques : centroïde ramené dans l'espace TF-IDF via les axes de la LSA."""
    termes = np.array(reps.tfidf.get_feature_names_out())
    centres_tfidf = res["kmeans"].cluster_centers_ @ reps.lsa.components_
    return {k: list(termes[np.argsort(c)[::-1][:n_mots]]) for k, c in enumerate(centres_tfidf)}


def plot_clustering(res: dict, mots: dict, profil: pd.DataFrame) -> None:
    s = res["scores"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, col, sens in zip(axes, s.columns, ["coude", "max", "min", "max"]):
        ax.plot(s.index, s[col], "o-")
        ax.axvline(res["best_k"], color="red", ls=":")
        ax.set_title(f"{col.replace('_', ' ').title()} ({sens})")
        ax.set_xlabel("k")
    fig.suptitle(f"Clustering K-Means des CV (espace LSA) — k retenu = {res['best_k']} (silhouette max)",
                 fontweight="bold")
    save(fig, "05_clustering_choix_k")

    coords = PCA(n_components=2, random_state=C.RANDOM_STATE).fit_transform(res["Z"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=res["labels"], cmap="tab10", s=14, alpha=0.75)
    for k in range(res["best_k"]):
        cx, cy = coords[res["labels"] == k].mean(axis=0)
        axes[0].text(cx, cy, f"{k}: {', '.join(mots[k][:3])}", fontsize=8, weight="bold",
                     bbox=dict(facecolor="white", alpha=0.8, lw=0))
    axes[0].set_title("Familles de CV (projection ACP du plan LSA)")
    axes[0].set_xlabel("Axe 1")
    axes[0].set_ylabel("Axe 2")
    linked = linkage(res["Z"], method="ward")
    dendrogram(linked, truncate_mode="lastp", p=25, ax=axes[1])
    axes[1].set_title(f"Dendrogramme CAH (Ward) — accord avec K-Means : ARI = {res['ari_cah']:.2f}")
    save(fig, "06_familles_de_cv")

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(profil) + 1.5))
    p = profil.iloc[::-1]
    ax.barh([f"{k} — {', '.join(mots[k][:4])}" for k in p.index], p["taux_adapte"] * 100, color="#27ae60")
    for i, (n, t) in enumerate(zip(p["n_cv"], p["taux_adapte"])):
        ax.text(t * 100 + 0.5, i, f"{t:.0%} ({n} CV)", va="center", fontsize=8)
    ax.set_title("Part des candidatures « adaptées » par famille de CV (Train)")
    ax.set_xlabel("% de paires « Good Fit »")
    save(fig, "07_profil_familles")


# =========================================================================== 3. anomalies
class GaussianAnomalyDetector:
    """
    Distribution gaussienne multivariée (Cours 2, slides 82-85) :
    μ et Σ estimés sur le Train, p(x) calculé pour chaque dossier, anomalie si p(x) < ε.
    ε = quantile bas (2 %) des densités du Train. On travaille en log-densité.
    """

    def __init__(self, variables: list, contamination: float = C.ANOMALY_CONTAMINATION):
        self.variables = variables
        self.contamination = contamination

    def fit(self, df: pd.DataFrame) -> "GaussianAnomalyDetector":
        self.scaler_ = StandardScaler().fit(df[self.variables])
        Z = self.scaler_.transform(df[self.variables])
        self.dist_ = multivariate_normal(Z.mean(axis=0), np.cov(Z, rowvar=False, bias=True), allow_singular=True)
        self.log_epsilon_ = float(np.quantile(self.dist_.logpdf(Z), self.contamination))
        return self

    def log_density(self, df: pd.DataFrame) -> np.ndarray:
        return np.atleast_1d(self.dist_.logpdf(self.scaler_.transform(df[self.variables])))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.log_density(df) < self.log_epsilon_

    def zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler_.transform(df[self.variables]), columns=self.variables, index=df.index)

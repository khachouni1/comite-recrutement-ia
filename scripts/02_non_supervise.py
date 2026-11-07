"""
ÉTAPE 2 — Apprentissage non supervisé : ACP, clustering, détection d'anomalies (Cours 2).

    python scripts/02_non_supervise.py

Sorties : reports/figures/06_* à 09_*, reports/resultats/02_*.csv / .json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recrutement import config as C
from recrutement import data, unsupervised as U

print("=" * 70, "\nÉTAPE 2 — APPRENTISSAGE NON SUPERVISÉ\n" + "=" * 70)
train, val, test = data.load_splits()

# ----------------------------------------------------------------- ACP
print("\n1) ACP sur les variables de profil (Train, données centrées-réduites)")
pca = U.run_pca(train)
print(pca["table"].round(2).to_string())
print(f"   Critère de Kaiser (λ > 1) : {pca['n_kaiser']} axes retenus "
      f"({pca['table']['variance_cumulee_%'].iloc[pca['n_kaiser'] - 1]:.1f} % de variance)")
print("   Corrélations variables / axes (interprétation) :")
print(pca["correlations"][["PC1", "PC2", "PC3"]].round(2).to_string())
U.plot_pca(pca, train[C.TARGET])
pca["table"].to_csv(C.RESULTS_DIR / "02_acp_valeurs_propres.csv")
pca["correlations"].to_csv(C.RESULTS_DIR / "02_acp_correlations.csv")

# ----------------------------------------------------------------- Clustering
print("\n2) Clustering K-Means (k = 2 … 8) et CAH de Ward")
clu = U.run_clustering(train)
print(clu["scores"].round(3).to_string())
print(f"   k retenu (parcimonie : plus petit k à ≥ 95 % de la silhouette max) = {clu['best_k']} ; "
      f"silhouette = {clu['scores'].loc[clu['best_k'], 'silhouette']:.3f}")
print(f"   Concordance K-Means / CAH : indice de Rand ajusté = {clu['ari']:.3f} (1 = partitions identiques)")
print(clu["agreement"].to_string())
print("   Profil des clusters (moyennes, taux d'embauche) :")
print(clu["profile"].to_string())
U.plot_clustering(clu, pca["coords"])
clu["scores"].to_csv(C.RESULTS_DIR / "02_clustering_scores.csv")
clu["profile"].to_csv(C.RESULTS_DIR / "02_clustering_profils.csv")
sil = clu["scores"].loc[clu["best_k"], "silhouette"]
if sil < 0.25:
    print(f"   ⚠ Silhouette faible ({sil:.2f} < 0.25) : les candidats ne forment pas de groupes naturels bien "
          "séparés. On utilise donc les clusters comme outil descriptif, pas comme critère de sélection.")

# ----------------------------------------------------------------- Anomalies
print("\n3) Détection d'anomalies — Gaussienne multivariée (μ, Σ estimés sur Train)")
det = U.GaussianAnomalyDetector().fit(train)
n_tr, n_te = int(det.predict(train).sum()), int(det.predict(test).sum())
print(f"   Seuil log ε = {det.log_epsilon_:.2f} | anomalies : Train {n_tr}/{len(train)}, Test {n_te}/{len(test)}")
U.plot_anomalies(det, train, test)

with open(C.RESULTS_DIR / "02_synthese.json", "w", encoding="utf-8") as f:
    json.dump({"acp_axes_kaiser": pca["n_kaiser"],
               "acp_variance_2_axes_%": round(float(pca["table"]["variance_cumulee_%"].iloc[1]), 2),
               "clustering_k": clu["best_k"], "clustering_silhouette": round(float(sil), 4),
               "anomalies_log_epsilon": round(det.log_epsilon_, 3),
               "anomalies_train": n_tr, "anomalies_test": n_te}, f, indent=2, ensure_ascii=False)
print("\nÉtape 2 terminée.")

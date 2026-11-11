"""
ÉTAPE 2 — Représentation du texte (TF-IDF, LSA, embeddings) et apprentissage non supervisé.

    python scripts/02_representation_non_supervise.py

Sorties : models/representations.joblib, data/processed/features_{train,val,test}.pkl,
          reports/figures/04_* à 08_*, reports/resultats/02_*
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from recrutement import config as C
from recrutement import data, features as F, unsupervised as U
from recrutement.plots import PALETTE_LABELS, save
from recrutement.representation import Representations

print("=" * 70, "\nÉTAPE 2 — REPRÉSENTATION DU TEXTE & NON SUPERVISÉ\n" + "=" * 70)
train, val, test = data.charger("train", "val", "test")

# ----------------------------------------------------------------- représentations
print("\n1) Apprentissage du TF-IDF et de la LSA sur les textes du TRAIN uniquement")
textes_train = pd.concat([train["cv"], train["offre"]]).drop_duplicates()
reps = Representations().fit(textes_train)
print(f"   Vocabulaire TF-IDF : {len(reps.tfidf.vocabulary_)} termes (mots et paires de mots)")
var = U.plot_lsa_variance(reps)

print("\n2) Variables de chaque paire (règles + similarités TF-IDF / LSA / embeddings)")
feats = {}
for nom, df in (("train", train), ("val", val), ("test", test)):
    feats[nom] = F.construire(df, reps)
    feats[nom].to_pickle(C.DATA_PROCESSED / f"features_{nom}.pkl")
    print(f"   {nom:5s} : {feats[nom].shape[0]} paires × {feats[nom].shape[1]} variables")
reps.sauver_cache()
print(f"   Embeddings : {'Hugging Face ' + C.EMBEDDING_MODEL if reps.utiliser_embeddings else 'INDISPONIBLES (repli LSA)'}")

print("\n3) Quelle représentation mesure le mieux l'adéquation ? (Validation)")
y_val = val["y"]
comp = []
for sim, nom in (("sim_tfidf", "TF-IDF (baseline)"), ("sim_lsa", "LSA (SVD)"), ("sim_embedding", "Embeddings HF")):
    comp.append({"représentation": nom,
                 "AUC adapté/potentiel vs non adapté": roc_auc_score(y_val > 0, feats["val"][sim]),
                 "AUC « adapté » vs reste": roc_auc_score(y_val == 2, feats["val"][sim]),
                 "Spearman avec l'étiquette": spearmanr(feats["val"][sim], y_val)[0]})
comp = pd.DataFrame(comp)
print(comp.round(3).to_string(index=False))
comp.to_csv(C.RESULTS_DIR / "02_comparaison_representations.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, sim, nom in zip(axes, ["sim_tfidf", "sim_lsa", "sim_embedding"], ["TF-IDF", "LSA", "Embeddings"]):
    for k, lab in enumerate(C.LABELS):
        ax.hist(feats["val"].loc[y_val == k, sim], bins=30, alpha=0.5, density=True,
                color=PALETTE_LABELS[lab], label=C.LABELS_FR[lab])
    ax.set_title(f"Similarité CV / offre — {nom} (Validation)")
    ax.legend(fontsize=8)
save(fig, "08_similarites_par_etiquette")

# ----------------------------------------------------------------- clustering des CV
print("\n4) Clustering des CV distincts du Train (espace LSA)")
cvs = train.drop_duplicates("cv_id")
res = U.clustering_cv(reps.lsa_vec(cvs["cv"]))
print(res["scores"].round(3).to_string())
mots = U.mots_cles_clusters(res, reps)
lab_cv = pd.Series(res["labels"], index=cvs["cv_id"])
profil = (train.assign(cluster=train["cv_id"].map(lab_cv)).groupby("cluster")
          .agg(n_cv=("cv_id", "nunique"), n_paires=("y", "size"), taux_adapte=("y", lambda s: (s == 2).mean())))
print(f"   k retenu = {res['best_k']} | silhouette = {res['scores'].loc[res['best_k'], 'silhouette']:.3f} | "
      f"accord K-Means / CAH (ARI) = {res['ari_cah']:.2f}")
for k in profil.index:
    print(f"   Famille {k} ({profil.loc[k, 'n_cv']} CV, {profil.loc[k, 'taux_adapte']:.0%} adaptés) : {', '.join(mots[k])}")
U.plot_clustering(res, mots, profil)
res["scores"].to_csv(C.RESULTS_DIR / "02_clustering_scores.csv")

joblib.dump(reps, C.MODELS_DIR / "representations.joblib")
with open(C.RESULTS_DIR / "02_synthese.json", "w", encoding="utf-8") as f:
    json.dump({"embeddings": reps.utiliser_embeddings, "lsa_variance_totale": round(float(var.sum()), 4),
               "clustering_k": res["best_k"], "silhouette": round(float(res["scores"].loc[res["best_k"], "silhouette"]), 4),
               "ari_kmeans_cah": round(res["ari_cah"], 4),
               "familles": {int(k): mots[k] for k in mots}}, f, indent=2, ensure_ascii=False)
print("\nÉtape 2 terminée.")

"""
ÉTAPE 3 — Apprentissage supervisé (3 classes), sélection de modèle et interprétabilité.

Deux jeux de variables sont comparés :
  A. « explicables » : les 17 variables extraites (compétences, années, diplômes, similarités) ;
  B. « explicables + embeddings » : A + interactions des embeddings CV / offre (768 dimensions).

    python scripts/03_supervise.py

Sorties : models/modele_final.joblib, reports/resultats/03_*, reports/figures/09_* à 16_*
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, learning_curve

from recrutement import config as C
from recrutement import data, explain, features as F, supervised as S
from recrutement.plots import save

print("=" * 70, "\nÉTAPE 3 — APPRENTISSAGE SUPERVISÉ & INTERPRÉTABILITÉ\n" + "=" * 70)
train, val, test = data.charger("train", "val", "test")
f_tr, f_va, f_te = data.charger("features_train", "features_val", "features_test")
y_tr, y_va, y_te = train["y"], val["y"], test["y"]
JEUX = {"explicables": C.FEATURES, "explicables + embeddings": C.FEATURES + F.colonnes_embeddings(f_tr)}
for nom, cols in JEUX.items():
    print(f"Jeu « {nom} » : {len(cols)} variables")

print("\n1) Validation croisée groupée par offre (Train), puis comparaison sur Validation")
table, modeles = S.regler_et_comparer(f_tr, y_tr, train["offre_id"], f_va, y_va, JEUX)
table.to_csv(C.RESULTS_DIR / "03_comparaison_modeles.csv", index=False)
meilleur = table.loc[0, "Modèle"]
cols = JEUX[table.loc[0, "Variables"]]
print(f"\n   => Modèle retenu (meilleure AUC en validation croisée groupée, 238 offres) : {meilleur} "
      f"{table.loc[0, 'Hyperparamètres']} | contrôle sur Validation : AUC = {table.loc[0, 'AUC macro (OvR) (Val)']:.3f}")

fig, ax = plt.subplots(figsize=(11, 5.5))
vue = table.set_index("Modèle")[["AUC macro (OvR) (Val)", "AUC adéquat vs non (Val)", "F1 macro (Val)"]]
vue.iloc[::-1].plot.barh(ax=ax)
ax.set_xlim(0, 1)
ax.set_title("Comparaison des modèles et des jeux de variables (Validation)")
ax.legend(fontsize=8, loc="lower right")
save(fig, "09_comparaison_modeles")

print("\n2) Courbe d'apprentissage (biais / variance, Cours 1 slides 45-49)")
tailles, sc_tr, sc_cv = learning_curve(
    clone(modeles[meilleur]), f_tr[cols], y_tr, groups=train["offre_id"],
    cv=StratifiedGroupKFold(5, shuffle=True, random_state=C.RANDOM_STATE), scoring="roc_auc_ovr",
    train_sizes=np.linspace(0.2, 1, 5), n_jobs=-1)
fig, ax = plt.subplots(figsize=(7, 4.5))
for s, lab in ((sc_tr, "Entraînement"), (sc_cv, "Validation croisée")):
    ax.plot(tailles, s.mean(1), "o-", label=lab)
    ax.fill_between(tailles, s.mean(1) - s.std(1), s.mean(1) + s.std(1), alpha=0.2)
ax.set_title(f"Courbe d'apprentissage — {meilleur}")
ax.set_xlabel("Nombre de paires d'entraînement")
ax.set_ylabel("AUC macro")
ax.legend()
save(fig, "10_courbe_apprentissage")
print(f"   AUC entraînement = {sc_tr.mean(1)[-1]:.3f} | validation croisée = {sc_cv.mean(1)[-1]:.3f}")

print("\n3) Ré-entraînement sur Train + Validation, évaluation UNIQUE sur Test")
final = S.reentrainer(modeles[meilleur], f_tr[cols], y_tr, f_va[cols], y_va)
cle_base = "Régression softmax (baseline) — explicables"
base = S.reentrainer(modeles[cle_base], f_tr[C.FEATURES], y_tr, f_va[C.FEATURES], y_va)
p_te = final.predict_proba(f_te[cols])
res = pd.DataFrame({meilleur: S.metriques(y_te, p_te),
                    cle_base: S.metriques(y_te, base.predict_proba(f_te[C.FEATURES]))}).T
print(res.round(3).to_string())
res.to_csv(C.RESULTS_DIR / "03_resultats_test.csv")

fig, ax = plt.subplots(figsize=(7, 4.5))
for k, lab in enumerate(C.LABELS):
    ax.hist(S.score_attendu(p_te)[y_te.values == k], bins=25, alpha=0.55, density=True, label=C.LABELS_FR[lab])
ax.set_title("Score d'adéquation prédit selon l'étiquette réelle (Test)")
ax.set_xlabel("Score attendu (0-100)")
ax.legend()
save(fig, "11_score_par_etiquette")

joblib.dump(final, C.MODELS_DIR / "modele_final.joblib")
with open(C.RESULTS_DIR / "03_modele_retenu.json", "w", encoding="utf-8") as f:
    json.dump({"modele": table.loc[0, "Algorithme"], "jeu_variables": table.loc[0, "Variables"],
               "hyperparametres": table.loc[0, "Hyperparamètres"], "variables": cols,
               "test": {k: round(float(v), 4) for k, v in res.loc[meilleur].items()}}, f, indent=2, ensure_ascii=False)

print("\n4) Interprétabilité")
valeurs, donnees = explain.shap_adapte(final, f_te[cols])
explain.plot_shap(valeurs, donnees, pd.Series(p_te[:, 2], index=f_te.index), y_te)
print("   SHAP — importance moyenne |SHAP| (classe « adapté ») :")
print(valeurs.abs().mean().sort_values(ascending=False).head(10).round(4).to_string())
perm = explain.permutation(final, f_te[cols], y_te)
perm.to_csv(C.RESULTS_DIR / "03_importance_permutation.csv", index=False)
print("   Permutation — perte d'AUC (6 premières) :")
print(perm.head(6).round(4).to_string(index=False))
prof = explain.erreurs(y_te, p_te, f_te[cols])
print("   Analyse des erreurs (profil moyen par type) :")
print(prof.to_string())
prof.to_csv(C.RESULTS_DIR / "03_analyse_erreurs.csv")
print("\nÉtape 3 terminée.")

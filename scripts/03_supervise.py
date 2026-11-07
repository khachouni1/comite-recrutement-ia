"""
ÉTAPE 3 — Apprentissage supervisé, sélection de modèle et interprétabilité (Cours 1).

    python scripts/03_supervise.py

Sorties : models/modele_final.joblib, reports/resultats/03_*, reports/figures/10_* à 17_*
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
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold, learning_curve

from recrutement import config as C
from recrutement import data, explain, supervised as S
from recrutement.plots import save

print("=" * 70, "\nÉTAPE 3 — APPRENTISSAGE SUPERVISÉ & INTERPRÉTABILITÉ\n" + "=" * 70)
train, val, test = data.load_splits()
X_tr, y_tr = data.X_y(train)
X_va, y_va = data.X_y(val)
X_te, y_te = data.X_y(test)
print(f"Variables utilisées ({len(C.MODEL_FEATURES)}) : {C.MODEL_FEATURES}")
print(f"Attributs protégés exclus des modèles : {C.PROTECTED}")

# ----------------------------------------------------------------- 1-2. réglage + comparaison
print("\n1) Hyperparamètres par validation croisée 5 plis (Train) puis comparaison sur Validation :")
table, fitted = S.tune_and_compare(X_tr, y_tr, X_va, y_va, C.MODEL_FEATURES)
table.to_csv(C.RESULTS_DIR / "03_comparaison_modeles.csv", index=False)
best_name = table.loc[0, "Modèle"]
best_params = table.loc[0, "Hyperparamètres"]
print(f"\n   => Modèle retenu (meilleure AUC sur Validation) : {best_name} {best_params}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
cols = ["ROC-AUC (Val)", "Accuracy équilibrée (Val)", "F1 (Val)"]
table.set_index("Modèle")[cols].iloc[::-1].plot.barh(ax=axes[0])
axes[0].set_title("Comparaison des modèles sur la Validation")
axes[0].set_xlim(0, 1)
axes[0].legend(fontsize=8, loc="lower right")
for name, model in fitted.items():
    RocCurveDisplay.from_estimator(model, X_va, y_va, name=name, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k:", lw=1)
axes[1].set_title("Courbes ROC sur la Validation")
axes[1].legend(fontsize=7)
save(fig, "10_comparaison_modeles")

# ----------------------------------------------------------------- équité : avec / sans attributs protégés
best_pipe = fitted[best_name]
with_protected = clone(best_pipe)
with_protected.set_params(prep=S.build_pipeline(best_pipe.named_steps["model"], C.ALL_FEATURES).named_steps["prep"])
with_protected.fit(train[C.ALL_FEATURES], y_tr)
auc_with = roc_auc_score(y_va, with_protected.predict_proba(val[C.ALL_FEATURES])[:, 1])
auc_without = table.loc[0, "ROC-AUC (Val)"]
print(f"\n2) Coût de l'équité : AUC Val sans genre/âge = {auc_without:.3f} | avec genre/âge = {auc_with:.3f}"
      f" (écart {auc_with - auc_without:+.3f}) -> on exclut les attributs protégés sans perte notable.")

# ----------------------------------------------------------------- courbe d'apprentissage
print("\n3) Courbe d'apprentissage du modèle retenu (diagnostic biais / variance, Cours 1 slides 45-49)")
sizes, tr_scores, cv_scores = learning_curve(
    clone(best_pipe), X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE),
    scoring="roc_auc", train_sizes=np.linspace(0.15, 1.0, 7), n_jobs=-1)
fig, ax = plt.subplots(figsize=(7, 4.5))
for scores, label in ((tr_scores, "Score d'entraînement"), (cv_scores, "Score de validation croisée")):
    ax.plot(sizes, scores.mean(axis=1), "o-", label=label)
    ax.fill_between(sizes, scores.mean(axis=1) - scores.std(axis=1), scores.mean(axis=1) + scores.std(axis=1), alpha=0.2)
ax.set_title(f"Courbe d'apprentissage — {best_name}")
ax.set_xlabel("Nombre d'exemples d'entraînement")
ax.set_ylabel("ROC-AUC")
ax.legend()
save(fig, "12_courbe_apprentissage")
gap = tr_scores.mean(axis=1)[-1] - cv_scores.mean(axis=1)[-1]
print(f"   AUC entraînement = {tr_scores.mean(axis=1)[-1]:.3f} | AUC validation croisée = "
      f"{cv_scores.mean(axis=1)[-1]:.3f} | écart = {gap:.3f}")

# ----------------------------------------------------------------- 3-4. ré-entraînement + test
print("\n4) Ré-entraînement sur Train + Validation, puis évaluation UNIQUE sur Test")
final = S.refit_on_train_val(best_pipe, X_tr, y_tr, X_va, y_va)
proba_te = final.predict_proba(X_te)[:, 1]
test_metrics = S.metrics(y_te, proba_te)
baseline = S.refit_on_train_val(fitted["Régression logistique (baseline)"], X_tr, y_tr, X_va, y_va)
baseline_metrics = S.metrics(y_te, baseline.predict_proba(X_te)[:, 1])
res = pd.DataFrame({best_name: test_metrics, "Régression logistique (baseline)": baseline_metrics}).T
print(res.round(3).to_string())
res.to_csv(C.RESULTS_DIR / "03_resultats_test.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ConfusionMatrixDisplay.from_predictions(y_te, (proba_te >= 0.5).astype(int), ax=axes[0], cmap="Blues",
                                        display_labels=["Rejeté", "Embauché"], colorbar=False)
axes[0].set_title(f"Matrice de confusion (Test) — {best_name}")
RocCurveDisplay.from_predictions(y_te, proba_te, name=best_name, ax=axes[1])
RocCurveDisplay.from_predictions(y_te, baseline.predict_proba(X_te)[:, 1], name="Baseline logistique", ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k:", lw=1)
axes[1].set_title("Courbe ROC (Test)")
save(fig, "11_evaluation_finale_test")

joblib.dump(final, C.MODELS_DIR / "modele_final.joblib")
with open(C.RESULTS_DIR / "03_modele_retenu.json", "w", encoding="utf-8") as f:
    json.dump({"modele": best_name, "hyperparametres": best_params, "variables": C.MODEL_FEATURES,
               "auc_val": round(float(auc_without), 4), "auc_val_avec_attributs_proteges": round(float(auc_with), 4),
               "test": {k: round(float(v), 4) for k, v in test_metrics.items()}}, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------- interprétabilité
print("\n5) Interprétabilité")
exp, Xt = explain.shap_values(final, X_te)
explain.plot_shap(exp, Xt, y_te, proba_te)
shap_imp = pd.Series(np.abs(exp.values).mean(axis=0), index=Xt.columns).sort_values(ascending=False)
print("   SHAP — importance globale moyenne |SHAP| :")
print(shap_imp.round(3).to_string())

perm = explain.permutation_table(final, X_te, y_te)
explain.plot_permutation(perm)
perm.to_csv(C.RESULTS_DIR / "03_importance_permutation.csv", index=False)
print("   Permutation — baisse d'AUC :")
print(perm.round(4).to_string(index=False))

err = explain.error_analysis(test, proba_te)
print("   Analyse des erreurs — profil moyen par type de prédiction :")
print(err["profil"][["effectif", "SkillScore", "InterviewScore", "PersonalityScore", "EducationLevel",
                     "RecruitmentStrategy", "proba"]].to_string())
print(f"   Taux d'erreur par groupe : {err['taux_erreur_par_groupe']}")
err["profil"].to_csv(C.RESULTS_DIR / "03_analyse_erreurs.csv")
print("\nÉtape 3 terminée.")

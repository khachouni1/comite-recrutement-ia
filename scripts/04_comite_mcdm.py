"""
ÉTAPE 4 — Comité multi-agents & aide à la décision multicritère (AHP, WSM, TOPSIS).

    python scripts/04_comite_mcdm.py

Protocole : agents entraînés sur Train, seuils et méthode MCDM vérifiés sur Validation,
évaluation finale sur Test (vérité terrain = décision d'embauche réelle).
Sorties : models/comite.joblib, data/processed/classement_test.csv, reports/resultats/04_*,
          reports/figures/17_* à 21_*
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from recrutement import config as C
from recrutement import data
from recrutement.agents import Coordinateur
from recrutement.mcdm import sensitivity
from recrutement.plots import DECISION_COLORS, save

print("=" * 70, "\nÉTAPE 4 — COMITÉ MULTI-AGENTS & MCDM\n" + "=" * 70)
train, val, test = data.load_splits()
LAB = C.CRITERIA_LABELS

# ----------------------------------------------------------------- 1. poids AHP + apprentissage des agents
comite = Coordinateur().fit(train)
ahp = comite.ahp.summary()
print("\n1) Pondération AHP des critères (vecteur propre principal) :")
for c, w in ahp["poids"].items():
    print(f"   {LAB[c]:18s} {w:6.1%}")
print(f"   λmax = {ahp['lambda_max']} | CI = {ahp['CI']} | CR = {ahp['CR']} -> cohérent : {ahp['coherent']}")

print("\n2) Ce que chaque agent a appris sur le Train, et sa qualité seul sur la Validation :")
for c, agent in comite.evaluateurs.items():
    auc = roc_auc_score(val[C.TARGET], agent.noter(val))
    appris = agent.poids_appris() if hasattr(agent, "poids_appris") else agent.config["modele"]
    print(f"   {agent.nom:15s} AUC Val = {auc:.3f} | {appris}")
print(f"   Seuils de véto (5e percentile Train) : { {LAB[c]: round(s, 1) for c, s in comite.seuils_veto.items()} }")

# ----------------------------------------------------------------- 2. vérification sur Validation
print("\n3) Vérifications sur la Validation (aucune décision prise avec le Test) :")
delib_val = comite.deliberer(val, expliquer=False)
dec_val = comite.decider(delib_val)
y_val = val.loc[dec_val.index, C.TARGET]
print(f"   Taux de véto = {(dec_val['veto'] != '').mean():.1%} | taux de désaccord = {dec_val['desaccord'].mean():.1%}")
print(f"   AUC Val : TOPSIS = {roc_auc_score(y_val, dec_val['TOPSIS_score']):.3f} | "
      f"WSM = {roc_auc_score(y_val, dec_val['WSM_score']):.3f}")

# ----------------------------------------------------------------- 3. délibération sur le Test
print("\n4) Délibération du comité sur le Test")
delib = comite.deliberer(test)
dec = comite.decider(delib)
classement = dec.join(test)
y = classement[C.TARGET]
k = comite.nb_places(len(dec))
print(classement["decision"].value_counts().to_string())
print(f"   Short-list : {k} places | dossiers atypiques : {int(dec['anomalie'].sum())} | "
      f"désaccords : {int(dec['desaccord'].sum())} | tests contrefactuels réussis : "
      f"{int(dec['contrefactuel_ok'].sum())}/{len(dec)}")

# ----------------------------------------------------------------- 4. confrontation à la réalité
print("\n5) Confrontation aux décisions réelles (Test)")
methodes = {"TOPSIS (comité)": "TOPSIS_score", "WSM (baseline)": "WSM_score"} | \
           {f"{LAB[c]} seul": c for c in C.CRITERIA}
lignes = []
for nom, col in methodes.items():
    ordre = classement.sort_values(col, ascending=False)[C.TARGET].to_numpy()
    lignes.append({"Méthode": nom, "ROC-AUC": roc_auc_score(y, classement[col]),
                   **{f"Précision@{n}": ordre[:n].mean() for n in (10, 25, 50, k)}})
evaluation = pd.DataFrame(lignes)
print(evaluation.round(3).to_string(index=False))
sel = classement["selectionne"]
print(f"   Sélection du comité : {int(sel.sum())} candidats, dont {y[sel].mean():.0%} réellement embauchés "
      f"(contre {y.mean():.0%} au hasard) ; rappel = {y[sel].sum() / y.sum():.0%}")
taux_decision = classement.groupby("decision")[C.TARGET].agg(["count", "mean"])
print("   Taux d'embauche réel par décision :")
print(taux_decision.round(3).to_string())
rho = spearmanr(classement["WSM_rang"], classement["TOPSIS_rang"])[0]
print(f"   Concordance WSM / TOPSIS : ρ de Spearman = {rho:.3f}")

# ----------------------------------------------------------------- 5. audit éthique
print("\n6) Audit éthique de la sélection")
audit = comite.ethique.audit_selection(classement)
for g, info in audit["disparate_impact"].items():
    print(f"   {g:12s} DI = {info['DI']:.2f} ({'conforme' if info['conforme'] else 'ALERTE'}) | {info['taux_selection']}")
print(f"   Proxies (ρ de Spearman note ~ attribut protégé) : max |ρ| = "
      f"{max(abs(v) for v in audit['proxies'].values()):.3f}")
print("   " + ("Aucune alerte." if audit["conforme"] else " | ".join(audit["alertes"])))

# ----------------------------------------------------------------- 6. sensibilité
print("\n7) Analyse de sensibilité (poids de chaque critère de −50 % à +50 %)")
sens = sensitivity(dec, comite.poids, top_k=k)
print(sens.pivot(index="critere", columns="variation", values="recouvrement_topK").round(2)
      .rename(index=LAB).to_string())

# ----------------------------------------------------------------- figures
print("\n8) Figures")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].bar([LAB[c] for c in C.CRITERIA], [ahp["poids"][c] * 100 for c in C.CRITERIA],
            color=["#2e86de", "#e67e22", "#27ae60", "#8e44ad"])
for i, c in enumerate(C.CRITERIA):
    axes[0].text(i, ahp["poids"][c] * 100 + 1, f"{ahp['poids'][c]:.1%}", ha="center")
axes[0].set_title(f"Poids AHP des critères (CR = {ahp['CR']:.3f} < 0.10)")
axes[0].set_ylabel("Poids (%)")
appris = pd.DataFrame({comite.evaluateurs[c].nom: comite.evaluateurs[c].poids_appris()
                       for c in C.CRITERIA if hasattr(comite.evaluateurs[c], "poids_appris")}).T.fillna(0)
appris.rename(columns=C.LABELS).plot.barh(stacked=True, ax=axes[1], colormap="tab20")
axes[1].set_title("Poids internes APPRIS par chaque agent expert (|β| normalisés)")
axes[1].legend(fontsize=7, loc="lower right")
save(fig, "17_poids_ahp_et_agents")

fig, ax = plt.subplots(figsize=(7, 6))
for d, col in DECISION_COLORS.items():
    m = classement["decision"] == d
    ax.scatter(classement.loc[m, "WSM_rang"], classement.loc[m, "TOPSIS_rang"], s=18, color=col, label=d)
ax.plot([1, len(dec)], [1, len(dec)], "k:", lw=1)
ax.set_title(f"Classement WSM (baseline) vs TOPSIS — ρ = {rho:.3f}")
ax.set_xlabel("Rang WSM")
ax.set_ylabel("Rang TOPSIS")
ax.legend(fontsize=8)
save(fig, "18_wsm_vs_topsis")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ks = np.arange(5, len(dec) + 1, 5)
for nom, col in methodes.items():
    ordre = classement.sort_values(col, ascending=False)[C.TARGET].to_numpy()
    axes[0].plot(ks, [ordre[:n].mean() for n in ks], "-" if "seul" not in nom else ":", lw=2, label=nom)
axes[0].axhline(y.mean(), color="grey", lw=1, label=f"Hasard ({y.mean():.0%})")
axes[0].axvline(k, color="black", ls="--", lw=0.8)
axes[0].set_title("Précision@K : part d'embauchés réels dans le Top K")
axes[0].set_xlabel("K")
axes[0].legend(fontsize=7)
ordre_dec = [d for d in DECISION_COLORS if d in taux_decision.index]
axes[1].bar(ordre_dec, taux_decision.loc[ordre_dec, "mean"] * 100, color=[DECISION_COLORS[d] for d in ordre_dec])
for i, d in enumerate(ordre_dec):
    axes[1].text(i, taux_decision.loc[d, "mean"] * 100 + 1.5,
                 f"{taux_decision.loc[d, 'mean']:.0%} (n={int(taux_decision.loc[d, 'count'])})", ha="center")
axes[1].set_title("Taux d'embauche réel selon la décision du comité")
axes[1].set_ylim(0, 100)
save(fig, "19_comite_vs_realite")

fig, ax = plt.subplots(figsize=(9, 3.8))
sns.heatmap(sens.pivot(index="critere", columns="variation", values="recouvrement_topK").rename(index=LAB),
            annot=True, fmt=".0%", cmap="Greens", vmin=0.5, vmax=1, ax=ax)
ax.set_title(f"Sensibilité : part de la short-list ({k} places) conservée quand un poids varie")
ax.set_xlabel("Variation relative du poids")
ax.set_ylabel("")
save(fig, "20_sensibilite_poids")

top = classement.head(12)
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.heatmap(top[C.CRITERIA].rename(columns=LAB).set_index(
    pd.Index([f"#{r} id {i} — {d}" for i, r, d in zip(top.index, top["TOPSIS_rang"], top["decision"])])),
    annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax)
ax.set_title("Notes des 4 agents évaluateurs pour le Top 12 TOPSIS")
save(fig, "21_top12_notes_agents")

# ----------------------------------------------------------------- sauvegardes
classement.to_csv(C.DATA_PROCESSED / "classement_test.csv", index_label="id")
evaluation.to_csv(C.RESULTS_DIR / "04_evaluation_mcdm.csv", index=False)
sens.to_csv(C.RESULTS_DIR / "04_sensibilite.csv", index=False)
with open(C.RESULTS_DIR / "04_synthese.json", "w", encoding="utf-8") as f:
    json.dump({"ahp": ahp, "nb_places": k, "decisions": classement["decision"].value_counts().to_dict(),
               "precision_selection": round(float(y[sel].mean()), 4),
               "auc_topsis": round(float(evaluation.loc[0, "ROC-AUC"]), 4),
               "auc_wsm": round(float(evaluation.loc[1, "ROC-AUC"]), 4),
               "audit_ethique": audit}, f, indent=2, ensure_ascii=False, default=str)
joblib.dump(comite, C.MODELS_DIR / "comite.joblib")
print("\n   comité -> models/comite.joblib | classement -> data/processed/classement_test.csv")
print("\nÉtape 4 terminée.")

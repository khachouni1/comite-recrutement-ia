"""
ÉTAPE 4 — Comité multi-agents & aide à la décision multicritère (AHP, WSM, TOPSIS).

    python scripts/04_comite_mcdm.py

Protocole : agents appris sur Train ; seuils de décision calibrés sur Validation ; évaluation sur Test.
Sorties : models/comite.joblib, reports/resultats/04_*, reports/figures/17_* à 21_*
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
from sklearn.metrics import ndcg_score, roc_auc_score

from recrutement import config as C
from recrutement import data
from recrutement.agents import Coordinateur
from recrutement.mcdm import sensitivity
from recrutement.plots import DECISION_COLORS, PALETTE_LABELS, save

print("=" * 70, "\nÉTAPE 4 — COMITÉ MULTI-AGENTS & MCDM\n" + "=" * 70)
train, val, test = data.charger("train", "val", "test")
f_tr, f_va, f_te = data.charger("features_train", "features_val", "features_test")
reps = joblib.load(C.MODELS_DIR / "representations.joblib")
LAB = C.CRITERIA_LABELS

# ----------------------------------------------------------------- 1. AHP + apprentissage
comite = Coordinateur(reps).fit(f_tr, train["y"])
ahp = comite.ahp.summary()
print("\n1) Poids AHP des critères (vecteur propre principal de la matrice de Saaty) :")
for c, w in ahp["poids"].items():
    print(f"   {LAB[c]:24s} {w:6.1%}")
print(f"   λmax = {ahp['lambda_max']} | CI = {ahp['CI']} | CR = {ahp['CR']} -> cohérent : {ahp['coherent']}")

print("\n2) Agents : poids internes appris (Train) et pouvoir discriminant seul (Validation)")
for c, a in comite.evaluateurs.items():
    auc = roc_auc_score(val["y"] >= 1, a.noter(f_va))
    appris = getattr(a, "poids_appris", None) or f"{a.config['modele']} ({a.config['jeu_variables']})"
    print(f"   {a.nom:24s} AUC Val = {auc:.3f} | {appris}")
print(f"   Seuils de véto (5e percentile des candidats « adaptés » du Train) : "
      f"{ {LAB[c]: round(s, 1) for c, s in comite.seuils_veto.items()} }")

# ----------------------------------------------------------------- 2. calibrage sur Validation
print("\n3) Calibrage des seuils de décision sur la Validation")
cal = comite.calibrer(f_va, val["y"])
print(f"   Recommandé si score ≥ {cal['seuil_recommande']:.1f} : {cal['part_recommandes']:.0%} des candidats, "
      f"dont {cal['precision_recommandes']:.0%} réellement adéquats (objectif ≥ {C.PRECISION_RECO:.0%})")
print(f"   Non retenu si score < {cal['seuil_non_retenu']:.1f} : {cal['part_non_retenus']:.0%} des candidats, "
      f"dont {cal['precision_non_retenus']:.0%} réellement non adaptés (objectif ≥ {C.PRECISION_REJET:.0%})")

# ----------------------------------------------------------------- 3. évaluation sur Test
print("\n4) Évaluation sur le Test (offres jamais vues)")
dec = comite.decider(f_te).join(test[["offre_id", "cv_id", "label", "y"]])
y = dec["y"]
methodes = {"TOPSIS (comité)": "score_global", "WSM (baseline)": "score_wsm"} | \
           {f"{LAB[c]} seul": c for c in C.CRITERIA}


def ndcg_par_offre(col: str, k: int = 5) -> float:
    vals = [ndcg_score([g["y"].to_numpy()], [g[col].to_numpy()], k=k) for _, g in dec.groupby("offre_id") if len(g) > 1]
    return float(np.mean(vals))


def precision_top3(col: str) -> float:
    return float(np.mean([(g.nlargest(3, col)["y"] == 2).mean() for _, g in dec.groupby("offre_id") if len(g) >= 3]))


lignes = [{"Méthode": nom, "AUC adéquat vs non": roc_auc_score(y >= 1, dec[col]),
           "AUC adapté vs reste": roc_auc_score(y == 2, dec[col]), "Spearman": spearmanr(dec[col], y)[0],
           "NDCG@5 par offre": ndcg_par_offre(col), "Précision top-3 par offre": precision_top3(col)}
          for nom, col in methodes.items()]
evaluation = pd.DataFrame(lignes)
print(evaluation.round(3).to_string(index=False))
print(f"   (hasard : précision top-3 ≈ {(y == 2).mean():.2f})")

tab = pd.crosstab(dec["decision"], dec["label"].map(C.LABELS_FR), normalize="index").reindex(
    [d for d in DECISION_COLORS if d in set(dec["decision"])])[[C.LABELS_FR[l] for l in C.LABELS]]
tab.insert(0, "effectif", dec["decision"].value_counts())
print("\n   Décisions du comité vs étiquette réelle (répartition par décision) :")
print(tab.round(3).to_string())

# ----------------------------------------------------------------- 4. éthique
print("\n5) Agent éthique : test contrefactuel sur 30 paires du Test (ajout de genre / âge / situation familiale)")
echantillon = test.sample(30, random_state=C.RANDOM_STATE)
noter = lambda c, o: comite.notes(comite.preparer_paire(c, o)[0]).iloc[0].to_dict()
ecarts = [comite.ethique.test_contrefactuel(noter, r.cv, r.offre)["ecart_max"] for r in echantillon.itertuples()]
print(f"   Écart maximal des notes : {max(ecarts):.3f} pt (tolérance {comite.ethique.TOLERANCE} pt) | "
      f"tests réussis : {sum(e <= comite.ethique.TOLERANCE for e in ecarts)}/{len(ecarts)}")

# ----------------------------------------------------------------- 5. sensibilité
print("\n6) Sensibilité du classement aux poids AHP (−50 % à +50 %), short-list = 25 % des paires")
sens = sensitivity(dec[C.CRITERIA], comite.poids, top_k=len(dec) // 4)
pivot = sens.pivot(index="critere", columns="variation", values="recouvrement_topK").rename(index=LAB)
print(pivot.round(2).to_string())

# ----------------------------------------------------------------- figures
print("\n7) Figures")
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].bar([LAB[c] for c in C.CRITERIA], [ahp["poids"][c] * 100 for c in C.CRITERIA], color="#2e86de")
for i, c in enumerate(C.CRITERIA):
    axes[0].text(i, ahp["poids"][c] * 100 + 0.8, f"{ahp['poids'][c]:.1%}", ha="center")
axes[0].set_title(f"Poids AHP des critères (CR = {ahp['CR']:.3f} < 0.10)")
axes[0].tick_params(axis="x", rotation=15)
auc_agents = [roc_auc_score(val["y"] >= 1, comite.evaluateurs[c].noter(f_va)) for c in C.CRITERIA]
axes[1].bar([LAB[c] for c in C.CRITERIA], auc_agents, color="#16a085")
axes[1].axhline(0.5, color="grey", ls=":")
axes[1].set_ylim(0.4, 1)
axes[1].set_title("Pouvoir discriminant de chaque agent seul (AUC, Validation)")
axes[1].tick_params(axis="x", rotation=15)
save(fig, "17_poids_ahp_et_agents")

fig, ax = plt.subplots(figsize=(9, 4.5))
for k, lab in enumerate(C.LABELS):
    ax.hist(dec.loc[y == k, "score_global"], bins=30, alpha=0.5, density=True, color=PALETTE_LABELS[lab],
            label=C.LABELS_FR[lab])
ax.axvline(comite.seuil_reco, color="green", ls="--", label=f"seuil Recommandé ({comite.seuil_reco:.0f})")
ax.axvline(comite.seuil_rejet, color="red", ls="--", label=f"seuil Non retenu ({comite.seuil_rejet:.0f})")
ax.set_title("Score global TOPSIS du comité selon l'étiquette réelle (Test)")
ax.set_xlabel("Score global (0-100)")
ax.legend(fontsize=8)
save(fig, "18_score_global_par_etiquette")

fig, ax = plt.subplots(figsize=(9, 4.5))
tab[[C.LABELS_FR[l] for l in C.LABELS]].mul(100).plot.barh(stacked=True, ax=ax,
                                                           color=[PALETTE_LABELS[l] for l in C.LABELS])
ax.set_title("Composition réelle de chaque décision du comité (Test)")
ax.set_xlabel("% des paires")
ax.set_yticks(range(len(tab)), [f"{d} (n={n})" for d, n in zip(tab.index, tab["effectif"])])
save(fig, "19_decisions_vs_realite")

fig, ax = plt.subplots(figsize=(10, 4.5))
evaluation.set_index("Méthode")[["AUC adéquat vs non", "AUC adapté vs reste", "NDCG@5 par offre"]].iloc[::-1] \
    .plot.barh(ax=ax)
ax.set_xlim(0.4, 1)
ax.set_title("Comité (TOPSIS) vs baseline WSM vs agents seuls (Test)")
ax.legend(fontsize=8, loc="lower right")
save(fig, "20_comparaison_methodes")

fig, ax = plt.subplots(figsize=(9, 3.8))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Greens", vmin=0.5, vmax=1, ax=ax)
ax.set_title("Sensibilité : part de la short-list conservée quand un poids varie")
ax.set_xlabel("Variation relative du poids")
ax.set_ylabel("")
save(fig, "21_sensibilite_poids")

# ----------------------------------------------------------------- sauvegardes
evaluation.to_csv(C.RESULTS_DIR / "04_evaluation_comite.csv", index=False)
tab.to_csv(C.RESULTS_DIR / "04_decisions_vs_realite.csv")
sens.to_csv(C.RESULTS_DIR / "04_sensibilite.csv", index=False)
with open(C.RESULTS_DIR / "04_synthese.json", "w", encoding="utf-8") as f:
    json.dump({"ahp": ahp, "calibrage": cal, "seuils_veto": comite.seuils_veto,
               "contrefactuel_ecart_max": max(ecarts),
               "evaluation": evaluation.set_index("Méthode").round(4).to_dict(orient="index")},
              f, indent=2, ensure_ascii=False)
joblib.dump(comite, C.MODELS_DIR / "comite.joblib")
data.creer_exemples()
print("\n   comité -> models/comite.joblib | exemples du site -> data/exemples_demo.csv")
print("\nÉtape 4 terminée.")

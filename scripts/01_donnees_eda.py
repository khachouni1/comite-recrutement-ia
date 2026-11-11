"""
ÉTAPE 1 — Données (CV + offres), découpage Train / Val / Test et analyse exploratoire.

    python scripts/01_donnees_eda.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from recrutement import config as C
from recrutement import data, eda, features as F

print("=" * 70, "\nÉTAPE 1 — DONNÉES & ANALYSE EXPLORATOIRE\n" + "=" * 70)

brut_tr, brut_te = data.charger_brut()
print(f"\n1) Données brutes : {len(brut_tr)} paires (train officiel) + {len(brut_te)} (test officiel), "
      f"{int(brut_tr.isna().sum().sum() + brut_te.isna().sum().sum())} valeur(s) manquante(s), "
      f"{int(brut_tr.duplicated().sum())} doublon(s) retiré(s)")

print("\n2) Nettoyage et ANONYMISATION des textes (e-mails, téléphones, âge, genre, situation familiale...)")
train_off, test = data.preparer(brut_tr), data.preparer(brut_te)

print("\n3) Découpage PAR OFFRE (aucune offre partagée entre les jeux)")
train, val, test = data.decouper(train_off, test)
data.sauvegarder(train=train, val=val, test=test)
struct = eda.structure({"Train": train, "Validation": val, "Test": test})
print(struct.to_string())
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    commun = set(locals()[a]["offre_id"]) & set(locals()[b]["offre_id"])
    assert not commun, f"offres partagées entre {a} et {b}"
print("   Vérifié : aucune offre commune entre Train, Validation et Test.")

print("\n4) Extraction des critères par règles et lexiques (Train)")
analyses_offres = [F.analyser_offre(t) for t in train.drop_duplicates("offre_id")["offre"]]
feats = F.table_regles(train)
qualite = F.resume_extraction(feats)
print(qualite.to_string())

print("\n5) Lien entre variables extraites et étiquette (Kruskal-Wallis, Train)")
variables = ["couverture_hard", "n_hard_communs", "couverture_soft", "annees_cv", "ratio_annees",
             "diplome_cv", "ecart_diplome", "longueur_cv"]
tests = eda.tests_kruskal(feats, train["y"], variables)
print(tests.round(3).to_string(index=False))
tests.to_csv(C.RESULTS_DIR / "01_tests_kruskal.csv", index=False)

print("\n6) Figures")
eda.plot_etiquettes_longueurs(train)
eda.plot_competences_demandees(train, analyses_offres)
eda.plot_variables_par_etiquette(feats, train["y"], variables)

struct.to_csv(C.RESULTS_DIR / "01_structure.csv")
with open(C.RESULTS_DIR / "01_qualite_extraction.json", "w", encoding="utf-8") as f:
    json.dump(qualite.to_dict(), f, indent=2, ensure_ascii=False)
print("\nÉtape 1 terminée.")

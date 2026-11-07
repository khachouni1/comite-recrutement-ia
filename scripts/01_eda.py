"""
ÉTAPE 1 — Données, découpage Train / Val / Test et analyse exploratoire.

    python scripts/01_eda.py

Sorties : data/processed/{train,val,test}.csv, reports/figures/01_* à 05_*,
          reports/resultats/01_qualite.json, 01_statistiques.csv, 01_tests.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recrutement import config as C
from recrutement import data, eda

print("=" * 70, "\nÉTAPE 1 — DONNÉES & ANALYSE EXPLORATOIRE\n" + "=" * 70)

df = data.load_raw()
quality = eda.data_quality(df)
print(f"\n1) Contrôle qualité : {quality['n_lignes']} lignes, {quality['n_colonnes']} colonnes, "
      f"{quality['valeurs_manquantes']} valeur(s) manquante(s), {quality['doublons']} doublon(s), "
      f"taux d'embauche = {quality['taux_embauche']:.1%}")

train, val, test = data.split_three_way(df)
data.save_splits(train, val, test)
quality["decoupage"] = {"train": len(train), "val": len(val), "test": len(test)}
print(f"\n2) Découpage AVANT tout prétraitement : Train {len(train)} | Val {len(val)} | Test {len(test)}")
for name, part in (("Train", train), ("Val", val), ("Test", test)):
    print(f"   {name:5s} : {part[C.TARGET].mean():.1%} d'embauchés (stratification)")

print("\n3) Statistiques descriptives (Train) :")
stats = eda.descriptive_stats(train)
print(stats[["mean", "std", "min", "max", "moyenne_rejetés", "moyenne_embauchés"]].to_string())
stats.to_csv(C.RESULTS_DIR / "01_statistiques.csv")

print("\n4) Tests de liaison avec la décision (Train) :")
tests = eda.statistical_tests(train)
print(tests.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
tests.to_csv(C.RESULTS_DIR / "01_tests.csv", index=False)

print("\n5) Figures :")
eda.run_all_figures(train)

with open(C.RESULTS_DIR / "01_qualite.json", "w", encoding="utf-8") as f:
    json.dump(quality, f, indent=2, ensure_ascii=False)
print("\nÉtape 1 terminée.")

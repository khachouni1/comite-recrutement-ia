# Données

## Source

- **Jeu de données :** *Predicting Hiring Decisions in Recruitment Data* (auteur : Rabie El Kharoua), données ouvertes Kaggle
- **Lien :** https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data
- **Fichier :** `raw/recruitment_data.csv` — 1 500 candidats × 11 colonnes, aucune valeur manquante, aucun doublon

> Les variables sont presque indépendantes entre elles : toutes les valeurs propres de l'ACP sont proches de 1
> (étape 2). Ce comportement est typique d'un jeu de données **synthétique**. Il faut le confirmer sur la page Kaggle
> et le mentionner dans le rapport.

## Dictionnaire des variables

| Variable | Type | Valeurs | Rôle dans le projet |
|---|---|---|---|
| `Age` | quantitative | 20 – 50 | **Attribut protégé** : utilisé uniquement pour l'audit d'équité |
| `Gender` | binaire | 0 = femme, 1 = homme | **Attribut protégé** : utilisé uniquement pour l'audit d'équité |
| `EducationLevel` | ordinale | 1 – 4 (plus élevé = diplôme plus élevé) | Agent Lead Tech, modèle ML |
| `ExperienceYears` | quantitative | 0 – 15 ans | Agent Expérience, modèle ML |
| `PreviousCompanies` | quantitative | 1 – 5 | Agent Expérience, modèle ML |
| `DistanceFromCompany` | quantitative | 1 – 51 km | Modèle ML |
| `InterviewScore` | quantitative | 0 – 100 | Agent RH, modèle ML |
| `SkillScore` | quantitative | 0 – 100 | Agent Lead Tech, modèle ML |
| `PersonalityScore` | quantitative | 0 – 100 | Agent RH, modèle ML |
| `RecruitmentStrategy` | catégorielle | 1, 2, 3 (libellés : voir Kaggle) | Modèle ML (codage one-hot) |
| `HiringDecision` | **cible** binaire | 0 = rejeté (69 %), 1 = embauché (31 %) | Vérité terrain |

## Traitements effectués

1. **Contrôle qualité** : valeurs manquantes, doublons et plages de valeurs (`reports/resultats/01_qualite.json`).
2. **Découpage stratifié, fait avant tout prétraitement** : Train 70 % (1 049), Validation 15 % (226), Test 15 % (225).
   Les fichiers produits sont `processed/train.csv`, `val.csv` et `test.csv`, et l'identifiant d'origine est conservé.
3. **Prétraitement appris sur Train uniquement, à l'intérieur des pipelines** :
   - standardisation des variables quantitatives et ordinales ;
   - codage one-hot de `RecruitmentStrategy` ;
   - SMOTE pour rééquilibrer les classes, appliqué seulement pendant l'entraînement.
4. **Exclusion de `Gender` et `Age`** de toutes les notes et de tous les modèles. Ces variables servent uniquement à l'audit.

Le dossier `processed/` est régénéré par `python scripts/01_eda.py`.

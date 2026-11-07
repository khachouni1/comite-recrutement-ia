# Comité multi-agents & méthodes MCDM pour l'aide à la sélection de candidats

**École Centrale de Casablanca — Ingénierie Machine Learning & Statistique — Sujet n° 11**

> *Système multi-agents fondé sur les méthodes MCDM pour l'aide à la sélection de candidats en entreprise.*

## L'idée en une phrase

Plusieurs **agents spécialisés** (technique, expérience, RH, prédictif) évaluent chaque candidat. Chacun a **appris
sur les données** et argumente son avis. Un **coordinateur** agrège ces avis avec des méthodes multicritères
(**AHP** pour les poids, **TOPSIS** pour le classement, comparé à la baseline **WSM**). Deux agents de **contrôle**
vérifient la qualité des dossiers et l'**équité** de la décision.

```
                 ┌──────────── AGENTS ÉVALUATEURS (notent /100 + confiance + arguments) ────────────┐
   candidat ───► │  Lead Tech   │  Expérience   │  RH & savoir-être   │  Prédictif ML (sans genre/âge) │
                 └──────────────────────────────────────┬──────────────────────────────────────────┘
                                                        ▼
   ┌───────────────── COORDINATEUR ─────────────────────────────────────────────────────┐
   │ percentiles → désaccords │ véto (Tech, RH) │ poids AHP (CR < 0.10) │ WSM vs TOPSIS   │──► Recommandé
   └────────────────────────────────────────────────────────────────────────────────────┘    À examiner
        ▲ signalements                                         │ sélection                    Écarté (véto)
   ┌────┴──────────────────┐                     ┌──────────────▼──────────────┐              Non retenu
   │ Agent Anomalies       │                     │ Agent Éthique : contrefactuel,│
   │ (gaussienne multiv.)  │                     │ règle des 4/5, proxies        │
   └───────────────────────┘                     └───────────────────────────────┘
```

## Démarche : 4 étapes, chacune reliée au cours

| Étape | Script | Contenu | Cours |
|---|---|---|---|
| 1 | `scripts/01_eda.py` | Contrôle qualité, découpage **Train/Val/Test avant prétraitement**, statistiques, tests de Mann-Whitney et du χ², figures | C1 slides 84-86 |
| 2 | `scripts/02_non_supervise.py` | **ACP** (Kaiser, cercle des corrélations), **K-Means & CAH de Ward** (coude, silhouette, Davies-Bouldin, Calinski-Harabasz), **détection d'anomalies gaussienne** | C2 slides 17-85 |
| 3 | `scripts/03_supervise.py` | Baselines (naïve, logistique) vs arbre, forêt, Gradient Boosting, XGBoost ; **SMOTE** dans le pipeline ; hyperparamètres choisis par **validation croisée**, modèle choisi sur **Val**, réentraînement sur Train+Val, **un seul passage sur Test** ; courbe d'apprentissage ; **SHAP, permutation, analyse des erreurs** | C1 slides 44-59, 84-88 |
| 4 | `scripts/04_comite_mcdm.py` | Agents, **AHP**, **WSM** (baseline) vs **TOPSIS**, confrontation aux décisions réelles, **audit éthique**, **sensibilité** des poids | Sujet 11 |

## Résultats clés (jeu de test, 225 candidats)

**Modèles supervisés** (le genre et l'âge sont exclus, ce qui ne coûte rien : AUC Val 0,933 contre 0,933 avec ces variables)

| Modèle | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Régression logistique (baseline) | 0,904 | 0,853 | 0,776 |
| **Gradient Boosting** (retenu sur Val) | **0,924** | **0,911** | **0,857** |

**Comité multi-agents**
- Sélection de 70 candidats : **60 % sont réellement embauchés**, contre 31 % pour une sélection au hasard. Parmi les « Recommandés », 65 % ont été embauchés, contre 18 % des « Non retenus ».
- Audit éthique conforme : Disparate Impact de 0,98 pour le genre et de 0,85 pour l'âge, et 225 tests contrefactuels sur 225 réussis.
- La short-list est robuste : elle est conservée à 89 % ou plus quand un poids varie de ± 50 %, sauf si le poids technique est divisé par deux.
- Classement face aux décisions réelles : AUC de 0,78 pour TOPSIS, 0,82 pour WSM et 0,92 pour le modèle ML seul (discuté ci-dessous).

**Ce que révèlent les données** (et que le rapport doit discuter honnêtement)
- La variable `RecruitmentStrategy` domine les décisions historiques (SHAP, permutation, χ²). Elle reflète la manière dont le candidat a été recruté, pas ses compétences.
- Les variables sont presque indépendantes : l'ACP ne compresse pas l'information et il n'y a pas de clusters naturels (silhouette de 0,11, accord K-Means/CAH avec un indice de Rand ajusté de 0,00).
- Le modèle ML **reproduit les décisions passées**, alors que le comité MCDM **applique la politique choisie par l'entreprise** (Technique > Expérience > RH). C'est pour cela que le comité a une AUC plus faible : il ne cherche pas à imiter l'historique, mais à rendre une décision justifiée, pondérée explicitement et auditable.

## Installation et exécution

```bash
pip install -r requirements.txt
python scripts/run_all.py          # les 4 étapes (environ 2 minutes)
streamlit run app/app.py           # démonstration interactive
python -m pytest                   # 16 tests unitaires
```

Chaque étape peut aussi être lancée seule (`python scripts/03_supervise.py`). Les sorties vont dans :
- `reports/figures/` : 21 figures numérotées dans l'ordre de la démarche ;
- `reports/resultats/` : tableaux CSV et synthèses JSON de chaque étape ;
- `models/` : le modèle final et le comité entraîné.

## Structure du projet

```text
Projet_ML_Sujet11/
├── data/
│   ├── README.md                 source, dictionnaire des variables, traitements
│   └── raw/recruitment_data.csv
├── src/recrutement/              le code, organisé en modules courts
│   ├── config.py                 TOUTES les constantes (chemins, variables, matrice AHP, seuils)
│   ├── data.py                   chargement + découpage Train/Val/Test
│   ├── preprocessing.py          pipeline : standardisation, one-hot, SMOTE
│   ├── eda.py                    analyse exploratoire
│   ├── unsupervised.py           ACP, clustering, anomalies gaussiennes
│   ├── supervised.py             catalogue de modèles, sélection, métriques
│   ├── explain.py                SHAP, permutation, analyse des erreurs
│   ├── mcdm/                     ahp.py · wsm.py · topsis.py · sensitivity.py
│   └── agents/
│       ├── base.py               classe Agent + Avis (note, confiance, arguments)
│       ├── evaluateurs.py        Lead Tech, Expérience, RH, Prédictif ML
│       ├── controle.py           Anomalies, Éthique
│       └── coordinateur.py       délibération, véto, désaccords, AHP, WSM/TOPSIS, décision
├── scripts/                      01 → 04 + run_all.py : la démarche pas à pas
├── app/app.py                    application Streamlit
├── tests/                        tests unitaires (MCDM, données, agents)
└── reports/figures, reports/resultats
```

## Choix méthodologiques à retenir pour l'oral

1. **Pas de fuite de données** : le découpage se fait sur les données brutes, et chaque transformation (standardisation, SMOTE) est apprise dans un pipeline, uniquement sur la partie entraînement.
2. **Rôle de chaque jeu** : Train sert à apprendre, Val à choisir (modèle, seuils, vérification de la méthode MCDM), Test à mesurer une seule fois.
3. **Des agents qui apprennent** : les poids internes de chaque expert sont des coefficients de régression logistique appris sur les données, pas des pondérations inventées.
4. **Équité par construction, puis vérifiée** : le genre et l'âge ne sont jamais utilisés pour noter. L'agent Éthique le vérifie par un test contrefactuel et contrôle la sélection avec la règle des 4/5.
5. **Humain dans la boucle** : quand les agents sont en désaccord ou qu'un dossier est atypique, la décision est « À examiner » au lieu d'une décision automatique.

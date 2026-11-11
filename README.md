# Comité de recrutement multi-agents : évaluer un CV face à une offre d'emploi

**École Centrale de Casablanca — Ingénierie Machine Learning & Statistique — Sujet n° 11**

> *Système multi-agents fondé sur les méthodes MCDM pour l'aide à la sélection de candidats en entreprise.*

## Ce que fait le projet

Un **site web** reçoit une **offre d'emploi** et un **CV** (texte, PDF ou Word). Un comité d'agents évalue le candidat
critère par critère, puis rend une décision argumentée :

```
 CV + offre ──► nettoyage + ANONYMISATION ──► extraction des critères + similarités textuelles
                                                         │
     ┌──────────────────── 6 AGENTS ÉVALUATEURS (note /100 + confiance + arguments) ─────────────────────┐
     │ 🛠️ Technique │ 📈 Expérience │ 🎓 Formation │ 🤝 Soft skills │ 🎯 Adéquation │ 🤖 Avis prédictif (ML) │
     └───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                 ▼
     COORDINATEUR : véto ▸ poids AHP (CR < 0.10) ▸ score TOPSIS (baseline : WSM) ▸ seuils calibrés
                    ▸ désaccord avec l'agent prédictif → « À examiner » ▸ contrôles (anomalies, éthique)
                                                 ▼
               ✅ Recommandé   🟠 À examiner   ❌ Non retenu   ⛔ Écarté (véto)
```

Le site permet aussi de **classer plusieurs CV pour une même offre** avec TOPSIS, et de saisir ses propres priorités.

## Démarche : 4 étapes, chacune reliée au cours

| Étape | Script | Contenu | Cours |
|---|---|---|---|
| 1 | `01_donnees_eda.py` | 8 000 paires CV/offre étiquetées ; anonymisation ; découpage **par offre** (pas de fuite) ; extraction de critères par règles ; tests de Kruskal-Wallis | C1 : préparation des données, Three-Way Holdout |
| 2 | `02_representation_non_supervise.py` | TF-IDF (baseline) → **LSA = ACP du texte** → embeddings Hugging Face ; comparaison des représentations ; **K-Means et CAH** des CV en familles de métiers (silhouette, Davies-Bouldin, Calinski-Harabasz) | C2 : ACP, clustering ; consigne : Hugging Face |
| 3 | `03_supervise.py` | Classification à 3 classes : baseline naïve, **régression softmax** (baseline), arbre, forêt, Gradient Boosting, XGBoost ; **SMOTE** ; validation croisée **groupée par offre** ; choix sur Val, réentraînement sur Train + Val, un seul passage sur Test ; courbe d'apprentissage ; **SHAP, permutation, analyse des erreurs** | C1 : softmax, SMOTE, validation croisée, interprétabilité |
| 4 | `04_comite_mcdm.py` | Agents appris sur les données, **AHP**, **TOPSIS** contre **WSM**, seuils calibrés sur Val, évaluation globale et **par offre** (NDCG), test éthique contrefactuel, sensibilité des poids, **anomalies gaussiennes** | Sujet 11 ; C2 : anomalies |

## Résultats clés (Test : 1 759 paires, 71 offres jamais vues)

**Représentation du texte (Validation)** : la similarité par embeddings Hugging Face (AUC 0,66) mesure mieux
l'adéquation que TF-IDF (0,66) et LSA (0,64), mais l'écart est faible. Le clustering des CV fait ressortir
**12 familles de métiers interprétables** (comptabilité, tests et qualité, Java/web, électricité, SQL/ETL, réseau,
big data…). L'accord entre K-Means et la CAH est de 0,65 (ARI).

**Modèles supervisés** (3 classes)

| Modèle | AUC macro | AUC adéquat vs non | F1 macro |
|---|---|---|---|
| Régression softmax (baseline), 17 variables explicables | 0,595 | 0,633 | 0,395 |
| **XGBoost, variables explicables + interactions d'embeddings** | **0,688** | **0,729** | **0,428** |

**Comité multi-agents (6 critères, AHP CR = 0,006)**

| Méthode | AUC adéquat vs non | NDCG@5 par offre |
|---|---|---|
| **TOPSIS (comité)** | **0,692** | **0,668** |
| WSM (baseline MCDM) | 0,680 | 0,643 |
| Meilleur expert explicable seul (adéquation) | 0,619 | 0,598 |
| Avis prédictif ML seul | 0,727 | 0,644 |

- **TOPSIS bat la baseline WSM**, et le comité fait mieux que chacun des 5 experts explicables pris seul.
- **Pour classer les candidats d'une même offre** (ce que fait un recruteur), le comité est le meilleur (NDCG@5).
- **Les décisions sont graduées** : parmi les « Recommandés », 71 % sont réellement adéquats ; parmi les « Écartés
  (véto) », 72 % ne sont pas adaptés.
- **Éthique** : 30 tests contrefactuels sur 30 réussis. Ajouter un genre, un âge ou une situation familiale au CV ne
  change aucune note.

**Lecture honnête** : prédire l'adéquation d'un CV à une offre reste difficile sur ce jeu de données. Les classes
« potentiel » et « adapté » sont très proches, ce qui donne une F1 macro d'environ 0,43. Le système est un **outil
d'aide** : il trie et argumente, et la décision finale revient à un humain (zone « À examiner »).

## Installation et exécution

```bash
pip install -r requirements.txt
python scripts/run_all.py          # télécharge les données puis exécute les 4 étapes (~30 min)
streamlit run app/app.py           # le site web
python -m pytest                   # tests unitaires
```

Au premier lancement, le modèle d'embeddings `all-MiniLM-L6-v2` (88 Mo) est téléchargé depuis Hugging Face.
Pour travailler hors ligne, placez-le dans `models/all-MiniLM-L6-v2/`.

Le comité déjà entraîné est fourni (`models/comite.joblib`) : `streamlit run app/app.py` fonctionne directement,
sans relancer le pipeline. **Mise en ligne** : voir [DEPLOIEMENT.md](DEPLOIEMENT.md).

## Structure du projet

```text
Projet_ML_Sujet11/
├── app/app.py                    le site web (Streamlit)
├── data/README.md                source, licence, traitements ; raw/cv_offres/{train,test}.csv
├── scripts/                      00 (téléchargement) → 04 + run_all.py : la démarche pas à pas
├── src/recrutement/
│   ├── config.py                 TOUTES les constantes (critères, matrice AHP, seuils)
│   ├── data.py                   chargement, anonymisation, découpage par offre
│   ├── texte.py · lexiques.py    lecture PDF / Word, nettoyage, anonymisation, extraction par règles
│   ├── features.py               variables de chaque paire CV / offre
│   ├── representation.py         TF-IDF, LSA, embeddings, similarité cosinus
│   ├── eda.py · unsupervised.py  analyse exploratoire, clustering, anomalies gaussiennes
│   ├── supervised.py · explain.py modèles, sélection, SHAP, permutation, erreurs
│   ├── mcdm/                     ahp.py · wsm.py · topsis.py · sensitivity.py
│   └── agents/                   base · evaluateurs · controle · coordinateur
├── tests/                        texte, MCDM, données, comité
└── reports/figures · reports/resultats
```

## Choix méthodologiques à retenir pour l'oral

1. **Pas de fuite de données** : un même CV apparaît dans plusieurs paires, donc le découpage et la validation croisée sont faits **par offre**. Le Test ne contient que des postes jamais vus, comme sur le site.
2. **Rôle de chaque jeu** : Train sert à apprendre, Validation à choisir (modèle, seuils de décision), Test à mesurer une seule fois.
3. **Des agents explicables qui apprennent** : chaque agent extrait des éléments lisibles (compétences trouvées et manquantes, années exigées et détectées…) et apprend sur les données comment les noter.
4. **Équité par construction** : le CV est anonymisé avant l'évaluation, et l'agent Éthique le vérifie par un test contrefactuel (ajout d'un genre, d'un âge ou d'une situation familiale → les notes ne bougent pas).
5. **Humain dans la boucle** : en cas de zone grise, de désaccord avec l'agent prédictif ou de dossier atypique, la décision est « À examiner ».

## Limites

- Les données et le comité sont **en anglais**.
- Le jeu de données ne déclare pas de licence et ne contient pas d'attributs démographiques : il n'y a donc pas d'audit d'équité par groupe, seulement des tests contrefactuels.
- Les noms propres ne sont pas anonymisés, car il faudrait un modèle de reconnaissance d'entités nommées.
- Les lexiques de compétences couvrent les métiers du jeu de données (informatique, data, finance, santé, ingénierie…).
- **Choix du modèle** : la Validation ne compte que 42 offres, ce qui rend son estimation bruitée. Le modèle est donc
  choisi sur l'AUC en validation croisée groupée (238 offres), et la Validation sert de contrôle. Ce critère a été fixé
  après une première exécution, et un essai exploratoire des interactions d'embeddings avait été mesuré sur le Test :
  c'est une petite entorse au protocole, signalée par transparence.
- **Surapprentissage** : XGBoost atteint une AUC de 1,00 sur ses données d'entraînement contre 0,68 en validation
  croisée. La généralisation reste stable, mais une régularisation plus forte serait une piste d'amélioration.

La version 1 du projet, sur des données tabulaires, reste disponible via le tag Git `v1-tabulaire`.

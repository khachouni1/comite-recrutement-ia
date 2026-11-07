"""
Configuration centrale du projet.

Toutes les constantes (chemins, variables, paramètres) sont définies ICI et nulle part ailleurs.
Pour modifier une hypothèse du projet (ex. la matrice AHP), il suffit de modifier ce fichier.
"""
from pathlib import Path

# --------------------------------------------------------------------------- chemins
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "recruitment_data.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "resultats"

for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- reproductibilité
RANDOM_STATE = 42

# --------------------------------------------------------------------------- découpage (Cours 1, slides 84-86)
TEST_SIZE = 0.15   # 15 % test
VAL_SIZE = 0.15    # 15 % validation  -> 70 % entraînement

# --------------------------------------------------------------------------- variables
TARGET = "HiringDecision"

NUMERIC_FEATURES = [
    "Age", "ExperienceYears", "PreviousCompanies", "DistanceFromCompany",
    "InterviewScore", "SkillScore", "PersonalityScore",
]
ORDINAL_FEATURES = ["EducationLevel"]           # 1 < 2 < 3 < 4 : gardée numérique
CATEGORICAL_FEATURES = ["RecruitmentStrategy"]  # 1 / 2 / 3 : codage one-hot
BINARY_FEATURES = ["Gender"]
ALL_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Attributs protégés : interdits pour NOTER un candidat, utilisés uniquement pour l'AUDIT d'équité
PROTECTED = ["Gender", "Age"]
MODEL_FEATURES = [f for f in ALL_FEATURES if f not in PROTECTED]

LABELS = {
    "Age": "Âge", "Gender": "Genre", "EducationLevel": "Niveau d'études",
    "ExperienceYears": "Années d'expérience", "PreviousCompanies": "Entreprises précédentes",
    "DistanceFromCompany": "Distance (km)", "InterviewScore": "Entretien",
    "SkillScore": "Test technique", "PersonalityScore": "Personnalité",
    "RecruitmentStrategy": "Stratégie de recrutement", "HiringDecision": "Embauché",
}

# --------------------------------------------------------------------------- comité multi-agents & MCDM
# Ordre des critères = ordre des lignes/colonnes de la matrice AHP (source unique de vérité).
CRITERIA = ["score_technique", "score_experience", "score_rh", "score_predictif"]
CRITERIA_LABELS = {
    "score_technique": "Technique",
    "score_experience": "Expérience",
    "score_rh": "RH & savoir-être",
    "score_predictif": "Prédictif (ML)",
}
# Échelle de Saaty : AHP_MATRIX[i][j] = importance du critère i par rapport au critère j
AHP_MATRIX = [
    # Tech  Exp   RH    Préd
    [1,    2,    3,    3],    # Technique
    [1/2,  1,    2,    2],    # Expérience
    [1/3,  1/2,  1,    1],    # RH
    [1/3,  1/2,  1,    1],    # Prédictif
]

VETO_CRITERIA = ["score_technique", "score_rh"]  # agents cœur de métier pouvant opposer un véto
VETO_PERCENTILE = 0.05       # véto si note < 5e percentile des notes du Train
DISAGREEMENT_GAP = 0.80      # désaccord si (percentile max - percentile min) >= 0.80
ANOMALY_CONTAMINATION = 0.02 # ε = quantile 2 % des densités du Train
DI_THRESHOLD = 0.80          # règle des 4/5 (Disparate Impact)

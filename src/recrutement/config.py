"""
Configuration centrale du projet (version 2 : CV et offres d'emploi en texte).

Toutes les constantes sont définies ICI : pour changer une hypothèse (matrice AHP, seuils...),
on modifie ce fichier et on relance le pipeline.
"""
from pathlib import Path

# --------------------------------------------------------------------------- chemins
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "cv_offres"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "resultats"
for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VAL_SIZE = 0.15  # part des OFFRES du jeu d'entraînement officiel réservée à la validation

# --------------------------------------------------------------------------- cible
LABELS = ["No Fit", "Potential Fit", "Good Fit"]          # ordre croissant d'adéquation
LABEL_TO_INT = {lab: i for i, lab in enumerate(LABELS)}   # 0, 1, 2
LABELS_FR = {"No Fit": "Non adapté", "Potential Fit": "Potentiel", "Good Fit": "Adapté"}

# --------------------------------------------------------------------------- représentation du texte
EMBEDDING_HUB_ID = "sentence-transformers/all-MiniLM-L6-v2"   # Hugging Face, 384 dimensions
_LOCAL_MODEL = MODELS_DIR / "all-MiniLM-L6-v2"                 # copie locale (fonctionne hors ligne)
EMBEDDING_MODEL = str(_LOCAL_MODEL) if (_LOCAL_MODEL / "model.safetensors").exists() else EMBEDDING_HUB_ID
LSA_COMPONENTS = 100                                         # réduction de dimension du TF-IDF (SVD)

# --------------------------------------------------------------------------- variables par agent
# Chaque agent n'observe que SES variables (extraites du texte anonymisé).
AGENT_FEATURES = {
    "score_technique": ["couverture_hard", "n_hard_communs", "n_hard_cv"],
    "score_experience": ["annees_cv", "ratio_annees", "exigence_annees_connue"],
    "score_formation": ["diplome_cv", "ecart_diplome", "exigence_diplome_connue"],
    "score_soft_skills": ["couverture_soft", "n_soft_communs", "n_soft_cv"],
    "score_adequation": ["sim_embedding", "sim_lsa", "sim_tfidf"],
}
FEATURES = sorted({f for fs in AGENT_FEATURES.values() for f in fs}) + ["longueur_cv", "longueur_offre"]

# --------------------------------------------------------------------------- MCDM
# 5 critères notés par des agents experts explicables + 1 critère noté par l'agent prédictif (ML).
# L'ajout du 6e critère a été décidé sur la VALIDATION : le comité à 6 critères (AUC 0.76) y dépasse
# chacun de ses agents pris seul, y compris l'agent ML (0.74), et le comité à 5 critères (0.73).
CRITERIA = list(AGENT_FEATURES) + ["score_predictif"]
CRITERIA_LABELS = {
    "score_technique": "Compétences techniques",
    "score_experience": "Expérience",
    "score_formation": "Formation",
    "score_soft_skills": "Soft skills",
    "score_adequation": "Adéquation au poste",
    "score_predictif": "Avis prédictif (ML)",
}
# Matrice de Saaty : AHP_MATRIX[i][j] = importance du critère i par rapport au critère j
# Jugement du comité : Technique ≈ Adéquation ≈ Avis prédictif > Expérience > Soft skills > Formation
AHP_MATRIX = [
    #  Tech  Exp   Form  Soft  Adéq  Préd
    [1,    2,    4,    3,    1,    1],     # Technique
    [1/2,  1,    3,    2,    1/2,  1/2],   # Expérience
    [1/4,  1/3,  1,    1/2,  1/4,  1/4],   # Formation
    [1/3,  1/2,  2,    1,    1/3,  1/3],   # Soft skills
    [1,    2,    4,    3,    1,    1],     # Adéquation
    [1,    2,    4,    3,    1,    1],     # Avis prédictif
]

VETO_CRITERIA = ["score_technique", "score_adequation"]  # critères éliminatoires
VETO_QUANTILE = 0.05        # véto si note < 5e percentile des candidats « Good Fit » du Train
DISAGREEMENT_PROBA = 0.60   # désaccord si l'agent prédictif contredit le comité avec p ≥ 0.60
PRECISION_RECO = 0.70       # zone « Recommandé » : ≥ 70 % de candidats adéquats (calibré sur Validation)
PRECISION_REJET = 0.65      # zone « Non retenu » : ≥ 65 % de candidats non adaptés (calibré sur Validation)
ANOMALY_CONTAMINATION = 0.02

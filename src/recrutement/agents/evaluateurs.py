"""
Agents ÉVALUATEURS : ils notent les candidats, chacun sur un critère MCDM.

┌─────────────────┬──────────────────────────────────────┬──────────────────────────────────────────┐
│ Agent           │ Variables observées                  │ Méthode (cours)                          │
├─────────────────┼──────────────────────────────────────┼──────────────────────────────────────────┤
│ Lead Tech       │ Test technique, niveau d'études      │ Régression logistique + SMOTE (Cours 1)  │
│ Expérience      │ Années d'expérience, nb d'entreprises│ Régression logistique + SMOTE (Cours 1)  │
│ RH              │ Entretien, test de personnalité      │ Régression logistique + SMOTE (Cours 1)  │
│ Prédictif (ML)  │ Toutes sauf genre et âge             │ Meilleur modèle de l'étape 3 + SHAP      │
└─────────────────┴──────────────────────────────────────┴──────────────────────────────────────────┘

Agents experts : au lieu de poids inventés à la main, chaque expert APPREND sur le Train
comment ses variables influencent l'embauche :
        p = σ(β0 + Σ βj · zj)           note = 100 · p
La contribution βj · zj (en log-odds) de chaque variable fournit les arguments de l'avis
(interprétabilité intrinsèque d'un modèle linéaire, Cours 1 slide 52).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import config as C
from ..preprocessing import build_pipeline
from ..supervised import model_catalog
from .base import Agent, Avis


class AgentExpert(Agent):
    SEUIL_ARGUMENT = 0.25  # contribution minimale (log-odds) pour devenir un argument

    def __init__(self, nom: str, critere: str, variables: list):
        super().__init__(variables)
        self.nom, self.critere = nom, critere

    def fit(self, train: pd.DataFrame) -> "AgentExpert":
        self.pipeline = build_pipeline(LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE),
                                       self.variables).fit(train[self.variables], train[C.TARGET])
        self.prep = self.pipeline.named_steps["prep"]
        noms = self.prep.get_feature_names_out()  # coefficients associés aux variables PAR NOM
        self.coefs = dict(zip(noms, self.pipeline.named_steps["model"].coef_[0]))
        self.entraine = True
        return self

    def poids_appris(self) -> dict:
        """Importance relative de chaque variable pour cet agent (|β| normalisés)."""
        tot = sum(abs(b) for b in self.coefs.values())
        return {v: round(float(abs(b) / tot), 3) for v, b in self.coefs.items()}

    def noter(self, df: pd.DataFrame) -> np.ndarray:
        return 100 * self.pipeline.predict_proba(df[self.variables])[:, 1]

    def evaluer(self, candidat: dict) -> Avis:
        x = pd.DataFrame([{v: candidat[v] for v in self.variables}])
        p = float(self.pipeline.predict_proba(x)[0, 1])
        z = dict(zip(self.prep.get_feature_names_out(), self.prep.transform(x)[0]))
        contrib = {v: self.coefs[v] * z[v] for v in self.variables}
        forts = [self.libelle(v, candidat[v]) for v, c in contrib.items() if c >= self.SEUIL_ARGUMENT]
        faibles = [self.libelle(v, candidat[v]) for v, c in contrib.items() if c <= -self.SEUIL_ARGUMENT]
        note = 100 * p
        verdict = ("très favorable" if note >= 70 else "favorable" if note >= 50
                   else "réservé" if note >= 30 else "défavorable")
        texte = f"Avis {verdict} ({note:.0f}/100)."
        if forts:
            texte += " Atouts : " + ", ".join(forts) + "."
        if faibles:
            texte += " Points faibles : " + ", ".join(faibles) + "."
        return Avis(self.nom, round(note, 1), self.confiance(p), forts, faibles, texte,
                    {"contributions_logodds": {k: round(v, 3) for k, v in contrib.items()}})


def agent_tech() -> AgentExpert:
    return AgentExpert("Lead Tech", "score_technique", ["SkillScore", "EducationLevel"])


def agent_experience() -> AgentExpert:
    return AgentExpert("Expérience", "score_experience", ["ExperienceYears", "PreviousCompanies"])


def agent_rh() -> AgentExpert:
    return AgentExpert("RH", "score_rh", ["InterviewScore", "PersonalityScore"])


class AgentPredictif(Agent):
    """
    « Mémoire » des décisions passées : reprend le modèle et les hyperparamètres retenus à l'étape 3
    (sélection sur Validation), ré-entraîné ici sur le Train seul. N'utilise ni le genre ni l'âge.
    Justification locale par valeurs SHAP.
    """
    nom, critere = "Prédictif (ML)", "score_predictif"
    SEUIL_SHAP = 0.15

    def __init__(self):
        super().__init__(C.MODEL_FEATURES)
        path = C.RESULTS_DIR / "03_modele_retenu.json"
        if not path.exists():
            raise FileNotFoundError("Exécutez d'abord  python scripts/03_supervise.py")
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def fit(self, train: pd.DataFrame) -> "AgentPredictif":
        model, _ = model_catalog()[self.config["modele"]]
        model.set_params(**self.config["hyperparametres"])
        self.pipeline = build_pipeline(model, self.variables).fit(train[self.variables], train[C.TARGET])
        self._explainer = None
        self.entraine = True
        return self

    def __getstate__(self):  # l'explainer SHAP n'est pas sauvegardé, il est recréé à la demande
        return {k: v for k, v in self.__dict__.items() if k != "_explainer"} | {"_explainer": None}

    def noter(self, df: pd.DataFrame) -> np.ndarray:
        return 100 * self.pipeline.predict_proba(df[self.variables])[:, 1]

    def _shap(self, x: pd.DataFrame) -> dict:
        import shap
        prep, model = self.pipeline.named_steps["prep"], self.pipeline.named_steps["model"]
        xt = pd.DataFrame(prep.transform(x), columns=prep.get_feature_names_out())
        if self._explainer is None:
            self._explainer = (shap.LinearExplainer(model, xt) if hasattr(model, "coef_")
                               else shap.TreeExplainer(model))
        vals = np.asarray(self._explainer.shap_values(xt))
        vals = vals[..., 1] if vals.ndim == 3 else vals
        # on regroupe les colonnes one-hot (RecruitmentStrategy_1, _2, _3) sous leur variable d'origine
        out = {}
        for col, v in zip(xt.columns, vals[0]):
            base = next(f for f in self.variables if col == f or col.startswith(f + "_"))
            out[base] = out.get(base, 0.0) + float(v)
        return out

    def evaluer(self, candidat: dict, expliquer: bool = True) -> Avis:
        x = pd.DataFrame([{v: candidat[v] for v in self.variables}])
        p = float(self.pipeline.predict_proba(x)[0, 1])
        contrib = self._shap(x) if expliquer else {}
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:4]
        forts = [self.libelle(v, candidat[v]) for v, c in top if c >= self.SEUIL_SHAP]
        faibles = [self.libelle(v, candidat[v]) for v, c in top if c <= -self.SEUIL_SHAP]
        texte = f"Probabilité d'embauche d'après l'historique : {100 * p:.0f} %."
        if forts:
            texte += " Facteurs favorables (SHAP) : " + ", ".join(forts) + "."
        if faibles:
            texte += " Facteurs défavorables (SHAP) : " + ", ".join(faibles) + "."
        return Avis(self.nom, round(100 * p, 1), self.confiance(p), forts, faibles, texte,
                    {"shap": {k: round(v, 3) for k, v in contrib.items()}})

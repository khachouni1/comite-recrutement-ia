"""
Agents ÉVALUATEURS : chacun note le candidat sur UN critère, à partir de ses propres variables.

┌──────────────────────┬─────────────────────────────────────────────┬───────────────────────────────────┐
│ Agent                │ Ce qu'il observe (extrait du texte)         │ Comment il note                   │
├──────────────────────┼─────────────────────────────────────────────┼───────────────────────────────────┤
│ Compétences techn.   │ compétences de l'offre retrouvées dans le CV│ régression logistique apprise     │
│ Expérience           │ années détectées / années exigées           │ régression logistique apprise     │
│ Formation            │ diplôme détecté / diplôme exigé             │ régression logistique apprise     │
│ Soft skills          │ savoir-être demandés retrouvés dans le CV   │ régression logistique apprise     │
│ Adéquation au poste  │ similarités TF-IDF, LSA, embeddings         │ régression logistique apprise     │
│ Prédictif (ML)       │ toutes les variables                        │ meilleur modèle de l'étape 3      │
└──────────────────────┴─────────────────────────────────────────────┴───────────────────────────────────┘

Chaque expert apprend sur le Train :  note = 100 · P(candidature au moins « potentielle » | ses variables).
Ses poids internes sont donc APPRIS sur les données, et non fixés à la main.
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as C
from ..lexiques import DEGREE_NAMES
from ..preprocessing import build_pipeline
from ..supervised import catalogue, score_attendu
from .base import Agent, Avis


def _verdict(note: float) -> str:
    return ("très favorable" if note >= 75 else "favorable" if note >= 55
            else "réservé" if note >= 35 else "défavorable")


class AgentExpert(Agent):
    def __init__(self, critere: str):
        self.critere = critere
        self.nom = C.CRITERIA_LABELS[critere]
        self.variables = C.AGENT_FEATURES[critere]

    def fit(self, feats: pd.DataFrame, y: pd.Series) -> "AgentExpert":
        self.modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE))
        self.modele.fit(feats[self.variables], (y >= 1).astype(int))
        coefs = self.modele[-1].coef_[0]
        self.poids_appris = {v: round(float(abs(b) / np.abs(coefs).sum()), 3) for v, b in zip(self.variables, coefs)}
        self.coefs = dict(zip(self.variables, coefs))
        return self

    def noter(self, feats: pd.DataFrame) -> np.ndarray:
        return 100 * self.modele.predict_proba(feats[self.variables])[:, 1]

    def evaluer(self, feats: pd.DataFrame, details: dict) -> Avis:
        p = float(self.modele.predict_proba(feats[self.variables])[0, 1])
        note = 100 * p
        forts, faibles, texte = self._arguments(feats.iloc[0], details)
        return Avis(self.nom, self.critere, round(note, 1), self.confiance(p), forts, faibles,
                    f"Avis {_verdict(note)} ({note:.0f}/100). {texte}".strip())

    # arguments propres à chaque critère, lisibles par un recruteur
    def _arguments(self, f: pd.Series, d: dict):
        c = self.critere
        if c == "score_technique":
            forts, faibles = d["hard_trouvees"], d["hard_manquantes"]
            n_req = len(forts) + len(faibles)
            texte = (f"{len(forts)} compétence(s) demandée(s) sur {n_req} retrouvée(s) dans le CV."
                     if n_req else "L'offre ne cite pas de compétence technique reconnue.")
            return forts, faibles, texte
        if c == "score_soft_skills":
            forts, faibles = d["soft_trouvees"], d["soft_manquantes"]
            n_req = len(forts) + len(faibles)
            return forts, faibles, (f"{len(forts)} savoir-être demandé(s) sur {n_req} mis en avant dans le CV."
                                    if n_req else "L'offre ne précise pas de savoir-être.")
        if c == "score_experience":
            a, req = d["annees_cv"], d["annees_exigees"]
            if req is None:
                return ([f"{a:.0f} ans d'expérience détectés"] if a else []), [], \
                    f"{a:.0f} an(s) d'expérience détecté(s) ; l'offre ne fixe pas de minimum."
            ok = a >= req
            msg = f"{a:.0f} an(s) détecté(s) pour {req:.0f} exigé(s)."
            return ([msg] if ok else []), ([] if ok else [msg]), msg
        if c == "score_formation":
            dc, dr = d["diplome_cv"], d["diplome_exige"]
            msg = f"Diplôme détecté : {DEGREE_NAMES[dc]} ; exigé : {DEGREE_NAMES[dr]}."
            ok = dr == 0 or dc >= dr
            return ([msg] if ok and dc else []), ([] if ok else [msg]), msg
        # adéquation
        sim = f["sim_embedding"]
        msg = f"Similarité sémantique CV / offre = {sim:.2f}."
        return ([msg] if sim >= 0.5 else []), ([msg] if sim < 0.3 else []), msg


class AgentPredictif(Agent):
    """
    Reprend le modèle et les hyperparamètres retenus à l'étape 3, ré-entraîné sur le Train seul.
    Il note le 6e critère (« avis prédictif ») : la mémoire des décisions passées.
    """
    nom, critere = "Avis prédictif (ML)", "score_predictif"

    def __init__(self):
        path = C.RESULTS_DIR / "03_modele_retenu.json"
        if not path.exists():
            raise FileNotFoundError("Exécutez d'abord  python scripts/03_supervise.py")
        self.config = json.loads(path.read_text(encoding="utf-8"))
        self.variables = self.config["variables"]

    def fit(self, feats: pd.DataFrame, y: pd.Series) -> "AgentPredictif":
        model, _ = catalogue(riche=len(self.variables) > 50)[self.config["modele"]]
        model = clone(model).set_params(**self.config["hyperparametres"])
        self.modele = build_pipeline(model).fit(feats[self.variables], y)
        return self

    def probas(self, feats: pd.DataFrame) -> np.ndarray:
        return self.modele.predict_proba(feats[self.variables])

    def noter(self, feats: pd.DataFrame) -> np.ndarray:
        return score_attendu(self.probas(feats))

    def evaluer(self, feats: pd.DataFrame, details: dict | None = None) -> Avis:
        p = self.probas(feats)[0]
        note = float(score_attendu(p[None, :])[0])
        texte = (f"D'après les décisions passées : {p[2]:.0%} adapté, {p[1]:.0%} potentiel, "
                 f"{p[0]:.0%} non adapté (score attendu {note:.0f}/100).")
        return Avis(self.nom, self.critere, round(note, 1), round(float(p.max()), 3), texte=texte,
                    details={"probas": {C.LABELS[k]: round(float(v), 3) for k, v in enumerate(p)}})


def evaluateurs() -> dict:
    """Les 6 agents notateurs, dans l'ordre des critères : 5 experts explicables + l'agent prédictif."""
    agents = {c: AgentExpert(c) for c in C.AGENT_FEATURES}
    agents["score_predictif"] = AgentPredictif()
    return agents

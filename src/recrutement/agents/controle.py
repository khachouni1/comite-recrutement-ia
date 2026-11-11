"""
Agents de CONTRÔLE : ils ne notent pas, ils vérifient.

1. Agent Anomalies (Cours 2, slides 82-85) : un CV dont le profil est très improbable sous la loi
   gaussienne multivariée apprise sur le Train (p(x) < ε) est signalé : extraction PDF ratée,
   CV vide, valeurs aberrantes... La décision est alors confiée à un humain.

2. Agent Éthique :
   a) anonymisation : il recense les informations personnelles retirées du CV avant l'évaluation ;
   b) test contrefactuel : on ajoute au CV des informations protégées (genre, âge, situation familiale)
      et on vérifie que les notes du comité ne bougent pas.
   Limite : le jeu de données ne contient pas d'attributs démographiques, donc pas d'audit par groupe.
"""
import re

import numpy as np
import pandas as pd

from .. import config as C
from ..texte import _ANONYMISATION
from ..unsupervised import GaussianAnomalyDetector
from .base import Agent, Avis

VARIABLES_CV = ["longueur_cv", "n_hard_cv", "n_soft_cv", "annees_cv", "diplome_cv"]


class AgentAnomalies(Agent):
    nom = "Contrôle qualité du dossier"
    variables = VARIABLES_CV

    def fit(self, feats: pd.DataFrame, y=None) -> "AgentAnomalies":
        self.detecteur = GaussianAnomalyDetector(VARIABLES_CV).fit(feats)
        return self

    def noter(self, feats):
        return self.detecteur.predict(feats)

    def evaluer(self, feats: pd.DataFrame, details=None) -> Avis:
        log_p = float(self.detecteur.log_density(feats)[0])
        anomalie = log_p < self.detecteur.log_epsilon_
        z = self.detecteur.zscores(feats).iloc[0]
        extremes = [f"{v} = {feats[v].iloc[0]:g} (z = {z[v]:+.1f})" for v in VARIABLES_CV if abs(z[v]) >= 2.5]
        if feats["longueur_cv"].iloc[0] < 80:
            anomalie, extremes = True, extremes + ["CV très court : texte illisible ou incomplet ?"]
        texte = ("Dossier atypique : vérification humaine recommandée." if anomalie
                 else "Dossier cohérent avec les CV de référence.")
        return Avis(self.nom, None, None, 1.0, [], extremes, texte,
                    {"anomalie": bool(anomalie), "log_densite": round(log_p, 2)})


_INJECTIONS = [
    "She is a 58 years old woman, married, mother of three. Date of birth: 1967. Nationality: Moroccan.",
    "He is a 24 years old man, single. Mr. Date of birth: 2001. Religion: none.",
]


class AgentEthique(Agent):
    nom = "Éthique & équité"
    TOLERANCE = 1.0   # écart maximal admis sur une note /100

    def fit(self, feats=None, y=None):
        return self

    def noter(self, feats):
        raise NotImplementedError("L'agent éthique ne note pas les candidats.")

    @staticmethod
    def infos_retirees(cv_brut: str) -> list:
        """Types d'informations personnelles trouvées (et retirées) dans le CV brut."""
        return [nom for nom, motif in _ANONYMISATION if re.search(motif, cv_brut, re.IGNORECASE)]

    def test_contrefactuel(self, noter_paire, cv_brut: str, offre_brute: str) -> dict:
        """noter_paire(cv, offre) -> dict des notes par critère ; on compare avec les CV « modifiés »."""
        ref = noter_paire(cv_brut, offre_brute)
        ecart = 0.0
        for phrase in _INJECTIONS:
            notes = noter_paire(phrase + " " + cv_brut, offre_brute)
            ecart = max(ecart, max(abs(notes[c] - ref[c]) for c in ref))
        return {"ok": ecart <= self.TOLERANCE, "ecart_max": round(float(ecart), 3)}

    def evaluer(self, feats=None, details: dict | None = None) -> Avis:
        details = details or {}
        retirees, cf = details.get("infos_retirees", []), details.get("contrefactuel")
        morceaux = [("Informations personnelles masquées avant évaluation : " + ", ".join(retirees) + ".")
                    if retirees else "Aucune information personnelle sensible détectée dans le CV."]
        if cf:
            morceaux.append("Test contrefactuel réussi : ajouter un genre, un âge ou une situation familiale "
                            f"ne change pas les notes (écart max {cf['ecart_max']:.2f} pt)." if cf["ok"] else
                            f"ALERTE : les notes varient de {cf['ecart_max']:.1f} pt selon des informations protégées.")
        return Avis(self.nom, None, None, 1.0, [], [] if not cf or cf["ok"] else ["contrefactuel"], " ".join(morceaux),
                    {"infos_retirees": retirees, "contrefactuel": cf})

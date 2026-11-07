"""
Agents de CONTRÔLE : ils ne notent pas les candidats, ils vérifient le travail du comité.

1. Agent Anomalies (Cours 2, slides 82-85) : un dossier hors distribution (densité gaussienne
   multivariée p(x) < ε) est signalé pour vérification humaine.

2. Agent Éthique : garantit que la décision ne dépend ni du genre ni de l'âge.
   a) Test contrefactuel : on change UNIQUEMENT le genre (ou l'âge ± 10 ans) d'un candidat ;
      les notes de tous les agents évaluateurs doivent rester identiques.
   b) Règle des 4/5 (Disparate Impact) : DI = taux de sélection min / max entre groupes ; alerte si DI < 0.80.
   c) Détection de variables « proxy » : corrélation de Spearman entre chaque note et genre / âge.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config as C
from ..unsupervised import GaussianAnomalyDetector
from .base import Agent, Avis


class AgentAnomalies(Agent):
    nom = "Contrôle anomalies"

    def __init__(self, seuil_z: float = 2.5):
        self.detecteur = GaussianAnomalyDetector()
        super().__init__(self.detecteur.features)
        self.seuil_z = seuil_z

    def fit(self, train: pd.DataFrame) -> "AgentAnomalies":
        self.detecteur.fit(train)
        self.entraine = True
        return self

    def anomalies(self, df: pd.DataFrame) -> np.ndarray:
        return self.detecteur.predict(df)

    def evaluer(self, candidat: dict) -> Avis:
        x = pd.DataFrame([candidat])
        log_p = float(self.detecteur.log_density(x)[0])
        anomalie = log_p < self.detecteur.log_epsilon_
        z = self.detecteur.zscores(x).iloc[0]
        extremes = [f"{C.LABELS[v]} = {candidat[v]:g} (z = {z[v]:+.1f})" for v in self.variables
                    if abs(z[v]) >= self.seuil_z]
        texte = ("Dossier atypique (p(x) < ε) : vérification humaine recommandée." if anomalie
                 else "Dossier cohérent avec la population de référence.")
        if anomalie and extremes:
            texte += " Valeurs extrêmes : " + ", ".join(extremes) + "."
        return Avis(self.nom, None, 1.0, [], extremes, texte,
                    {"anomalie": bool(anomalie), "log_densite": round(log_p, 2),
                     "log_epsilon": round(self.detecteur.log_epsilon_, 2)})


def tranche_age(age) -> str:
    return "< 30" if age < 30 else ("30-39" if age < 40 else "≥ 40")


class AgentEthique(Agent):
    nom = "Éthique & équité"
    TOLERANCE = 1e-6

    def __init__(self, evaluateurs: list):
        super().__init__(C.PROTECTED)
        self.evaluateurs = evaluateurs
        self.entraine = True

    @staticmethod
    def _variantes(df: pd.DataFrame) -> list:
        return [df.assign(Gender=1 - df["Gender"]),
                df.assign(Age=(df["Age"] - 10).clip(lower=18)),
                df.assign(Age=(df["Age"] + 10).clip(upper=70))]

    def test_contrefactuel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Écart maximal de note provoqué par un changement de genre ou d'âge, pour chaque candidat."""
        ecart = np.zeros(len(df))
        sensibles = [set() for _ in range(len(df))]
        for agent in self.evaluateurs:
            ref = agent.noter(df)
            for var in self._variantes(df):
                d = np.abs(agent.noter(var) - ref)
                ecart = np.maximum(ecart, d)
                for i in np.where(d > self.TOLERANCE)[0]:
                    sensibles[i].add(agent.nom)
        return pd.DataFrame({"contrefactuel_ok": ecart <= self.TOLERANCE, "ecart_contrefactuel": ecart.round(4),
                             "agents_sensibles": [", ".join(sorted(s)) for s in sensibles]}, index=df.index)

    def evaluer(self, candidat: dict) -> Avis:
        r = self.test_contrefactuel(pd.DataFrame([candidat])).iloc[0]
        texte = ("Test contrefactuel réussi : changer le genre ou l'âge ne modifie aucune note."
                 if r["contrefactuel_ok"] else f"ALERTE : notes sensibles au genre/âge ({r['agents_sensibles']}).")
        return Avis(self.nom, None, 1.0, [], [] if r["contrefactuel_ok"] else [r["agents_sensibles"]], texte,
                    {"contrefactuel_ok": bool(r["contrefactuel_ok"])})

    def audit_selection(self, df: pd.DataFrame, colonne_selection: str = "selectionne") -> dict:
        data = df.assign(Genre=df["Gender"].map({0: "Femme", 1: "Homme"}),
                         Tranche_age=df["Age"].apply(tranche_age))
        rapport = {"disparate_impact": {}, "proxies": {}, "alertes": []}
        for groupe in ("Genre", "Tranche_age"):
            taux = data.groupby(groupe)[colonne_selection].mean()
            di = float(taux.min() / taux.max()) if taux.max() > 0 else 1.0
            rapport["disparate_impact"][groupe] = {"taux_selection": taux.round(3).to_dict(),
                                                   "DI": round(di, 3), "conforme": di >= C.DI_THRESHOLD}
            if di < C.DI_THRESHOLD:
                rapport["alertes"].append(f"Règle des 4/5 non respectée pour {groupe} (DI = {di:.2f}).")
        for crit in C.CRITERIA:
            for attr in C.PROTECTED:
                rho = spearmanr(data[crit], data[attr])[0]
                rapport["proxies"][f"{crit} ~ {attr}"] = round(float(rho), 3)
                if abs(rho) > 0.2:
                    rapport["alertes"].append(f"Proxy possible : {crit} corrélé à {attr} (ρ = {rho:.2f}).")
        rapport["conforme"] = not rapport["alertes"]
        return rapport

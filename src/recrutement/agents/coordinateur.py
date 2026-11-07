"""
COORDINATEUR : organise la délibération du comité et rend la décision finale.

    ┌──────────────┐   avis (note, confiance, arguments)   ┌──────────────────────────────┐
    │ Lead Tech    │──┐                                     │ COORDINATEUR                 │
    │ Expérience   │──┼────────────────────────────────────►│ 1. percentiles & désaccords  │
    │ RH           │──┤                                     │ 2. droit de véto             │
    │ Prédictif ML │──┘                                     │ 3. poids AHP (CR < 0.10)     │
    └──────────────┘                                        │ 4. WSM (baseline) & TOPSIS   │
    ┌──────────────┐   signalements                         │ 5. décision                  │
    │ Anomalies    │──────────────────────────────────────►│ 6. audit éthique             │
    │ Éthique      │◄──── audit de la sélection ────────────└──────────────────────────────┘
    └──────────────┘

Décisions possibles :
  • Écarté (véto) : un agent cœur de métier (Tech, RH) a une note sous le 5e percentile du Train ;
  • Recommandé    : dans la short-list TOPSIS (K places) et avis convergents ;
  • À examiner    : dans la short-list mais désaccord fort entre agents ou dossier atypique ;
  • Non retenu    : hors de la short-list.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..mcdm import AHP, topsis, wsm
from .controle import AgentAnomalies, AgentEthique
from .evaluateurs import AgentPredictif, agent_experience, agent_rh, agent_tech

RECOMMANDE, EXAMINER, VETO, NON_RETENU = "Recommandé", "À examiner", "Écarté (véto)", "Non retenu"


class Coordinateur:
    def __init__(self, matrice_ahp=None):
        self.evaluateurs = {a.critere: a for a in (agent_tech(), agent_experience(), agent_rh(), AgentPredictif())}
        self.anomalies = AgentAnomalies()
        self.ethique = AgentEthique([self.evaluateurs[c] for c in C.CRITERIA])
        self.ahp = AHP(C.CRITERIA, C.AHP_MATRIX if matrice_ahp is None else matrice_ahp)
        if not self.ahp.is_consistent:
            raise ValueError(f"Jugements AHP incohérents (CR = {self.ahp.cr:.3f} ≥ 0.10).")
        self.poids = self.ahp.weights

    # ------------------------------------------------------------ apprentissage (Train uniquement)
    def fit(self, train: pd.DataFrame) -> "Coordinateur":
        for agent in self.evaluateurs.values():
            agent.fit(train)
        self.anomalies.fit(train)
        # distribution des notes de chaque agent sur le Train -> percentiles comparables entre agents
        self.distributions = {c: np.sort(self.evaluateurs[c].noter(train)) for c in C.CRITERIA}
        self.seuils_veto = {c: float(np.quantile(self.distributions[c], C.VETO_PERCENTILE))
                            for c in C.VETO_CRITERIA}
        self.taux_embauche = float(train[C.TARGET].mean())
        return self

    def percentile(self, critere: str, notes) -> np.ndarray:
        d = self.distributions[critere]
        return np.searchsorted(d, np.asarray(notes, dtype=float), side="right") / len(d)

    def nb_places(self, n: int) -> int:
        """Taille de la short-list : autant de places que le taux d'embauche historique."""
        return int(round(n * self.taux_embauche))

    # ------------------------------------------------------------ 1. délibération
    def deliberer(self, df: pd.DataFrame, expliquer: bool = True) -> pd.DataFrame:
        lignes = []
        for idx, cand in zip(df.index, df.to_dict("records")):
            ligne = {"id": idx}
            for c in C.CRITERIA:
                agent = self.evaluateurs[c]
                avis = agent.evaluer(cand, expliquer) if isinstance(agent, AgentPredictif) else agent.evaluer(cand)
                court = c.replace("score_", "")
                ligne[c], ligne[f"confiance_{court}"], ligne[f"avis_{court}"] = avis.note, avis.confiance, avis.texte
            ctrl = self.anomalies.evaluer(cand)
            ligne["anomalie"], ligne["avis_anomalie"] = ctrl.details["anomalie"], ctrl.texte
            lignes.append(ligne)
        out = pd.DataFrame(lignes).set_index("id")

        pct = pd.DataFrame({c: self.percentile(c, out[c]) for c in C.CRITERIA}, index=out.index)
        out["ecart_avis"] = (pct.max(axis=1) - pct.min(axis=1)).round(3)
        out["desaccord"] = out["ecart_avis"] >= C.DISAGREEMENT_GAP
        out["veto"] = [", ".join(C.CRITERIA_LABELS[c] for c, s in self.seuils_veto.items() if out.at[i, c] < s)
                       for i in out.index]
        return out.join(self.ethique.test_contrefactuel(df))

    # ------------------------------------------------------------ 2. agrégation MCDM + décision
    def decider(self, delib: pd.DataFrame, poids: dict | None = None, nb_places: int | None = None) -> pd.DataFrame:
        poids = poids or self.poids
        k = nb_places or self.nb_places(len(delib))
        out = delib.join(wsm(delib, poids)).join(topsis(delib, poids)[["TOPSIS_score", "TOPSIS_rang"]])
        eligibles = out["veto"] == ""
        out["rang_eligible"] = out["TOPSIS_score"].where(eligibles).rank(ascending=False, method="first")

        decisions, justifications = [], []
        for _, r in out.iterrows():
            if r["veto"]:
                d, j = VETO, f"Véto ({r['veto']}) : note sous le minimum requis pour le poste."
            elif r["rang_eligible"] <= k:
                motifs = (["désaccord fort entre agents"] if r["desaccord"] else []) + \
                         (["dossier atypique"] if r["anomalie"] else [])
                if motifs:
                    d, j = EXAMINER, f"Dans la short-list ({k} places) mais {' et '.join(motifs)} : arbitrage humain."
                else:
                    d, j = RECOMMANDE, f"Rang TOPSIS {int(r['TOPSIS_rang'])}, avis convergents des agents."
            else:
                d, j = NON_RETENU, f"Rang TOPSIS {int(r['TOPSIS_rang'])} : hors des {k} places."
            decisions.append(d)
            justifications.append(j)
        out["decision"], out["justification"] = decisions, justifications
        out["selectionne"] = out["decision"].isin([RECOMMANDE, EXAMINER])
        return out.sort_values("TOPSIS_rang")

    # ------------------------------------------------------------ 3. candidat isolé (application)
    def evaluer_candidat(self, candidat: dict, vivier: pd.DataFrame, poids: dict | None = None) -> dict:
        """Évalue un nouveau candidat et le classe par rapport à un vivier déjà évalué."""
        poids = poids or self.poids
        avis = {c: self.evaluateurs[c].evaluer(candidat) for c in C.CRITERIA}
        notes = {c: avis[c].note for c in C.CRITERIA}
        pct = {c: float(self.percentile(c, [notes[c]])[0]) for c in C.CRITERIA}
        veto = [C.CRITERIA_LABELS[c] for c, s in self.seuils_veto.items() if notes[c] < s]
        ecart = max(pct.values()) - min(pct.values())
        anomalie = self.anomalies.evaluer(candidat)

        ensemble = pd.concat([vivier[C.CRITERIA], pd.DataFrame([notes], index=["NOUVEAU"])])
        res = topsis(ensemble, poids)
        rang = int(res.loc["NOUVEAU", "TOPSIS_rang"])
        k = self.nb_places(len(ensemble))
        if veto:
            decision = VETO
        elif rang <= k:
            decision = EXAMINER if (ecart >= C.DISAGREEMENT_GAP or anomalie.details["anomalie"]) else RECOMMANDE
        else:
            decision = NON_RETENU
        return {"avis": avis, "percentiles": pct, "ecart_avis": ecart, "veto": veto, "anomalie": anomalie,
                "ethique": self.ethique.evaluer(candidat), "topsis_score": float(res.loc["NOUVEAU", "TOPSIS_score"]),
                "rang": rang, "taille_vivier": len(ensemble), "nb_places": k, "decision": decision}

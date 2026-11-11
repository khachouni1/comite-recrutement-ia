"""
COORDINATEUR : organise la délibération et rend la décision.

          CV + offre (texte brut)
                  │  nettoyage + ANONYMISATION
                  ▼
     extraction des critères + similarités
                  │
   ┌──────────────┼───────────────────────────────────────────────┐
   │ 5 agents évaluateurs (technique, expérience, formation,      │
   │ soft skills, adéquation) : note /100 + confiance + arguments │
   └──────────────┬───────────────────────────────────────────────┘
                  ▼
   COORDINATEUR : 1. véto (technique ou adéquation sous le minimum)
                  2. poids AHP (CR < 0.10)  ->  score TOPSIS (référence fixe = Train) ; WSM = baseline
                  3. seuils de décision calibrés sur la Validation
                  4. second avis de l'agent prédictif ML : en cas de désaccord -> « À examiner »
                  5. contrôles : anomalies du dossier, éthique (anonymisation, contrefactuel)
                  ▼
   Recommandé  |  À examiner  |  Non retenu  |  Écarté (véto)
"""
import numpy as np
import pandas as pd

from .. import config as C
from .. import features as F
from ..mcdm import AHP, topsis, wsm
from ..texte import anonymiser, nettoyer
from .controle import AgentAnomalies, AgentEthique
from .evaluateurs import evaluateurs

RECOMMANDE, EXAMINER, NON_RETENU, VETO = "Recommandé", "À examiner", "Non retenu", "Écarté (véto)"


class Coordinateur:
    def __init__(self, representations, matrice_ahp=None):
        self.reps = representations
        self.evaluateurs = evaluateurs()                 # 6 notateurs (5 experts + agent prédictif)
        self.predictif = self.evaluateurs["score_predictif"]
        self.anomalies = AgentAnomalies()
        self.ethique = AgentEthique()
        self.ahp = AHP(C.CRITERIA, C.AHP_MATRIX if matrice_ahp is None else matrice_ahp)
        if not self.ahp.is_consistent:
            raise ValueError(f"Jugements AHP incohérents (CR = {self.ahp.cr:.3f} ≥ 0.10).")
        self.poids = self.ahp.weights

    # =================================================================== apprentissage & calibrage
    def fit(self, feats: pd.DataFrame, y: pd.Series) -> "Coordinateur":
        """Apprentissage de tous les agents sur le Train."""
        for agent in self.evaluateurs.values():
            agent.fit(feats, y)
        self.anomalies.fit(feats)
        self.reference = self.notes(feats)  # référence fixe de TOPSIS (idéal / anti-idéal du Train)
        bons = self.reference[y.values == 2]
        self.seuils_veto = {c: float(np.quantile(bons[c], C.VETO_QUANTILE)) for c in C.VETO_CRITERIA}
        return self

    def calibrer(self, feats: pd.DataFrame, y: pd.Series) -> dict:
        """
        Seuils de décision choisis sur la VALIDATION, avec des objectifs lisibles par un recruteur :
          - « Recommandé » (score ≥ seuil_reco) : le plus bas seuil tel qu'au moins PRECISION_RECO des
            candidats au-dessus soient réellement adéquats (potentiel ou adapté) ;
          - « Non retenu » (score < seuil_rejet) : le plus haut seuil tel qu'au moins PRECISION_REJET des
            candidats en dessous soient réellement non adaptés ;
          - entre les deux : « À examiner ».
        """
        s = self.score_global(self.notes(feats))
        y = np.asarray(y)
        grille = np.quantile(s, np.linspace(0.02, 0.98, 97))
        ok_reco = [t for t in grille if (y[s >= t] >= 1).mean() >= C.PRECISION_RECO]
        ok_rejet = [t for t in grille if (s < t).sum() >= 10 and (y[s < t] == 0).mean() >= C.PRECISION_REJET]
        self.seuil_reco = float(min(ok_reco)) if ok_reco else float(grille[-1])
        self.seuil_rejet = float(min(max(ok_rejet) if ok_rejet else grille[0], self.seuil_reco))
        reco, rejet = s >= self.seuil_reco, s < self.seuil_rejet
        return {"seuil_recommande": self.seuil_reco, "seuil_non_retenu": self.seuil_rejet,
                "part_recommandes": float(reco.mean()), "precision_recommandes": float((y[reco] >= 1).mean()),
                "part_non_retenus": float(rejet.mean()), "precision_non_retenus": float((y[rejet] == 0).mean())}

    # =================================================================== notes & décision (lot)
    def notes(self, feats: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: a.noter(feats) for c, a in self.evaluateurs.items()}, index=feats.index)

    def score_global(self, notes: pd.DataFrame, poids: dict | None = None) -> np.ndarray:
        """Score TOPSIS à référence fixe, ramené sur 100."""
        return 100 * topsis(notes, poids or self.poids, reference=self.reference)["TOPSIS_score"].to_numpy()

    def decider(self, feats: pd.DataFrame, poids: dict | None = None) -> pd.DataFrame:
        poids = poids or self.poids
        out = self.notes(feats)
        out["score_global"] = self.score_global(out, poids)
        out["score_wsm"] = 100 * wsm(out, poids, reference=self.reference)["WSM_score"]
        probas = self.predictif.probas(feats)
        out["p_non_adapte"], out["p_adapte"] = probas[:, 0], probas[:, 2]
        out["anomalie"] = self.anomalies.noter(feats)
        out["veto"] = [", ".join(C.CRITERIA_LABELS[c] for c, s in self.seuils_veto.items() if r[c] < s)
                       for _, r in out.iterrows()]
        decisions, motifs = zip(*[self._regle(r) for _, r in out.iterrows()])
        out["decision"], out["justification"] = decisions, motifs
        return out

    def _regle(self, r) -> tuple[str, str]:
        if r["veto"]:
            return VETO, f"Critère éliminatoire non atteint : {r['veto']}."
        s = r["score_global"]
        if s >= self.seuil_reco:
            base = RECOMMANDE
        elif s >= self.seuil_rejet:
            return EXAMINER, f"Score {s:.0f}/100 dans la zone intermédiaire : un recruteur doit trancher."
        else:
            base = NON_RETENU
        # second avis de l'agent prédictif : désaccord fort -> arbitrage humain
        if base == RECOMMANDE and r["p_non_adapte"] >= C.DISAGREEMENT_PROBA:
            return EXAMINER, (f"Score {s:.0f}/100 mais l'agent prédictif juge le profil non adapté "
                              f"({r['p_non_adapte']:.0%}) : désaccord, arbitrage humain.")
        if base == NON_RETENU and r["p_adapte"] >= C.DISAGREEMENT_PROBA:
            return EXAMINER, (f"Score {s:.0f}/100 mais l'agent prédictif juge le profil adapté "
                              f"({r['p_adapte']:.0%}) : désaccord, arbitrage humain.")
        if r["anomalie"]:
            return EXAMINER, "Dossier atypique signalé par le contrôle qualité : vérification humaine."
        if base == RECOMMANDE:
            return RECOMMANDE, f"Score {s:.0f}/100 au-dessus du seuil de recommandation ({self.seuil_reco:.0f})."
        return NON_RETENU, f"Score {s:.0f}/100 sous le seuil ({self.seuil_rejet:.0f})."

    # =================================================================== site web : une paire
    def preparer_paire(self, cv_brut: str, offre_brute: str):
        cv, offre = anonymiser(nettoyer(cv_brut)), anonymiser(nettoyer(offre_brute))
        return F.construire_une(cv, offre, self.reps)

    def evaluer(self, cv_brut: str, offre_brute: str, poids: dict | None = None,
                test_ethique: bool = True) -> dict:
        """Rapport complet pour un CV face à une offre (texte brut, tel que déposé sur le site)."""
        feats, det = self.preparer_paire(cv_brut, offre_brute)
        avis = {c: a.evaluer(feats, det) for c, a in self.evaluateurs.items()}
        dec = self.decider(feats, poids).iloc[0]
        cf = None
        if test_ethique:
            cf = self.ethique.test_contrefactuel(
                lambda c, o: self.notes(self.preparer_paire(c, o)[0]).iloc[0].to_dict(), cv_brut, offre_brute)
        ethique = self.ethique.evaluer(details={"infos_retirees": self.ethique.infos_retirees(cv_brut),
                                                "contrefactuel": cf})
        return {"avis": avis, "predictif": self.predictif.evaluer(feats), "anomalie": self.anomalies.evaluer(feats),
                "ethique": ethique, "decision": dec["decision"], "justification": dec["justification"],
                "score_global": float(dec["score_global"]), "score_wsm": float(dec["score_wsm"]),
                "veto": dec["veto"], "details": det, "variables": feats.iloc[0].to_dict()}

    # =================================================================== site web : plusieurs CV, une offre
    def classer(self, cvs_bruts: dict, offre_brute: str, poids: dict | None = None) -> pd.DataFrame:
        """Classe plusieurs candidats pour une même offre (TOPSIS classique entre eux + décision absolue)."""
        poids = poids or self.poids
        feats = pd.concat([self.preparer_paire(t, offre_brute)[0] for t in cvs_bruts.values()])
        feats.index = list(cvs_bruts)
        out = self.decider(feats, poids)
        out["rang"] = topsis(out[C.CRITERIA], poids)["TOPSIS_rang"]
        return out.sort_values("rang")

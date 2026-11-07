"""
Analyse de sensibilité : le classement TOPSIS est-il robuste aux poids choisis ?

Pour chaque critère, on multiplie son poids par (1 + δ), δ ∈ [−50 %, +50 %], puis on renormalise
les autres poids proportionnellement pour garder Σ w = 1. On mesure :
  - la corrélation de Spearman entre le nouveau classement et le classement de référence ;
  - le recouvrement de la short-list : |Top-K nouveau ∩ Top-K référence| / K.
"""
import pandas as pd
from scipy.stats import spearmanr

from .topsis import topsis


def perturb(weights: dict, criterion: str, delta: float) -> dict:
    new_w = min(max(weights[criterion] * (1 + delta), 1e-6), 0.999)
    rest = sum(v for k, v in weights.items() if k != criterion)
    return {k: (new_w if k == criterion else v / rest * (1 - new_w)) for k, v in weights.items()}


def sensitivity(scores: pd.DataFrame, weights: dict, top_k: int,
                deltas=(-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5)) -> pd.DataFrame:
    ref = topsis(scores, weights)
    ref_top = set(ref.nsmallest(top_k, "TOPSIS_rang").index)
    rows = []
    for crit in weights:
        for d in deltas:
            w = perturb(weights, crit, d)
            res = topsis(scores, w)
            top = set(res.nsmallest(top_k, "TOPSIS_rang").index)
            rows.append({"critere": crit, "variation": d, "poids": w[crit],
                         "spearman": spearmanr(ref["TOPSIS_rang"], res["TOPSIS_rang"])[0],
                         "recouvrement_topK": len(top & ref_top) / top_k})
    return pd.DataFrame(rows)

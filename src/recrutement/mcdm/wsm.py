"""
WSM — Weighted Sum Model : la baseline MCDM simple.

    1. Normalisation min-max de chaque critère :  r_ij = (x_ij − min_j) / (max_j − min_j)
    2. Score :  S_i = Σ_j  w_j · r_ij
Limite : méthode compensatoire — une très bonne note peut masquer une très mauvaise.
"""
import pandas as pd


def wsm(scores: pd.DataFrame, weights: dict) -> pd.DataFrame:
    crit = list(weights)
    x = scores[crit].astype(float)
    r = (x - x.min()) / (x.max() - x.min()).replace(0, 1)
    out = pd.DataFrame(index=scores.index)
    out["WSM_score"] = r.mul(pd.Series(weights)).sum(axis=1)
    out["WSM_rang"] = out["WSM_score"].rank(ascending=False, method="min").astype(int)
    return out

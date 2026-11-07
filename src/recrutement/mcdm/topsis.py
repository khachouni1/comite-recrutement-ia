"""
TOPSIS — Technique for Order Preference by Similarity to Ideal Solution.

    1. Normalisation vectorielle :         r_ij = x_ij / sqrt(Σ_i x_ij²)
    2. Pondération :                       v_ij = w_j · r_ij
    3. Solution idéale A+ = max_i v_ij ;   solution anti-idéale A− = min_i v_ij  (critères à maximiser)
    4. Distances euclidiennes :            D_i+ = ||v_i − A+||,   D_i− = ||v_i − A−||
    5. Coefficient de proximité :          C_i = D_i− / (D_i+ + D_i−)  ∈ [0, 1]   (1 = idéal)
Tous nos critères sont des critères « bénéfice » (plus la note est haute, mieux c'est).
"""
import numpy as np
import pandas as pd


def topsis(scores: pd.DataFrame, weights: dict) -> pd.DataFrame:
    crit = list(weights)
    x = scores[crit].to_numpy(dtype=float)
    w = np.array([weights[c] for c in crit])
    norm = np.sqrt((x ** 2).sum(axis=0))
    v = x / np.where(norm == 0, 1, norm) * w
    ideal, anti = v.max(axis=0), v.min(axis=0)
    d_plus = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((v - anti) ** 2).sum(axis=1))
    out = pd.DataFrame(index=scores.index)
    out["D_plus"] = d_plus
    out["D_moins"] = d_minus
    out["TOPSIS_score"] = d_minus / np.where(d_plus + d_minus == 0, 1, d_plus + d_minus)
    out["TOPSIS_rang"] = out["TOPSIS_score"].rank(ascending=False, method="min").astype(int)
    return out
